import math
import os
from dataclasses import dataclass, replace
from typing import Optional

import mujoco
import numpy as np
from gymnasium import spaces

from .drone_aero_config import DEFAULT_DRONE_MODEL, get_drone_model_profile
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
from orca_gym.devices.xbox_joystick import XboxJoystick, XboxJoystickManager
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.utils import rotations

# 抑制 pygame 启动 banner
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

_logger = get_orca_logger()


class ControlDevice:
    KEYBOARD = "keyboard"
    XBOX = "xbox"


def _normalize_axis(vec: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return np.asarray(fallback, dtype=np.float64)
    return arr / norm


@dataclass(frozen=True)
class RotorSpec:
    joint_suffix: str
    spin_sign: float


class DroneOrcaEnv(OrcaGymEulerEnv):
    """自由飞行刚体：旋翼关节仅动画。

    - **vertical_z_only**：世界 +Z 标量推力（经体轴换算写入 xfrc），可选 WASD 世界系水平力；姿态可锁。
    - **全量（非 vertical_z_only）**：集体升力沿 **drone_frame 机体系 +Z**，经 **R 旋到世界系** 写入 `xfrc_applied`（MuJoCo 为世界系）；W/A/S/D 倾转 PD + 可选世界系水平键盘力；Q/E 偏航阻尼。"""

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,
        scene_binding: Optional[dict] = None,
        autoplay: bool = False,
        vertical_z_only_physics: bool = False,
        vertical_thrust_ramp: bool = False,
        vertical_ramp_t0_factor: float = 0.65,
        vertical_ramp_t1_factor: float = 2.05,
        vertical_ramp_duration_s: float = 25.0,
        vertical_lock_quat_world_up: bool = True,
        vertical_fixed_thrust_over_hover: float = -1.0,
        vertical_keyboard_baseline_tmg: float = 1.0022,
        vertical_keyboard_xy_force_factor: Optional[float] = None,
        reset_height_offset_m: float = 0.25,
        fullmode_reset_thrust_ramp_s: float = 0.8,
        fullmode_reset_thrust_start_factor: float = 0.2,
        fullmode_reset_minimal_stab_s: float = 0.35,
        drone_model: str = DEFAULT_DRONE_MODEL,
        diag_logs_enabled: bool = True,
        diag_every_env_steps: int = 0,
        ctrl_device: str = ControlDevice.KEYBOARD,
        **kwargs,
    ):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

        self.nu = int(self.model.nu)
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self._physics_dt = float(time_step)
        self._control_dt = float(time_step) * int(frame_skip)
        self._scene_binding = scene_binding or {}
        self._autoplay_enabled = bool(autoplay)
        self._autoplay_time = 0.0
        self._diag_logs_enabled = bool(diag_logs_enabled)
        self._model_profile = get_drone_model_profile(drone_model)
        self._reset_height_offset_m = max(0.0, float(reset_height_offset_m))
        self._fullmode_reset_thrust_ramp_s = max(0.0, float(fullmode_reset_thrust_ramp_s))
        self._fullmode_reset_thrust_start_factor = float(np.clip(fullmode_reset_thrust_start_factor, 0.0, 1.0))
        self._fullmode_reset_minimal_stab_s = max(0.0, float(fullmode_reset_minimal_stab_s))

        self._ctrl_device = ctrl_device
        self._keyboard: Optional[KeyboardInput] = None
        self._joystick_manager: Optional[XboxJoystickManager] = None
        self._joystick: Optional[XboxJoystick] = None
        if self._ctrl_device == ControlDevice.XBOX:
            self._joystick_manager = XboxJoystickManager()
            joystick_names = self._joystick_manager.get_joystick_names()
            if len(joystick_names) == 0:
                _logger.warning("[DroneOrcaEnv] 未检测到手柄，回退到键盘控制")
                self._ctrl_device = ControlDevice.KEYBOARD
                self._joystick_manager = None
            else:
                self._joystick = self._joystick_manager.get_joystick(joystick_names[0])
                _logger.info(f"[DroneOrcaEnv] 手柄控制已启用: {joystick_names[0]}")
        if self._ctrl_device == ControlDevice.KEYBOARD:
            self._keyboard = KeyboardInput(KeyboardInputSourceType.ORCASTUDIO, orcagym_addr)
        self._last_space_state = 0
        self._last_a_button_state = 0

        self._free_joint_suffix = "drone_free"
        self._free_joint = self._resolve_name("joints", self._free_joint_suffix)
        self._primary_rotor_specs = [
            RotorSpec("FL_joint", 1.0),
            RotorSpec("FR_joint", -1.0),
            RotorSpec("BL_joint", -1.0),
            RotorSpec("BR_joint", 1.0),
        ]
        self._secondary_rotor_map: dict[str, str] = {}
        joints_by_suffix = self._scene_binding.get("joints_by_suffix", {})
        for ps in self._primary_rotor_specs:
            sec_name = ps.joint_suffix.replace("_joint", "2_joint")
            # _resolve_name 的 fallback (self.joint) 只做前缀拼接不做存在性校验，
            # 所以必须先查 scene_binding 确认场景中确实存在该关节
            if sec_name in joints_by_suffix:
                self._secondary_rotor_map[ps.joint_suffix] = sec_name
        self._rotor_specs = list(self._primary_rotor_specs)
        for primary_name, sec_name in self._secondary_rotor_map.items():
            primary_spec = next(s for s in self._primary_rotor_specs if s.joint_suffix == primary_name)
            self._rotor_specs.append(RotorSpec(sec_name, -primary_spec.spin_sign))
        self._rotor_joints = {
            spec.joint_suffix: self._resolve_name("joints", spec.joint_suffix) for spec in self._rotor_specs
        }
        actuators_by_suffix = self._scene_binding.get("actuators_by_suffix", {})
        self._rotor_actuators = {
            spec.joint_suffix: actuators_by_suffix[spec.joint_suffix]
            for spec in self._rotor_specs
            if spec.joint_suffix in actuators_by_suffix
        }
        self._drone_body = self._resolve_name("bodies", "Drone")
        self._frame_body = self._resolve_name("bodies", "drone_frame")

        self._drone_body_id = int(self.model.body_name2id(self._drone_body))
        self._frame_body_id = int(self.model.body_name2id(self._frame_body))

        self._gripper_joints: dict[str, str] = {}
        _gripper_resolve_errors: list[str] = []
        for gname in ("gripper_left_joint", "gripper_right_joint"):
            # _resolve_name 的 fallback (self.joint) 只做前缀拼接不做存在性校验，
            # 所以必须先查 scene_binding 确认场景中确实存在该关节
            if gname in joints_by_suffix:
                resolved = joints_by_suffix[gname]
                self._gripper_joints[gname] = resolved
                _logger.info(f"[DroneOrcaEnv] gripper joint resolved via scene_binding: {gname} -> {resolved}")
            else:
                _gripper_resolve_errors.append(f"scene_binding: {gname} not found")
                try:
                    jid = self.model.joint_name2id(gname)
                    self._gripper_joints[gname] = gname
                    _logger.info(f"[DroneOrcaEnv] gripper joint resolved via model: {gname} (jid={jid})")
                except (KeyError, ValueError) as e2:
                    _gripper_resolve_errors.append(f"model direct: {e2}")
        self._gripper_enabled = len(self._gripper_joints) == 2
        if self._gripper_enabled:
            _logger.info("[DroneOrcaEnv] 抓取机构已启用: Z=收拢(抓取), X=扩张(释放) (键盘) / LB=收拢, RB=扩张 (手柄)")
        else:
            _logger.warning(f"[DroneOrcaEnv] 抓取机构未启用, resolved={len(self._gripper_joints)}/2, errors={_gripper_resolve_errors}")
            _logger.info(f"[DroneOrcaEnv] scene_binding joints_by_suffix={self._scene_binding.get('joints_by_suffix', {})}")
            try:
                all_joints = list(self.model.get_joint_dict().keys())
                _logger.info(f"[DroneOrcaEnv] all model joints: {all_joints}")
            except Exception:
                pass
        self._gripper_target: float = 0.0
        self._gripper_close_speed: float = 0.5
        fixed_r = float(vertical_fixed_thrust_over_hover)
        use_fixed_thrust = fixed_r >= 0.0
        ramp_on = bool(vertical_thrust_ramp) and not use_fixed_thrust
        bvz = self._model_profile.aero.vertical_z_only
        if vertical_keyboard_xy_force_factor is not None:
            bvz = replace(bvz, keyboard_world_xy_force_factor=float(vertical_keyboard_xy_force_factor))
        if bool(vertical_z_only_physics):
            bvz = replace(bvz, apply_thrust_at_free_frame=False)
        vz_cfg = replace(
            bvz,
            enabled=bool(vertical_z_only_physics),
            thrust_ramp_enabled=ramp_on,
            thrust_ramp_t0_factor=float(vertical_ramp_t0_factor),
            thrust_ramp_t1_factor=float(vertical_ramp_t1_factor),
            thrust_ramp_duration_s=float(vertical_ramp_duration_s),
            lock_quat_world_up=bool(vertical_lock_quat_world_up),
            fixed_thrust_over_hover=fixed_r if use_fixed_thrust else -1.0,
            keyboard_baseline_thrust_over_hover=float(vertical_keyboard_baseline_tmg),
        )
        self._aero = replace(self._model_profile.aero, vertical_z_only=vz_cfg)

        subtree_mass = float(self.data.body_subtree_mass(self._frame_body))
        full_cfg = self._model_profile.full_mode
        # 与 subtree 重力平衡；略高会持续爬升。接触/地面效应可用键盘垂直通道微调。
        self._hover_thrust = subtree_mass * 9.81
        # R/F 集体推力增益保守一些，避免大杆量时先把姿态/高度瞬间拉爆
        self._thrust_cmd_scale = float(full_cfg.thrust_cmd_scale_over_hover) * self._hover_thrust
        # 全量模式滚转/俯仰改由「目标推力方向」PD（见 _apply_thrust_and_drag）；偏航杆量保守些，避免在 WASD/RF 下被航向环带偏
        self._tau_yaw = float(full_cfg.tau_yaw_over_hover) * self._hover_thrust
        self._planar_forward_axis_body = _normalize_axis(
            np.asarray(full_cfg.planar_forward_axis_body, dtype=np.float64),
            (0.0, 1.0, 0.0),
        )
        self._planar_right_axis_body = _normalize_axis(
            np.asarray(full_cfg.planar_right_axis_body, dtype=np.float64),
            (1.0, 0.0, 0.0),
        )
        self._attitude_kp_scale = float(full_cfg.attitude_kp_scale)
        self._attitude_kd_scale = float(full_cfg.attitude_kd_scale)
        self._attitude_rate_cap_scale = float(full_cfg.attitude_rate_cap_scale)
        self._attitude_torque_limit_scale = float(full_cfg.attitude_torque_limit_scale)
        self._idle_attitude_kp_scale = float(full_cfg.idle_attitude_kp_scale)
        self._idle_attitude_torque_limit_scale = float(full_cfg.idle_attitude_torque_limit_scale)
        self._thrust_min = max(0.12 * self._hover_thrust, 0.02)
        self._thrust_max = float(full_cfg.thrust_max_over_hover) * self._hover_thrust
        if self._aero.vertical_z_only.enabled:
            # 爬升扫描 t1 可能 >2；键盘竖直通道也需要余量，避免顶到上限后「假悬停」
            t1 = float(self._aero.vertical_z_only.thrust_ramp_t1_factor)
            self._thrust_max = max(self._thrust_max, (t1 + 0.35) * self._hover_thrust, 3.0 * self._hover_thrust)

        self._hover_rotor_speed = float(full_cfg.hover_rotor_speed)
        self._rotor_speed_delta = float(full_cfg.rotor_speed_delta)
        self._rotor_ramp_rate = float(full_cfg.rotor_ramp_rate)
        demo_bias = tuple(float(v) for v in full_cfg.demo_rotor_bias)
        self._demo_rotor_bias: dict[str, float] = {
            "FL_joint": demo_bias[0],
            "FR_joint": demo_bias[1],
            "BL_joint": demo_bias[2],
            "BR_joint": demo_bias[3],
        }
        for primary_name, sec_name in self._secondary_rotor_map.items():
            self._demo_rotor_bias[sec_name] = self._demo_rotor_bias[primary_name]

        qfree = self.query_joint_qpos([self._free_joint])[self._free_joint]
        self._initial_free_qpos = np.asarray(qfree, dtype=np.float64).reshape(-1).copy()
        self._initial_rotor_qpos = self._capture_joint_positions(self._rotor_joints.values())
        self._rotor_phases = {
            spec.joint_suffix: self._initial_rotor_qpos[self._rotor_joints[spec.joint_suffix]]
            for spec in self._rotor_specs
        }
        self._rotor_speeds = {spec.joint_suffix: 0.0 for spec in self._rotor_specs}
        self._last_command = np.zeros(4, dtype=np.float32)
        self._takeoff_crossing_logged = False
        self._takeoff_z_ref = 0.0
        self._takeoff_sustain_accum_s = 0.0
        self._takeoff_sustained_logged = False
        self._vertical_quiet_diag_logs = False
        self._unstable_contact_logged_this_reset = False
        self._last_reset_sim_time = 0.0

        self._v_dof_labels: list[str] = [f"v[{i}]" for i in range(int(self.nv))]

        self._diag_env_steps = 0
        # periodic 动力学长日志默认关闭；需要排查 full 模式乱飘时可通过参数打开
        self._diag_every_env_steps = max(0, int(diag_every_env_steps))
        self._last_xfrc_body = np.zeros(6, dtype=np.float64)
        self._last_thrust_scalar = 0.0
        self._last_tau_cmd = np.zeros(3, dtype=np.float64)
        self._last_ground_effect_factor = 1.0
        # 全量模式：松杆 XY 抱死用滤波速度，削弱 Orca/步进噪声 → 水平来回晃
        self._full_mode_filt_vxy = np.zeros(2, dtype=np.float64)
        self._full_mode_thrust_lpf = float(self._hover_thrust)

        try:
            di = self.model.get_body_byname(self._drone_body)["Inertia"]
            self._body_J = np.diag(np.asarray(di, dtype=np.float64))
        except (KeyError, ValueError):
            self._body_J = np.diag(np.array([6.0, 4.0, 3.0], dtype=np.float64))
        self._body_J_inv = np.linalg.inv(self._body_J)
        self._ctbr_omega_max_rp = float(full_cfg.ctbr_omega_max_rp)
        self._ctbr_omega_max_yaw = float(full_cfg.ctbr_omega_max_yaw)
        self._ctbr_Kp_scalar = float(full_cfg.ctbr_Kp)
        self._ctbr_Kp = np.diag(np.array([self._ctbr_Kp_scalar] * 3, dtype=np.float64))
        self._ctbr_use_inertia_scaling = bool(full_cfg.ctbr_use_inertia_scaling)
        self._ctbr_max_torque_norm = float(self._aero.drag.max_body_torque_norm)
        self._attitude_recover_kp_idle = float(full_cfg.attitude_recover_kp_idle)
        self._attitude_recover_kp_active = float(full_cfg.attitude_recover_kp_active)
        self._torque_warmup_s = float(full_cfg.torque_warmup_s)

        free_lo, free_hi = self._compute_joint_dof_bounds(self._free_joint)
        self._free_dof_lo = free_lo
        self._free_dof_hi = free_hi
        rotor_bounds = [self._compute_joint_dof_bounds(self._rotor_joints[s.joint_suffix]) for s in self._rotor_specs]
        self._rotor_dof_bounds = rotor_bounds
        drone_idx: list[int] = []
        for lo, hi in ((free_lo, free_hi), *rotor_bounds):
            drone_idx.extend(range(lo, hi))
        self._drone_dof_indices = np.asarray(drone_idx, dtype=np.int32)
        self._drone_actor_roots = {
            self._drone_body.split("/")[0],
            self._frame_body.split("/")[0],
        }

        rotor_jids: set[int] = set()
        for spec in self._rotor_specs:
            try:
                rotor_jids.add(self.model.joint_name2id(self._rotor_joints[spec.joint_suffix]))
            except (KeyError, ValueError):
                pass
        rotor_actuator_names: list[str] = []
        for aname, act in self.model.get_actuator_dict().items():
            trnjid = int(act.get("TrnId", -1))
            if trnjid in rotor_jids:
                rotor_actuator_names.append(aname)
        if rotor_actuator_names:
            self._diag_warning(
                "[DroneOrcaEnv] 模型仍含绑定桨关节的执行器 "
                f"{rotor_actuator_names}，会与脚本写 qpos/qvel 冲突并导致极大 qacc；"
                "请使用当前仓库无桨执行器的 drone-v1.xml 并在 OrcaStudio 中重新导入/替换资产。"
            )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = self.generate_observation_space(self._get_obs())

        if int(self.nv) != len(self._drone_dof_indices) or self._free_dof_lo != 0:
            self._diag_warning(
                f"[DroneOrcaEnv] 场景 nv={self.nv}，无人机相关 dof 索引为 {self._drone_dof_indices.tolist()} "
                f"(drone_free=[{self._free_dof_lo},{self._free_dof_hi}))；physics 日志按关节名切片，不再使用 qacc[0:6] 假定。"
            )
        if int(self.nv) > 8:
            self._diag_warning(f"[DroneOrcaEnv] 场景 DOF 观察点 {self._v_dof_labels[8]}")
        if self._reset_height_offset_m > 0.0:
            self._diag_warning(
                f"[DroneOrcaEnv] reset 将额外抬高出生点 dz={self._reset_height_offset_m:.4f}m，"
                "用于排查复杂场景中的初始接触/穿插问题。"
            )
        if self._fullmode_reset_thrust_ramp_s > 0.0:
            self._diag_warning(
                "[DroneOrcaEnv] full 模式 reset 推力渐入 "
                f"T: {self._fullmode_reset_thrust_start_factor:.2f}·hover -> 目标 / {self._fullmode_reset_thrust_ramp_s:.3f}s"
            )
        if self._fullmode_reset_minimal_stab_s > 0.0:
            self._diag_warning(
                "[DroneOrcaEnv] full 模式 reset 最小稳定窗口 "
                f"{self._fullmode_reset_minimal_stab_s:.3f}s（零输入时仅保留集体推力主链，用于隔离姿态/阻尼耦合）"
            )
        self._diag_warning(
            f"[DroneOrcaEnv] 使用无人机参数配置 {self._model_profile.display_name} ({self._model_profile.key})"
        )
        if self._diag_every_env_steps > 0:
            self._diag_warning(
                f"[DroneOrcaEnv] 已启用 periodic 动力学日志：每 {self._diag_every_env_steps} 个 env step 输出一次"
            )

        if self._aero.vertical_z_only.enabled:
            self._diag_every_env_steps = 0
            self._last_vertical_ramp_log_t = -1.0e9
            self._vertical_ramp_t1_logged = False
            vz0 = self._aero.vertical_z_only
            if vz0.thrust_ramp_enabled:
                thrust_mode = (
                    f"线性爬升 {vz0.thrust_ramp_t0_factor}·mg→{vz0.thrust_ramp_t1_factor}·mg / {vz0.thrust_ramp_duration_s}s"
                )
            elif float(vz0.fixed_thrust_over_hover) >= 0.0:
                thrust_mode = f"固定 T/(mg)={vz0.fixed_thrust_over_hover}"
            else:
                thrust_mode = f"键盘 R/F 微调推力（杆量零时 T/(mg)={vz0.keyboard_baseline_thrust_over_hover}）"
            self._diag_warning(
                "[DroneOrcaEnv] 已启用 vertical_z_only：仅世界 +Z 推力与 vz 阻尼；"
                f"{thrust_mode}；"
                f"姿态锁={'世界朝上' if vz0.lock_quat_world_up else '场景初值'}；"
                f"推力作用体={'drone_frame' if vz0.apply_thrust_at_free_frame else 'Drone'}；"
                f"持续起飞判据 Δz≥{vz0.takeoff_sustain_dz_m}m ∧ vz≥{vz0.takeoff_sustain_vz_threshold}m/s 连续 {vz0.takeoff_sustain_time_s}s；"
                f"WASD 水平力系数={vz0.keyboard_world_xy_force_factor}（0=仅 Z）；"
                "已关闭 periodic 动力学长日志。"
            )

    def _resolve_name(self, category: str, suffix: str) -> str:
        matched = self._scene_binding.get(f"{category}_by_suffix", {})
        if suffix in matched:
            return matched[suffix]

        if category == "joints":
            return self.joint(suffix)
        if category == "actuators":
            return self.actuator(suffix)
        if category == "bodies":
            return self.body(suffix)
        if category == "sites":
            return self.site(suffix)
        raise KeyError(f"Unsupported category: {category}")

    def _capture_joint_positions(self, joint_names) -> dict[str, float]:
        qpos_dict = self.query_joint_qpos(list(joint_names))
        return {joint_name: float(np.asarray(qpos_dict[joint_name]).reshape(-1)[0]) for joint_name in joint_names}

    def _should_emit_diag_logs(self) -> bool:
        return self._diag_logs_enabled and not self._vertical_quiet_diag_logs

    def _diag_warning(self, message: str) -> None:
        if self._should_emit_diag_logs():
            _logger.warning(message)

    def apply_joint_qpos_dict(self, joint_qpos_dict: dict) -> None:
        """将 {joint_name: qpos} dict 合并为完整 qpos 数组后写入（Euler 体系 set_joint_qpos 仅接受数组）。"""
        full_qpos = self.data.qpos.copy()
        for jname, jqpos in joint_qpos_dict.items():
            addr = self.jnt_qposadr(jname)
            arr = np.atleast_1d(np.asarray(jqpos, dtype=full_qpos.dtype))
            full_qpos[addr:addr + len(arr)] = arr
        self.set_joint_qpos(full_qpos)

    def apply_joint_qvel_dict(self, joint_qvel_dict: dict) -> None:
        """将 {joint_name: qvel} dict 合并为完整 qvel 数组后写入。"""
        full_qvel = self.data.qvel.copy()
        for jname, jqvel in joint_qvel_dict.items():
            addr = self.jnt_dofadr(jname)
            arr = np.atleast_1d(np.asarray(jqvel, dtype=full_qvel.dtype))
            full_qvel[addr:addr + len(arr)] = arr
        self.set_joint_qvel(full_qvel)

    def _compute_joint_dof_bounds(self, joint_name: str) -> tuple[int, int]:
        """用 Euler 公共 API 计算 joint 的 dof 地址范围 [lo, hi)。"""
        try:
            adr = self.jnt_dofadr(joint_name)
        except (KeyError, ValueError):
            return 0, 0
        jtype = int(self.model.get_joint_byname(joint_name)["Type"])
        if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
            n = 6
        elif jtype == int(mujoco.mjtJoint.mjJNT_BALL):
            n = 3
        elif jtype in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)):
            n = 1
        else:
            n = 0
        return adr, adr + n

    def _ctrl_align_rotor_position_actuators(self) -> np.ndarray:
        """若旧资产仍带桨 position 执行器：ctrl 与相位对齐；无执行器时全零即可（桨纯脚本驱动）。"""
        ctrl = np.zeros(self.nu, dtype=np.float32)
        if not self._rotor_actuators:
            return ctrl
        for spec in self._rotor_specs:
            aname = self._rotor_actuators.get(spec.joint_suffix)
            if not aname:
                continue
            try:
                aid = self.model.actuator_name2id(aname)
            except (KeyError, ValueError):
                continue
            ctrlrange = self.model.get_actuator_byname(aname)["CtrlRange"]
            lo, hi = float(ctrlrange[0]), float(ctrlrange[1])
            ph = float(self._rotor_phases[spec.joint_suffix])
            ctrl[aid] = float(np.clip(ph, lo, hi))
        return ctrl

    def render_callback(self, mode="human") -> None:
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")

    def close(self) -> None:
        if self._joystick_manager is not None:
            self._joystick_manager.close()
            self._joystick_manager = None
            self._joystick = None
        if hasattr(self, "_keyboard") and self._keyboard is not None:
            try:
                if hasattr(self._keyboard, "_source") and hasattr(self._keyboard._source, "close"):
                    self._keyboard._source.close()
            except Exception:
                pass
            self._keyboard = None
        super().close()

    def reset_model(self):
        self._takeoff_crossing_logged = False
        self._takeoff_sustained_logged = False
        self._takeoff_sustain_accum_s = 0.0
        self._vertical_ramp_t1_logged = False
        self._last_vertical_ramp_log_t = -1.0e9
        self._unstable_contact_logged_this_reset = False
        self._last_reset_sim_time = float(self.data.time)
        self._rotor_phases = {
            spec.joint_suffix: self._initial_rotor_qpos[self._rotor_joints[spec.joint_suffix]]
            for spec in self._rotor_specs
        }
        self._rotor_speeds = {spec.joint_suffix: 0.0 for spec in self._rotor_specs}
        self._last_command[:] = 0.0
        self._autoplay_time = 0.0
        self._last_space_state = 0
        self._last_a_button_state = 0
        self._gripper_target = 0.0

        free_q = self._initial_free_qpos.copy()
        if self._reset_height_offset_m > 0.0:
            free_q[2] += self._reset_height_offset_m
        if self._aero.vertical_z_only.enabled and self._aero.vertical_z_only.lock_quat_world_up:
            free_q[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        elif not self._aero.vertical_z_only.enabled:
            # 全量四旋翼：frame 世界朝上，集体升力沿 frame +Z ≈ 世界 +Z；否则推力加在歪斜的 Drone 子体会等效「侧向喷气」
            free_q[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        qpos_update = {self._free_joint: free_q}
        qvel_update = {self._free_joint: np.zeros(6, dtype=np.float64)}
        for spec in self._rotor_specs:
            jn = self._rotor_joints[spec.joint_suffix]
            qpos_update[jn] = np.array([self._initial_rotor_qpos[jn]], dtype=np.float64)
            qvel_update[jn] = np.array([0.0], dtype=np.float64)
        if self._gripper_enabled:
            for gname, full_name in self._gripper_joints.items():
                qpos_update[full_name] = np.array([0.0], dtype=np.float64)
                qvel_update[full_name] = np.array([0.0], dtype=np.float64)

        self.clear_body_force(self._drone_body)
        if self._frame_body_id != self._drone_body_id:
            self.clear_body_force(self._frame_body)
        self.apply_joint_qpos_dict(qpos_update)
        self.apply_joint_qvel_dict(qvel_update)
        self.set_ctrl(self._ctrl_align_rotor_position_actuators())
        self.mj_forward()
        if not self._aero.vertical_z_only.enabled:
            self._full_mode_filt_vxy[:] = 0.0
            self._full_mode_thrust_lpf = float(self._hover_thrust)
        # Orca 同步偶发在 mj_forward 后写回非零关节速度，再清一次 free
        self.apply_joint_qvel_dict({self._free_joint: np.zeros(6, dtype=np.float64)})
        self.mj_forward()
        self._takeoff_z_ref = float(self.data.body_xpos(self._frame_body)[2])
        if self._reset_height_offset_m > 0.0:
            self._diag_warning(
                f"[DroneOrcaEnv] reset 后 frame z={self._takeoff_z_ref:.4f}m "
                f"(包含 dz={self._reset_height_offset_m:.4f}m 安全偏移)"
            )
        self._diag_warning(
            "[DroneOrcaEnv] reset 刚体参考 "
            f"{self._body_pose_summary(self._frame_body, 'frame')} "
            f"{self._body_pose_summary(self._drone_body, 'drone')}"
        )
        contact_summary = self._format_contact_summary()
        if contact_summary is not None:
            self._diag_warning(f"[DroneOrcaEnv] reset 初始接触 {contact_summary}")
        vz_cfg = self._aero.vertical_z_only
        if vz_cfg.enabled and vz_cfg.thrust_ramp_enabled:
            self._diag_warning(
                "[DroneOrcaEnv] ramp 扫描开始 "
                f"{vz_cfg.thrust_ramp_t0_factor}·mg→{vz_cfg.thrust_ramp_t1_factor}·mg "
                f"/ {vz_cfg.thrust_ramp_duration_s}s，每 {vz_cfg.ramp_progress_log_interval_s}s 一条进度；"
                "满足持续起飞判据时打「持续起飞临界(精估)」。"
            )
        elif (
            vz_cfg.enabled
            and float(vz_cfg.fixed_thrust_over_hover) >= 0.0
            and self._should_emit_diag_logs()
        ):
            self._diag_warning(
                f"[DroneOrcaEnv] 固定推力试验 T/(mg)={vz_cfg.fixed_thrust_over_hover}，"
                f"z_ref(frame)={self._takeoff_z_ref:.4f}m"
            )
        return self._get_obs(), self._get_info(reset_requested=False)

    def step(self, action):
        del action

        command, reset_requested = self._read_keyboard_command()
        self._last_command[:] = command

        unstable_logged_this_step = False
        for _ in range(self.frame_skip):
            self.mj_forward()
            self._apply_thrust_and_drag(command, self._physics_dt)
            self.set_ctrl(self._ctrl_align_rotor_position_actuators())
            self.mj_step(nstep=1)
            # mj_step 不触发 sync_to_view，data.time（float 快照）不会更新。
            # _apply_thrust_and_drag 依赖 data.time 计算 sim_since_reset 控制推力爬升；
            # render() 也依赖 data.time 推进 OrcaStudio 画面。每次 mj_step 后必须同步。
            self._sync_view()
            if self._aero.vertical_z_only.enabled:
                if self._aero.vertical_z_only.lock_pose_and_planar_velocity:
                    self._enforce_vertical_only_kinematics()
                else:
                    self._apply_free_joint_velocity_safety()
                self._maybe_log_takeoff_first_vz_spike()
                self._update_takeoff_sustain_detector()
            else:
                self._apply_free_joint_velocity_safety()
            rotor_cmd = command
            vz_o = self._aero.vertical_z_only
            if vz_o.enabled and (
                vz_o.thrust_ramp_enabled or float(vz_o.fixed_thrust_over_hover) >= 0.0
            ):
                rotor_cmd = np.zeros(4, dtype=np.float32)
            self._update_rotors(rotor_cmd, self._physics_dt)
            if self._gripper_enabled:
                self._update_gripper(self._physics_dt)
            if self._diag_logs_enabled and not unstable_logged_this_step and self._drone_physics_should_warn_immediate():
                include_contacts = not self._unstable_contact_logged_this_reset
                self._emit_drone_physics_warning("unstable_post_mj_step", include_contacts=include_contacts)
                if include_contacts:
                    self._unstable_contact_logged_this_reset = True
                unstable_logged_this_step = True

        self._diag_env_steps += 1
        if (
            self._diag_logs_enabled
            and self._diag_every_env_steps > 0
            and not unstable_logged_this_step
            and self._diag_env_steps % self._diag_every_env_steps == 0
        ):
            self._emit_drone_physics_warning("periodic")

        self._maybe_log_vertical_ramp_progress()

        obs = self._get_obs()
        info = self._get_info(reset_requested=reset_requested)
        return obs, 0.0, False, False, info

    def _apply_free_joint_velocity_safety(self) -> None:
        """对 free joint 速度做硬钳制，防止场景耦合把系统带入数值发散区。"""
        lo, hi = self._free_dof_lo, self._free_dof_hi
        if hi - lo != 6:
            return
        qv = np.asarray(self.data.qvel[lo:hi], dtype=np.float64).reshape(6)
        dcfg = self._aero.drag
        lv_cap = float(dcfg.free_linear_speed_cap)
        av_cap = float(dcfg.free_angular_speed_cap)
        clipped = np.array(
            [
                float(np.clip(qv[0], -lv_cap, lv_cap)),
                float(np.clip(qv[1], -lv_cap, lv_cap)),
                float(np.clip(qv[2], -lv_cap, lv_cap)),
                float(np.clip(qv[3], -av_cap, av_cap)),
                float(np.clip(qv[4], -av_cap, av_cap)),
                float(np.clip(qv[5], -av_cap, av_cap)),
            ],
            dtype=np.float64,
        )
        if np.max(np.abs(clipped - qv)) > 1e-9:
            self.apply_joint_qvel_dict({self._free_joint: clipped})
            self.mj_forward()

    def _read_keyboard_command(self) -> tuple[np.ndarray, bool]:
        if self._ctrl_device == ControlDevice.XBOX and self._joystick_manager is not None:
            return self._read_joystick_command()
        return self._read_keyboard_only_command()

    def _read_joystick_command(self) -> tuple[np.ndarray, bool]:
        self._joystick_manager.update()
        pos_ctrl = self._joystick.capture_joystick_pos_ctrl()
        rot_ctrl = self._joystick.capture_joystick_rot_ctrl()
        state = self._joystick.get_state()

        a_pressed = int(state["buttons"]["A"])
        reset_requested = self._last_a_button_state == 0 and a_pressed == 1
        self._last_a_button_state = a_pressed

        if self._autoplay_enabled:
            command = self._build_autoplay_command()
        else:
            planar_scale = 0.5
            # pos_ctrl["y"] 已对 pygame 上推取反为 +1，但键盘约定 ws_cmd = S - W
            # （W = -0.5 = 前进，因 CTBR 中 cmd_xy_w = -ws_cmd * fwd_ref_w）。
            # 故此处再取反，使左摇杆前推得到负 ws_cmd，与 W 键一致。
            # rot_ctrl["yaw"] = RightStickX：右推=+1，但 CTBR 中 omega_z>0=逆时针(左转)，
            # 故取反使右推=右转，与键盘 E 键（右转=-1→omega_z<0）一致。
            command = np.array(
                [
                    -planar_scale * float(pos_ctrl["y"]),
                    planar_scale * float(pos_ctrl["x"]),
                    float(pos_ctrl["z"]),
                    -float(rot_ctrl["yaw"]),
                ],
                dtype=np.float32,
            )
        if self._gripper_enabled:
            lb = int(state["buttons"].get("LB", 0))
            rb = int(state["buttons"].get("RB", 0))
            new_target = float(np.clip(rb - lb, -1.0, 1.0))
            if abs(new_target - self._gripper_target) > 0.01:
                _logger.info(f"[DroneOrcaEnv] gripper_target changed: {self._gripper_target:.2f} -> {new_target:.2f} (LB={lb} RB={rb})")
            self._gripper_target = new_target
        return command, reset_requested

    def _read_keyboard_only_command(self) -> tuple[np.ndarray, bool]:
        self._keyboard.update()
        state = self._keyboard.get_state()

        space_pressed = int(state["Space"])
        reset_requested = self._last_space_state == 0 and space_pressed == 1
        self._last_space_state = space_pressed

        if self._autoplay_enabled:
            command = self._build_autoplay_command()
        else:
            planar_scale = 0.5
            command = np.array(
                [
                    planar_scale * (state["S"] - state["W"]),
                    planar_scale * (state["D"] - state["A"]),
                    state["R"] - state["F"],
                    state["Q"] - state["E"],
                ],
                dtype=np.float32,
            )
        if self._gripper_enabled:
            close_val = float(state.get("Z", 0))
            open_val = float(state.get("X", 0))
            new_target = float(np.clip(open_val - close_val, -1.0, 1.0))
            if abs(new_target - self._gripper_target) > 0.01:
                _logger.info(f"[DroneOrcaEnv] gripper_target changed: {self._gripper_target:.2f} -> {new_target:.2f} (Z=close={close_val:.1f} X=open={open_val:.1f})")
            self._gripper_target = new_target
        return command, reset_requested

    def _build_autoplay_command(self) -> np.ndarray:
        t = self._autoplay_time
        self._autoplay_time += self._control_dt

        forward = 0.35 + 0.10 * math.sin(0.23 * t)
        lateral = 0.15 * math.sin(0.47 * t + 0.5)
        vertical = 0.08 * math.sin(0.31 * t + 1.0)
        yaw = 0.12 * math.sin(0.37 * t + 0.2)
        return np.array([forward, lateral, vertical, yaw], dtype=np.float32)

    def _apply_thrust_vertical_z_only(self, command: np.ndarray) -> None:
        """世界 +Z 标量推力 + 仅用 qvel 的 vz 阻尼；无力矩、无地面效应。free joint qvel 布局为 [vx,vy,vz,ωx,ωy,ωz]。"""
        vz_cfg = self._aero.vertical_z_only
        thrust_body = self._frame_body if vz_cfg.apply_thrust_at_free_frame else self._drone_body

        ws_cmd, ad_cmd, vertical_cmd, _yaw_cmd = [float(np.clip(v, -1.0, 1.0)) for v in command]

        if float(vz_cfg.fixed_thrust_over_hover) >= 0.0:
            r = float(vz_cfg.fixed_thrust_over_hover)
            thrust = float(np.clip(r * self._hover_thrust, self._thrust_min, self._thrust_max))
        elif vz_cfg.thrust_ramp_enabled:
            dur = max(float(vz_cfg.thrust_ramp_duration_s), 1e-6)
            u = min(1.0, float(self.data.time) / dur)
            t0 = float(vz_cfg.thrust_ramp_t0_factor)
            t1 = float(vz_cfg.thrust_ramp_t1_factor)
            thrust = self._hover_thrust * (t0 + u * (t1 - t0))
            thrust = float(np.clip(thrust, self._thrust_min, self._thrust_max))
        else:
            base = float(vz_cfg.keyboard_baseline_thrust_over_hover) * self._hover_thrust
            thrust = float(
                np.clip(
                    base + vertical_cmd * self._thrust_cmd_scale,
                    self._thrust_min,
                    self._thrust_max,
                )
            )

        lo, hi = self._free_dof_lo, self._free_dof_hi
        if hi - lo == 6:
            vz = float(self.data.qvel[lo + 2])
        else:
            vz = float(self.data.body_cvel(self._drone_body)[5])
        k_l = float(vz_cfg.vz_linear_k)
        k_q = float(vz_cfg.vz_quadratic_k)
        fz = thrust - k_l * vz - k_q * abs(vz) * vz

        mg_h = max(float(self._hover_thrust), 1e-9)
        kxy = float(vz_cfg.keyboard_world_xy_force_factor)
        fx = 0.0
        fy = 0.0
        if kxy > 1e-12 and hi - lo == 6:
            # 用推力作用体的 COM 世界系线速度做阻尼/刹停，与全量模式 drag 一致；勿与 qvel 混用以免原点/COM 不一致
            cvel = self.data.body_cvel(thrust_body)
            vx_w = float(cvel[3])
            vy_w = float(cvel[4])
            kxd = float(vz_cfg.keyboard_world_xy_vel_damping)
            dead = float(self._aero.drag.zero_cmd_hold_deadband)
            cap_xy = float(self._aero.drag.zero_cmd_xy_hold_force_cap)
            planar_idle = abs(ws_cmd) < dead and abs(ad_cmd) < dead
            if planar_idle:
                kxh = kxd + float(self._aero.drag.zero_cmd_xy_hold_k)
                fx = float(np.clip(-kxh * vx_w, -cap_xy, cap_xy))
                fy = float(np.clip(-kxh * vy_w, -cap_xy, cap_xy))
            else:
                fcx = ad_cmd * kxy * mg_h
                fcy = ws_cmd * kxy * mg_h
                fcap = 1.55 * kxy * mg_h
                hc = float(math.hypot(fcx, fcy))
                if hc > fcap and hc > 1e-12:
                    s = fcap / hc
                    fcx *= s
                    fcy *= s
                fx = fcx - kxd * vx_w
                fy = fcy - kxd * vy_w

        f_w = np.array([fx, fy, fz], dtype=np.float64)

        self.clear_body_force(self._drone_body)
        if self._frame_body_id != self._drone_body_id:
            self.clear_body_force(self._frame_body)
        # xfrc_applied 线力为世界系；竖直模式在 lock 姿态下 R≈I，写 f_w 与旧版写 R^T f_w 等价，解锁姿态时亦正确
        self.apply_body_force(thrust_body, f_w, np.zeros(3, dtype=np.float64))

        r_w_b = self.data.body_xmat(thrust_body).reshape(3, 3)
        self._last_xfrc_body = np.concatenate([r_w_b.T @ f_w, np.zeros(3, dtype=np.float64)])
        self._last_thrust_scalar = thrust
        self._last_tau_cmd[:] = 0.0
        self._last_ground_effect_factor = 1.0

    def _enforce_vertical_only_kinematics(self) -> None:
        lo, hi = self._free_dof_lo, self._free_dof_hi
        if hi - lo != 6:
            return
        qcur = np.asarray(self.query_joint_qpos([self._free_joint])[self._free_joint], dtype=np.float64).reshape(-1)
        if qcur.size != 7:
            return
        new_q = qcur.copy()
        if self._aero.vertical_z_only.lock_quat_world_up:
            new_q[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            new_q[3:7] = self._initial_free_qpos[3:7]
        self.apply_joint_qpos_dict({self._free_joint: new_q})
        qv = np.asarray(self.data.qvel[lo:hi], dtype=np.float64).reshape(6)
        new_v = np.zeros(6, dtype=np.float64)
        new_v[2] = float(qv[2])
        vz_xy = self._aero.vertical_z_only
        if float(vz_xy.keyboard_world_xy_force_factor) > 1e-12:
            new_v[0] = float(qv[0])
            new_v[1] = float(qv[1])
            vcap = float(vz_xy.keyboard_world_xy_max_speed)
            if vcap > 1e-9:
                hxy = float(math.hypot(new_v[0], new_v[1]))
                if hxy > vcap and hxy > 1e-12:
                    s = vcap / hxy
                    new_v[0] *= s
                    new_v[1] *= s
        self.apply_joint_qvel_dict({self._free_joint: new_v})
        self.mj_forward()

    def _maybe_log_takeoff_first_vz_spike(self) -> None:
        if not self._should_emit_diag_logs():
            return
        vz_cfg = self._aero.vertical_z_only
        if not vz_cfg.takeoff_log_first_vz_spike or self._takeoff_crossing_logged:
            return
        if not (vz_cfg.thrust_ramp_enabled or float(vz_cfg.fixed_thrust_over_hover) >= 0.0):
            return
        lo, hi = self._free_dof_lo, self._free_dof_hi
        if hi - lo != 6:
            return
        vz = float(self.data.qvel[lo + 2])
        thr = float(self._last_thrust_scalar)
        mg = max(self._hover_thrust, 1e-9)
        if vz > float(vz_cfg.takeoff_vz_threshold):
            self._takeoff_crossing_logged = True
            om_m = float(np.mean([self._rotor_speeds[s.joint_suffix] for s in self._rotor_specs]))
            self._diag_warning(
                "[DroneOrcaEnv] 首次 vz 过阈(易与弹跳混淆): "
                f"sim_t={float(self.data.time):.4f}s vz={vz:.4f}m/s z={float(self.data.body_xpos(self._frame_body)[2]):.4f}m "
                f"thrust={thr:.4f}N T/(mg)={thr/mg:.4f} 桨ω_mean≈{om_m:.2f}rad/s"
            )

    def _update_takeoff_sustain_detector(self) -> None:
        if not self._should_emit_diag_logs():
            return
        vz_cfg = self._aero.vertical_z_only
        if self._takeoff_sustained_logged:
            return
        if not (vz_cfg.thrust_ramp_enabled or float(vz_cfg.fixed_thrust_over_hover) >= 0.0):
            return
        lo, hi = self._free_dof_lo, self._free_dof_hi
        if hi - lo != 6:
            return
        vz = float(self.data.qvel[lo + 2])
        z = float(self.data.body_xpos(self._frame_body)[2])
        v_th = float(vz_cfg.takeoff_sustain_vz_threshold)
        dz_need = float(vz_cfg.takeoff_sustain_dz_m)
        need_t = float(vz_cfg.takeoff_sustain_time_s)
        if vz >= v_th and (z - self._takeoff_z_ref) >= dz_need:
            self._takeoff_sustain_accum_s += float(self._physics_dt)
        else:
            self._takeoff_sustain_accum_s = 0.0
        if self._takeoff_sustain_accum_s < need_t:
            return
        self._takeoff_sustained_logged = True
        thr = float(self._last_thrust_scalar)
        mg = max(float(self._hover_thrust), 1e-9)
        om_m = float(np.mean([self._rotor_speeds[s.joint_suffix] for s in self._rotor_specs]))
        self._diag_warning(
            "[DroneOrcaEnv] 持续起飞临界(精估): "
            f"sim_t={float(self.data.time):.4f}s Δz={(z - self._takeoff_z_ref):.4f}m vz={vz:.4f}m/s "
            f"thrust={thr:.4f}N T/(mg)={thr/mg:.6f} 桨ω_mean≈{om_m:.2f}rad/s"
        )

    def get_vertical_takeoff_z_reference(self) -> float:
        return float(self._takeoff_z_ref)

    def set_vertical_quiet_diag_logs(self, quiet: bool) -> None:
        """关闭竖直模式 reset/持续起飞等诊断 WARNING（供脚本二分等批量试验）。"""
        self._vertical_quiet_diag_logs = bool(quiet)

    def set_vertical_fixed_thrust_over_hover(self, ratio: float) -> None:
        """运行时切换固定 T/(mg)；ratio<0 关闭固定推力并恢复由键盘竖直通道控制（爬升 ramp 需重新建 env）。"""
        vz0 = self._aero.vertical_z_only
        if float(ratio) < 0.0:
            vz = replace(vz0, fixed_thrust_over_hover=-1.0)
        else:
            vz = replace(vz0, fixed_thrust_over_hover=float(ratio), thrust_ramp_enabled=False)
        self._aero = replace(self._aero, vertical_z_only=vz)

    def _apply_thrust_and_drag(self, command: np.ndarray, dt: float) -> None:
        dt = float(max(dt, 1e-6))
        if self._aero.vertical_z_only.enabled:
            self._apply_thrust_vertical_z_only(command)
            return

        # full 模式：推力方向仍按 drone_frame 姿态解算，但实际 wrench 打在 Drone 子体质心。
        # 诊断表明某些场景中 frame 的等效 COM/参考点异常，直接打在 frame 会引入巨大的假力矩。
        pose_body = self._frame_body
        bid_body = self._drone_body
        self.clear_body_force(self._drone_body)
        self.clear_body_force(self._frame_body)
        sim_since_reset = max(0.0, float(self.data.time) - float(self._last_reset_sim_time))
        startup_minimal_stab = (
            self._fullmode_reset_minimal_stab_s > 0.0
            and sim_since_reset < self._fullmode_reset_minimal_stab_s
        )

        ws_cmd, ad_cmd, vertical_cmd, yaw_cmd = [float(np.clip(v, -1.0, 1.0)) for v in command]

        planar_mag = min(1.0, float(math.hypot(ws_cmd, ad_cmd)))
        max_tilt_rad = math.radians(float(self._model_profile.full_mode.max_tilt_deg))
        R_d = self.data.body_xmat(pose_body).reshape(3, 3)
        zb_w_cur = R_d @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # WASD 时按当前真实倾角补 hover，而不是按目标倾角补。
        # 之前实际只倾了 1~2 度，却按更大的目标角度多补了推力，导致平移时持续上飘/下沉。
        thrust_hover_comp = 1.0
        thrust_target = float(
            np.clip(
                (self._hover_thrust * thrust_hover_comp) + vertical_cmd * self._thrust_cmd_scale,
                self._thrust_min,
                self._thrust_max,
            )
        )
        if self._fullmode_reset_thrust_ramp_s > 0.0 and sim_since_reset < self._fullmode_reset_thrust_ramp_s:
            a_thr = sim_since_reset / self._fullmode_reset_thrust_ramp_s
            thrust_start = self._fullmode_reset_thrust_start_factor * self._hover_thrust
            thrust_target = float(thrust_start + a_thr * (thrust_target - thrust_start))
        tau_t = float(max(self._aero.drag.full_mode_thrust_lpf_tau_s, 0.02))
        lpf_a = min(1.0, dt / tau_t)
        self._full_mode_thrust_lpf = float(
            self._full_mode_thrust_lpf + lpf_a * (thrust_target - self._full_mode_thrust_lpf)
        )
        thrust = float(self._full_mode_thrust_lpf)

        cvel = self.data.body_cvel(bid_body)
        omega_w_raw = np.asarray(cvel[0:3], dtype=np.float64).reshape(3)
        omega_b_raw = R_d.T @ omega_w_raw
        dcfg = self._aero.drag
        v_raw = np.asarray(cvel[3:6], dtype=np.float64).reshape(3)
        vc = float(dcfg.aero_model_velocity_clip)
        v_w = np.clip(v_raw, -vc, vc)
        dead = float(dcfg.zero_cmd_hold_deadband)
        planar_idle = abs(ws_cmd) < dead and abs(ad_cmd) < dead
        vert_idle = abs(vertical_cmd) < dead
        yaw_idle = abs(yaw_cmd) < dead
        full_idle = planar_idle and vert_idle and yaw_idle
        vert_active = not vert_idle
        k_xy_base = float(dcfg.world_xy_velocity_damping)
        cap_xy = float(dcfg.zero_cmd_xy_hold_force_cap)
        if planar_idle:
            ema_a = 0.22
            self._full_mode_filt_vxy = (1.0 - ema_a) * self._full_mode_filt_vxy + ema_a * np.asarray(
                v_raw[0:2], dtype=np.float64
            )
            vxy_h = self._full_mode_filt_vxy.copy()
            vxy_dead = 0.075
            if float(np.linalg.norm(vxy_h)) < vxy_dead:
                vxy_h[:] = 0.0
            # 键盘全松时保留一层轻微世界系 XY 抱死，抑制残余小倾角带来的持续平移。
            if full_idle:
                kxh = k_xy_base + float(dcfg.zero_cmd_xy_hold_k) + 0.22
                cap_xy_full = min(1.05, cap_xy + 0.20)
                f_xy_w = np.array(
                    [
                        float(np.clip(-kxh * vxy_h[0], -cap_xy_full, cap_xy_full)),
                        float(np.clip(-kxh * vxy_h[1], -cap_xy_full, cap_xy_full)),
                        0.0,
                    ],
                    dtype=np.float64,
                )
            else:
                kxh = k_xy_base + float(dcfg.zero_cmd_xy_hold_k)
                f_xy_w = np.array(
                    [
                        float(np.clip(-kxh * vxy_h[0], -cap_xy, cap_xy)),
                        float(np.clip(-kxh * vxy_h[1], -cap_xy, cap_xy)),
                        0.0,
                    ],
                    dtype=np.float64,
                )
        else:
            self._full_mode_filt_vxy[:] = np.asarray(v_raw[0:2], dtype=np.float64)
            stick_xy = min(1.0, float(math.hypot(ws_cmd, ad_cmd)))
            k_xy_eff = k_xy_base * (1.0 - 0.5 * stick_xy)
            f_xy_w = np.array([-k_xy_eff * v_w[0], -k_xy_eff * v_w[1], 0.0], dtype=np.float64)
        f_z_hold_w = np.zeros(3, dtype=np.float64)
        if vert_idle:
            kzh = float(dcfg.zero_cmd_z_hold_k)
            cap_z = float(dcfg.zero_cmd_z_hold_force_cap)
            f_z_hold_w[2] = float(np.clip(-kzh * v_raw[2], -cap_z, cap_z))
        if startup_minimal_stab and planar_idle and vert_idle and yaw_idle:
            f_xy_w[:] = 0.0
            f_z_hold_w[:] = 0.0

        # 用 frame 的水平朝向定义“前/右”基向量，使 WASD 跟随机头/绑定相机旋转，而不是固定在世界坐标里。
        fwd_ref_w = R_d @ self._planar_forward_axis_body
        fwd_ref_w[2] = 0.0
        nf = float(np.linalg.norm(fwd_ref_w))
        if nf < 1e-8:
            fwd_ref_w = self._planar_forward_axis_body.copy()
        else:
            fwd_ref_w /= nf
        right_ref_w = R_d @ self._planar_right_axis_body
        right_ref_w[2] = 0.0
        nr = float(np.linalg.norm(right_ref_w))
        if nr < 1e-8:
            right_ref_w = self._planar_right_axis_body.copy()
        else:
            right_ref_w /= nr

        # CTBR 控制器：command → 目标体角速率 → P+FB 力矩
        # 用 fwd/right 参考轴计算目标倾斜方向，投影到机体 roll/pitch 轴
        omega_des_rp = float(self._ctbr_omega_max_rp)
        omega_des_y = float(self._ctbr_omega_max_yaw)
        if startup_minimal_stab:
            omega_des_rp *= 0.3
            omega_des_y *= 0.3
        cmd_xy_w = -ws_cmd * fwd_ref_w + ad_cmd * right_ref_w
        cmd_xy_body = R_d.T @ cmd_xy_w
        omega_des_b = np.array([
            cmd_xy_body[1] * omega_des_rp,
            -cmd_xy_body[0] * omega_des_rp,
            yaw_cmd * omega_des_y,
        ], dtype=np.float64)

        # 姿态回正：计算机体倾斜角，在 omega_des 上叠加回正角速率
        # 操控时（WASD 按下）轻微回正，松键时强力回正
        zb_body = R_d.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tilt_err_rp = np.array([-zb_body[1], zb_body[0]], dtype=np.float64)
        tilt_rad = float(math.acos(np.clip(zb_body[2], -1.0, 1.0)))
        max_tilt_rad = math.radians(float(self._model_profile.full_mode.max_tilt_deg))
        if planar_idle:
            Kp_att = self._attitude_recover_kp_idle
        else:
            Kp_att = self._attitude_recover_kp_active
        omega_des_b[0:2] += tilt_err_rp[0:2] * Kp_att

        # 倾角硬限：超过 max_tilt 时强力回正
        if tilt_rad > max_tilt_rad:
            over_ratio = min((tilt_rad - max_tilt_rad) / math.radians(10.0), 1.0)
            omega_des_b[0:2] += tilt_err_rp[0:2] * omega_des_rp * over_ratio * 3.0

        omega_err = omega_des_b - omega_b_raw
        if self._ctbr_use_inertia_scaling:
            prop = self._body_J @ (self._ctbr_Kp @ omega_err)
            Jw = self._body_J @ omega_b_raw
            fb_lin = np.cross(omega_b_raw, Jw)
        else:
            # 小惯量机体：Kp 直接为力矩/角速率误差，不乘 body_J
            prop = self._ctbr_Kp @ omega_err
            fb_lin = np.zeros(3, dtype=np.float64)
        tau_b_cmd = prop + fb_lin

        tnorm = float(np.linalg.norm(tau_b_cmd))
        tmax = self._ctbr_max_torque_norm
        if tnorm > tmax and tnorm > 1e-12:
            tau_b_cmd = tau_b_cmd * (tmax / tnorm)
        if startup_minimal_stab and full_idle:
            tau_b_cmd[:] = 0.0

        f_collective_b = np.array([0.0, 0.0, thrust], dtype=np.float64)
        f_stab_w = f_xy_w + f_z_hold_w
        f_w = R_d @ f_collective_b + f_stab_w
        kq_xy = float(dcfg.quad_world_xy_stick_force_factor)
        if kq_xy > 1e-12 and not planar_idle:
            mg_h = max(float(self._hover_thrust), 1e-9)
            f_w += (ad_cmd * right_ref_w + ws_cmd * fwd_ref_w) * (kq_xy * mg_h)

        tcap = float(dcfg.angular_drag_torque_axis_max)
        tau_drag_b = np.array(
            [
                float(np.clip(-dcfg.angular_xy * omega_b_raw[0], -tcap, tcap)),
                float(np.clip(-dcfg.angular_xy * omega_b_raw[1], -tcap, tcap)),
                float(np.clip(-dcfg.angular_z * omega_b_raw[2], -tcap, tcap)),
            ],
            dtype=np.float64,
        )
        if startup_minimal_stab and full_idle:
            tau_drag_b[:] = 0.0

        f_drag_w = np.array(
            [
                -dcfg.linear_xy * v_w[0] - dcfg.quadratic_xy * abs(v_w[0]) * v_w[0],
                -dcfg.linear_xy * v_w[1] - dcfg.quadratic_xy * abs(v_w[1]) * v_w[1],
                -dcfg.linear_z * v_w[2] - dcfg.quadratic_z * abs(v_w[2]) * v_w[2],
            ],
            dtype=np.float64,
        )
        f_w = f_w + f_drag_w

        tau_b_tot = tau_b_cmd + tau_drag_b
        # reset 后先软启动 torque，避免小惯量机体在同步/初始误差尚未消散时被第一帧大力矩打翻。
        warmup_s = self._torque_warmup_s
        if warmup_s > 0.0 and sim_since_reset < warmup_s:
            a = sim_since_reset / warmup_s
            torque_scale = 0.25 + 0.75 * a
            tau_b_tot *= torque_scale
        tnorm = float(np.linalg.norm(tau_b_tot))
        tmax = float(dcfg.max_body_torque_norm)
        if tnorm > tmax and tnorm > 1e-12:
            tau_b_tot = tau_b_tot * (tmax / tnorm)
        tau_w = R_d @ tau_b_tot
        tnw = float(np.linalg.norm(tau_w))
        if tnw > tmax and tnw > 1e-12:
            tau_w = tau_w * (tmax / tnw)

        max_fw = float(dcfg.max_total_linear_force_over_hover) * float(self._hover_thrust)
        fn = float(np.linalg.norm(f_w))
        if fn > max_fw and fn > 1e-12:
            f_w = f_w * (max_fw / fn)

        self._last_ground_effect_factor = 1.0
        if self._aero.ground_effect.enabled:
            z = float(self.data.body_xpos(pose_body)[2])
            gcfg = self._aero.ground_effect
            z_body_w = R_d @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            cos_up = float(np.clip(z_body_w[2], -1.0, 1.0))
            tilt_ok = cos_up >= math.cos(math.radians(float(gcfg.max_tilt_deg)))
            if tilt_ok and z < gcfg.active_height and z > gcfg.min_height:
                factor = 1.0 + gcfg.gain * (1.0 - z / gcfg.active_height)
                factor = float(np.clip(factor, 1.0, gcfg.max_factor))
                if vert_active:
                    factor = 1.0 + (factor - 1.0) * 0.45
                f_w[2] *= factor
                self._last_ground_effect_factor = factor

        fn2 = float(np.linalg.norm(f_w))
        if fn2 > max_fw and fn2 > 1e-12:
            f_w = f_w * (max_fw / fn2)

        self.apply_body_force(bid_body, f_w, tau_w)

        # 诊断用：与机体系控制律对齐的等效 wrench（世界系施加值旋回机体系）
        self._last_xfrc_body = np.concatenate(
            [R_d.T @ f_w.reshape(3), R_d.T @ tau_w.reshape(3)]
        ).astype(np.float64, copy=False)
        self._last_thrust_scalar = thrust
        self._last_tau_cmd = tau_b_cmd.copy()

    def _drone_physics_should_warn_immediate(self) -> bool:
        nv = int(self.model.nv)
        if nv <= 0:
            return False
        qacc = np.asarray(self.data.qacc, dtype=np.float64).reshape(-1)
        qvel = np.asarray(self.data.qvel, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(qacc)) or not np.all(np.isfinite(qvel)):
            return True
        if self._aero.vertical_z_only.enabled:
            return False
        di = self._drone_dof_indices
        if di.size == 0:
            return False
        qa_d = qacc[di]
        qv_d = qvel[di]
        if not np.all(np.isfinite(qa_d)) or not np.all(np.isfinite(qv_d)):
            return True
        if float(np.max(np.abs(qa_d))) > 5.0e4:
            return True
        if float(np.max(np.abs(qv_d))) > 80.0:
            return True
        return False

    def _format_contact_summary(self, max_items: int = 6) -> Optional[str]:
        contacts = self.data.contact
        ncon = len(contacts)
        if ncon <= 0:
            return None
        items: list[str] = []
        for i, con in enumerate(contacts):
            if i >= max_items:
                break
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            item = f"geom[{g1}]<->geom[{g2}] dist={float(con.dist):.5f}"
            items.append(item)
        suffix = ""
        omitted = ncon - len(items)
        if omitted > 0:
            suffix = f" ... +{omitted} more"
        return f"ncon={ncon} contacts=[" + "; ".join(items) + "]" + suffix

    def _body_pose_summary(self, body_name: str, label: str) -> str:
        origin = np.asarray(self.data.body_xpos(body_name), dtype=np.float64).reshape(3)
        return f"{label}_origin={np.round(origin, 4)}"

    def _emit_drone_physics_warning(self, reason: str, include_contacts: bool = False) -> None:
        """在全局仅开启 WARNING 时仍可看见：用 WARNING 输出一帧关键动力学量。"""
        if not self._should_emit_diag_logs():
            return
        qacc = np.asarray(self.data.qacc, dtype=np.float64).reshape(-1)
        qvel = np.asarray(self.data.qvel, dtype=np.float64).reshape(-1)
        nv = int(self.model.nv)
        imax = int(np.argmax(np.abs(qacc))) if nv > 0 else 0
        vmax = int(np.argmax(np.abs(qvel))) if nv > 0 else 0
        di = self._drone_dof_indices
        qa_d = qacc[di] if di.size else np.array([])
        qv_d = qvel[di] if di.size else np.array([])
        gidx_d_acc = int(di[int(np.argmax(np.abs(qa_d)))]) if qa_d.size else -1
        gidx_d_vel = int(di[int(np.argmax(np.abs(qv_d)))]) if qv_d.size else -1
        qacc_free = qacc[self._free_dof_lo : self._free_dof_hi]
        qvel_free = qvel[self._free_dof_lo : self._free_dof_hi]
        qacc_rot_flat = (
            np.concatenate([qacc[lo:hi] for lo, hi in self._rotor_dof_bounds]) if self._rotor_dof_bounds else np.array([])
        )
        qvel_rot_flat = np.concatenate([qvel[lo:hi] for lo, hi in self._rotor_dof_bounds]) if self._rotor_dof_bounds else np.array([])
        euler = rotations.quat2euler(np.asarray(self.data.body_xquat(self._drone_body), dtype=np.float64).reshape(1, -1))
        euler_deg = np.rad2deg(euler.reshape(-1))
        frame_euler_suffix = ""
        if not self._aero.vertical_z_only.enabled:
            fe = rotations.quat2euler(
                np.asarray(self.data.body_xquat(self._frame_body), dtype=np.float64).reshape(1, -1)
            )
            frame_euler_suffix = f" frame_euler_deg={np.round(np.rad2deg(fe.reshape(-1)), 2)}"
        cmd = np.asarray(self._last_command, dtype=np.float64).reshape(-1)
        contact_suffix = ""
        if include_contacts:
            contact_summary = self._format_contact_summary()
            if contact_summary is not None:
                contact_suffix = f" {contact_summary}"
        pose_suffix = (
            f" {self._body_pose_summary(self._frame_body, 'frame')}"
            f" {self._body_pose_summary(self._drone_body, 'drone')}"
        )
        msg = (
            f"[DroneOrcaEnv] {reason} sim_t={self.data.time:.5f} "
            f"hover_thrust={self._hover_thrust:.4f} subtree_kg={self._hover_thrust / 9.81:.4f} "
            f"cmd[ws,ad,rf,qe]={cmd.round(3)} "
            f"thrust_cmd={self._last_thrust_scalar:.4f} tau_cmd={np.round(self._last_tau_cmd, 5)} "
            f"ge_fac={self._last_ground_effect_factor:.3f} "
            f"xfrc_body[Fxyz,Txyz]={np.round(self._last_xfrc_body, 4)} "
            f"Drone_xyz={np.round(self.data.body_xpos(self._drone_body), 4)} euler_drone_deg={np.round(euler_deg, 2)}{frame_euler_suffix} "
            f"scene_max|qacc|={float(np.max(np.abs(qacc))) if nv else 0:.4e}@{imax}({self._v_dof_labels[imax] if nv else ''}) "
            f"scene_max|qvel|={float(np.max(np.abs(qvel))) if nv else 0:.4e}@{vmax}({self._v_dof_labels[vmax] if nv else ''}) "
            f"drone_max|qacc|={float(np.max(np.abs(qa_d))) if qa_d.size else 0:.4e}@{gidx_d_acc}({self._v_dof_labels[gidx_d_acc] if nv and gidx_d_acc >= 0 else ''}) "
            f"drone_max|qvel|={float(np.max(np.abs(qv_d))) if qv_d.size else 0:.4e}@{gidx_d_vel}({self._v_dof_labels[gidx_d_vel] if nv and gidx_d_vel >= 0 else ''}) "
            f"qacc_drone_free={np.round(qacc_free, 3)} qacc_drone_rotors={np.round(qacc_rot_flat, 3)} "
            f"qvel_drone_free={np.round(qvel_free, 3)} qvel_drone_rotors={np.round(qvel_rot_flat, 3)}"
            f"{pose_suffix}{contact_suffix}"
        )
        self._diag_warning(msg)

    def _mean_rotor_omega(self) -> float:
        if not self._rotor_specs:
            return 0.0
        return float(np.mean([self._rotor_speeds[s.joint_suffix] for s in self._rotor_specs]))

    def _maybe_log_vertical_ramp_progress(self) -> None:
        vz_cfg = self._aero.vertical_z_only
        if not self._should_emit_diag_logs() or not vz_cfg.enabled or not vz_cfg.thrust_ramp_enabled:
            return
        dt_log = float(vz_cfg.ramp_progress_log_interval_s)
        if dt_log <= 0.0:
            return
        t = float(self.data.time)
        if t - self._last_vertical_ramp_log_t < dt_log:
            return
        self._last_vertical_ramp_log_t = t
        mg = max(float(self._hover_thrust), 1e-9)
        lo, hi = self._free_dof_lo, self._free_dof_hi
        vz = float(self.data.qvel[lo + 2]) if hi - lo == 6 else float(self.data.body_cvel(self._drone_body)[5])
        om_m = self._mean_rotor_omega()
        self._diag_warning(
            f"[DroneOrcaEnv] ramp sim_t={t:.2f}s T/(mg)={self._last_thrust_scalar/mg:.4f} "
            f"thrust_N={self._last_thrust_scalar:.4f} vz={vz:.5f} z={float(self.data.body_xpos(self._drone_body)[2]):.4f}m "
            f"omega_mean={om_m:.2f}rad/s"
        )
        dur = max(float(vz_cfg.thrust_ramp_duration_s), 1e-6)
        if t >= dur - 1e-9 and not self._vertical_ramp_t1_logged:
            self._vertical_ramp_t1_logged = True
            self._diag_warning(
                f"[DroneOrcaEnv] ramp 已达时长上限 sim_t={t:.2f}s，当前 T/(mg)={self._last_thrust_scalar/mg:.4f} "
                f"(目标系数 {vz_cfg.thrust_ramp_t1_factor})"
            )

    def _update_rotors(self, command: np.ndarray, dt: float) -> None:
        """在 mj_step 之后更新桨角：仅改 qpos（相位积分），qvel 恒为 0，避免与积分步内耦合产生巨大广义加速度。"""
        ws_cmd, ad_cmd, command_z, command_yaw = [float(v) for v in command]
        if self._aero.vertical_z_only.enabled:
            ratio = float(self._last_thrust_scalar) / max(float(self._hover_thrust), 1e-9)
            collective = self._hover_rotor_speed * float(np.clip(ratio, 0.0, 3.5))
            vz_cfg = self._aero.vertical_z_only
            if float(vz_cfg.keyboard_world_xy_force_factor) > 1e-12:
                pitch_term = ws_cmd * 0.35 * self._rotor_speed_delta
                roll_term = ad_cmd * 0.35 * self._rotor_speed_delta
            else:
                pitch_term = roll_term = 0.0
            yaw_term = 0.0
        else:
            planar = min(1.0, math.hypot(ws_cmd, ad_cmd))
            collective = self._hover_rotor_speed + command_z * self._rotor_speed_delta + planar * 0.25 * self._rotor_speed_delta
            pitch_term = ws_cmd * 0.35 * self._rotor_speed_delta
            roll_term = ad_cmd * 0.35 * self._rotor_speed_delta
            yaw_term = command_yaw * 0.22 * self._rotor_speed_delta
        targets = {
            "FL_joint": collective + pitch_term - roll_term + yaw_term,
            "FR_joint": collective + pitch_term + roll_term - yaw_term,
            "BL_joint": collective - pitch_term - roll_term - yaw_term,
            "BR_joint": collective - pitch_term + roll_term + yaw_term,
        }
        for primary_name, sec_name in self._secondary_rotor_map.items():
            targets[sec_name] = targets[primary_name]
        if not self._aero.vertical_z_only.enabled:
            for joint_suffix, bias in self._demo_rotor_bias.items():
                targets[joint_suffix] += bias

        omega_cap = self._hover_rotor_speed + self._rotor_speed_delta * 2.0
        if self._aero.vertical_z_only.enabled:
            omega_cap = self._hover_rotor_speed * 3.5

        max_delta = self._rotor_ramp_rate * dt
        qpos_update = {}
        qvel_update = {}
        for spec in self._rotor_specs:
            joint_suffix = spec.joint_suffix
            current = self._rotor_speeds[joint_suffix]
            target = float(np.clip(targets[joint_suffix], 0.0, omega_cap))
            delta = float(np.clip(target - current, -max_delta, max_delta))
            new_speed = current + delta
            self._rotor_speeds[joint_suffix] = new_speed

            signed_speed = spec.spin_sign * new_speed
            self._rotor_phases[joint_suffix] = math.remainder(
                self._rotor_phases[joint_suffix] + signed_speed * dt,
                2.0 * math.pi,
            )
            qpos_update[self._rotor_joints[joint_suffix]] = np.array(
                [self._rotor_phases[joint_suffix]], dtype=np.float64
            )
            qvel_update[self._rotor_joints[joint_suffix]] = np.array([0.0], dtype=np.float64)

        self.apply_joint_qpos_dict(qpos_update)
        self.apply_joint_qvel_dict(qvel_update)
        self.mj_forward()

    def _update_gripper(self, dt: float) -> None:
        if not self._gripper_enabled:
            return
        max_delta = self._gripper_close_speed * dt
        qpos_update = {}
        qvel_update = {}
        if not hasattr(self, '_gripper_diag_logged'):
            self._gripper_diag_logged = True
            _logger.info(f"[DroneOrcaEnv] _update_gripper first call: target={self._gripper_target:.2f}, joints={self._gripper_joints}")
        for gname, full_name in self._gripper_joints.items():
            try:
                qadr = self.jnt_qposadr(full_name)
                _ = self.jnt_dofadr(full_name)
            except (KeyError, ValueError):
                _logger.warning(f"[DroneOrcaEnv] gripper joint '{gname}' -> full_name='{full_name}' not found in model")
                continue
            jrange_arr = self.model.get_joint_qposrange([full_name])[0]
            jrange = (float(jrange_arr[0]), float(jrange_arr[1]))
            current_pos = float(self.data.qpos[qadr])
            if self._gripper_target < 0:
                lo, hi = jrange
                target_pos = hi if abs(hi) > abs(lo) else lo
                target_pos *= min(abs(self._gripper_target), 1.0)
            else:
                target_pos = 0.0
            target_pos = float(np.clip(target_pos, jrange[0], jrange[1]))
            delta = float(np.clip(target_pos - current_pos, -max_delta, max_delta))
            new_pos = float(np.clip(current_pos + delta, jrange[0], jrange[1]))
            qpos_update[full_name] = np.array([new_pos], dtype=np.float64)
            qvel_update[full_name] = np.array([delta / max(dt, 1e-6)], dtype=np.float64)
        if qpos_update:
            self.apply_joint_qpos_dict(qpos_update)
            self.apply_joint_qvel_dict(qvel_update)

    def _get_obs(self) -> np.ndarray:
        self.mj_forward()
        _drone_pose = self.get_body_xpos_xmat_xquat([self._drone_body])[self._drone_body]
        position = _drone_pose["xpos"]
        quat = _drone_pose["xquat"]
        euler = rotations.quat2euler(quat)
        cvel = self.data.body_cvel(self._drone_body)
        linear_velocity = np.array(cvel[3:6], dtype=np.float32)
        angular_velocity = np.array(cvel[0:3], dtype=np.float32)
        rotor_speed = np.array([self._rotor_speeds[spec.joint_suffix] for spec in self._rotor_specs], dtype=np.float32)
        return np.concatenate(
            [
                np.array(position, dtype=np.float32).flatten(),
                np.array(euler, dtype=np.float32).flatten(),
                linear_velocity,
                angular_velocity,
                rotor_speed,
                self._last_command.astype(np.float32),
            ]
        )

    def _get_info(self, *, reset_requested: bool) -> dict:
        _drone_pose = self.get_body_xpos_xmat_xquat([self._drone_body])[self._drone_body]
        position = _drone_pose["xpos"]
        quat = _drone_pose["xquat"]
        mg = max(float(self._hover_thrust), 1e-9)
        return {
            "position": np.array(position, dtype=np.float32).copy(),
            "euler": rotations.quat2euler(quat).astype(np.float32),
            "rotor_speeds": np.array(
                [self._rotor_speeds[spec.joint_suffix] for spec in self._rotor_specs],
                dtype=np.float32,
            ),
            "command": self._last_command.copy(),
            "reset_requested": reset_requested,
            "autoplay_enabled": self._autoplay_enabled,
            "vertical_z_only": self._aero.vertical_z_only.enabled,
            "takeoff_crossing_logged": self._takeoff_crossing_logged,
            "takeoff_sustained_logged": self._takeoff_sustained_logged,
            "takeoff_z_ref_frame": float(self._takeoff_z_ref),
            "thrust_scalar": float(self._last_thrust_scalar),
            "thrust_over_hover": float(self._last_thrust_scalar / mg),
        }
