import math
from dataclasses import dataclass, replace
from typing import Optional

import mujoco
import numpy as np
from gymnasium import spaces

from envs.drone.drone_aero_config import DEFAULT_DRONE_AERO_CONFIG
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.utils import rotations

_logger = get_orca_logger()


def _joint_dof_bounds(mjm: mujoco.MjModel, joint_name: str) -> tuple[int, int]:
    jid = int(mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
    if jid < 0:
        return 0, 0
    adr = int(mjm.jnt_dofadr[jid])
    jt = int(mjm.jnt_type[jid])
    if jt == mujoco.mjtJoint.mjJNT_FREE:
        n = 6
    elif jt == mujoco.mjtJoint.mjJNT_BALL:
        n = 3
    elif jt in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
        n = 1
    else:
        n = 0
    return adr, adr + n


@dataclass(frozen=True)
class RotorSpec:
    joint_suffix: str
    spin_sign: float


class DroneOrcaEnv(OrcaGymLocalEnv):
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

        self._keyboard = KeyboardInput(KeyboardInputSourceType.ORCASTUDIO, orcagym_addr)
        self._last_space_state = 0

        self._free_joint_suffix = "drone_free"
        self._free_joint = self._resolve_name("joints", self._free_joint_suffix)
        self._rotor_specs = [
            RotorSpec("FL_joint", 1.0),
            RotorSpec("FR_joint", -1.0),
            RotorSpec("BL_joint", -1.0),
            RotorSpec("BR_joint", 1.0),
        ]
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
        self._drone_frame_body = self._resolve_name("bodies", "drone_frame")

        self._drone_body_id = int(self.model.body_name2id(self._drone_body))
        self._frame_body_id = int(self.model.body_name2id(self._drone_frame_body))
        fixed_r = float(vertical_fixed_thrust_over_hover)
        use_fixed_thrust = fixed_r >= 0.0
        ramp_on = bool(vertical_thrust_ramp) and not use_fixed_thrust
        bvz = DEFAULT_DRONE_AERO_CONFIG.vertical_z_only
        if vertical_keyboard_xy_force_factor is not None:
            bvz = replace(bvz, keyboard_world_xy_force_factor=float(vertical_keyboard_xy_force_factor))
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
        self._aero = replace(DEFAULT_DRONE_AERO_CONFIG, vertical_z_only=vz_cfg)

        subtree_mass = float(self.gym._mjModel.body_subtreemass[self._frame_body_id])
        # 与 subtree 重力平衡；略高会持续爬升。接触/地面效应可用键盘垂直通道微调。
        self._hover_thrust = subtree_mass * 9.81
        self._thrust_cmd_scale = 0.55 * self._hover_thrust
        # 全量模式滚转/俯仰改由「目标推力方向」PD（见 _apply_thrust_and_drag）；偏航杆量保守些，避免在 WASD/RF 下被航向环带偏
        self._tau_yaw = 0.012 * self._hover_thrust
        self._thrust_min = max(0.12 * self._hover_thrust, 0.02)
        self._thrust_max = 2.2 * self._hover_thrust
        if self._aero.vertical_z_only.enabled:
            # 爬升扫描 t1 可能 >2；键盘竖直通道也需要余量，避免顶到上限后「假悬停」
            t1 = float(self._aero.vertical_z_only.thrust_ramp_t1_factor)
            self._thrust_max = max(self._thrust_max, (t1 + 0.35) * self._hover_thrust, 3.0 * self._hover_thrust)

        self._hover_rotor_speed = 42.0
        self._rotor_speed_delta = 24.0
        self._rotor_ramp_rate = 80.0
        self._demo_rotor_bias = {
            "FL_joint": 60.0,
            "FR_joint": 60.0,
            "BL_joint": 60.0,
            "BR_joint": 60.0,
        }

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

        mjm = self.gym._mjModel
        self._v_dof_labels: list[str] = []
        for i in range(int(mjm.nv)):
            jid = int(mjm.dof_jntid[i])
            jn = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint{jid}"
            self._v_dof_labels.append(f"v[{i}]<{jn}>")

        self._diag_env_steps = 0
        self._diag_every_env_steps = 20
        self._last_xfrc_body = np.zeros(6, dtype=np.float64)
        self._last_thrust_scalar = 0.0
        self._last_tau_cmd = np.zeros(3, dtype=np.float64)
        self._last_ground_effect_factor = 1.0
        # 全量模式：松杆 XY 抱死用滤波速度，削弱 Orca/步进噪声 → 水平来回晃
        self._full_mode_filt_vxy = np.zeros(2, dtype=np.float64)
        self._full_mode_thrust_lpf = float(self._hover_thrust)

        free_lo, free_hi = _joint_dof_bounds(mjm, self._free_joint)
        self._free_dof_lo = free_lo
        self._free_dof_hi = free_hi
        rotor_bounds = [_joint_dof_bounds(mjm, self._rotor_joints[s.joint_suffix]) for s in self._rotor_specs]
        self._rotor_dof_bounds = rotor_bounds
        drone_idx: list[int] = []
        for lo, hi in ((free_lo, free_hi), *rotor_bounds):
            drone_idx.extend(range(lo, hi))
        self._drone_dof_indices = np.asarray(drone_idx, dtype=np.int32)

        rotor_jids: set[int] = set()
        for spec in self._rotor_specs:
            jid = int(mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, self._rotor_joints[spec.joint_suffix]))
            if jid >= 0:
                rotor_jids.add(jid)
        rotor_actuator_names: list[str] = []
        for ia in range(int(mjm.nu)):
            if int(mjm.actuator_trntype[ia]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            trnjid = int(mjm.actuator_trnid[ia])
            if trnjid in rotor_jids:
                aname = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, ia) or f"actuator_{ia}"
                rotor_actuator_names.append(aname)
        if rotor_actuator_names:
            _logger.warning(
                "[DroneOrcaEnv] 模型仍含绑定桨关节的执行器 "
                f"{rotor_actuator_names}，会与脚本写 qpos/qvel 冲突并导致极大 qacc；"
                "请使用当前仓库无桨执行器的 drone-v1.xml 并在 OrcaStudio 中重新导入/替换资产。"
            )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = self.generate_observation_space(self._get_obs())

        if int(self.nv) != len(self._drone_dof_indices) or self._free_dof_lo != 0:
            _logger.warning(
                f"[DroneOrcaEnv] 场景 nv={self.nv}，无人机相关 dof 索引为 {self._drone_dof_indices.tolist()} "
                f"(drone_free=[{self._free_dof_lo},{self._free_dof_hi}))；physics 日志按关节名切片，不再使用 qacc[0:6] 假定。"
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
            _logger.warning(
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

    def _ctrl_align_rotor_position_actuators(self) -> np.ndarray:
        """若旧资产仍带桨 position 执行器：ctrl 与相位对齐；无执行器时全零即可（桨纯脚本驱动）。"""
        ctrl = np.zeros(self.nu, dtype=np.float32)
        if not self._rotor_actuators:
            return ctrl
        mjm = self.gym._mjModel
        for spec in self._rotor_specs:
            aname = self._rotor_actuators.get(spec.joint_suffix)
            if not aname:
                continue
            aid = int(mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, aname))
            if aid < 0:
                continue
            lo, hi = float(mjm.actuator_ctrlrange[aid, 0]), float(mjm.actuator_ctrlrange[aid, 1])
            ph = float(self._rotor_phases[spec.joint_suffix])
            ctrl[aid] = float(np.clip(ph, lo, hi))
        return ctrl

    def render_callback(self, mode="human") -> None:
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")

    def reset_model(self):
        self._takeoff_crossing_logged = False
        self._takeoff_sustained_logged = False
        self._takeoff_sustain_accum_s = 0.0
        self._vertical_ramp_t1_logged = False
        self._last_vertical_ramp_log_t = -1.0e9
        self._rotor_phases = {
            spec.joint_suffix: self._initial_rotor_qpos[self._rotor_joints[spec.joint_suffix]]
            for spec in self._rotor_specs
        }
        self._rotor_speeds = {spec.joint_suffix: 0.0 for spec in self._rotor_specs}
        self._last_command[:] = 0.0
        self._autoplay_time = 0.0
        self._last_space_state = 0

        free_q = self._initial_free_qpos.copy()
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

        self.gym._mjData.xfrc_applied[self._drone_body_id].fill(0.0)
        self.set_joint_qpos(qpos_update)
        self.set_joint_qvel(qvel_update)
        self.set_ctrl(self._ctrl_align_rotor_position_actuators())
        self.mj_forward()
        if not self._aero.vertical_z_only.enabled:
            self._full_mode_filt_vxy[:] = 0.0
            self._full_mode_thrust_lpf = float(self._hover_thrust)
        # Orca 同步偶发在 mj_forward 后写回非零关节速度，再清一次 free
        self.set_joint_qvel({self._free_joint: np.zeros(6, dtype=np.float64)})
        self.mj_forward()
        self.gym.update_data()
        self._takeoff_z_ref = float(self.gym._mjData.xpos[self._frame_body_id, 2])
        vz_cfg = self._aero.vertical_z_only
        if vz_cfg.enabled and vz_cfg.thrust_ramp_enabled:
            _logger.warning(
                "[DroneOrcaEnv] ramp 扫描开始 "
                f"{vz_cfg.thrust_ramp_t0_factor}·mg→{vz_cfg.thrust_ramp_t1_factor}·mg "
                f"/ {vz_cfg.thrust_ramp_duration_s}s，每 {vz_cfg.ramp_progress_log_interval_s}s 一条进度；"
                "满足持续起飞判据时打「持续起飞临界(精估)」。"
            )
        elif (
            vz_cfg.enabled
            and float(vz_cfg.fixed_thrust_over_hover) >= 0.0
            and not self._vertical_quiet_diag_logs
        ):
            _logger.warning(
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
            self.gym.update_data()
            if not unstable_logged_this_step and self._drone_physics_should_warn_immediate():
                self._emit_drone_physics_warning("unstable_post_mj_step")
                unstable_logged_this_step = True

        self._diag_env_steps += 1
        if (
            self._diag_every_env_steps > 0
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
        data = self.gym._mjData
        lo, hi = self._free_dof_lo, self._free_dof_hi
        if hi - lo != 6:
            return
        qv = np.asarray(data.qvel[lo:hi], dtype=np.float64).reshape(6)
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
            self.set_joint_qvel({self._free_joint: clipped})
            self.mj_forward()

    def _read_keyboard_command(self) -> tuple[np.ndarray, bool]:
        self._keyboard.update()
        state = self._keyboard.get_state()

        space_pressed = int(state["Space"])
        reset_requested = self._last_space_state == 0 and space_pressed == 1
        self._last_space_state = space_pressed

        if self._autoplay_enabled:
            command = self._build_autoplay_command()
        else:
            command = np.array(
                [
                    state["W"] - state["S"],
                    state["A"] - state["D"],
                    state["R"] - state["F"],
                    state["Q"] - state["E"],
                ],
                dtype=np.float32,
            )
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
        data = self.gym._mjData
        vz_cfg = self._aero.vertical_z_only
        thrust_bid = (
            self._frame_body_id if vz_cfg.apply_thrust_at_free_frame else self._drone_body_id
        )

        forward_cmd, lateral_cmd, vertical_cmd, _yaw_cmd = [float(np.clip(v, -1.0, 1.0)) for v in command]

        if float(vz_cfg.fixed_thrust_over_hover) >= 0.0:
            r = float(vz_cfg.fixed_thrust_over_hover)
            thrust = float(np.clip(r * self._hover_thrust, self._thrust_min, self._thrust_max))
        elif vz_cfg.thrust_ramp_enabled:
            dur = max(float(vz_cfg.thrust_ramp_duration_s), 1e-6)
            u = min(1.0, float(data.time) / dur)
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
            vz = float(data.qvel[lo + 2])
        else:
            vz = float(data.cvel[self._drone_body_id][5])
        k_l = float(vz_cfg.vz_linear_k)
        k_q = float(vz_cfg.vz_quadratic_k)
        fz = thrust - k_l * vz - k_q * abs(vz) * vz

        mg_h = max(float(self._hover_thrust), 1e-9)
        kxy = float(vz_cfg.keyboard_world_xy_force_factor)
        fx = 0.0
        fy = 0.0
        if kxy > 1e-12 and hi - lo == 6:
            # 用推力作用体的 COM 世界系线速度做阻尼/刹停，与全量模式 drag 一致；勿与 qvel 混用以免原点/COM 不一致
            cvel = data.cvel[thrust_bid]
            vx_w = float(cvel[3])
            vy_w = float(cvel[4])
            kxd = float(vz_cfg.keyboard_world_xy_vel_damping)
            dead = float(self._aero.drag.zero_cmd_hold_deadband)
            cap_xy = float(self._aero.drag.zero_cmd_xy_hold_force_cap)
            planar_idle = abs(forward_cmd) < dead and abs(lateral_cmd) < dead
            if planar_idle:
                kxh = kxd + float(self._aero.drag.zero_cmd_xy_hold_k)
                fx = float(np.clip(-kxh * vx_w, -cap_xy, cap_xy))
                fy = float(np.clip(-kxh * vy_w, -cap_xy, cap_xy))
            else:
                fcx = lateral_cmd * kxy * mg_h
                fcy = forward_cmd * kxy * mg_h
                fcap = 1.55 * kxy * mg_h
                hc = float(math.hypot(fcx, fcy))
                if hc > fcap and hc > 1e-12:
                    s = fcap / hc
                    fcx *= s
                    fcy *= s
                fx = fcx - kxd * vx_w
                fy = fcy - kxd * vy_w

        f_w = np.array([fx, fy, fz], dtype=np.float64)

        data.xfrc_applied[self._drone_body_id].fill(0.0)
        if self._frame_body_id != self._drone_body_id:
            data.xfrc_applied[self._frame_body_id].fill(0.0)
        # xfrc_applied 线力为世界系；竖直模式在 lock 姿态下 R≈I，写 f_w 与旧版写 R^T f_w 等价，解锁姿态时亦正确
        data.xfrc_applied[thrust_bid, 0:3] = f_w
        data.xfrc_applied[thrust_bid, 3:6] = 0.0

        r_w_b = data.xmat[thrust_bid].reshape(3, 3)
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
        self.set_joint_qpos({self._free_joint: new_q})
        qv = np.asarray(self.gym._mjData.qvel[lo:hi], dtype=np.float64).reshape(6)
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
        self.set_joint_qvel({self._free_joint: new_v})
        self.mj_forward()

    def _maybe_log_takeoff_first_vz_spike(self) -> None:
        if self._vertical_quiet_diag_logs:
            return
        vz_cfg = self._aero.vertical_z_only
        if not vz_cfg.takeoff_log_first_vz_spike or self._takeoff_crossing_logged:
            return
        if not (vz_cfg.thrust_ramp_enabled or float(vz_cfg.fixed_thrust_over_hover) >= 0.0):
            return
        lo, hi = self._free_dof_lo, self._free_dof_hi
        if hi - lo != 6:
            return
        data = self.gym._mjData
        vz = float(data.qvel[lo + 2])
        thr = float(self._last_thrust_scalar)
        mg = max(self._hover_thrust, 1e-9)
        if vz > float(vz_cfg.takeoff_vz_threshold):
            self._takeoff_crossing_logged = True
            om_m = float(np.mean([self._rotor_speeds[s.joint_suffix] for s in self._rotor_specs]))
            _logger.warning(
                "[DroneOrcaEnv] 首次 vz 过阈(易与弹跳混淆): "
                f"sim_t={float(data.time):.4f}s vz={vz:.4f}m/s z={float(data.xpos[self._frame_body_id, 2]):.4f}m "
                f"thrust={thr:.4f}N T/(mg)={thr/mg:.4f} 桨ω_mean≈{om_m:.2f}rad/s"
            )

    def _update_takeoff_sustain_detector(self) -> None:
        if self._vertical_quiet_diag_logs:
            return
        vz_cfg = self._aero.vertical_z_only
        if self._takeoff_sustained_logged:
            return
        if not (vz_cfg.thrust_ramp_enabled or float(vz_cfg.fixed_thrust_over_hover) >= 0.0):
            return
        lo, hi = self._free_dof_lo, self._free_dof_hi
        if hi - lo != 6:
            return
        data = self.gym._mjData
        vz = float(data.qvel[lo + 2])
        z = float(data.xpos[self._frame_body_id, 2])
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
        _logger.warning(
            "[DroneOrcaEnv] 持续起飞临界(精估): "
            f"sim_t={float(data.time):.4f}s Δz={(z - self._takeoff_z_ref):.4f}m vz={vz:.4f}m/s "
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

        data = self.gym._mjData
        # 与竖直模式一致：力/力矩加在 drone_frame（free 关节体），不用 Drone 子体（USD/网格装配角会使子体 +Z 非竖直）
        bid = self._frame_body_id
        data.xfrc_applied[self._drone_body_id].fill(0.0)
        data.xfrc_applied[self._frame_body_id].fill(0.0)

        forward_cmd, lateral_cmd, vertical_cmd, yaw_cmd = [float(np.clip(v, -1.0, 1.0)) for v in command]

        planar_mag = min(1.0, float(math.hypot(forward_cmd, lateral_cmd)))
        max_tilt_rad = math.radians(18.0)
        tilt_mag = max_tilt_rad * planar_mag
        # 倾转时按 cos(theta) 轻补集体升力，保留微倾斜视觉同时减少 WASD 触发的掉高/补偿振荡。
        thrust_hover_comp = 1.0 / max(math.cos(tilt_mag), math.cos(max_tilt_rad))
        thrust_target = float(
            np.clip(
                (self._hover_thrust * thrust_hover_comp) + vertical_cmd * self._thrust_cmd_scale,
                self._thrust_min,
                self._thrust_max,
            )
        )
        tau_t = float(max(self._aero.drag.full_mode_thrust_lpf_tau_s, 0.02))
        lpf_a = min(1.0, dt / tau_t)
        self._full_mode_thrust_lpf = float(
            self._full_mode_thrust_lpf + lpf_a * (thrust_target - self._full_mode_thrust_lpf)
        )
        thrust = float(self._full_mode_thrust_lpf)

        R_d = data.xmat[bid].reshape(3, 3)
        cvel = data.cvel[bid]
        omega_w_raw = np.asarray(cvel[0:3], dtype=np.float64).reshape(3)
        omega_b_raw = R_d.T @ omega_w_raw
        dcfg = self._aero.drag
        v_raw = np.asarray(cvel[3:6], dtype=np.float64).reshape(3)
        vc = float(dcfg.aero_model_velocity_clip)
        v_w = np.clip(v_raw, -vc, vc)
        dead = float(dcfg.zero_cmd_hold_deadband)
        planar_idle = abs(forward_cmd) < dead and abs(lateral_cmd) < dead
        vert_idle = abs(vertical_cmd) < dead
        yaw_idle = abs(yaw_cmd) < dead
        full_idle = planar_idle and vert_idle and yaw_idle
        vert_active = not vert_idle
        planar_active = not planar_idle
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
            # 键盘全松：不要世界系 XY 抱死，否则小幅 v 噪声 + 倾侧推力分力易形成极限环晃
            if full_idle:
                f_xy_w = np.zeros(3, dtype=np.float64)
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
            stick_xy = min(1.0, float(math.hypot(forward_cmd, lateral_cmd)))
            k_xy_eff = k_xy_base * (1.0 - 0.5 * stick_xy)
            f_xy_w = np.array([-k_xy_eff * v_w[0], -k_xy_eff * v_w[1], 0.0], dtype=np.float64)
        f_z_hold_w = np.zeros(3, dtype=np.float64)
        if vert_idle:
            kzh = float(dcfg.zero_cmd_z_hold_k)
            cap_z = float(dcfg.zero_cmd_z_hold_force_cap)
            f_z_hold_w[2] = float(np.clip(-kzh * v_raw[2], -cap_z, cap_z))

        # 目标推力方向（世界系）：杆量把机体 +Z 从竖直倾向 (lateral, forward)，产生稳定水平分力；松杆拉回竖直并阻尼角速度，减轻晃振
        ez_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        zb_w = R_d @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        h = float(math.hypot(lateral_cmd, forward_cmd))
        if planar_idle or h < 1e-8:
            zb_des_w = ez_w.copy()
        else:
            ux = lateral_cmd / h
            uy = forward_cmd / h
            sm = min(h, 1.0)
            ang = max_tilt_rad * sm
            ca, sa = math.cos(ang), math.sin(ang)
            zb_des_w = np.array([sa * ux, sa * uy, ca], dtype=np.float64)
        e_w = np.cross(zb_w, zb_des_w)
        en = float(np.linalg.norm(e_w))
        if en > 1e-12:
            e_cap = math.sin(math.radians(20.0))
            e_w = e_w * (min(en, e_cap) / en)
        if full_idle:
            # 全松杆：小姿态误差不进 P，避免与 D 一起顶满 trp；压低 ω 裁剪上限防 D 项单独饱和
            en_db = math.sin(math.radians(2.5))
            if en < en_db:
                e_w[:] = 0.0
            om_cap_rp = 2.0
            om_cap_y = 3.2
            Kp_align = 0.11
            Kd_rp = 0.035
            trp_max = 0.08
            Kd_yaw = 0.03
        elif planar_idle and vert_idle and not yaw_idle:
            # 仅 Q/E：full_idle 为假会误走 aggressive，滚仰顶满 trp 与偏航耦合成剧烈抖
            en_db = math.sin(math.radians(2.2))
            if en < en_db:
                e_w[:] = 0.0
            om_cap_rp = 2.2
            om_cap_y = 5.0
            Kp_align = 0.12
            Kd_rp = 0.04
            trp_max = 0.09
            Kd_yaw = 0.05
        elif vert_active and not planar_active:
            # 仅 R/F：推力在爬升/下降，硬滚仰 PD 会与 T 突变强耦合 → 与全松杆同级软环
            en_db = math.sin(math.radians(2.2))
            if en < en_db:
                e_w[:] = 0.0
            om_cap_rp = 2.2
            om_cap_y = 3.4
            Kp_align = 0.12
            Kd_rp = 0.04
            trp_max = 0.09
            Kd_yaw = 0.03
        elif vert_active and planar_active:
            om_cap_rp = 2.8
            om_cap_y = 4.0
            Kp_align = 0.30
            Kd_rp = 0.05
            trp_max = 0.14
            Kd_yaw = 0.04
        elif planar_active:
            # 仅有 WASD（可无 Q/E、无 R/F）：原 else aggressive 易与倾转/阻尼打架，按键即抖
            om_cap_rp = 2.8
            om_cap_y = 4.0
            Kp_align = 0.28
            Kd_rp = 0.05
            trp_max = 0.14
            Kd_yaw = 0.04
        else:
            om_cap_rp = 2.2
            om_cap_y = 3.4
            Kp_align = 0.12
            Kd_rp = 0.04
            trp_max = 0.09
            Kd_yaw = 0.03
        wxc = float(np.clip(omega_b_raw[0], -om_cap_rp, om_cap_rp))
        wyc = float(np.clip(omega_b_raw[1], -om_cap_rp, om_cap_rp))
        wzc = float(np.clip(omega_b_raw[2], -om_cap_y, om_cap_y))
        tau_b_tilt = Kp_align * (R_d.T @ e_w)
        tau_x = float(np.clip(tau_b_tilt[0] - Kd_rp * wxc, -trp_max, trp_max))
        tau_y = float(np.clip(tau_b_tilt[1] - Kd_rp * wyc, -trp_max, trp_max))
        tau_z = 0.0 if yaw_idle else float(-yaw_cmd * self._tau_yaw - Kd_yaw * wzc)
        # 额外世界系 ψ 稳定见 tau_w 合成后（避免子体欧拉第三角万向节问题）

        # MuJoCo mjData.xfrc_applied 的线力与力矩均为**世界系**（作用在质心）。机体系集体升力须用 R@f_b 再写入，否则 [0,0,T] 会被当成世界 +Z，倾转不产生水平分力，WASD 几乎无法平移。
        f_collective_b = np.array([0.0, 0.0, thrust], dtype=np.float64)
        f_stab_w = f_xy_w + f_z_hold_w
        f_w = R_d @ f_collective_b + f_stab_w
        kq_xy = float(dcfg.quad_world_xy_stick_force_factor)
        if kq_xy > 1e-12 and not planar_idle:
            mg_h = max(float(self._hover_thrust), 1e-9)
            f_w[0] += lateral_cmd * kq_xy * mg_h
            f_w[1] += forward_cmd * kq_xy * mg_h

        tau_b_cmd = np.array([tau_x, tau_y, tau_z], dtype=np.float64)

        tcap = float(dcfg.angular_drag_torque_axis_max)
        tau_drag_b = np.array(
            [
                float(np.clip(-dcfg.angular_xy * omega_b_raw[0], -tcap, tcap)),
                float(np.clip(-dcfg.angular_xy * omega_b_raw[1], -tcap, tcap)),
                float(np.clip(-dcfg.angular_z * omega_b_raw[2], -tcap, tcap)),
            ],
            dtype=np.float64,
        )
        # 全松杆时不叠这一层，避免与「软倾转 PD + angular_xy」重复阻尼导致滚仰力矩来回顶格
        if planar_idle and yaw_idle and not full_idle:
            k_ah = float(dcfg.zero_cmd_angular_hold_k)
            c_ah = float(dcfg.zero_cmd_angular_torque_axis_max)
            tau_drag_b += np.array(
                [
                    float(np.clip(-k_ah * omega_b_raw[0], -c_ah, c_ah)),
                    float(np.clip(-k_ah * omega_b_raw[1], -c_ah, c_ah)),
                    float(np.clip(-k_ah * omega_b_raw[2], -c_ah, c_ah)),
                ],
                dtype=np.float64,
            )

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
        tnorm = float(np.linalg.norm(tau_b_tot))
        tmax = float(dcfg.max_body_torque_norm)
        if tnorm > tmax and tnorm > 1e-12:
            tau_b_tot = tau_b_tot * (tmax / tnorm)
        tau_w = R_d @ tau_b_tot
        # 世界系偏航弱稳定：用机头在水平面投影角 + 世界 ωz，避免去掉欧拉回正后航向慢漂带动水平晃
        if yaw_idle:
            fwd = R_d @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
            nxy = float(math.hypot(fwd[0], fwd[1]))
            if nxy > 1e-8:
                psi = float(math.atan2(fwd[1], fwd[0]))
                wz_w = float(omega_w_raw[2])
                if full_idle or (vert_active and not planar_active):
                    tau_w[2] += float(np.clip(-0.012 * psi - 0.03 * wz_w, -0.04, 0.04))
                else:
                    tau_w[2] += float(np.clip(-0.008 * psi - 0.02 * wz_w, -0.03, 0.03))
        tnw = float(np.linalg.norm(tau_w))
        if tnw > tmax and tnw > 1e-12:
            tau_w = tau_w * (tmax / tnw)

        max_fw = float(dcfg.max_total_linear_force_over_hover) * float(self._hover_thrust)
        fn = float(np.linalg.norm(f_w))
        if fn > max_fw and fn > 1e-12:
            f_w = f_w * (max_fw / fn)

        self._last_ground_effect_factor = 1.0
        if self._aero.ground_effect.enabled:
            z = float(data.xpos[bid, 2])
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

        data.xfrc_applied[bid, 0:3] = f_w
        data.xfrc_applied[bid, 3:6] = tau_w

        # 诊断用：与机体系控制律对齐的等效 wrench（世界系施加值旋回机体系）
        self._last_xfrc_body = np.concatenate(
            [R_d.T @ f_w.reshape(3), R_d.T @ tau_w.reshape(3)]
        ).astype(np.float64, copy=False)
        self._last_thrust_scalar = thrust
        self._last_tau_cmd = np.array([tau_x, tau_y, tau_z], dtype=np.float64)

    def _drone_physics_should_warn_immediate(self) -> bool:
        mjd = self.gym._mjData
        nv = int(self.nv)
        if nv <= 0:
            return False
        qacc = np.asarray(mjd.qacc, dtype=np.float64).reshape(-1)
        qvel = np.asarray(mjd.qvel, dtype=np.float64).reshape(-1)
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

    def _emit_drone_physics_warning(self, reason: str) -> None:
        """在全局仅开启 WARNING 时仍可看见：用 WARNING 输出一帧关键动力学量。"""
        mjd = self.gym._mjData
        bid = self._drone_body_id
        qacc = np.asarray(mjd.qacc, dtype=np.float64).reshape(-1)
        qvel = np.asarray(mjd.qvel, dtype=np.float64).reshape(-1)
        nv = int(self.nv)
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
        euler = rotations.quat2euler(np.asarray(mjd.xquat[bid], dtype=np.float64).reshape(1, -1))
        euler_deg = np.rad2deg(euler.reshape(-1))
        frame_euler_suffix = ""
        if not self._aero.vertical_z_only.enabled:
            fe = rotations.quat2euler(
                np.asarray(mjd.xquat[self._frame_body_id], dtype=np.float64).reshape(1, -1)
            )
            frame_euler_suffix = f" frame_euler_deg={np.round(np.rad2deg(fe.reshape(-1)), 2)}"
        cmd = np.asarray(self._last_command, dtype=np.float64).reshape(-1)
        msg = (
            f"[DroneOrcaEnv] {reason} sim_t={mjd.time:.5f} "
            f"hover_thrust={self._hover_thrust:.4f} subtree_kg={self._hover_thrust / 9.81:.4f} "
            f"cmd[ws,ad,rf,qe]={cmd.round(3)} "
            f"thrust_cmd={self._last_thrust_scalar:.4f} tau_cmd={np.round(self._last_tau_cmd, 5)} "
            f"ge_fac={self._last_ground_effect_factor:.3f} "
            f"xfrc_body[Fxyz,Txyz]={np.round(self._last_xfrc_body, 4)} "
            f"Drone_xyz={np.round(mjd.xpos[bid], 4)} euler_drone_deg={np.round(euler_deg, 2)}{frame_euler_suffix} "
            f"scene_max|qacc|={float(np.max(np.abs(qacc))) if nv else 0:.4e}@{imax}({self._v_dof_labels[imax] if nv else ''}) "
            f"scene_max|qvel|={float(np.max(np.abs(qvel))) if nv else 0:.4e}@{vmax}({self._v_dof_labels[vmax] if nv else ''}) "
            f"drone_max|qacc|={float(np.max(np.abs(qa_d))) if qa_d.size else 0:.4e}@{gidx_d_acc}({self._v_dof_labels[gidx_d_acc] if nv and gidx_d_acc >= 0 else ''}) "
            f"drone_max|qvel|={float(np.max(np.abs(qv_d))) if qv_d.size else 0:.4e}@{gidx_d_vel}({self._v_dof_labels[gidx_d_vel] if nv and gidx_d_vel >= 0 else ''}) "
            f"qacc_drone_free={np.round(qacc_free, 3)} qacc_drone_rotors={np.round(qacc_rot_flat, 3)} "
            f"qvel_drone_free={np.round(qvel_free, 3)} qvel_drone_rotors={np.round(qvel_rot_flat, 3)}"
        )
        _logger.warning(msg)

    def _mean_rotor_omega(self) -> float:
        if not self._rotor_specs:
            return 0.0
        return float(np.mean([self._rotor_speeds[s.joint_suffix] for s in self._rotor_specs]))

    def _maybe_log_vertical_ramp_progress(self) -> None:
        vz_cfg = self._aero.vertical_z_only
        if not vz_cfg.enabled or not vz_cfg.thrust_ramp_enabled:
            return
        dt_log = float(vz_cfg.ramp_progress_log_interval_s)
        if dt_log <= 0.0:
            return
        data = self.gym._mjData
        t = float(data.time)
        if t - self._last_vertical_ramp_log_t < dt_log:
            return
        self._last_vertical_ramp_log_t = t
        mg = max(float(self._hover_thrust), 1e-9)
        lo, hi = self._free_dof_lo, self._free_dof_hi
        vz = float(data.qvel[lo + 2]) if hi - lo == 6 else float(data.cvel[self._drone_body_id][5])
        om_m = self._mean_rotor_omega()
        _logger.warning(
            f"[DroneOrcaEnv] ramp sim_t={t:.2f}s T/(mg)={self._last_thrust_scalar/mg:.4f} "
            f"thrust_N={self._last_thrust_scalar:.4f} vz={vz:.5f} z={float(data.xpos[self._drone_body_id, 2]):.4f}m "
            f"omega_mean={om_m:.2f}rad/s"
        )
        dur = max(float(vz_cfg.thrust_ramp_duration_s), 1e-6)
        if t >= dur - 1e-9 and not self._vertical_ramp_t1_logged:
            self._vertical_ramp_t1_logged = True
            _logger.warning(
                f"[DroneOrcaEnv] ramp 已达时长上限 sim_t={t:.2f}s，当前 T/(mg)={self._last_thrust_scalar/mg:.4f} "
                f"(目标系数 {vz_cfg.thrust_ramp_t1_factor})"
            )

    def _update_rotors(self, command: np.ndarray, dt: float) -> None:
        """在 mj_step 之后更新桨角：仅改 qpos（相位积分），qvel 恒为 0，避免与积分步内耦合产生巨大广义加速度。"""
        command_x, command_y, command_z, command_yaw = [float(v) for v in command]
        if self._aero.vertical_z_only.enabled:
            ratio = float(self._last_thrust_scalar) / max(float(self._hover_thrust), 1e-9)
            collective = self._hover_rotor_speed * float(np.clip(ratio, 0.0, 3.5))
            vz_cfg = self._aero.vertical_z_only
            if float(vz_cfg.keyboard_world_xy_force_factor) > 1e-12:
                pitch_term = command_x * 0.35 * self._rotor_speed_delta
                roll_term = command_y * 0.35 * self._rotor_speed_delta
            else:
                pitch_term = roll_term = 0.0
            yaw_term = 0.0
        else:
            planar = min(1.0, math.hypot(command_x, command_y))
            collective = self._hover_rotor_speed + command_z * self._rotor_speed_delta + planar * 0.25 * self._rotor_speed_delta
            pitch_term = command_x * 0.35 * self._rotor_speed_delta
            roll_term = command_y * 0.35 * self._rotor_speed_delta
            yaw_term = command_yaw * 0.22 * self._rotor_speed_delta
        targets = {
            "FL_joint": collective + pitch_term - roll_term + yaw_term,
            "FR_joint": collective + pitch_term + roll_term - yaw_term,
            "BL_joint": collective - pitch_term - roll_term - yaw_term,
            "BR_joint": collective - pitch_term + roll_term + yaw_term,
        }
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

        self.set_joint_qpos(qpos_update)
        self.set_joint_qvel(qvel_update)
        self.mj_forward()

    def _get_obs(self) -> np.ndarray:
        self.mj_forward()
        position, _, quat = self.get_body_xpos_xmat_xquat([self._drone_body])
        euler = rotations.quat2euler(quat)
        cvel = self.gym._mjData.cvel[self._drone_body_id]
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
        position, _, quat = self.get_body_xpos_xmat_xquat([self._drone_body])
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
