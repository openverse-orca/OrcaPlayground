import logging
import numpy as np
from typing import Any, Optional, Tuple

import mujoco
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

from .trajectory.trajectory_frame import HumanTrajectoryStepConfig

_logger = get_orca_logger()
# 轨迹回放分析：grep "[SimEnv.trajectory]"
TRAJ_LOG = logging.getLogger("examples.embodied.fluid.sim_env.trajectory")


class SimEnv(OrcaGymEulerEnv):
    """
    A class to represent the ORCA Gym environment for the Replicator scene.
    """

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        import sys

        print("[PRINT-DEBUG] SimEnv.__init__() - START", file=sys.stderr, flush=True)

        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )
        print(
            "[PRINT-DEBUG] SimEnv.__init__() - super().__init__() completed",
            file=sys.stderr,
            flush=True,
        )

        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

        self._pending_human: Optional[HumanTrajectoryStepConfig] = None
        self._human_trajectory_step_count: int = 0
        self._last_eq_replay_key: Optional[Tuple[Tuple[str, str, int, int, Tuple[float, ...]], ...]] = (
            None
        )

        print("[PRINT-DEBUG] SimEnv.__init__() - Setting obs/action spaces", file=sys.stderr, flush=True)
        self._set_obs_space()
        self._set_action_space()
        print("[PRINT-DEBUG] SimEnv.__init__() - END", file=sys.stderr, flush=True)

    def _set_obs_space(self):
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self):
        if self.nu > 0:
            scaled_action_range = np.concatenate([[[-1.0, 1.0]] for _ in range(self.nu)])
            self.action_space = self.generate_action_space(scaled_action_range)
        else:
            self.action_space = spaces.Box(
                low=np.array([]),
                high=np.array([]),
                dtype=np.float32,
            )
            _logger.info("No action space defined, nu is 0.")

    def apply_joint_qpos_dict(self, joint_qpos_dict: dict) -> None:
        """将 {joint_name: qpos} 字典合并到完整 qpos 数组并应用（Euler 兼容）。"""
        full_qpos = self.data.qpos.copy()
        for jname, jqpos in joint_qpos_dict.items():
            addr = self.jnt_qposadr(jname)
            arr = np.atleast_1d(np.asarray(jqpos, dtype=full_qpos.dtype))
            full_qpos[addr:addr + len(arr)] = arr
        self.set_joint_qpos(full_qpos)

    def apply_joint_qvel_dict(self, joint_qvel_dict: dict) -> None:
        """将 {joint_name: qvel} 字典合并到完整 qvel 数组并应用（Euler 兼容）。"""
        full_qvel = self.data.qvel.copy()
        for jname, jqvel in joint_qvel_dict.items():
            addr = self.jnt_dofadr(jname)
            arr = np.atleast_1d(np.asarray(jqvel, dtype=full_qvel.dtype))
            full_qvel[addr:addr + len(arr)] = arr
        self.set_joint_qvel(full_qvel)

    def set_pending_human_trajectory_step(self, config: HumanTrajectoryStepConfig) -> None:
        """在 step 前写入一帧人类轨迹；step 内消费后清除。覆盖此前未消费的 pending。"""
        self._validate_human_trajectory_step_config(config)
        self._pending_human = config

    def clear_pending_human_trajectory_step(self) -> None:
        """清除未消费的人类轨迹 pending（如 reset / 异常恢复）。"""
        self._pending_human = None

    def _validate_human_trajectory_step_config(self, cfg: HumanTrajectoryStepConfig) -> None:
        ctrl = np.asarray(cfg.ctrl, dtype=np.float32).reshape(-1)
        if ctrl.size != self.nu:
            raise ValueError(
                f"pending ctrl size {ctrl.size} != nu={self.nu}"
            )
        K = len(cfg.mocap_names)
        if cfg.mocap_pos.shape[0] != K or cfg.mocap_quat.shape[0] != K:
            raise ValueError(
                f"mocap rows {cfg.mocap_pos.shape[0]}/{cfg.mocap_quat.shape[0]} != K={K}"
            )
        E = len(cfg.eq_indices)
        ew = self.model.equality_data_width()
        if cfg.eq_active.shape[0] != E:
            raise ValueError("eq_active length mismatch vs eq_indices")
        if len(cfg.eq_obj1_name) != E or len(cfg.eq_obj2_name) != E:
            raise ValueError("eq_obj1_name / eq_obj2_name length mismatch vs eq_indices")
        if cfg.eq_type.shape[0] != E:
            raise ValueError("eq_type length mismatch vs eq_indices")
        if cfg.eq_data.shape[1] != ew:
            raise ValueError(f"eq_data width {cfg.eq_data.shape[1]} != model {ew}")
        if cfg.eq_data.shape[0] != E:
            raise ValueError(f"eq_data rows {cfg.eq_data.shape[0]} != E={E}")

    def render_callback(self, mode="human") -> None:
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")

    def step(self, action: Optional[np.ndarray]) -> tuple:
        """
        若已通过 set_pending_human_trajectory_step 配置本步人类轨迹，则 step 内应用
        mocap/equality/ctrl（此时忽略 action）；否则 action 为外围执行器指令；
        action 为 None 且无 pending 时使用默认搅拌棒占位。
        """
        if self._pending_human is not None:
            cfg = self._pending_human
            self._pending_human = None
            self._human_trajectory_step_count += 1
            self._apply_human_trajectory_mocap_eq(cfg)
            # ctrl 与 mocap/eq 同帧：在 do_simulation 中下发（见 _apply_human_trajectory_mocap_eq 说明）
            ctrl = np.asarray(cfg.ctrl, dtype=np.float32).reshape(self.nu)
        else:
            if action is None:
                ctrl = np.zeros(self.nu, dtype=np.float32)
                for i in range(self.nu):
                    ctrl[i] = 20
            else:
                ctrl = np.asarray(action, dtype=np.float32).reshape(self.nu)

        self.do_simulation(ctrl, self.frame_skip)
        obs = self._get_obs().copy()

        info: dict[str, Any] = {}
        terminated = False
        truncated = False
        reward = 0.0

        return obs, reward, terminated, truncated, info

    def _mj_body_name(self, body_id: int) -> str:
        if body_id < 0:
            return f"<invalid:{body_id}>"
        try:
            n = self.model.body_id2name(body_id)
            return n if n else f"<id:{body_id}>"
        except Exception:
            return f"<id:{body_id}>"

    def _resolve_body_id_from_name(self, body_name: str) -> int:
        if not body_name:
            raise ValueError("empty equality endpoint body name in trajectory")
        try:
            return int(self.model.body_name2id(body_name))
        except Exception as e:
            raise ValueError(
                f"unknown equality endpoint body name {body_name!r}"
            ) from e

    def _equality_replay_key(
        self, cfg: HumanTrajectoryStepConfig
    ) -> Tuple[Tuple[str, str, int, int, Tuple[float, ...]], ...]:
        """与轨迹文件一致的逻辑状态，用于跳过未变更帧的 modify/update。"""
        if not cfg.eq_indices:
            return ()
        rows: list[Tuple[str, str, int, int, Tuple[float, ...]]] = []
        for j in range(len(cfg.eq_indices)):
            n1 = cfg.eq_obj1_name[j]
            n2 = cfg.eq_obj2_name[j]
            ea = int(cfg.eq_active[j])
            et = int(cfg.eq_type[j])
            data_t = tuple(
                float(x) for x in np.asarray(cfg.eq_data[j], dtype=np.float64).ravel()
            )
            rows.append((n1, n2, ea, et, data_t))
        return tuple(rows)

    def _apply_human_trajectory_mocap_eq(self, cfg: HumanTrajectoryStepConfig) -> None:
        """
        仅应用本帧人类 mocap 与 equality，并 mj_forward。

        执行器指令 **ctrl** 不在此设置：`step()` 在调用本方法之后用同一份
        ``cfg.ctrl`` 调用 ``do_simulation(ctrl, frame_skip)``（内部会 ``set_ctrl`` 再步进），
        与计划里「先 SPH/人类 mocap+eq，再物理步进」的顺序一致。
        """
        si = self._human_trajectory_step_count
        if cfg.mocap_names:
            dct: dict[str, dict[str, np.ndarray]] = {}
            for j, name in enumerate(cfg.mocap_names):
                dct[name] = {
                    "pos": np.array(cfg.mocap_pos[j], dtype=np.float64, copy=True),
                    "quat": np.array(cfg.mocap_quat[j], dtype=np.float64, copy=True),
                }
            self.set_mocap_pos_and_quat(dct)
            for name in cfg.mocap_names:
                p = dct[name]["pos"]
                TRAJ_LOG.debug(
                    "[SimEnv.trajectory] step=%s mocap=%s pos=[%.4f,%.4f,%.4f]",
                    si,
                    name,
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                )
        if cfg.eq_indices:
            self._apply_equality_human_row(cfg, step_index=si)
        self.mj_forward()
        ctrl_preview = (
            float(np.asarray(cfg.ctrl, dtype=np.float64).ravel()[0])
            if self.nu > 0
            else 0.0
        )
        TRAJ_LOG.debug(
            "[SimEnv.trajectory] step=%s after_mj_forward ctrl[0]=%.6f (full ctrl in do_simulation next)",
            si,
            ctrl_preview,
        )

    def _apply_equality_human_row(
        self, cfg: HumanTrajectoryStepConfig, step_index: int
    ) -> None:
        eqt_row = cfg.eq_type
        eqd_row = cfg.eq_data
        ea_row = cfg.eq_active

        key = self._equality_replay_key(cfg)
        if self._last_eq_replay_key is not None and key == self._last_eq_replay_key:
            return

        for j, gi in enumerate(cfg.eq_indices):
            name1 = cfg.eq_obj1_name[j]
            name2 = cfg.eq_obj2_name[j]
            f1 = self._resolve_body_id_from_name(name1)
            f2 = self._resolve_body_id_from_name(name2)
            c1, c2 = self.model.equality_object_ids(gi)  # TODO(euler-migration): 原 mj.eq_obj1id/eq_obj2id
            TRAJ_LOG.debug(
                "[SimEnv.trajectory] step=%s eq[%s] target=(%s|%s) id=(%d,%d) "
                "model_before=(%s|%s) id=(%d,%d) active_file=%s",
                step_index,
                gi,
                name1,
                name2,
                f1,
                f2,
                self._mj_body_name(c1),
                self._mj_body_name(c2),
                c1,
                c2,
                int(ea_row[j]),
            )
            if c1 != f1 or c2 != f2:
                TRAJ_LOG.info(
                    "[SimEnv.trajectory] step=%s eq[%s] modify_equality_objects: "
                    "model(%s|%s) id=(%d,%d) -> target(%s|%s) id=(%d,%d)",
                    step_index,
                    gi,
                    self._mj_body_name(c1),
                    self._mj_body_name(c2),
                    c1,
                    c2,
                    name1,
                    name2,
                    f1,
                    f2,
                )
                # Euler 体系用 equality_update 替代 modify_equality_objects：
                # 按 slot 直接更新 obj1/obj2（底层 update_equality_constraints 按
                # 当前 (obj1_id, obj2_id) 匹配槽位写入新值）
                self.equality_update(
                    gi,
                    obj1_name=name1,
                    obj2_name=name2,
                    forward=False,
                )

        # Euler 体系用 equality_update 逐槽位替代批量 update_equality_constraints。
        # equality_update 内部会调用 update_equality_constraints 处理 type/obj/data，
        # 同时支持 active/solref/solimp 的按 slot 写入。
        for j, gi in enumerate(cfg.eq_indices):
            eq_type_val = int(eqt_row[j])
            eq_data_val = np.array(eqd_row[j], dtype=np.float64, copy=True)
            ea_val = bool(ea_row[j])
            self.equality_update(
                gi,
                eq_type=eq_type_val,
                data=eq_data_val,
                active=ea_val,
                forward=False,
            )
        # 批量写入完成后统一 mj_forward 同步派生量
        self.mj_forward()

        for j, gi in enumerate(cfg.eq_indices):
            m1, m2 = self.model.equality_object_ids(gi)
            m1 = int(m1)
            m2 = int(m2)
            f1 = self._resolve_body_id_from_name(cfg.eq_obj1_name[j])
            f2 = self._resolve_body_id_from_name(cfg.eq_obj2_name[j])
            if m1 != f1 or m2 != f2:
                TRAJ_LOG.warning(
                    "[SimEnv.trajectory] step=%s eq[%s] after update_equality_constraints "
                    "model id=(%d,%d) still != target id=(%d,%d) — update may not have matched",
                    step_index,
                    gi,
                    m1,
                    m2,
                    f1,
                    f2,
                )

        self._last_eq_replay_key = key

    def _get_obs(self) -> dict:
        obs = {
            "joint_pos": self.data.qpos[: self.nq].copy(),
            "joint_vel": self.data.qvel[: self.nv].copy(),
            "joint_acc": self.data.qacc[: self.nv].copy(),
        }
        return obs

    def reset_model(self) -> tuple[dict, dict]:
        self.ctrl = np.zeros(self.nu, dtype=np.float32)
        self._pending_human = None
        self._human_trajectory_step_count = 0
        self._last_eq_replay_key = None

        obs = self._get_obs().copy()
        return obs, {}

    def get_observation(self, obs=None) -> dict:
        if obs is not None:
            return obs
        return self._get_obs().copy()

    def apply_force_to_body(
        self,
        body_name: str,
        force: np.ndarray,
        torque: np.ndarray,
    ) -> None:
        """
        在刚体上施加世界系外力/外力矩（通过 OrcaGymEulerEnv.apply_body_force 写入仿真器）。

        OrcaLink force_position 模式下 SPH 经 Ch1 下发的流体力通过此方法写入。
        """
        f = np.asarray(force, dtype=np.float64).reshape(3)
        tau = np.asarray(torque, dtype=np.float64).reshape(3)
        self.apply_body_force(body_name, f, tau)
