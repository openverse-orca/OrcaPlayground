"""短链 / 全链路 waterjug 运动学轨迹：写入 free joint qpos/qvel，可选 fluidblock 刚性随动。"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..utils.sph_scene_initial_offset import _hint_to_entity_name
from .water_jug_trajectory_math import (
    rotation_from_mjc_body,
    sample_pose_vel_constant_lift_world_rotate,
    trajectory_duration_mode2,
    wxyz_to_scipy,
)

logger = logging.getLogger(__name__)

_WATERJUG_MARKERS = ("5382871157918", "waterjug")
_FLUID_BLOCK_DEFAULT = "water_block_water_block"
_MJ_JNT_FREE = 0
_MJ_GEOM_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)


def _free_joint_for_body(env: Any, body_name: str) -> Optional[str]:
    """返回挂在指定 body 上的 free joint 名称；无则 None。"""
    jd = env.model.get_joint_dict()
    if not jd:
        return None
    try:
        body_id = env.model.body_name2id(body_name)
    except Exception:
        return None
    for name, info in jd.items():
        if info.get("Type") != _MJ_JNT_FREE:
            continue
        if info.get("BodyID") == body_id:
            return name
    return None


def _bodies_with_mesh_token(
    env: Any,
    token: str,
    *,
    exclude_tokens: tuple[str, ...] = ("cup",),
) -> List[str]:
    """
    按 mesh 资源文件名或 mesh 名中的子串查找 body。

    全链路 OrcaStudio 导出的 joint 名常为 ``[entityId]_joint_...``，不含 ``waterjug`` 字面量，
    需经 mesh（如 ``waterjug_02_b_*.obj``）反查 body，再取 free joint。
    """
    token_low = token.lower()
    mesh_dict = env.model.get_mesh_dict() or {}
    matching_mesh_ids: set[int] = set()
    for mesh_name, info in mesh_dict.items():
        file_path = str(info.get("File") or "").lower()
        name_low = mesh_name.lower()
        if any(ex in file_path or ex in name_low for ex in exclude_tokens):
            continue
        if token_low in file_path or token_low in name_low:
            mesh_id = info.get("ID")
            if mesh_id is not None:
                matching_mesh_ids.add(int(mesh_id))

    if not matching_mesh_ids:
        return []

    mj_model = env.gym._mjModel
    bodies: set[str] = set()
    for i in range(mj_model.ngeom):
        geom = mj_model.geom(i)
        if int(geom.type[0]) != _MJ_GEOM_MESH:
            continue
        if int(geom.dataid[0]) not in matching_mesh_ids:
            continue
        bodies.add(mj_model.body(int(geom.bodyid[0])).name)
    return sorted(bodies)


def resolve_waterjug_joint_name(env: Any, hint: str | None = None) -> str | None:
    """
    在 MuJoCo 关节表中解析 waterjug 的 free joint 名称。

    解析顺序：
      1. hint 精确匹配关节名；
      2. hint 为数字 entityId → ``[{id}]`` 出现在关节名中；
      3. hint → ``_hint_to_entity_name`` 得 body，再取该 body 的 free joint；
      4. hint 子串匹配关节名（排除 cup）；
      5. hint 含 ``waterjug`` 或缺省时，按 mesh 名/文件含 ``waterjug`` 找 body → free joint；
      6. 短链回退：关节名含 ``5382871157918`` / ``waterjug``。
    """
    jd = env.model.get_joint_dict()
    if not jd:
        return None

    hint_str = str(hint).strip() if hint else ""

    if hint_str:
        if hint_str in jd:
            return hint_str

        if hint_str.isdigit():
            entity_token = f"[{hint_str}]"
            for name, info in jd.items():
                if info.get("Type") != _MJ_JNT_FREE:
                    continue
                if entity_token in name:
                    return name

        body_name = _hint_to_entity_name(hint_str)
        if body_name and body_name != hint_str:
            joint = _free_joint_for_body(env, body_name)
            if joint:
                return joint

        hint_low = hint_str.lower()
        for name in jd:
            if hint_low in name.lower() and _is_waterjug_joint(name):
                return name

        if "waterjug" in hint_low:
            for body in _bodies_with_mesh_token(env, "waterjug"):
                joint = _free_joint_for_body(env, body)
                if joint:
                    logger.info(
                        "resolve_waterjug_joint_name: hint=%r → body %s → joint %s",
                        hint_str,
                        body,
                        joint,
                    )
                    return joint

    for body in _bodies_with_mesh_token(env, "waterjug"):
        joint = _free_joint_for_body(env, body)
        if joint:
            logger.info(
                "resolve_waterjug_joint_name: mesh fallback → body %s → joint %s",
                body,
                joint,
            )
            return joint

    for name, info in jd.items():
        if info.get("Type") != _MJ_JNT_FREE:
            continue
        if _is_waterjug_joint(name):
            return name
    return None


def _is_waterjug_joint(joint_name: str) -> bool:
    low = joint_name.lower()
    if "cup" in low and "waterjug" not in low:
        return False
    return any(m in low for m in _WATERJUG_MARKERS)


def _set_fixed_body_world_pose(env: Any, body_name: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    """
    将无关节 body（世界子节点）的 model.body_pos/body_quat 设为给定世界位姿，并 mj_forward。

    用于 fluidblock 等静态 body 随 waterjug 刚性运动。
    """
    mj_model = env.gym._mjModel
    body_id = env.model.body_name2id(body_name)
    mj_model.body_pos[body_id] = np.asarray(pos, dtype=np.float64).reshape(3)
    mj_model.body_quat[body_id] = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    env.mj_forward()
    if hasattr(env.gym, "update_data"):
        env.gym.update_data()


class WaterJugTrajectoryController:
    """
    短链 waterjug 轨迹控制器。

    - mode 2：独立初始化（可含 initial_z_offset_m）、匀速抬升、世界轴恒角速度翻转
    - mode 3：高度 H 固定（initial_z_offset_m=0），位姿/速度恒为零运动学
    - 每帧 set_joint_qpos/set_joint_qvel；reapply 阶段可清零外力后再次对齐
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.mode = int(cfg.get("mode", 2))
        self.joint_hint = cfg.get("joint_hint", "5382871157918")
        self.lift_m = float(cfg.get("lift_m", 0.6))
        self.lift_speed_mps = float(cfg.get("lift_speed_mps", 0.6))
        self.phase1_sec = float(cfg.get("phase1_sec", 1.0))
        self.rotate_deg = float(cfg.get("rotate_deg", 90.0))
        self.rotate_speed_deg_s = float(cfg.get("rotate_speed_deg_s", 45.0))
        self.phase2_sec = float(cfg.get("phase2_sec", 2.0))
        self.hold_sec = float(cfg.get("hold_sec", 0.0))
        self.world_axis = str(cfg.get("rotate_world_axis", "x")).lower()
        self.initial_z_offset_m = float(cfg.get("initial_z_offset_m", -0.35))
        self.fluid_block_body = cfg.get("fluid_block_body", _FLUID_BLOCK_DEFAULT)
        self.reapply_after_step = bool(cfg.get("reapply_after_step", True))
        self.clear_external_forces = bool(cfg.get("clear_external_forces", True))
        self.skip_sph_forces_on_mujoco = bool(cfg.get("skip_sph_forces_on_mujoco", False))

        self.kettle_joint: Optional[str] = None
        self._p0: Optional[np.ndarray] = None
        self._q0_wxyz: Optional[np.ndarray] = None
        self._t0: float = 0.0
        self._block_rel_pos: Optional[np.ndarray] = None
        self._block_rel_rot: Optional[R] = None
        self._jug_body_name: Optional[str] = None

    @property
    def move_duration_sec(self) -> float:
        """运动段总时长（不含 hold）。"""
        if self.mode == 3:
            return 0.0
        return trajectory_duration_mode2(self.phase1_sec, self.phase2_sec, 0.0)

    @property
    def total_duration_sec(self) -> float:
        """轨迹总时长（含 hold）。"""
        if self.mode == 3:
            return float("inf")
        return trajectory_duration_mode2(self.phase1_sec, self.phase2_sec, self.hold_sec)

    def reset(self, env: Any) -> None:
        """
        绑定 joint、记录初值并施加 initial_z_offset_m（独立模式：从 H−0.35 m 平面开始）。
        """
        joint = resolve_waterjug_joint_name(env, self.joint_hint)
        if not joint:
            raise RuntimeError(
                f"无法解析 waterjug joint（hint={self.joint_hint!r}）"
            )
        self.kettle_joint = joint
        qpos = np.asarray(env.query_joint_qpos([joint])[joint], dtype=np.float64).ravel()
        if qpos.size != 7:
            raise ValueError(f"joint {joint} 期望 7D free qpos，实际 {qpos.size}")

        self._p0 = qpos[:3].copy()
        self._q0_wxyz = qpos[3:7].copy()
        n = np.linalg.norm(self._q0_wxyz)
        if n > 1e-12:
            self._q0_wxyz /= n

        if abs(self.initial_z_offset_m) > 1e-12:
            self._p0[2] += self.initial_z_offset_m

        self._t0 = float(env.data.time)
        self._jug_body_name = self._body_name_for_joint(env, joint)
        # 先写入带 z 偏移的初值并 mj_forward，再采 fluidblock 相对位姿
        self.apply(env, phase="init")
        self._capture_fluidblock_relative_pose(env)

    def _body_name_for_joint(self, env: Any, joint_name: str) -> str:
        jd = env.model.get_joint_dict()
        info = jd.get(joint_name, {})
        bid = info.get("BodyID")
        if bid is None:
            return ""
        try:
            return env.model.body_id2name(int(bid))
        except Exception:
            return ""

    def _capture_fluidblock_relative_pose(self, env: Any) -> None:
        """记录 fluidblock 相对 waterjug 的刚体变换，供后续每帧随动。"""
        block = self.fluid_block_body
        jug_body = self._jug_body_name
        if not block or not jug_body:
            self._block_rel_pos = None
            self._block_rel_rot = None
            return
        try:
            jug_pos, jug_mat, jug_quat = env.get_body_xpos_xmat_xquat([jug_body])
            block_pos, block_mat, block_quat = env.get_body_xpos_xmat_xquat([block])
        except Exception as e:
            logger.warning("fluidblock 相对位姿采集失败: %s", e)
            self._block_rel_pos = None
            self._block_rel_rot = None
            return

        p_j = np.asarray(jug_pos, dtype=np.float64).reshape(3)
        p_b = np.asarray(block_pos, dtype=np.float64).reshape(3)
        # waterjug 优先用 free joint qpos 四元数（已归一化）；fluidblock 用 xmat 回退
        assert self._q0_wxyz is not None
        r_j = R.from_quat(wxyz_to_scipy(self._q0_wxyz))
        r_b = rotation_from_mjc_body(block_mat, block_quat)
        self._block_rel_pos = r_j.inv().apply(p_b - p_j)
        self._block_rel_rot = r_j.inv() * r_b

    def _sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        assert self._p0 is not None and self._q0_wxyz is not None
        if self.mode == 1:
            raise ValueError("mode 1 为自由落体，不应使用运动学轨迹控制器")
        if self.mode == 3:
            qpos = np.concatenate([self._p0, self._q0_wxyz])
            qvel = np.zeros(6, dtype=np.float64)
            return qpos, qvel
        return sample_pose_vel_constant_lift_world_rotate(
            t,
            self._p0,
            self._q0_wxyz,
            lift_m=self.lift_m,
            lift_speed_mps=self.lift_speed_mps,
            phase1_sec=self.phase1_sec,
            rotate_deg=self.rotate_deg,
            rotate_speed_deg_s=self.rotate_speed_deg_s,
            phase2_sec=self.phase2_sec,
            world_axis=self.world_axis,
        )

    def _clear_jug_external_forces(self, env: Any) -> None:
        if not self.clear_external_forces or not self._jug_body_name:
            return
        try:
            body_id = env.model.body_name2id(self._jug_body_name)
            env.gym._mjData.xfrc_applied[body_id] = 0.0
        except Exception as e:
            logger.debug("clear xfrc on %s: %s", self._jug_body_name, e)

    def _update_fluidblock(self, env: Any, jug_pos: np.ndarray, jug_quat_wxyz: np.ndarray) -> None:
        if self._block_rel_pos is None or self._block_rel_rot is None:
            return
        block = self.fluid_block_body
        if not block:
            return
        r_j = R.from_quat(wxyz_to_scipy(jug_quat_wxyz))
        p_b = np.asarray(jug_pos, dtype=np.float64) + r_j.apply(self._block_rel_pos)
        r_b = r_j * self._block_rel_rot
        q_xyzw = r_b.as_quat()
        q_b = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
        _set_fixed_body_world_pose(env, block, p_b, q_b)

    def apply(self, env: Any, *, phase: str = "pre") -> None:
        """
        按仿真时间写入轨迹位姿。

        phase:
          - init/pre：mj_step 前（与 CP1 发布前对齐）
          - post：mj_step 后 reapply，可选清零外力
        """
        if self.kettle_joint is None:
            self.reset(env)
            return

        sim_t = float(env.data.time)
        if self._p0 is None or sim_t < self._t0 - 1e-9:
            self.reset(env)
            sim_t = float(env.data.time)

        if phase == "post" and self.clear_external_forces:
            self._clear_jug_external_forces(env)

        t = sim_t - self._t0
        qpos, qvel = self._sample(t)
        env.set_joint_qpos({self.kettle_joint: qpos})
        env.set_joint_qvel({self.kettle_joint: qvel})
        env.mj_forward()
        if hasattr(env.gym, "update_data"):
            env.gym.update_data()
        self._update_fluidblock(env, qpos[:3], qpos[3:7])


def maybe_create_water_jug_controller(
    config: Dict[str, Any], env: Any
) -> Optional[WaterJugTrajectoryController]:
    """
    若配置启用 water_jug_trajectory 且 mode≠1，创建并 reset 控制器；否则返回 None。
    """
    cfg = config.get("water_jug_trajectory") or {}
    if not cfg.get("enabled"):
        return None
    mode = int(cfg.get("mode", 0))
    if mode == 1:
        return None

    ctrl = WaterJugTrajectoryController(cfg)
    ctrl.reset(env)
    if mode == 3:
        logger.info(
            "waterjug 轨迹 mode=3: 高度 H 固定 (z_offset=%.3fm), "
            "reapply=%s skip_sph_forces=%s",
            ctrl.initial_z_offset_m,
            ctrl.reapply_after_step,
            ctrl.skip_sph_forces_on_mujoco,
        )
    else:
        logger.info(
            "waterjug 轨迹 mode=%s: lift=%.2fm@%.2fm/s %.1fs, "
            "rotate %.0f°@%.0f°/s world_%s %.1fs, z_offset=%.3fm, "
            "reapply=%s skip_sph_forces=%s",
            mode,
            ctrl.lift_m,
            ctrl.lift_speed_mps,
            ctrl.phase1_sec,
            ctrl.rotate_deg,
            ctrl.rotate_speed_deg_s,
            ctrl.world_axis,
            ctrl.phase2_sec,
            ctrl.initial_z_offset_m,
            ctrl.reapply_after_step,
            ctrl.skip_sph_forces_on_mujoco,
        )
    return ctrl
