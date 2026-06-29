"""
从 Studio 导出的 MJCF 读取 test20260508 场景布局，并规划布–夹爪抓取关键帧。

关键帧轨迹从 ``cloth_robot_gripper_keyframes.json`` 读取（可经 session / 环境变量覆盖）。
权威坐标：**MuJoCo Z-up 世界系**（``mj_forward`` 后 ``xpos`` / ``site_xpos``）。
Pico 回放 JSON 使用 **B 系（base_link）相对位移**，与 ``ControllerArm.update_goal`` 合同一致。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

_CLOTH_3D_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GRIPPER_KEYFRAMES_JSON = _CLOTH_3D_DIR / "cloth_robot_gripper_keyframes.json"

GRIP_CMD_NAMES: dict[str, int] = {
    "open": 0,
    "closing": 1,
    "closed": 2,
    "opening": 3,
}

# 由 ``reload_gripper_trajectory`` 从 JSON 同步；保留模块级名供旧 import。
SLEEVE_HALF_X_M = 0.55
DURATION_SEC = 15.0
CLOSE_T0 = 4.0
CLOSE_T1 = 6.0
OPEN_T0 = 12.0
OPEN_T1 = 15.0

# 与 OrcaManipulation ``conf/openloong_conf.py`` 双臂 neutral 一致（data_collection_cloth_tele 复位用）
OPENLOONG_TELE_ARM_JOINT_VALUES: dict[str, float] = {
    "J_arm_l_01": 1.9,
    "J_arm_l_02": -0.5,
    "J_arm_l_03": 0.0,
    "J_arm_l_04": 2.0,
    "J_arm_l_05": 1.5708,
    "J_arm_l_06": 0.0,
    "J_arm_l_07": 0.0,
    "J_arm_r_01": -1.9,
    "J_arm_r_02": 0.5,
    "J_arm_r_03": 0.0,
    "J_arm_r_04": 2.0,
    "J_arm_r_05": -1.5708,
    "J_arm_r_06": 0.0,
    "J_arm_r_07": 0.0,
}

# 与 ``g1_omnipicker_conf`` / ``data_collection_cloth_tele`` 双臂 neutral 一致
G1_TELE_ARM_JOINT_VALUES: dict[str, float] = {
    "idx21_arm_l_joint1": 0.0,
    "idx22_arm_l_joint2": 0.0,
    "idx23_arm_l_joint3": 0.0,
    "idx24_arm_l_joint4": -0.87,
    "idx25_arm_l_joint5": 0.0,
    "idx26_arm_l_joint6": 0.0,
    "idx27_arm_l_joint7": 0.0,
    "idx61_arm_r_joint1": 0.0,
    "idx62_arm_r_joint2": 0.0,
    "idx63_arm_r_joint3": 0.0,
    "idx64_arm_r_joint4": 0.87,
    "idx65_arm_r_joint5": 0.0,
    "idx66_arm_r_joint6": 0.0,
    "idx67_arm_r_joint7": 0.0,
}


def tele_joint_values_for_session(session: dict[str, Any]) -> dict[str, float]:
    """
    返回与 ``data_collection_cloth_tele`` 复位一致的 tele 关节 neutral。

    优先读 session ``orcagym.default_joint_values``；否则按 ``mjc_agent_prefix`` /
    ``agent_name`` 在 G1 与 openloong 表之间选择。
    """
    og = session.get("orcagym") or {}
    from_session = og.get("default_joint_values")
    if isinstance(from_session, dict) and from_session:
        return {str(k): float(v) for k, v in from_session.items()}

    prefix = str(og.get("mjc_agent_prefix", "")).lower()
    agent = str(og.get("agent_name", "")).lower()
    if "g1" in prefix or "g1" in agent or "omnipicker" in prefix:
        return dict(G1_TELE_ARM_JOINT_VALUES)
    return dict(OPENLOONG_TELE_ARM_JOINT_VALUES)


@dataclass(frozen=True)
class ClothRobotGripperKeyframe:
    """单条掌位关键帧（时刻 + 左右目标 + 夹爪 cmd）。"""

    t_sec: float
    grip_cmd: int
    neutral: bool = False
    left_yup: tuple[float, float, float] | None = None
    right_yup: tuple[float, float, float] | None = None
    left_offset_mjc: tuple[float, float, float] | None = None
    right_offset_mjc: tuple[float, float, float] | None = None
    comment: str = ""


@dataclass(frozen=True)
class ClothRobotGripperTrajectory:
    """从 JSON 加载的完整双掌关键帧轨迹与 FSM 时序。"""

    path: Path
    coordinate_system: str
    target_body: str
    keyframes: tuple[ClothRobotGripperKeyframe, ...]
    close_t0_sec: float
    close_t1_sec: float
    open_t0_sec: float
    open_t1_sec: float
    approach_t_sec: float
    max_palm_err_m: float
    sleeve_half_x_m: float = 0.55
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        if not self.keyframes:
            return 0.0
        return float(self.keyframes[-1].t_sec)


_trajectory: ClothRobotGripperTrajectory | None = None
_trajectory_path: Path | None = None


def yup_to_mjc(pos_yup: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    """
    Y-up 世界坐标 → MuJoCo Z-up 世界坐标。

    与 ``identify_xpbd_cloth.enrich_cloth_discovery_pose`` 互逆：
    ``mjc = (yup_x, -yup_z, yup_y)``。
    """
    p = np.asarray(pos_yup, dtype=np.float64).reshape(3)
    return np.array([p[0], -p[2], p[1]], dtype=np.float64)


def mjc_to_yup(pos_mjc: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    """MuJoCo Z-up 世界坐标 → Y-up 世界坐标。"""
    p = np.asarray(pos_mjc, dtype=np.float64).reshape(3)
    return np.array([p[0], p[2], -p[1]], dtype=np.float64)


def _parse_grip_cmd(value: Any) -> int:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in GRIP_CMD_NAMES:
            return GRIP_CMD_NAMES[key]
        raise ValueError(f"unknown grip_cmd: {value!r} (use {list(GRIP_CMD_NAMES)})")
    raise TypeError(f"grip_cmd must be int or str, got {type(value)}")


def _parse_vec3(raw: Any, *, field_name: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{field_name} must be [x,y,z], got {raw!r}")
    return float(raw[0]), float(raw[1]), float(raw[2])


def resolve_keyframes_json_path(
    session: dict[str, Any] | None = None,
    *,
    override: Path | str | None = None,
) -> Path:
    """
    解析掌位关键帧 JSON 路径（优先级：``override`` > 环境变量 > session > 默认文件）。

    环境变量：``CLOTH_ROBOT_KEYFRAMES_JSON``。
    Session：``cloth_robot.gripper_keyframes_json``（相对 ``cloth_3d`` 或绝对路径）。
    """
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"gripper keyframes JSON not found: {p}")
        return p.resolve()

    env = os.environ.get("CLOTH_ROBOT_KEYFRAMES_JSON", "").strip()
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"CLOTH_ROBOT_KEYFRAMES_JSON not found: {p}")
        return p.resolve()

    if session:
        cr = session.get("cloth_robot") or {}
        rel = cr.get("gripper_keyframes_json")
        if rel:
            p = Path(str(rel))
            if not p.is_absolute():
                p = (_CLOTH_3D_DIR / p).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"session cloth_robot.gripper_keyframes_json not found: {p}")
            return p

    if not DEFAULT_GRIPPER_KEYFRAMES_JSON.is_file():
        raise FileNotFoundError(f"default gripper keyframes missing: {DEFAULT_GRIPPER_KEYFRAMES_JSON}")
    return DEFAULT_GRIPPER_KEYFRAMES_JSON.resolve()


def load_cloth_robot_gripper_keyframes(path: Path) -> ClothRobotGripperTrajectory:
    """
    从 JSON 加载双掌关键帧轨迹。

    支持 ``coordinate_system``：
    - ``yup_world``：``left_yup`` / ``right_yup`` 为掌 body 世界坐标（Y-up）；
    - ``offset_from_cloth_center_mjc``：``left_offset_mjc`` / ``right_offset_mjc`` 相对布心 MJC 偏移。
    ``neutral: true`` 时保持 tele neutral 掌位（零 ``delta_B``）。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    coord = str(data.get("coordinate_system", "yup_world")).strip()
    if coord not in ("yup_world", "offset_from_cloth_center_mjc"):
        raise ValueError(f"unsupported coordinate_system: {coord}")

    fsm = data.get("gripper_fsm") or {}
    ver = data.get("verification") or {}
    meta = data.get("metadata") or {}
    raw_keys = data.get("keyframes")
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ValueError(f"{path}: keyframes must be a non-empty list")

    parsed: list[ClothRobotGripperKeyframe] = []
    for i, item in enumerate(raw_keys):
        if not isinstance(item, dict):
            raise ValueError(f"keyframes[{i}] must be an object")
        t_sec = float(item["t_sec"])
        cmd = _parse_grip_cmd(item.get("grip_cmd", "open"))
        neutral = bool(item.get("neutral", False))
        comment = str(item.get("comment", ""))

        left_yup = right_yup = None
        left_off = right_off = None
        if neutral:
            pass
        elif coord == "yup_world":
            if "left_yup" not in item or "right_yup" not in item:
                raise ValueError(f"keyframes[{i}]: yup_world requires left_yup and right_yup")
            left_yup = _parse_vec3(item["left_yup"], field_name="left_yup")
            right_yup = _parse_vec3(item["right_yup"], field_name="right_yup")
        else:
            if "left_offset_mjc" not in item or "right_offset_mjc" not in item:
                raise ValueError(
                    f"keyframes[{i}]: offset_from_cloth_center_mjc requires "
                    "left_offset_mjc and right_offset_mjc"
                )
            left_off = _parse_vec3(item["left_offset_mjc"], field_name="left_offset_mjc")
            right_off = _parse_vec3(item["right_offset_mjc"], field_name="right_offset_mjc")

        parsed.append(
            ClothRobotGripperKeyframe(
                t_sec=t_sec,
                grip_cmd=cmd,
                neutral=neutral,
                left_yup=left_yup,
                right_yup=right_yup,
                left_offset_mjc=left_off,
                right_offset_mjc=right_off,
                comment=comment,
            )
        )

    parsed.sort(key=lambda k: k.t_sec)
    for i in range(len(parsed) - 1):
        if parsed[i].t_sec >= parsed[i + 1].t_sec:
            raise ValueError(f"{path}: keyframes must be strictly increasing by t_sec")

    return ClothRobotGripperTrajectory(
        path=path.resolve(),
        coordinate_system=coord,
        target_body=str(data.get("target_body", "palm")),
        keyframes=tuple(parsed),
        close_t0_sec=float(fsm.get("close_t0_sec", 4.0)),
        close_t1_sec=float(fsm.get("close_t1_sec", 6.0)),
        open_t0_sec=float(fsm.get("open_t0_sec", 12.0)),
        open_t1_sec=float(fsm.get("open_t1_sec", 15.0)),
        approach_t_sec=float(ver.get("approach_t_sec", 2.0)),
        max_palm_err_m=float(ver.get("max_palm_err_m", 0.05)),
        sleeve_half_x_m=float(meta.get("sleeve_half_x_m", 0.55)),
        metadata=dict(meta),
    )


def _sync_module_timing_from_trajectory(traj: ClothRobotGripperTrajectory) -> None:
    """将 JSON 中的 FSM / 时长 / 袖口半宽写入模块级变量（兼容旧 import）。"""
    global SLEEVE_HALF_X_M, DURATION_SEC, CLOSE_T0, CLOSE_T1, OPEN_T0, OPEN_T1
    SLEEVE_HALF_X_M = traj.sleeve_half_x_m
    DURATION_SEC = traj.duration_sec
    CLOSE_T0 = traj.close_t0_sec
    CLOSE_T1 = traj.close_t1_sec
    OPEN_T0 = traj.open_t0_sec
    OPEN_T1 = traj.open_t1_sec


def reload_gripper_trajectory(
    path: Path | str | None = None,
    *,
    session: dict[str, Any] | None = None,
) -> ClothRobotGripperTrajectory:
    """
    加载（或重载）掌位关键帧 JSON，并更新模块级 ``DURATION_SEC`` / ``CLOSE_T0`` 等。

    返回 ``ClothRobotGripperTrajectory`` 供 ``build_ee_delta_keyframes_mjc`` 使用。
    """
    global _trajectory, _trajectory_path
    resolved = resolve_keyframes_json_path(session, override=path)
    traj = load_cloth_robot_gripper_keyframes(resolved)
    _trajectory = traj
    _trajectory_path = resolved
    _sync_module_timing_from_trajectory(traj)
    logger.info("Loaded gripper keyframes: %s (%d keys, %.1fs)", resolved, len(traj.keyframes), traj.duration_sec)
    return traj


def get_gripper_trajectory() -> ClothRobotGripperTrajectory:
    """返回已缓存的关键帧轨迹；若未加载则读默认 JSON。"""
    if _trajectory is None:
        return reload_gripper_trajectory()
    return _trajectory


def palm_targets_mjc_for_keyframe(
    kf: ClothRobotGripperKeyframe,
    layout: ClothRobotSceneLayout,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    将单条关键帧解析为左右掌 MJC 世界目标；``neutral`` 时返回 ``(None, None)``。
    """
    if kf.neutral:
        return None, None
    if kf.left_yup is not None and kf.right_yup is not None:
        return yup_to_mjc(kf.left_yup), yup_to_mjc(kf.right_yup)
    if kf.left_offset_mjc is not None and kf.right_offset_mjc is not None:
        cx, cy, cz = layout.cloth_center_mjc
        lo = kf.left_offset_mjc
        ro = kf.right_offset_mjc
        tgt_l = np.array([cx + lo[0], cy + lo[1], cz + lo[2]], dtype=np.float64)
        tgt_r = np.array([cx + ro[0], cy + ro[1], cz + ro[2]], dtype=np.float64)
        return tgt_l, tgt_r
    raise ValueError(f"keyframe t={kf.t_sec}s has no palm targets")


def keyframes_as_grasp_offsets_mjc(
    traj: ClothRobotGripperTrajectory,
    layout: ClothRobotSceneLayout,
) -> list[tuple[float, tuple[float, float, float], tuple[float, float, float], int]]:
    """
    将 JSON 轨迹转为相对布心的 MJC 偏移表（与旧 ``_REL_GRASP_OFFSETS_MJC`` 同形）。

    仅用于调试打印；``build_ee_delta_keyframes_mjc`` 直接读 ``ClothRobotGripperKeyframe``。
    """
    cx, cy, cz = layout.cloth_center_mjc
    out: list[tuple[float, tuple[float, float, float], tuple[float, float, float], int]] = []
    for kf in traj.keyframes:
        tgt_l, tgt_r = palm_targets_mjc_for_keyframe(kf, layout)
        if tgt_l is None:
            out.append((kf.t_sec, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), kf.grip_cmd))
            continue
        out.append(
            (
                kf.t_sec,
                (float(tgt_l[0] - cx), float(tgt_l[1] - cy), float(tgt_l[2] - cz)),
                (float(tgt_r[0] - cx), float(tgt_r[1] - cy), float(tgt_r[2] - cz)),
                kf.grip_cmd,
            )
        )
    return out


@dataclass
class ClothRobotSceneLayout:
    """test20260508 单帧场景几何快照（MuJoCo Z-up）。"""

    mjcf_path: str
    agent_prefix: str
    cloth_body: str
    cloth_center_mjc: tuple[float, float, float]
    cloth_center_yup: tuple[float, float, float]
    base_link: str
    base_pos_mjc: tuple[float, float, float]
    left_palm_body: str
    right_palm_body: str
    left_palm_mjc: tuple[float, float, float]
    right_palm_mjc: tuple[float, float, float]
    left_ee_site: str
    right_ee_site: str
    left_ee_B: tuple[float, float, float]
    right_ee_B: tuple[float, float, float]
    tele_neutral_applied: bool = False
    mjcf_default_left_palm_mjc: tuple[float, float, float] | None = None
    mjcf_default_right_palm_mjc: tuple[float, float, float] | None = None


def _resolve_mjcf_from_session(session: dict[str, Any]) -> Path:
    meta = session.get("_cloth_robot_session_meta") or {}
    src = meta.get("source_mjcf") or session.get("mujoco", {}).get("model_path")
    if not src:
        raise ValueError("session missing mujoco.model_path or source_mjcf")
    path = Path(str(src)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MJCF not found: {path}")
    return path.resolve()


def _body_xpos(data: mujoco.MjData, model: mujoco.MjModel, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"body not in MJCF: {name}")
    return np.array(data.xpos[bid], dtype=np.float64)


def _site_xpos(data: mujoco.MjData, model: mujoco.MjModel, name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise KeyError(f"site not in MJCF: {name}")
    return np.array(data.site_xpos[sid], dtype=np.float64)


def _world_to_base_B(pos_w: np.ndarray, base_pos: np.ndarray, base_quat_wxyz: np.ndarray) -> np.ndarray:
    """世界系位置 → base_link 局部 B 系（与 OrcaGym ``query_site_pos_and_quat_B`` 同族）。"""
    rot = R.from_quat(base_quat_wxyz[[1, 2, 3, 0]])
    return rot.inv().apply(pos_w - base_pos)


def delta_b_to_unity_position(delta_b: np.ndarray) -> dict[str, float]:
    """
    将 B 系位移编码为 Pico JSON Unity 位置。

    与 ``abstract_device.transform_event`` 互逆：
    ``relative_B = [unity_z, -unity_x, unity_y]``。
    """
    db = np.asarray(delta_b, dtype=np.float64).reshape(3)
    return {"x": float(-db[1]), "y": float(db[2]), "z": float(db[0])}


def apply_agent_joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    agent_prefix: str,
    joint_values: dict[str, float],
) -> int:
    """
    将 tele 使用的关节角写入 ``qpos``（短名或 ``{prefix}_{short}``），并 ``mj_forward``。

    返回成功写入的关节数。
    """
    if not joint_values:
        return 0
    applied = 0
    for short_name, value in joint_values.items():
        for candidate in (short_name, f"{agent_prefix}_{short_name}"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate)
            if jid < 0:
                continue
            adr = model.jnt_qposadr[jid]
            data.qpos[adr] = float(value)
            applied += 1
            break
    if applied > 0:
        mujoco.mj_forward(model, data)
    return applied


def prepare_mjcf_model_data(
    session: dict[str, Any],
    *,
    default_joint_values: dict[str, float] | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, ClothRobotSceneLayout]:
    """
    加载 session 指向的 MJCF，可选施加 tele 关节角，返回与 ``ControllerArm`` 复位一致的 model/data/layout。
    """
    layout = load_scene_layout_from_session(
        session,
        default_joint_values=default_joint_values,
    )
    model = mujoco.MjModel.from_xml_path(layout.mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if default_joint_values:
        apply_agent_joint_qpos(model, data, layout.agent_prefix, default_joint_values)
    else:
        mujoco.mj_forward(model, data)
    return model, data, layout


def load_scene_layout_from_session(
    session: dict[str, Any],
    *,
    session_path: Path | None = None,
    default_joint_values: dict[str, float] | None = None,
) -> ClothRobotSceneLayout:
    """
    加载 session 指向的 MJCF，``mj_forward`` 后读取布心、base_link、双掌与末端 site 的 B 系初态。

    ``default_joint_values`` 非空时先记录 MJCF 默认掌位，再写入 tele 关节角作为 replay neutral
    （须与 ``data_collection_cloth_tele`` 的 ``default_joint_values`` 一致）。
    """
    mjcf_path = _resolve_mjcf_from_session(session)
    og = session.get("orcagym") or {}
    prefix = str(og.get("mjc_agent_prefix", "openloong_gripper_2f85_fix_base_usda"))

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    left_palm = f"{prefix}_zbll_base_link"
    right_palm = f"{prefix}_zbr_base_link"
    raw_left_palm: tuple[float, float, float] | None = None
    raw_right_palm: tuple[float, float, float] | None = None
    tele_applied = False
    if default_joint_values:
        raw_left_palm = tuple(float(x) for x in _body_xpos(data, model, left_palm))
        raw_right_palm = tuple(float(x) for x in _body_xpos(data, model, right_palm))
        n = apply_agent_joint_qpos(model, data, prefix, default_joint_values)
        if n <= 0:
            logger.warning("tele default_joint_values: no joints applied (check MJCF names)")
        else:
            tele_applied = True

    from modules.identify_xpbd_cloth import enrich_cloth_discovery_pose, identify_xpbd_cloth  # noqa: WPS433

    cloths = enrich_cloth_discovery_pose(model, data, identify_xpbd_cloth(model))
    if not cloths:
        raise RuntimeError("no XPBD cloth in MJCF")
    cloth = cloths[0]
    cmjc = tuple(float(x) for x in cloth["center_mjc"])
    cyup = tuple(float(x) for x in cloth["center_yup"])
    cloth_body = str(cloth["body_name"])

    base_link = f"{prefix}_base_link"
    left_ee = "ee_center_site"
    right_ee = "ee_center_site_r"
    for site in (left_ee, right_ee):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) < 0:
            alt = f"{prefix}_{site}"
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, alt) >= 0:
                if site == left_ee:
                    left_ee = alt
                else:
                    right_ee = alt

    base_pos = _body_xpos(data, model, base_link)
    base_quat = np.array(data.xquat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_link)])

    left_ee_B = _world_to_base_B(_site_xpos(data, model, left_ee), base_pos, base_quat)
    right_ee_B = _world_to_base_B(_site_xpos(data, model, right_ee), base_pos, base_quat)

    return ClothRobotSceneLayout(
        mjcf_path=str(mjcf_path),
        agent_prefix=prefix,
        cloth_body=cloth_body,
        cloth_center_mjc=cmjc,
        cloth_center_yup=cyup,
        base_link=base_link,
        base_pos_mjc=tuple(float(x) for x in base_pos),
        left_palm_body=left_palm,
        right_palm_body=right_palm,
        left_palm_mjc=tuple(float(x) for x in _body_xpos(data, model, left_palm)),
        right_palm_mjc=tuple(float(x) for x in _body_xpos(data, model, right_palm)),
        left_ee_site=left_ee,
        right_ee_site=right_ee,
        left_ee_B=tuple(float(x) for x in left_ee_B),
        right_ee_B=tuple(float(x) for x in right_ee_B),
        tele_neutral_applied=tele_applied,
        mjcf_default_left_palm_mjc=raw_left_palm,
        mjcf_default_right_palm_mjc=raw_right_palm,
    )


def load_scene_layout_from_session_path(path: Path) -> ClothRobotSceneLayout:
    """从 ``cloth_sim_session_*.json`` 文件加载场景布局（tele 关节 neutral）。"""
    session = json.loads(path.read_text(encoding="utf-8"))
    return load_scene_layout_from_session(
        session,
        session_path=path,
        default_joint_values=OPENLOONG_TELE_ARM_JOINT_VALUES,
    )


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _interp_keyframes(
    t: float,
    keys: list[tuple[float, np.ndarray, np.ndarray, int]],
) -> tuple[np.ndarray, np.ndarray, int]:
    if t >= keys[-1][0]:
        _, dl, dr, cmd = keys[-1]
        return dl.copy(), dr.copy(), cmd
    for i in range(len(keys) - 1):
        t0, d0l, d0r, c0 = keys[i]
        t1, d1l, d1r, _ = keys[i + 1]
        if t0 <= t < t1:
            u = _smoothstep((t - t0) / max(1e-6, t1 - t0))
            left = d0l + u * (d1l - d0l)
            right = d0r + u * (d1r - d0r)
            return left, right, c0
    _, dl, dr, cmd = keys[0]
    return dl.copy(), dr.copy(), cmd


def build_ee_delta_keyframes_mjc(
    layout: ClothRobotSceneLayout,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    trajectory: ClothRobotGripperTrajectory | None = None,
) -> list[tuple[float, np.ndarray, np.ndarray, int]]:
    """
    根据 JSON 关键帧掌目标，计算每关键帧左右 ``ee_center_site`` 的 B 系目标位移。

    目标世界位姿：``target_ee_world = ee_site_world_neutral + (target_palm_world - palm_world_neutral)``。
    """
    traj = trajectory or get_gripper_trajectory()
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, layout.base_link)
    base_pos = np.array(data.xpos[base_bid], dtype=np.float64)
    base_quat = np.array(data.xquat[base_bid], dtype=np.float64)

    ee_l_w0 = _site_xpos(data, model, layout.left_ee_site)
    ee_r_w0 = _site_xpos(data, model, layout.right_ee_site)
    palm_l0 = np.array(layout.left_palm_mjc, dtype=np.float64)
    palm_r0 = np.array(layout.right_palm_mjc, dtype=np.float64)
    ee_l_B0 = np.array(layout.left_ee_B, dtype=np.float64)
    ee_r_B0 = np.array(layout.right_ee_B, dtype=np.float64)

    keys_out: list[tuple[float, np.ndarray, np.ndarray, int]] = []

    for kf in traj.keyframes:
        tgt_l_w, tgt_r_w = palm_targets_mjc_for_keyframe(kf, layout)
        if tgt_l_w is None:
            keys_out.append((kf.t_sec, np.zeros(3), np.zeros(3), kf.grip_cmd))
            continue
        ee_l_w = ee_l_w0 + (tgt_l_w - palm_l0)
        ee_r_w = ee_r_w0 + (tgt_r_w - palm_r0)
        ee_l_B = _world_to_base_B(ee_l_w, base_pos, base_quat)
        ee_r_B = _world_to_base_B(ee_r_w, base_pos, base_quat)
        keys_out.append((kf.t_sec, ee_l_B - ee_l_B0, ee_r_B - ee_r_B0, kf.grip_cmd))
    return keys_out


def interp_ee_deltas_at(
    t: float,
    delta_keys: list[tuple[float, np.ndarray, np.ndarray, int]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """插值得到时刻 t 的左右 B 系 ee 位移与 grip cmd。"""
    return _interp_keyframes(t, delta_keys)


def grip_cmd_to_pico_trigger(t: float, cmd: int) -> tuple[float, bool, bool, bool]:
    """夹爪 cmd → Pico 扳机/按键（时序来自 JSON ``gripper_fsm``）。"""
    traj = get_gripper_trajectory()
    if cmd == 0:
        return 0.0, False, False, False
    if cmd == 1:
        if t < traj.close_t0_sec:
            return 0.0, False, False, False
        if t < traj.close_t1_sec:
            u = (t - traj.close_t0_sec) / max(1e-6, traj.close_t1_sec - traj.close_t0_sec)
            u = u * u * (3.0 - 2.0 * u)
            return u, True, False, False
        return 1.0, True, False, False
    if cmd == 2:
        return 1.0, True, False, True
    if cmd == 3:
        if t < traj.open_t0_sec:
            return 1.0, True, False, False
        if t < traj.open_t1_sec:
            u = (t - traj.open_t0_sec) / max(1e-6, traj.open_t1_sec - traj.open_t0_sec)
            u = u * u * (3.0 - 2.0 * u)
            return 1.0 - u, False, True, False
        return 0.0, False, True, False
    return 0.0, False, False, False


def _keyframe_at_or_before(traj: ClothRobotGripperTrajectory, t_sec: float) -> ClothRobotGripperKeyframe:
    chosen = traj.keyframes[0]
    for kf in traj.keyframes:
        if kf.t_sec <= t_sec + 1e-9:
            chosen = kf
        else:
            break
    return chosen


def verify_replay_approach_palm_targets(
    layout: ClothRobotSceneLayout,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    delta_keys: list[tuple[float, np.ndarray, np.ndarray, int]],
    *,
    approach_t: float | None = None,
    max_palm_err_m: float | None = None,
    trajectory: ClothRobotGripperTrajectory | None = None,
) -> tuple[bool, str]:
    """
    校验 approach 时刻 ``delta_B`` 是否将掌导向 JSON 规划袖口（与 ``ControllerArm`` 合同一致）。

    默认 ``approach_t`` / ``max_palm_err_m`` 取自 JSON ``verification`` 块。
    """
    traj = trajectory or get_gripper_trajectory()
    t_check = traj.approach_t_sec if approach_t is None else approach_t
    err_thr = traj.max_palm_err_m if max_palm_err_m is None else max_palm_err_m

    d_l, d_r, _ = interp_ee_deltas_at(t_check, delta_keys)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, layout.base_link)
    base_pos = np.array(data.xpos[base_bid], dtype=np.float64)
    base_quat = np.array(data.xquat[base_bid], dtype=np.float64)
    rot = R.from_quat(base_quat[[1, 2, 3, 0]])

    ee_l_w0 = _site_xpos(data, model, layout.left_ee_site)
    ee_r_w0 = _site_xpos(data, model, layout.right_ee_site)
    palm_l0 = np.array(layout.left_palm_mjc, dtype=np.float64)
    palm_r0 = np.array(layout.right_palm_mjc, dtype=np.float64)

    kf = _keyframe_at_or_before(traj, t_check)
    tgt_l_w, tgt_r_w = palm_targets_mjc_for_keyframe(kf, layout)
    if tgt_l_w is None or tgt_r_w is None:
        return False, f"approach t={t_check}s: keyframe t={kf.t_sec}s is neutral (no palm targets)"

    def _implied_palm(
        delta_b: np.ndarray, ee_b0: np.ndarray, ee_w0: np.ndarray, palm0: np.ndarray
    ) -> np.ndarray:
        ee_w_goal = rot.apply(ee_b0 + delta_b) + base_pos
        return palm0 + (ee_w_goal - ee_w0)

    ee_l_b0 = np.array(layout.left_ee_B, dtype=np.float64)
    ee_r_b0 = np.array(layout.right_ee_B, dtype=np.float64)
    err_l = float(np.linalg.norm(_implied_palm(d_l, ee_l_b0, ee_l_w0, palm_l0) - tgt_l_w))
    err_r = float(np.linalg.norm(_implied_palm(d_r, ee_r_b0, ee_r_w0, palm_r0) - tgt_r_w))
    ok = err_l <= err_thr and err_r <= err_thr
    msg = (
        f"approach t={t_check}s palm err L={err_l:.4f}m R={err_r:.4f}m "
        f"(thr {err_thr:.3f}m) tele_neutral={layout.tele_neutral_applied} "
        f"keyframes={traj.path.name}"
    )
    return ok, msg


def format_layout_report(layout: ClothRobotSceneLayout) -> str:
    """生成人类可读场景表（含 MJC / Y-up）。"""
    traj = get_gripper_trajectory()
    lines = [
        "=== ClothRobot test20260508 scene layout (from MJCF) ===",
        f"MJCF: {layout.mjcf_path}",
        f"Gripper keyframes JSON: {traj.path}",
        f"Agent prefix: {layout.agent_prefix}",
        f"Replay neutral: {'tele default_joint_values' if layout.tele_neutral_applied else 'MJCF default qpos (WARN)'}",
        "",
        "Entity (MuJoCo Z-up world | Y-up for XPBD)",
        f"  Cloth {layout.cloth_body}:",
        f"    mjc center = {layout.cloth_center_mjc}",
        f"    yup center = {layout.cloth_center_yup}",
        f"  Robot base {layout.base_link}:",
        f"    mjc pos    = {layout.base_pos_mjc}",
        f"  Left palm  {layout.left_palm_body}: mjc = {layout.left_palm_mjc}",
        f"  Right palm {layout.right_palm_body}: mjc = {layout.right_palm_mjc}",
        f"  Left ee {layout.left_ee_site} B-frame = {layout.left_ee_B}",
        f"  Right ee {layout.right_ee_site} B-frame = {layout.right_ee_B}",
    ]
    if layout.mjcf_default_left_palm_mjc is not None:
        dl = np.linalg.norm(
            np.array(layout.left_palm_mjc) - np.array(layout.mjcf_default_left_palm_mjc)
        )
        dr = np.linalg.norm(
            np.array(layout.right_palm_mjc) - np.array(layout.mjcf_default_right_palm_mjc or layout.right_palm_mjc)
        )
        lines.extend(
            [
                "",
                "MJCF default vs tele neutral palm shift:",
                f"  left  default {layout.mjcf_default_left_palm_mjc} -> tele {layout.left_palm_mjc}  |d|={dl:.3f}m",
                f"  right default {layout.mjcf_default_right_palm_mjc} -> tele {layout.right_palm_mjc}  |d|={dr:.3f}m",
            ]
        )
    lines.extend(
        [
            "",
            "Note: replay delta_B must use tele neutral (openloong_conf arm joints), not Studio Play qpos.",
        ]
    )
    return "\n".join(lines)


# 模块 import 时加载默认 JSON，同步 DURATION_SEC / CLOSE_T0 等（文件缺失时跳过）。
if DEFAULT_GRIPPER_KEYFRAMES_JSON.is_file():
    reload_gripper_trajectory()
