"""
RectCloth 双掌关键帧自动生成。

默认 **短边 ±Z 外夹并向布心折叠**（``pinch_short_edges_pmZ_fold_center``）：
左掌夹 -Z 短边、右掌夹 +Z 短边，X 对齐布心，收拢后沿 Z 向 ``cz`` 对折。

用法::
  python -m modules.cloth_robot_rect_keyframe_builder \\
    --session-json cloth_sim_session_p23c_*.json \\
    --output cloth_robot_gripper_keyframes.test20260508_RectCloth.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from modules.cloth_robot_scene_layout import (
    OPENLOONG_TELE_ARM_JOINT_VALUES,
    _site_xpos,
    build_ee_delta_keyframes_mjc,
    interp_ee_deltas_at,
    load_cloth_robot_gripper_keyframes,
    mjc_to_yup,
    prepare_mjcf_model_data,
    reload_gripper_trajectory,
    yup_to_mjc,
)


def _cloth_rect_meta(session: dict[str, Any]) -> dict[str, float]:
    cloth = session.get("cloth") or {}
    spacing = float(cloth.get("cloth_spacing_m", 0.0618))
    nx = int(cloth.get("cloth_nx", 20))
    ny = int(cloth.get("cloth_ny", 11))
    cx, cy, cz = (float(x) for x in (cloth.get("center_yup") or [-0.6, 11.7, 0.0]))
    half_x = max(0.0, (nx - 1) * spacing * 0.5)
    half_z = max(0.0, (ny - 1) * spacing * 0.5)
    return {
        "cx": cx,
        "cy": cy,
        "cz": cz,
        "spacing": spacing,
        "half_x": half_x,
        "half_z": half_z,
        "span_x": 2.0 * half_x,
        "span_z": 2.0 * half_z,
    }


def _neutral_lateral_yup(layout) -> tuple[float, float, float, float]:
    """tele neutral 下左右掌 Y-up 坐标（用于保持左右分列）。"""
    l = mjc_to_yup(layout.left_palm_mjc)
    r = mjc_to_yup(layout.right_palm_mjc)
    return float(l[0]), float(l[1]), float(l[2]), float(r[2])


def _make_keyframe(
    t_sec: float,
    left_yup: list[float] | None,
    right_yup: list[float] | None,
    grip_cmd: str,
    *,
    neutral: bool = False,
    comment: str = "",
) -> dict[str, Any]:
    kf: dict[str, Any] = {"t_sec": t_sec, "grip_cmd": grip_cmd}
    if neutral:
        kf["neutral"] = True
    else:
        kf["left_yup"] = left_yup
        kf["right_yup"] = right_yup
    if comment:
        kf["comment"] = comment
    return kf


def build_rect_cloth_keyframes_short_edge_fold_pmZ(
    session: dict[str, Any],
    layout,
    *,
    margin_m: float = 0.08,
    fold_hold_z_m: float = 0.06,
) -> dict[str, Any]:
    """
    夹取矩形布两条 **短边中点**（X=cx，Z=cz±half_z），折痕沿长边 X，沿 Z 向布心对折。

    - 左掌 → 短边中点 (cx, cy, cz - half_z)
    - 右掌 → 短边中点 (cx, cy, cz + half_z)
    - 4–6 s 收爪；10 s 沿 Z 内收至 ``fold_hold_z_m``
    """
    m = _cloth_rect_meta(session)
    cx, cy, cz = m["cx"], m["cy"], m["cz"]
    hx, hz = m["half_x"], m["half_z"]
    approach_z = hz + margin_m
    grasp_z = hz
    fold_z = max(0.03, fold_hold_z_m)

    keyframes = [
        _make_keyframe(0.0, None, None, "open", neutral=True, comment="tele neutral"),
        _make_keyframe(
            2.0,
            [cx, cy + 0.02, cz - approach_z],
            [cx, cy + 0.02, cz + approach_z],
            "open",
            comment="approach：X 对齐布心，Z 分列两条短边外侧",
        ),
        _make_keyframe(
            4.0,
            [cx, cy + 0.005, cz - grasp_z],
            [cx, cy + 0.005, cz + grasp_z],
            "closing",
            comment="收爪至短边中点 (cx, cz±half_z)",
        ),
        _make_keyframe(
            6.0,
            [cx, cy, cz - grasp_z],
            [cx, cy, cz + grasp_z],
            "closed",
            comment="短边中点夹紧",
        ),
        _make_keyframe(
            8.0,
            [cx, cy + 0.05, cz - grasp_z],
            [cx, cy + 0.05, cz + grasp_z],
            "closed",
            comment="微抬",
        ),
        _make_keyframe(
            10.0,
            [cx, cy + 0.15, cz - fold_z],
            [cx, cy + 0.15, cz + fold_z],
            "closed",
            comment="沿 Z 向布心对折（折痕平行长边 X）",
        ),
        _make_keyframe(
            12.0,
            [cx, cy + 0.15, cz - fold_z],
            [cx, cy + 0.15, cz + fold_z],
            "opening",
            comment="张爪",
        ),
        _make_keyframe(
            15.0,
            [cx, cy + 0.18, cz - approach_z],
            [cx, cy + 0.18, cz + approach_z],
            "open",
            comment="外撤",
        ),
    ]

    return {
        "description": "RectCloth 短边中点外夹，折痕沿长边 X 向布心对折",
        "coordinate_system": "yup_world",
        "target_body": "palm",
        "metadata": {
            "level": (session.get("orcagym") or {}).get("level", "test20260508_RectCloth"),
            "cloth_center_yup": [cx, cy, cz],
            "cloth_nx": int((session.get("cloth") or {}).get("cloth_nx", 20)),
            "cloth_ny": int((session.get("cloth") or {}).get("cloth_ny", 11)),
            "cloth_spacing_m": m["spacing"],
            "rect_span_x_m": m["span_x"],
            "rect_span_z_m": m["span_z"],
            "rect_half_extent_x_m": hx,
            "rect_half_extent_z_m": hz,
            "grasp_mode": "pinch_short_edge_midpoints_fold_along_long_X",
            "short_edge_axis_yup": "Z",
            "long_edge_axis_yup": "X",
            "fold_crease_axis_yup": "X",
            "fold_motion_axis_yup": "Z",
            "sleeve_half_x_m": hz,
        },
        "gripper_fsm": {
            "close_t0_sec": 4.0,
            "close_t1_sec": 6.0,
            "open_t0_sec": 12.0,
            "open_t1_sec": 15.0,
        },
        "verification": {
            "approach_t_sec": 2.0,
            "max_palm_err_m": 0.08,
        },
        "keyframes": keyframes,
    }


def build_rect_cloth_keyframes_long_edge_pmX(
    session: dict[str, Any],
    layout,
    *,
    margin_m: float = 0.08,
    lateral_z_scale: float = 1.15,
) -> dict[str, Any]:
    """
    沿 **长边 ±X** 外夹（历史策略；``nx`` 沿世界 X，跨度约 1.17 m）。

    左掌在 -X 长边、右掌在 +X 长边；Z 保持 neutral 分列。openloong 在布 -X 侧时常不可达 +X 目标。
    """
    m = _cloth_rect_meta(session)
    _lx0, ly0, lz0, rz0 = _neutral_lateral_yup(layout)
    lz = lz0 * lateral_z_scale
    rz = rz0 * lateral_z_scale
    if lz > -0.12:
        lz = -0.22
    if rz < 0.12:
        rz = 0.22

    cx, cy = m["cx"], m["cy"]
    hx, hz = m["half_x"], m["half_z"]
    out_x = hx + margin_m
    inset_x = max(0.04, hx - 0.04)

    keyframes = [
        _make_keyframe(0.0, None, None, "open", neutral=True, comment="tele neutral"),
        _make_keyframe(
            2.0,
            [cx - out_x, cy + 0.02, lz],
            [cx + out_x, cy + 0.02, rz],
            "open",
            comment="approach：分列短边外侧，Z 保持左负右正",
        ),
        _make_keyframe(
            4.0,
            [cx - inset_x, cy + 0.005, lz],
            [cx + inset_x, cy + 0.005, rz],
            "closing",
            comment="收拢至短边内侧",
        ),
        _make_keyframe(
            6.0,
            [cx - inset_x + 0.02, cy, lz],
            [cx + inset_x - 0.02, cy, rz],
            "closed",
            comment="夹紧",
        ),
        _make_keyframe(
            8.0,
            [cx - inset_x + 0.02, cy + 0.05, lz],
            [cx + inset_x - 0.02, cy + 0.05, rz],
            "closed",
            comment="微抬",
        ),
        _make_keyframe(
            10.0,
            [cx - inset_x + 0.02, cy + 0.15, lz],
            [cx + inset_x - 0.02, cy + 0.15, rz],
            "closed",
            comment="抬升",
        ),
        _make_keyframe(
            12.0,
            [cx - inset_x + 0.02, cy + 0.15, lz],
            [cx + inset_x - 0.02, cy + 0.15, rz],
            "opening",
            comment="张爪",
        ),
        _make_keyframe(
            15.0,
            [cx - out_x, cy + 0.18, lz],
            [cx + out_x, cy + 0.18, rz],
            "open",
            comment="外撤",
        ),
    ]

    return {
        "description": "RectCloth 长边 ±X 外夹（旧策略，openloong 常不可达 +X 侧）",
        "coordinate_system": "yup_world",
        "target_body": "palm",
        "metadata": {
            "level": (session.get("orcagym") or {}).get("level", "test20260508_RectCloth"),
            "cloth_center_yup": [cx, cy, m["cz"]],
            "cloth_nx": int((session.get("cloth") or {}).get("cloth_nx", 20)),
            "cloth_ny": int((session.get("cloth") or {}).get("cloth_ny", 11)),
            "cloth_spacing_m": m["spacing"],
            "rect_span_x_m": m["span_x"],
            "rect_span_z_m": m["span_z"],
            "rect_half_extent_x_m": hx,
            "rect_half_extent_z_m": hz,
            "grasp_mode": "pinch_long_edges_pmX_no_cross",
            "neutral_left_z_yup": lz0,
            "neutral_right_z_yup": rz0,
            "sleeve_half_x_m": hz,
        },
        "gripper_fsm": {
            "close_t0_sec": 4.0,
            "close_t1_sec": 6.0,
            "open_t0_sec": 12.0,
            "open_t1_sec": 15.0,
        },
        "verification": {
            "approach_t_sec": 2.0,
            "max_palm_err_m": 0.08,
        },
        "keyframes": keyframes,
    }


_GRASP_BUILDERS = {
    "pmZ_fold": build_rect_cloth_keyframes_short_edge_fold_pmZ,
    "pmX_long": build_rect_cloth_keyframes_long_edge_pmX,
}


def sample_implied_palm_mjc(
    layout,
    model,
    data,
    delta_keys: list,
    duration_sec: float,
    samples: int = 40,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """按 replay 插值采样 implied 掌位（MJC 世界系）。"""
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, layout.base_link)
    base_pos = np.array(data.xpos[base_bid], dtype=np.float64)
    base_quat = np.array(data.xquat[base_bid], dtype=np.float64)
    rot = R.from_quat(base_quat[[1, 2, 3, 0]])

    ee_l_w0 = _site_xpos(data, model, layout.left_ee_site)
    ee_r_w0 = _site_xpos(data, model, layout.right_ee_site)
    palm_l0 = np.array(layout.left_palm_mjc, dtype=np.float64)
    palm_r0 = np.array(layout.right_palm_mjc, dtype=np.float64)
    ee_l_b0 = np.array(layout.left_ee_B, dtype=np.float64)
    ee_r_b0 = np.array(layout.right_ee_B, dtype=np.float64)

    out: list[tuple[float, np.ndarray, np.ndarray]] = []
    for i in range(samples):
        t = duration_sec * float(i) / max(1, samples - 1)
        d_l, d_r, _ = interp_ee_deltas_at(t, delta_keys)
        ee_l_w = rot.apply(ee_l_b0 + d_l) + base_pos
        ee_r_w = rot.apply(ee_r_b0 + d_r) + base_pos
        palm_l = palm_l0 + (ee_l_w - ee_l_w0)
        palm_r = palm_r0 + (ee_r_w - ee_r_w0)
        out.append((t, palm_l, palm_r))
    return out


def check_no_arm_cross_mjc(
    samples: list[tuple[float, np.ndarray, np.ndarray]],
    *,
    min_mjc_y_sep_m: float = 0.05,
    min_yup_z_sep_m: float = 0.10,
) -> tuple[bool, str]:
    """
    校验全程无「左右臂交叉」。

    openloong：MJC Y 轴为左右分列（左掌 MJC Y > 右掌 MJC Y）；Y-up Z 亦左负右正。
    """
    worst_y = float("inf")
    worst_z = float("inf")
    worst_t = 0.0
    for t, pl, pr in samples:
        sep_y = float(pl[1] - pr[1])
        sep_z = float(mjc_to_yup(pr)[2] - mjc_to_yup(pl)[2])
        worst_y = min(worst_y, sep_y)
        if sep_z < worst_z:
            worst_z = sep_z
            worst_t = t
        if sep_y < min_mjc_y_sep_m:
            return False, f"t={t:.2f}s MJC Y sep={sep_y:.3f}m < {min_mjc_y_sep_m}m (cross risk)"
        if sep_z < min_yup_z_sep_m:
            return False, f"t={t:.2f}s Y-up Z sep={sep_z:.3f}m < {min_yup_z_sep_m}m (cross risk)"
    return True, f"OK min MJC-Y sep={worst_y:.3f}m min Y-up-Z sep={worst_z:.3f}m @t≈{worst_t:.1f}s"


def validate_keyframe_doc(
    doc: dict[str, Any],
    session: dict[str, Any],
    *,
    min_mjc_y_sep_m: float = 0.05,
) -> tuple[bool, str]:
    """写入 JSON 前：加载轨迹并做无交叉采样校验。"""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(doc, tf, indent=2, ensure_ascii=False)
        tmp = Path(tf.name)
    try:
        traj = load_cloth_robot_gripper_keyframes(tmp)
        model, data, layout = prepare_mjcf_model_data(session, default_joint_values=dict(OPENLOONG_TELE_ARM_JOINT_VALUES))
        reload_gripper_trajectory(tmp, session=session)
        delta_keys = build_ee_delta_keyframes_mjc(layout, model, data, trajectory=traj)
        samples = sample_implied_palm_mjc(layout, model, data, delta_keys, traj.duration_sec)
        return check_no_arm_cross_mjc(samples, min_mjc_y_sep_m=min_mjc_y_sep_m)
    finally:
        tmp.unlink(missing_ok=True)


def _session_with_local_mjcf(session: dict[str, Any]) -> dict[str, Any]:
    """Studio tmp 路径不存在时，回退 ``~/.orcagym/tmp`` 同名或最新 MJCF。"""
    mj = dict(session.get("mujoco") or {})
    path = Path(str(mj.get("model_path", "")))
    if path.is_file():
        return session
    alt = Path.home() / ".orcagym/tmp" / path.name
    if not alt.is_file():
        tmp_dir = Path.home() / ".orcagym/tmp"
        cands = sorted(tmp_dir.glob("*.xml"), key=lambda p: p.stat().st_mtime, reverse=True)
        alt = cands[0] if cands else alt
    if alt.is_file():
        out = dict(session)
        out["mujoco"] = {**mj, "model_path": str(alt)}
        meta = dict(out.get("_cloth_robot_session_meta") or {})
        meta["source_mjcf"] = str(alt)
        out["_cloth_robot_session_meta"] = meta
        print(f"MJCF fallback: {alt}")
        return out
    return session


def find_latest_session(cloth_3d: Path, tag: str = "p23c") -> Path | None:
    pattern = f"cloth_sim_session_{tag}_*.json" if tag else "cloth_sim_session_*.json"
    cands = sorted(cloth_3d.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def main() -> int:
    cloth_3d = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate RectCloth gripper keyframes (no arm cross)")
    parser.add_argument("--session-json", type=Path, default=None)
    parser.add_argument("--session-tag", type=str, default="p23c")
    parser.add_argument(
        "--output",
        type=Path,
        default=cloth_3d / "cloth_robot_gripper_keyframes.test20260508_RectCloth.json",
    )
    parser.add_argument("--margin-m", type=float, default=0.08)
    parser.add_argument(
        "--grasp-mode",
        choices=sorted(_GRASP_BUILDERS.keys()),
        default="pmZ_fold",
        help="pmZ_fold=短边±Z对折（默认）；pmX_long=长边±X外夹（旧）",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    session_path = args.session_json or find_latest_session(cloth_3d, args.session_tag)
    if session_path is None or not session_path.is_file():
        print("ERROR: no session JSON; Studio Play + refresh session first", file=sys.stderr)
        return 1

    session = _session_with_local_mjcf(json.loads(session_path.read_text(encoding="utf-8")))
    model, data, layout = prepare_mjcf_model_data(session, default_joint_values=dict(OPENLOONG_TELE_ARM_JOINT_VALUES))
    build_fn = _GRASP_BUILDERS[args.grasp_mode]
    doc = build_fn(session, layout, margin_m=args.margin_m)
    ok, msg = validate_keyframe_doc(doc, session)
    print(f"Session: {session_path}")
    print(f"Cross check: {msg}")

    if not ok:
        print("ERROR: generated keyframes still fail cross check", file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps(doc, indent=2, ensure_ascii=False))
        return 0

    args.output.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote: {args.output}")
    print("Next: REGEN_REPLAY=1 bash XPBD/Cloth_robot/run_cloth_robot_p23c.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
