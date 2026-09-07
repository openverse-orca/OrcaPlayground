"""
RectCloth 关键帧自动更新 + 机器手动作变化追踪。

流程：
  1. 从最新 p23c session 写入/刷新关键帧 JSON（track 轨迹模板）
  2. 对比写入前后关键帧差异（指纹 + 逐帧掌位）
  3. 可选联调后从 CLOTH_DEBUG CSV 采样掌位，与基线对比是否随关键帧变化

用法::
  python -m modules.cloth_robot_keyframe_motion_track update --write
  python -m modules.cloth_robot_keyframe_motion_track diff --before path.bak --after path.json
  python -m modules.cloth_robot_keyframe_motion_track track --debug-dir logs/cloth_debug_*
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from modules.cloth_robot_rect_keyframe_builder import (
    _cloth_rect_meta,
    _make_keyframe,
    _session_with_local_mjcf,
    find_latest_session,
    validate_keyframe_doc,
)
from modules.cloth_robot_scene_layout import (
    OPENLOONG_TELE_ARM_JOINT_VALUES,
    prepare_mjcf_model_data,
)

_DEFAULT_OUT = (
    Path(__file__).resolve().parents[1]
    / "cloth_robot_gripper_keyframes.test20260508_RectCloth.json"
)
_MANIP_LOGS = (
    Path(__file__).resolve().parents[5]
    / "OrcaManipulation"
    / "src"
    / "examples"
    / "dataCollection_cloth"
    / "logs"
)
_BASELINE_NAME = "rect_cloth_hand_motion_baseline.json"
_DEFAULT_LEFT_PALM = "openloong_gripper_2f85_fix_base_usda_zbll_base_link"
_DEFAULT_RIGHT_PALM = "openloong_gripper_2f85_fix_base_usda_zbr_base_link"


def build_rect_cloth_keyframes_short_edge_midpoint_fold_along_long(
    session: dict[str, Any],
    layout,
    *,
    margin_m: float = 0.08,
    fold_hold_z_m: float = 0.06,
    lift_y_m: float = 0.15,
) -> dict[str, Any]:
    """
    夹取两条 **短边中点**（X=cx，Z=cz±half_z），折痕沿 **长边 X**，掌位沿 Z 向布心收拢对折。

    几何（Y-up）：
      - 短边在 Z=±half_z，中点 (cx, cy, cz±half_z)
      - 长边沿 X；对折时折痕平行 X，左右掌 Z 从 ±half_z 收到 ±fold_hold_z_m
    """
    del layout
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
            comment="微抬（仍保持短边中点 X/Z）",
        ),
        _make_keyframe(
            10.0,
            [cx, cy + lift_y_m, cz - fold_z],
            [cx, cy + lift_y_m, cz + fold_z],
            "closed",
            comment="沿 Z 向布心对折（折痕平行长边 X）",
        ),
        _make_keyframe(
            12.0,
            [cx, cy + lift_y_m, cz - fold_z],
            [cx, cy + lift_y_m, cz + fold_z],
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
            "grasp_margin_m": margin_m,
            "fold_hold_z_m": fold_hold_z_m,
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


# 兼容旧脚本名
build_rect_cloth_keyframes_short_edge_fold_pmZ_track = (
    build_rect_cloth_keyframes_short_edge_midpoint_fold_along_long
)


def keyframes_fingerprint(doc: dict[str, Any]) -> str:
    """
    对关键帧中所有掌位与 grip 指令做稳定哈希，用于快速判断 JSON 是否变化。
    """
    parts: list[str] = []
    for kf in doc.get("keyframes") or []:
        parts.append(f"t={kf.get('t_sec')}|g={kf.get('grip_cmd')}|n={kf.get('neutral', False)}")
        for side in ("left_yup", "right_yup"):
            if side in kf:
                parts.append(f"{side}=" + ",".join(f"{v:.6f}" for v in kf[side]))
    raw = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def extract_palm_targets(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """
    从关键帧文档提取每帧时间戳与左右掌 Y-up 目标，便于 diff 与报告。
    """
    rows: list[dict[str, Any]] = []
    for kf in doc.get("keyframes") or []:
        row: dict[str, Any] = {
            "t_sec": float(kf.get("t_sec", 0.0)),
            "grip_cmd": kf.get("grip_cmd"),
            "neutral": bool(kf.get("neutral", False)),
        }
        if "left_yup" in kf:
            row["left_yup"] = [float(x) for x in kf["left_yup"]]
        if "right_yup" in kf:
            row["right_yup"] = [float(x) for x in kf["right_yup"]]
        rows.append(row)
    return rows


def diff_keyframe_docs(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """
    逐帧对比两份关键帧文档，返回人类可读差异行（仅掌位与 grip 变化）。
    """
    b_map = {r["t_sec"]: r for r in extract_palm_targets(before)}
    a_map = {r["t_sec"]: r for r in extract_palm_targets(after)}
    lines: list[str] = []
    fp_b = keyframes_fingerprint(before)
    fp_a = keyframes_fingerprint(after)
    if fp_b != fp_a:
        lines.append(f"指纹: {fp_b} -> {fp_a}")
    else:
        lines.append(f"指纹未变: {fp_a}")

    all_t = sorted(set(b_map) | set(a_map))
    for t in all_t:
        b = b_map.get(t)
        a = a_map.get(t)
        if b is None:
            lines.append(f"t={t:.1f}s: 新增帧")
            continue
        if a is None:
            lines.append(f"t={t:.1f}s: 删除帧")
            continue
        if b.get("grip_cmd") != a.get("grip_cmd"):
            lines.append(f"t={t:.1f}s: grip {b.get('grip_cmd')} -> {a.get('grip_cmd')}")
        for side in ("left_yup", "right_yup"):
            pb, pa = b.get(side), a.get(side)
            if pb is None and pa is None:
                continue
            if pb is None or pa is None:
                lines.append(f"t={t:.1f}s {side}: {pb} -> {pa}")
                continue
            d = np.linalg.norm(np.array(pb) - np.array(pa))
            if d > 1e-6:
                lines.append(f"t={t:.1f}s {side}: Δ={d:.4f} m  {pb} -> {pa}")
    return lines


def mjc_to_yup(p: np.ndarray) -> np.ndarray:
    """MuJoCo 世界坐标转 Y-up（与 verify_replay_osc_tracking 一致）。"""
    return np.array([p[0], p[2], -p[1]], dtype=np.float64)


def load_mjc_palm_at_mf(units_csv: Path, logical_name: str, mf: int) -> np.ndarray | None:
    """
    从 ``mujoco_orcalink_units.csv`` 读取指定宏步的掌位（Y-up）。
    """
    if not units_csv.is_file():
        return None
    for row in csv.DictReader(units_csv.open(encoding="utf-8")):
        if (
            row.get("logical_name") == logical_name
            and row.get("data_type") == "POSITION"
            and str(row.get("object_id", "")).endswith("_body_p")
            and int(row["macro_frame"]) == mf
        ):
            return mjc_to_yup(np.array([float(row["x"]), float(row["y"]), float(row["z"])]))
    return None


def detect_palm_logical_names(units_csv: Path) -> tuple[str, str]:
    """
    从 ``mujoco_orcalink_units.csv`` 推断左右掌 ``logical_name``（``zbll`` / ``zbr`` 掌体）。
    """
    left, right = _DEFAULT_LEFT_PALM, _DEFAULT_RIGHT_PALM
    if not units_csv.is_file():
        return left, right
    for row in csv.DictReader(units_csv.open(encoding="utf-8")):
        if row.get("data_type") != "POSITION" or not str(row.get("object_id", "")).endswith("_body_p"):
            continue
        ln = row.get("logical_name") or ""
        if ln.endswith("zbll_base_link"):
            left = ln
        elif ln.endswith("zbr_base_link") and "zbrl" not in ln:
            right = ln
    return left, right


def sample_palm_trajectory(
    debug_dir: Path,
    macro_frames: list[int],
    *,
    left_name: str | None = None,
    right_name: str | None = None,
) -> dict[str, Any]:
    """
    在 CLOTH_DEBUG 目录中按宏步列表采样左右掌实际 Y-up 轨迹。
    """
    units_csv = debug_dir / "mujoco_orcalink_units.csv"
    if left_name is None or right_name is None:
        det_l, det_r = detect_palm_logical_names(units_csv)
        left_name = left_name or det_l
        right_name = right_name or det_r
    samples: dict[str, dict[str, list[float] | None]] = {}
    for mf in macro_frames:
        left = load_mjc_palm_at_mf(units_csv, left_name, mf)
        right = load_mjc_palm_at_mf(units_csv, right_name, mf)
        samples[str(mf)] = {
            "left_yup": None if left is None else left.tolist(),
            "right_yup": None if right is None else right.tolist(),
        }
    return {"debug_dir": str(debug_dir), "samples": samples}


def baseline_path() -> Path:
    """掌位运动基线 JSON 的默认路径。"""
    return _MANIP_LOGS / _BASELINE_NAME


def load_baseline(path: Path | None = None) -> dict[str, Any] | None:
    """读取上次保存的掌位运动基线；不存在则返回 None。"""
    p = path or baseline_path()
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_baseline(
    doc: dict[str, Any],
    *,
    keyframes_fp: str,
    debug_dir: Path,
    macro_frames: list[int],
    path: Path | None = None,
) -> Path:
    """
    将当前关键帧指纹 + 调试目录掌位采样写入基线文件，供下次对比。
    """
    p = path or baseline_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "keyframes_fingerprint": keyframes_fp,
        "trajectory": doc,
        **sample_palm_trajectory(debug_dir, macro_frames),
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def compare_hand_motion(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    motion_threshold_m: float = 0.005,
) -> dict[str, Any]:
    """
    对比两次运行的掌位采样，判断机器手是否随关键帧/联调发生变化。

    返回字段：
      - keyframes_changed: 指纹是否不同
      - hands_moved: 任一采样点相对基线位移是否超过 ``motion_threshold_m``
      - max_delta_m / per_mf: 各宏步最大位移
    """
    fp_b = baseline.get("keyframes_fingerprint", "")
    fp_c = current.get("keyframes_fingerprint", "")
    keyframes_changed = fp_b != fp_c and bool(fp_b) and bool(fp_c)

    per_mf: dict[str, float] = {}
    max_delta = 0.0
    b_samples = (baseline.get("trajectory") or baseline).get("samples") or baseline.get("samples") or {}
    c_samples = current.get("samples") or {}

    for mf_key in sorted(set(b_samples) | set(c_samples), key=lambda x: int(x)):
        b = b_samples.get(mf_key) or {}
        c = c_samples.get(mf_key) or {}
        deltas: list[float] = []
        for side in ("left_yup", "right_yup"):
            pb, pc = b.get(side), c.get(side)
            if pb is None or pc is None:
                continue
            deltas.append(float(np.linalg.norm(np.array(pb) - np.array(pc))))
        if deltas:
            d = max(deltas)
            per_mf[mf_key] = d
            max_delta = max(max_delta, d)

    hands_moved = max_delta >= motion_threshold_m
    followed = keyframes_changed and hands_moved
    stale = keyframes_changed and not hands_moved

    return {
        "keyframes_changed": keyframes_changed,
        "hands_moved": hands_moved,
        "hands_followed_keyframes": followed,
        "hands_stale_after_keyframe_edit": stale,
        "max_delta_m": max_delta,
        "per_mf_max_delta_m": per_mf,
        "baseline_fingerprint": fp_b,
        "current_fingerprint": fp_c,
        "baseline_debug_dir": baseline.get("debug_dir"),
        "current_debug_dir": current.get("debug_dir"),
    }


def find_latest_debug_dir(logs_dir: Path | None = None) -> Path | None:
    """在 dataCollection_cloth/logs 下找最新 cloth_debug_* 目录。"""
    root = logs_dir or _MANIP_LOGS
    if not root.is_dir():
        return None
    cands = sorted(root.glob("cloth_debug_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def backup_keyframes_json(target: Path) -> Path:
    """备份关键帧 JSON，文件名带时间戳。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_suffix(f".{ts}.bak.json")
    shutil.copy2(target, backup)
    return backup


def update_keyframes_from_session(
    *,
    session_path: Path | None,
    session_tag: str,
    output: Path,
    write: bool,
    margin_m: float,
    fold_hold_z_m: float,
) -> tuple[dict[str, Any], Path, bool, str]:
    """
    从 session 生成短边中点对折关键帧并可选写入 ``output``。

    返回 (新文档, session_path, validate_ok, validate_msg)。
    """
    embodied/cloth = Path(__file__).resolve().parents[1]
    sp = session_path or find_latest_session(cloth_3d, session_tag)
    if sp is None or not sp.is_file():
        raise FileNotFoundError("未找到 session JSON；请先 Studio Play 并刷新 session")

    session = _session_with_local_mjcf(json.loads(sp.read_text(encoding="utf-8")))
    model, data, layout = prepare_mjcf_model_data(
        session, default_joint_values=dict(OPENLOONG_TELE_ARM_JOINT_VALUES)
    )
    del model, data
    doc = build_rect_cloth_keyframes_short_edge_midpoint_fold_along_long(
        session,
        layout,
        margin_m=margin_m,
        fold_hold_z_m=fold_hold_z_m,
    )
    ok, msg = validate_keyframe_doc(doc, session)
    if write:
        output.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc, sp, ok, msg


def print_track_report(
    kf_diff: list[str],
    motion: dict[str, Any] | None,
    *,
    validate_ok: bool,
    validate_msg: str,
) -> None:
    """打印关键帧 diff 与掌位追踪结论。"""
    print("=== 关键帧变化 ===")
    for line in kf_diff:
        print(f"  {line}")
    print(f"Cross check: {validate_msg} ({'OK' if validate_ok else 'FAIL'})")
    if motion is None:
        print("=== 掌位追踪 ===")
        print("  无历史基线；本次联调结果已写入基线，下次运行可对比机器手是否变化")
        return
    print("=== 掌位追踪 ===")
    print(f"  关键帧是否变化: {motion['keyframes_changed']}")
    print(f"  机器手是否移动: {motion['hands_moved']} (max Δ={motion['max_delta_m']:.4f} m)")
    if motion.get("hands_followed_keyframes"):
        print("  结论: 关键帧已变，机器手轨迹随之变化 ✓")
    elif motion.get("hands_stale_after_keyframe_edit"):
        print("  结论: 关键帧已变但机器手几乎未动 — 请确认 REGEN_REPLAY=1 且 replay 已重生成")
    elif not motion["keyframes_changed"] and motion["hands_moved"]:
        print("  结论: 关键帧未变但掌位有漂移（可能为 OSC/初始态差异）")
    else:
        print("  结论: 关键帧与掌位相对基线均无显著变化")
    for mf, d in sorted(motion.get("per_mf_max_delta_m", {}).items(), key=lambda x: int(x[0])):
        print(f"    mf={mf}: max Δ={d:.4f} m")


def cmd_update(args: argparse.Namespace) -> int:
    output: Path = args.output
    before_doc: dict[str, Any] | None = None
    backup: Path | None = None
    if output.is_file() and args.write:
        backup = backup_keyframes_json(output)
        before_doc = json.loads(output.read_text(encoding="utf-8"))
        print(f"已备份: {backup}")

    doc, sp, ok, msg = update_keyframes_from_session(
        session_path=args.session_json,
        session_tag=args.session_tag,
        output=output,
        write=args.write,
        margin_m=args.margin_m,
        fold_hold_z_m=args.fold_hold_z_m,
    )
    print(f"Session: {sp}")
    print(f"指纹: {keyframes_fingerprint(doc)}")
    if args.write:
        print(f"已写入: {output}")
    else:
        print(json.dumps(doc, indent=2, ensure_ascii=False))

    if before_doc is not None:
        print_track_report(diff_keyframe_docs(before_doc, doc), None, validate_ok=ok, validate_msg=msg)
    return 0 if ok else 1


def cmd_diff(args: argparse.Namespace) -> int:
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    for line in diff_keyframe_docs(before, after):
        print(line)
    return 0


def cmd_track(args: argparse.Namespace) -> int:
    output: Path = args.output
    kf_doc = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
    kf_fp = keyframes_fingerprint(kf_doc) if kf_doc else ""

    debug_dir = Path(args.debug_dir) if args.debug_dir else find_latest_debug_dir()
    if debug_dir is None or not debug_dir.is_dir():
        print("ERROR: 未找到 cloth_debug_* 目录", file=sys.stderr)
        return 1

    macro_frames = [int(x) for x in args.macro_frames.split(",") if x.strip()]
    current_traj = sample_palm_trajectory(debug_dir, macro_frames)
    current_payload = {
        "keyframes_fingerprint": kf_fp,
        **current_traj,
    }

    baseline = load_baseline(Path(args.baseline)) if args.baseline else load_baseline()
    motion: dict[str, Any] | None = None
    kf_diff: list[str] = []
    if baseline is not None:
        motion = compare_hand_motion(baseline, current_payload, motion_threshold_m=args.motion_threshold_m)
        if baseline.get("keyframes_fingerprint") != kf_fp:
            # 基线里可能没有完整 keyframes，用指纹说明即可
            kf_diff = [f"当前指纹 {kf_fp} vs 基线 {baseline.get('keyframes_fingerprint')}"]
        else:
            kf_diff = ["当前关键帧指纹与基线相同"]
    else:
        kf_diff = ["无掌位基线，本次将写入新基线"]

    print_track_report(kf_diff, motion, validate_ok=True, validate_msg="n/a (track only)")

    if args.save_baseline or baseline is None:
        saved = save_baseline(
            current_payload,
            keyframes_fp=kf_fp,
            debug_dir=debug_dir,
            macro_frames=macro_frames,
        )
        print(f"基线已保存: {saved}")
    return 0


def main() -> int:
    embodied/cloth = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="RectCloth 关键帧更新与机器手动作追踪")
    sub = parser.add_subparsers(dest="command", required=True)

    p_up = sub.add_parser("update", help="从 session 刷新关键帧 JSON")
    p_up.add_argument("--session-json", type=Path, default=None)
    p_up.add_argument("--session-tag", type=str, default="p23c")
    p_up.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    p_up.add_argument("--write", action="store_true", help="写入 JSON（默认仅打印）")
    p_up.add_argument("--margin-m", type=float, default=0.08, help="短边外侧 approach 外扩（m）")
    p_up.add_argument("--fold-hold-z-m", type=float, default=0.06, help="对折后 |Z| 目标（m）")
    p_up.set_defaults(func=cmd_update)

    p_df = sub.add_parser("diff", help="对比两份关键帧 JSON")
    p_df.add_argument("--before", type=Path, required=True)
    p_df.add_argument("--after", type=Path, required=True)
    p_df.set_defaults(func=cmd_diff)

    p_tr = sub.add_parser("track", help="对比最新 CLOTH_DEBUG 掌位与基线")
    p_tr.add_argument("--debug-dir", type=Path, default=None)
    p_tr.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    p_tr.add_argument("--baseline", type=Path, default=None)
    p_tr.add_argument("--macro-frames", type=str, default="50,100,150,200,250,300")
    p_tr.add_argument("--motion-threshold-m", type=float, default=0.005)
    p_tr.add_argument("--save-baseline", action="store_true")
    p_tr.set_defaults(func=cmd_track)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
