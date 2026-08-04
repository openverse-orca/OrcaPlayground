import copy
import json
import os
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from yaml import safe_load


def _to_vec3(val, key: str) -> np.ndarray:
    return np.asarray(val, dtype=np.float64).reshape(3)


def _to_quat_xyzw(val, key: str) -> np.ndarray:
    return np.asarray(val, dtype=np.float64).reshape(4)


def _to_rot_from_offset(payload: dict, arm_prefix: str, context: str) -> R:
    quat_key = f"{arm_prefix}_frame_offset_quat_o"
    euler_key = f"{arm_prefix}_frame_offset_euler_deg"
    if payload.get(quat_key) is not None and payload.get(euler_key) is not None:
        raise ValueError(f"{context}: cannot set both {quat_key} and {euler_key}")
    if payload.get(quat_key) is not None:
        return R.from_quat(_to_quat_xyzw(payload[quat_key], f"{context}.{quat_key}"))
    if payload.get(euler_key) is not None:
        euler_deg = _to_vec3(payload[euler_key], f"{context}.{euler_key}")
        return R.from_euler("xyz", euler_deg, degrees=True)
    return R.identity()


def _parse_gripper_token(val, prev: float, g_open: float, g_close: float) -> float:
    if val is None:
        return prev
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower()
    if s == "open":
        return g_open
    if s == "close":
        return g_close
    if s == "hold":
        return prev
    return float(s)


def _interp_quat_seq(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    key = R.from_quat(np.stack([q0_xyzw, q1_xyzw], axis=0))
    slerp = Slerp([0.0, 1.0], key)
    return slerp(alphas).as_quat().astype(np.float32)


def _resolve_body_name(env, object_name: str) -> str:
    if not object_name:
        raise ValueError("object name is empty")
    body_names = [name for name in env.model.get_body_names() if name]
    if object_name in body_names:
        return object_name
    lower = object_name.lower()
    exact_ci = [name for name in body_names if name.lower() == lower]
    if len(exact_ci) == 1:
        return exact_ci[0]
    if len(exact_ci) > 1:
        raise ValueError(f"Ambiguous body name {object_name}: {exact_ci}")
    contains = [name for name in body_names if lower in name.lower()]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        primary = [name for name in contains if name.lower().count(lower) >= 2]
        if len(primary) == 1:
            return primary[0]
        if len(primary) > 1:
            shortest = min(primary, key=len)
            if sum(1 for n in primary if len(n) == len(shortest)) == 1:
                return shortest
        shortest = min(contains, key=len)
        if sum(1 for n in contains if len(n) == len(shortest)) == 1:
            return shortest
        raise ValueError(f"Ambiguous body name {object_name}, candidates: {contains[:10]}")
    raise ValueError(f"Body {object_name} not found in scene")


def _query_body_pose_B(env, body_name: str, base_body: str) -> tuple[np.ndarray, np.ndarray]:
    pos_b = np.asarray(env.query_position_body_B(body_name, base_body), dtype=np.float64).reshape(3)
    quat_b = np.asarray(env.query_orientation_body_B(body_name, base_body), dtype=np.float64).reshape(4)
    return pos_b, quat_b


def _object_frame_to_base(
    obj_pos_b: np.ndarray,
    obj_quat_b: np.ndarray,
    target_o: Optional[np.ndarray],
    quat_o: Optional[np.ndarray],
    frame_offset_pos_o: np.ndarray,
    frame_offset_rot_o: R,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    obj_rot_b = R.from_quat(obj_quat_b)
    target_b = None
    quat_b = None
    if target_o is not None:
        target_o_effective = frame_offset_pos_o + frame_offset_rot_o.apply(target_o)
        target_b = obj_pos_b + obj_rot_b.apply(target_o_effective)
    if quat_o is not None:
        quat_o_effective = frame_offset_rot_o * R.from_quat(quat_o)
        quat_b = (obj_rot_b * quat_o_effective).as_quat()
    return target_b, quat_b


def _resolve_arm_object_fields(env, base_body: str, payload: dict, arm_prefix: str, context: str):
    object_key = f"{arm_prefix}_object_frame"
    target_b_key = f"{arm_prefix}_target_b"
    quat_b_key = f"{arm_prefix}_quat_b"
    target_o_key = f"{arm_prefix}_target_o"
    quat_o_key = f"{arm_prefix}_quat_o"
    frame_offset_pos_key = f"{arm_prefix}_frame_offset_o"

    if payload.get(object_key) is None:
        return

    if payload.get(target_b_key) is not None and payload.get(target_o_key) is not None:
        raise ValueError(f"{context}: cannot set both {target_b_key} and {target_o_key}")
    if payload.get(quat_b_key) is not None and payload.get(quat_o_key) is not None:
        raise ValueError(f"{context}: cannot set both {quat_b_key} and {quat_o_key}")

    resolved_body = _resolve_body_name(env, str(payload[object_key]))
    obj_pos_b, obj_quat_b = _query_body_pose_B(env, resolved_body, base_body)

    target_o = (
        _to_vec3(payload[target_o_key], f"{context}.{target_o_key}")
        if payload.get(target_o_key) is not None
        else None
    )
    quat_o = (
        _to_quat_xyzw(payload[quat_o_key], f"{context}.{quat_o_key}")
        if payload.get(quat_o_key) is not None
        else None
    )
    frame_offset_pos_o = (
        _to_vec3(payload[frame_offset_pos_key], f"{context}.{frame_offset_pos_key}")
        if payload.get(frame_offset_pos_key) is not None
        else np.zeros(3, dtype=np.float64)
    )
    frame_offset_rot_o = _to_rot_from_offset(payload, arm_prefix, context)
    target_b, quat_b = _object_frame_to_base(
        obj_pos_b, obj_quat_b, target_o, quat_o, frame_offset_pos_o, frame_offset_rot_o,
    )

    if target_b is not None:
        payload[target_b_key] = target_b.tolist()
    if quat_b is not None:
        payload[quat_b_key] = quat_b.tolist()
    payload[f"{arm_prefix}_resolved_body"] = resolved_body


def resolve_object_frames_in_segments(env, base_body: str, segments: list[dict]) -> list[dict]:
    resolved_segments = copy.deepcopy(segments)
    for index, segment in enumerate(resolved_segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segments[{index}] must be a dict")
        _resolve_arm_object_fields(env, base_body, segment, "l", f"segments[{index}]")
        _resolve_arm_object_fields(env, base_body, segment, "r", f"segments[{index}]")
    return resolved_segments


def resolve_pose_spec_for_current_scene(env, base_body: str, spec: dict) -> dict:
    resolved_spec = copy.deepcopy(spec)
    if resolved_spec.get("segments"):
        resolved_spec["segments"] = resolve_object_frames_in_segments(
            env, base_body, resolved_spec["segments"]
        )
    else:
        _resolve_arm_object_fields(env, base_body, resolved_spec, "l", "pose_file")
        _resolve_arm_object_fields(env, base_body, resolved_spec, "r", "pose_file")
    return resolved_spec


def build_segmented_trajectory(
    env,
    base_body: str,
    ee_site_l: str,
    ee_site_r: str,
    segments: list[dict],
    g_open: float,
    g_close: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ee_names = [ee_site_l, ee_site_r]
    ee_b = env.query_site_pos_and_quat_B(ee_names, [base_body])
    l0 = ee_b[ee_site_l]["xpos"].astype(np.float64)
    r0 = ee_b[ee_site_r]["xpos"].astype(np.float64)
    lq0 = ee_b[ee_site_l]["xquat"][[1, 2, 3, 0]].astype(np.float64)
    rq0 = ee_b[ee_site_r]["xquat"][[1, 2, 3, 0]].astype(np.float64)

    l_pos_all: list[np.ndarray] = []
    l_quat_all: list[np.ndarray] = []
    r_pos_all: list[np.ndarray] = []
    r_quat_all: list[np.ndarray] = []
    l_grip_all: list[np.ndarray] = []
    r_grip_all: list[np.ndarray] = []

    gl_prev, gr_prev = g_open, g_open

    for si, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise ValueError(f"segments[{si}] must be a dict")
        n_steps = int(seg["steps"])
        if n_steps < 1:
            raise ValueError(f"segments[{si}].steps must be >= 1")

        l_hold = bool(seg.get("l_hold", False))
        r_hold = bool(seg.get("r_hold", False))

        if l_hold:
            l1 = l0.copy()
        elif seg.get("l_target_b") is not None:
            l1 = _to_vec3(seg["l_target_b"], f"segments[{si}].l_target_b")
        elif seg.get("l_delta_b") is not None:
            l1 = l0 + _to_vec3(seg["l_delta_b"], f"segments[{si}].l_delta_b")
        else:
            l1 = l0.copy()

        if r_hold:
            r1 = r0.copy()
        elif seg.get("r_target_b") is not None:
            r1 = _to_vec3(seg["r_target_b"], f"segments[{si}].r_target_b")
        elif seg.get("r_delta_b") is not None:
            r1 = r0 + _to_vec3(seg["r_delta_b"], f"segments[{si}].r_delta_b")
        else:
            r1 = r0.copy()

        lq1 = (
            _to_quat_xyzw(seg["l_quat_b"], f"segments[{si}].l_quat_b")
            if seg.get("l_quat_b") is not None
            else lq0.copy()
        )
        rq1 = (
            _to_quat_xyzw(seg["r_quat_b"], f"segments[{si}].r_quat_b")
            if seg.get("r_quat_b") is not None
            else rq0.copy()
        )

        alphas = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
        l_pos_seg = np.stack([(1 - a) * l0 + a * l1 for a in alphas], axis=0).astype(np.float32)
        r_pos_seg = np.stack([(1 - a) * r0 + a * r1 for a in alphas], axis=0).astype(np.float32)
        l_quat_seg = _interp_quat_seq(lq0, lq1, alphas)
        r_quat_seg = _interp_quat_seq(rq0, rq1, alphas)

        gl_prev = _parse_gripper_token(seg.get("gripper_l"), gl_prev, g_open, g_close)
        gr_prev = _parse_gripper_token(seg.get("gripper_r"), gr_prev, g_open, g_close)
        l_grip_seg = np.full(n_steps, gl_prev, dtype=np.float32)
        r_grip_seg = np.full(n_steps, gr_prev, dtype=np.float32)

        l_pos_all.append(l_pos_seg)
        l_quat_all.append(l_quat_seg)
        r_pos_all.append(r_pos_seg)
        r_quat_all.append(r_quat_seg)
        l_grip_all.append(l_grip_seg)
        r_grip_all.append(r_grip_seg)

        l0 = l1.copy()
        r0 = r1.copy()
        lq0 = lq1.copy()
        rq0 = rq1.copy()

    return (
        np.concatenate(l_pos_all, axis=0),
        np.concatenate(l_quat_all, axis=0),
        np.concatenate(r_pos_all, axis=0),
        np.concatenate(r_quat_all, axis=0),
        np.concatenate(l_grip_all, axis=0),
        np.concatenate(r_grip_all, axis=0),
    )


def load_pose_spec_from_file(path: str) -> dict:
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yaml", ".yml")):
            spec = safe_load(f)
        else:
            spec = json.load(f)
    if not isinstance(spec, dict):
        raise ValueError("pose file root must be a dict")
    return spec


def dump_object_poses(env, base_body: str, object_name_filters: list[str]) -> None:
    for keyword in object_name_filters:
        try:
            resolved = _resolve_body_name(env, keyword)
            pos_b, quat_b = _query_body_pose_B(env, resolved, base_body)
            print(f"  {keyword} -> {resolved}: pos_B={pos_b}, quat_B_xyzw={quat_b}")
        except Exception as ex:
            print(f"  {keyword}: query failed: {ex}")
