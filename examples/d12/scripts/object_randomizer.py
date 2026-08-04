import copy
import os
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
from yaml import safe_load

MJ_FREE_JOINT_QPOS_SIZE = 7


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
        ends_with = [name for name in contains if name.lower().endswith(lower)]
        if len(ends_with) == 1:
            return ends_with[0]
        if len(ends_with) > 1:
            shortest = min(ends_with, key=len)
            if sum(1 for n in ends_with if len(n) == len(shortest)) == 1:
                return shortest
        else:
            shortest = min(contains, key=len)
            if sum(1 for n in contains if len(n) == len(shortest)) == 1:
                return shortest
        raise ValueError(f"Ambiguous body name {object_name}, candidates: {contains[:10]}")
    raise ValueError(f"Body {object_name} not found in scene")


def _resolve_joint_name(env, joint_name: str) -> str:
    if not joint_name:
        raise ValueError("joint name is empty")
    joint_dict = env.model.get_joint_dict() or {}
    joint_names = list(joint_dict.keys())
    if joint_name in joint_names:
        return joint_name
    lower = joint_name.lower()
    exact_ci = [name for name in joint_names if name.lower() == lower]
    if len(exact_ci) == 1:
        return exact_ci[0]
    if len(exact_ci) > 1:
        raise ValueError(f"Ambiguous joint name {joint_name}: {exact_ci}")
    contains = [name for name in joint_names if lower in name.lower()]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        raise ValueError(f"Ambiguous joint name {joint_name}, candidates: {contains[:10]}")
    raise ValueError(f"Joint {joint_name} not found in scene")


def _find_free_joint_for_body(env, body_name: str) -> str:
    joint_dict = env.model.get_joint_dict() or {}
    body_id = env.model.body_name2id(body_name)
    candidates = []
    for joint_name, joint_info in joint_dict.items():
        if joint_info.get("BodyID") == body_id:
            candidates.append((joint_name, joint_info))
    if not candidates:
        raise ValueError(f"Body {body_name} has no joint in current model")
    for joint_name, _ in candidates:
        qpos = np.asarray(env.query_joint_qpos([joint_name])[joint_name], dtype=np.float64).reshape(-1)
        if qpos.size == MJ_FREE_JOINT_QPOS_SIZE:
            return joint_name
    candidate_names = [joint_name for joint_name, _ in candidates]
    raise ValueError(f"Body {body_name} has no free joint, candidate joints: {candidate_names}")


def _sample_uniform_vec3(bounds, key: str, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.shape != (3, 2):
        raise ValueError(f"{key} must have shape [3][2]")
    low, high = arr[:, 0], arr[:, 1]
    if np.any(high < low):
        raise ValueError(f"{key} has upper bound smaller than lower bound")
    return rng.uniform(low, high)


def _sample_rotation_delta_xyzw(entry: dict, context: str, rng: np.random.Generator) -> np.ndarray:
    if entry.get("rotation_delta_quat_xyzw") is not None and entry.get("rotation_delta_euler_deg") is not None:
        raise ValueError(f"{context}: cannot set both rotation_delta_quat_xyzw and rotation_delta_euler_deg")
    if entry.get("rotation_delta_quat_xyzw") is not None:
        quat_xyzw = np.asarray(entry["rotation_delta_quat_xyzw"], dtype=np.float64).reshape(4)
        return quat_xyzw / np.linalg.norm(quat_xyzw)
    if entry.get("rotation_delta_euler_deg") is not None:
        euler = np.asarray(entry["rotation_delta_euler_deg"], dtype=np.float64).reshape(3)
        return R.from_euler("xyz", euler, degrees=True).as_quat()
    if entry.get("rotation_range_deg") is not None:
        euler = _sample_uniform_vec3(entry["rotation_range_deg"], f"{context}.rotation_range_deg", rng)
        return R.from_euler("xyz", euler, degrees=True).as_quat()
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _sample_translation_delta(entry: dict, context: str, rng: np.random.Generator) -> np.ndarray:
    if entry.get("position_delta") is not None and entry.get("position_range") is not None:
        raise ValueError(f"{context}: cannot set both position_delta and position_range")
    if entry.get("position_delta") is not None:
        return np.asarray(entry["position_delta"], dtype=np.float64).reshape(3)
    if entry.get("position_range") is not None:
        return _sample_uniform_vec3(entry["position_range"], f"{context}.position_range", rng)
    return np.zeros(3, dtype=np.float64)


def _build_joint_qpos_update_from_rand_entry(
    env, entry: dict, context: str, rng: np.random.Generator
) -> tuple[str, np.ndarray]:
    object_name = entry.get("object") or entry.get("body")
    joint_name = entry.get("joint")
    if object_name is None and joint_name is None:
        raise ValueError(f"{context}: one of object/body or joint is required")
    if object_name is not None and joint_name is not None:
        raise ValueError(f"{context}: provide object/body or joint, not both")

    if joint_name is None:
        body_name = _resolve_body_name(env, str(object_name))
        joint_name = _find_free_joint_for_body(env, body_name)
    else:
        joint_name = _resolve_joint_name(env, str(joint_name))

    base_qpos = np.asarray(env.query_joint_qpos([joint_name])[joint_name], dtype=np.float64).reshape(-1).copy()
    if base_qpos.size != MJ_FREE_JOINT_QPOS_SIZE:
        raise ValueError(f"{context}: joint {joint_name} is not a free joint, qpos size={base_qpos.size}")

    pos_delta = _sample_translation_delta(entry, context, rng)
    quat_delta_xyzw = _sample_rotation_delta_xyzw(entry, context, rng)
    base_pos = base_qpos[:3]
    base_quat_xyzw = base_qpos[[4, 5, 6, 3]]

    if entry.get("position_frame", "world").lower() == "local":
        pos_delta_world = R.from_quat(base_quat_xyzw).apply(pos_delta)
    else:
        pos_delta_world = pos_delta

    new_pos = base_pos + pos_delta_world
    new_quat_xyzw = (R.from_quat(base_quat_xyzw) * R.from_quat(quat_delta_xyzw)).as_quat()
    new_qpos = np.concatenate([new_pos, new_quat_xyzw[[3, 0, 1, 2]]]).astype(np.float64)
    return joint_name, new_qpos


def apply_object_randomization(env, rand_spec: dict) -> dict[str, np.ndarray]:
    randomize_entries = rand_spec.get("objects", [])
    if not randomize_entries:
        return {}
    if not isinstance(randomize_entries, list):
        raise ValueError("rand.yaml field `objects` must be a list")

    seed = rand_spec.get("seed")
    rng = np.random.default_rng(int(seed) if seed is not None else None)
    joint_qpos_updates: dict[str, np.ndarray] = {}
    for index, entry in enumerate(randomize_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"rand.yaml objects[{index}] must be a dict")
        joint_name, qpos = _build_joint_qpos_update_from_rand_entry(
            env, entry, f"rand.yaml.objects[{index}]", rng
        )
        joint_qpos_updates[joint_name] = qpos

    if joint_qpos_updates:
        env.apply_joint_qpos_dict(joint_qpos_updates)
        env.mj_forward()
        for joint_name, qpos in joint_qpos_updates.items():
            quat_xyzw = qpos[[4, 5, 6, 3]]
            print(f"  randomized {joint_name}: pos={qpos[:3]}, quat_xyzw={quat_xyzw}")
    return joint_qpos_updates


def advance_rand_spec_seed(rand_spec: dict, episode_index: int) -> dict:
    if not rand_spec:
        return {}
    updated = copy.deepcopy(rand_spec)
    seed = updated.get("seed")
    if seed is not None:
        updated["seed"] = int(seed) + int(episode_index)
    return updated


def load_rand_spec_from_file(path: str) -> dict:
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "r", encoding="utf-8") as f:
        data = safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} root must be a dict")
    return data
