"""
MuJoCo 世界 Z 初值偏移 ↔ SPH scene 垂直分量同步（Y-up 下 translation[1]）。

防止仅改 MuJoCo qpos、OrcaSPH 仍按原始 sph_scene.json 起机导致的初值错位。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# MuJoCo Z-up 竖直位移 → SPH scene.json translation 的下标（Y-up：[x, y_vertical, z]）
SPH_VERTICAL_TRANSLATION_INDEX = 1


@dataclass(frozen=True)
class SphSceneInitialOffsetPlan:
    """
    SPH 场景初值竖直偏移计划。

    delta_mujoco_z_m：与 MuJoCo 世界 +Z 同向的位移（米）；写入 SPH translation[1]。
    rigid_body_entity_names：要平移的 RigidBodies[].entityName 集合。
    offset_fluid_blocks：是否平移所有 FluidBlocks[].translation。
    """

    delta_mujoco_z_m: float
    rigid_body_entity_names: frozenset[str]
    offset_fluid_blocks: bool = True


def _hint_to_entity_name(hint: str) -> str:
    """joint_hint / 数字 ID → 常见 SPH entityName。"""
    h = (hint or "").strip()
    if not h:
        return ""
    if h.startswith("[") and "_bodyjoint" in h:
        return h
    if h.isdigit() or "waterjug" in h.lower():
        return f"[{h}]_bodyjoint" if h.isdigit() else h
    return h


def _load_entity_names_from_aliases(config: Dict[str, Any]) -> List[str]:
    dbg = config.get("debug", {}).get("force_position_trace", {})
    path = dbg.get("body_aliases_file")
    if not path:
        return []
    p = Path(path).expanduser()
    if not p.is_file():
        return []
    try:
        aliases = json.loads(p.read_text(encoding="utf-8"))
        return list(aliases.keys()) if isinstance(aliases, dict) else []
    except (json.JSONDecodeError, OSError):
        return []


def resolve_sph_scene_initial_offset_plan(
    config: Dict[str, Any],
    *,
    waterjug_entity_override: Optional[str] = None,
) -> Optional[SphSceneInitialOffsetPlan]:
    """
    从 fluid 配置解析 SPH 场景初值偏移计划。

    优先级：
      1. 显式 ``sph_scene_initial_offset``（通用，适用任意场景）
      2. ``water_jug_trajectory.initial_z_offset_m`` + 自动 entity / FluidBlocks
    """
    explicit = config.get("sph_scene_initial_offset") or {}
    if explicit.get("enabled") is False:
        return None

    traj = config.get("water_jug_trajectory") or {}

    delta: Optional[float] = None
    if "delta_mujoco_z_m" in explicit:
        delta = float(explicit["delta_mujoco_z_m"])
    elif explicit.get("sync_from_trajectory_z") and traj.get("enabled"):
        delta = float(traj.get("initial_z_offset_m", 0.0))
    elif traj.get("enabled") and int(traj.get("mode", 0)) != 1:
        delta = float(traj.get("initial_z_offset_m", 0.0))

    if delta is None or abs(delta) < 1e-12:
        return None

    offset_fluid_blocks = bool(
        explicit.get("offset_fluid_blocks", traj.get("offset_fluid_blocks", True))
    )

    names: List[str] = []
    if waterjug_entity_override:
        names.append(waterjug_entity_override)

    for key in ("rigid_body_entity_names", "entity_names"):
        raw = explicit.get(key)
        if isinstance(raw, list):
            names.extend(str(x) for x in raw if str(x).strip())

    if not names and explicit.get("auto_rigid_bodies_from_aliases"):
        names.extend(_load_entity_names_from_aliases(config))

    if not names:
        hint = traj.get("joint_hint") or ""
        ent = _hint_to_entity_name(str(hint))
        if ent:
            names.append(ent)

    if not names:
        logger.warning(
            "sph_scene_initial_offset: 未解析到 rigid_body_entity_names，仅偏移 FluidBlocks"
        )

    return SphSceneInitialOffsetPlan(
        delta_mujoco_z_m=delta,
        rigid_body_entity_names=frozenset(names),
        offset_fluid_blocks=offset_fluid_blocks,
    )


def _apply_delta_to_translation(
    translation: Sequence[float], delta_mujoco_z_m: float
) -> List[float]:
    """对 SPH translation 数组施加 MuJoCo Z 方向位移（映射到 index 1）。"""
    out = [float(x) for x in translation]
    idx = SPH_VERTICAL_TRANSLATION_INDEX
    if len(out) <= idx:
        while len(out) <= idx:
            out.append(0.0)
    out[idx] += float(delta_mujoco_z_m)
    return out


def patch_sph_scene_dict(
    scene: Dict[str, Any], plan: SphSceneInitialOffsetPlan
) -> Dict[str, int]:
    """
    就地修改 scene 字典中的 RigidBodies / FluidBlocks 竖直平移。

    返回统计：{"rigid_bodies": n, "fluid_blocks": m}。
    """
    stats = {"rigid_bodies": 0, "fluid_blocks": 0}
    delta = plan.delta_mujoco_z_m
    names = plan.rigid_body_entity_names

    for rb in scene.get("RigidBodies") or []:
        ent = str(rb.get("entityName") or "")
        if ent and ent in names:
            tr = rb.get("translation")
            if isinstance(tr, list) and len(tr) >= 3:
                rb["translation"] = _apply_delta_to_translation(tr, delta)
                stats["rigid_bodies"] += 1
                logger.info(
                    "SPH scene offset: RigidBody %s translation[1] %+0.4f m",
                    ent,
                    delta,
                )

    if plan.offset_fluid_blocks:
        for fb in scene.get("FluidBlocks") or []:
            tr = fb.get("translation")
            if isinstance(tr, list) and len(tr) >= 3:
                fb["translation"] = _apply_delta_to_translation(tr, delta)
                stats["fluid_blocks"] += 1
        if stats["fluid_blocks"]:
            logger.info(
                "SPH scene offset: %d FluidBlock(s) translation[1] %+0.4f m",
                stats["fluid_blocks"],
                delta,
            )

    return stats


def write_offset_sph_scene(
    source_scene_path: Path,
    output_scene_path: Path,
    plan: SphSceneInitialOffsetPlan,
) -> Dict[str, int]:
    """
    读取 source_scene_path，应用 plan，写入 output_scene_path（父目录自动创建）。

    若 output 与 source 相同则直接覆写；否则先复制再改，保留未改字段。
    """
    source_scene_path = source_scene_path.expanduser().resolve()
    output_scene_path = output_scene_path.expanduser().resolve()
    output_scene_path.parent.mkdir(parents=True, exist_ok=True)

    scene = json.loads(source_scene_path.read_text(encoding="utf-8"))
    stats = patch_sph_scene_dict(scene, plan)
    output_scene_path.write_text(
        json.dumps(scene, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Wrote offset SPH scene: %s (delta_mujoco_z=%+.4f m, rb=%d, fb=%d)",
        output_scene_path,
        plan.delta_mujoco_z_m,
        stats["rigid_bodies"],
        stats["fluid_blocks"],
    )
    return stats


def apply_sph_scene_initial_offset_to_context(
    config: Dict[str, Any],
    *,
    scene_path: Optional[Path],
    output_dir: Path,
    session_timestamp: str,
    waterjug_entity_override: Optional[str] = None,
) -> Optional[Path]:
    """
    若配置需要初值偏移：生成带偏移的 scene 副本并返回新路径；否则返回 None。
    """
    if scene_path is None or not scene_path.is_file():
        return None

    plan = resolve_sph_scene_initial_offset_plan(
        config, waterjug_entity_override=waterjug_entity_override
    )
    if plan is None:
        return None

    out = output_dir / f"sph_scene_initoffset_{session_timestamp}.json"
    write_offset_sph_scene(scene_path, out, plan)
    return out
