"""Scene registry for doubleGripper_towel package."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class SceneSpec:
    """A scene item that can be selected by name."""

    name: str
    xml_path: str
    default_seconds: float
    description: str
    dual_gripper: bool = True


def _packaged_xml_path(filename: str) -> str:
    return str(files("doubleGripper_towel.scenes").joinpath(filename))


SCENE_REGISTRY: dict[str, SceneSpec] = {
    "packaged_fastfall_dualgripper": SceneSpec(
        name="packaged_fastfall_dualgripper",
        xml_path=_packaged_xml_path("towel_pickup_dualgripper_fastfall.xml"),
        default_seconds=5.5,
        description="Packaged FastFall dual-gripper towel scene.",
        dual_gripper=True,
    ),
    "workspace_fastfall_dualgripper": SceneSpec(
        name="workspace_fastfall_dualgripper",
        xml_path="/home/hjadmin/Mujoco/mujoco/model/"
        "towel_pickup_frank_gripper_auto6step_cornerpick_mesh_single_calibrated_dualgripper_fastfall_scene.xml",
        default_seconds=5.5,
        description="Workspace FastFall dual-gripper towel scene.",
        dual_gripper=True,
    ),
}


def resolve_scene_path(scene_name: str) -> Path:
    if scene_name not in SCENE_REGISTRY:
        known = ", ".join(sorted(SCENE_REGISTRY))
        raise KeyError(f"Unknown scene '{scene_name}'. Available: {known}")
    return Path(SCENE_REGISTRY[scene_name].xml_path).expanduser().resolve()
