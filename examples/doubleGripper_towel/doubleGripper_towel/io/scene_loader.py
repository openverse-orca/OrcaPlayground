"""Load scene xml into MuJoCo model/data."""

from __future__ import annotations

from pathlib import Path

import mujoco

from doubleGripper_towel.config.scene_data import SCENE_REGISTRY, SceneSpec, resolve_scene_path


def load_model_data_from_xml(xml_path: str | Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load model and data from xml path."""
    xml = Path(xml_path).expanduser().resolve()
    if not xml.exists():
        raise FileNotFoundError(f"Scene xml not found: {xml}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    return model, data


def load_model_data_from_scene_name(scene_name: str) -> tuple[mujoco.MjModel, mujoco.MjData, SceneSpec]:
    """Resolve scene name then load model/data."""
    if scene_name not in SCENE_REGISTRY:
        known = ", ".join(sorted(SCENE_REGISTRY))
        raise KeyError(f"Unknown scene '{scene_name}'. Available: {known}")
    scene = SCENE_REGISTRY[scene_name]
    model, data = load_model_data_from_xml(resolve_scene_path(scene_name))
    return model, data, scene
