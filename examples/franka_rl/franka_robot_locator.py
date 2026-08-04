from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from examples._common.model_scanner import (
    build_suffix_template,
    match_robot_instances,
    probe_scene_model,
    require_complete_matches,
)
from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()


@dataclass(frozen=True)
class SceneFrankaBinding:
    agent_names: list[str]
    robot_config: dict
    matched_prefixes: list[str]


def _build_runtime_franka_config(robot_config: dict, agent_names: list[str]) -> dict:
    runtime_robot_config = deepcopy(robot_config)
    runtime_robot_config["log_agent_names"] = agent_names[:1] if agent_names else []
    runtime_robot_config["visualize_command_agent_names"] = agent_names[:1] if agent_names else []
    if agent_names:
        runtime_robot_config["playable_agent_name"] = agent_names[0]
    runtime_robot_config["model_name"] = robot_config.get("model_name", "")
    return runtime_robot_config


def locate_scene_franka(
    orcagym_addr: str,
    model_name: str,
    robot_config: dict,
    time_step: float,
    min_count: int = 1,
    max_count: int | None = None,
) -> SceneFrankaBinding:
    scene_names = probe_scene_model(orcagym_addr=orcagym_addr, time_step=time_step)
    template = build_suffix_template(
        model_name=model_name,
        joints=[
            *robot_config["arm_joint_names"],
            *robot_config["gripper_joint_names"],
            robot_config["obj_joint_name"],
        ],
        actuators=list(robot_config["actuator_names"]),
        sites=[
            robot_config["ee_site_name"],
            robot_config["obj_site_name"],
        ],
        bodies=[robot_config["base_joint_name"]],
        sensors=[],
    )
    report = match_robot_instances(template, scene_names)
    complete_matches = require_complete_matches(
        report,
        min_count=min_count,
        max_count=max_count,
        allow_empty_prefix=False,
        orcagym_addr=orcagym_addr,
    )

    matched_prefixes = [match.prefix for match in complete_matches]
    agent_names = matched_prefixes
    runtime_robot_config = _build_runtime_franka_config(robot_config, agent_names)
    runtime_robot_config["model_name"] = model_name

    _logger.info(
        "Resolved scene franka binding. "
        f"model_name={model_name}, matched_prefixes={matched_prefixes}, agent_names={agent_names}"
    )

    return SceneFrankaBinding(
        agent_names=agent_names,
        robot_config=runtime_robot_config,
        matched_prefixes=matched_prefixes,
    )
