from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from envs.common.model_scanner import (
    build_suffix_template,
    match_robot_instances,
    probe_scene_model,
    require_complete_matches,
)
from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()


@dataclass(frozen=True)
class SceneRobotBinding:
    agent_names: list[str]
    robot_config: dict
    matched_prefixes: list[str]


def _remap_named_agents(template_names: list[str], agent_names: list[str]) -> list[str]:
    if not template_names:
        return []
    if not agent_names:
        return []

    remapped = []
    for index, _template_name in enumerate(template_names):
        if index >= len(agent_names):
            break
        remapped.append(agent_names[index])

    return remapped or [agent_names[0]]


def _build_runtime_robot_config(robot_config: dict, agent_names: list[str]) -> dict:
    runtime_robot_config = deepcopy(robot_config)
    runtime_robot_config["log_agent_names"] = _remap_named_agents(
        robot_config.get("log_agent_names", []),
        agent_names,
    )
    runtime_robot_config["visualize_command_agent_names"] = _remap_named_agents(
        robot_config.get("visualize_command_agent_names", []),
        agent_names,
    )
    if agent_names:
        runtime_robot_config["playable_agent_name"] = agent_names[0]
    runtime_robot_config["model_name"] = robot_config.get("model_name", "")
    return runtime_robot_config


def locate_scene_robot(
    orcagym_addr: str,
    model_name: str,
    robot_config: dict,
    time_step: float,
    min_count: int = 1,
    max_count: int | None = None,
) -> SceneRobotBinding:
    scene_names = probe_scene_model(orcagym_addr=orcagym_addr, time_step=time_step)
    template = build_suffix_template(
        model_name=model_name,
        joints=[robot_config["base_joint_name"], *list(robot_config["leg_joint_names"])],
        actuators=list(robot_config["actuator_names"]),
        sites=[robot_config["imu_site_name"], *list(robot_config["contact_site_names"])],
        bodies=list(robot_config.get("foot_body_names", [])),
        sensors=list(robot_config.get("sensor_foot_touch_names", [])),
    )
    report = match_robot_instances(template, scene_names)
    complete_matches = require_complete_matches(
        report,
        min_count=min_count,
        max_count=max_count,
        allow_empty_prefix=False,
    )

    matched_prefixes = [match.prefix for match in complete_matches]
    agent_names = matched_prefixes
    runtime_robot_config = _build_runtime_robot_config(robot_config, agent_names)
    runtime_robot_config["model_name"] = model_name

    _logger.info(
        "Resolved scene robot binding. "
        f"model_name={model_name}, matched_prefixes={matched_prefixes}, agent_names={agent_names}"
    )

    return SceneRobotBinding(
        agent_names=agent_names,
        robot_config=runtime_robot_config,
        matched_prefixes=matched_prefixes,
    )
