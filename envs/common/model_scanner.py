from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from envs.fluid.sim_env import SimEnv
from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()


@dataclass(frozen=True)
class SceneModelNames:
    bodies: set[str]
    joints: set[str]
    actuators: set[str]
    sites: set[str]
    sensors: set[str]


@dataclass(frozen=True)
class SuffixTemplate:
    model_name: str
    joints: list[str]
    actuators: list[str]
    sites: list[str]
    bodies: list[str]
    sensors: list[str]


@dataclass(frozen=True)
class InstanceMatch:
    prefix: str
    matched_names: dict[str, dict[str, str]]
    missing_suffixes: dict[str, list[str]]

    @property
    def is_complete(self) -> bool:
        return all(len(missing) == 0 for missing in self.missing_suffixes.values())

    @property
    def agent_name(self) -> str:
        return self.prefix


@dataclass(frozen=True)
class SceneScanReport:
    model_name: str
    complete_matches: list[InstanceMatch]
    partial_matches: list[InstanceMatch]
    scene_names: SceneModelNames

    @property
    def detected_count(self) -> int:
        return len(self.complete_matches)


def scan_scene_for_template(
    orcagym_addr: str,
    time_step: float,
    template: SuffixTemplate,
) -> SceneScanReport:
    scene_names = probe_scene_model(orcagym_addr=orcagym_addr, time_step=time_step)
    return match_robot_instances(template, scene_names)


def probe_scene_model(orcagym_addr: str, time_step: float) -> SceneModelNames:
    probe_env = SimEnv(
        frame_skip=1,
        orcagym_addr=orcagym_addr,
        agent_names=["SceneProbe"],
        time_step=time_step,
    )

    try:
        site_dict = probe_env.model.get_site_dict() if hasattr(probe_env.model, "get_site_dict") else {}
        sensor_dict = getattr(probe_env.model, "_sensor_dict", {})
        return SceneModelNames(
            bodies=set(probe_env.model.get_body_names()),
            joints=set(probe_env.model.get_joint_dict().keys()),
            actuators=set(probe_env.model.get_actuator_dict().keys()),
            sites=set(site_dict.keys()),
            sensors=set(sensor_dict.keys()),
        )
    finally:
        probe_env.close()


def build_suffix_template(
    model_name: str,
    joints: Iterable[str] = (),
    actuators: Iterable[str] = (),
    sites: Iterable[str] = (),
    bodies: Iterable[str] = (),
    sensors: Iterable[str] = (),
) -> SuffixTemplate:
    return SuffixTemplate(
        model_name=model_name,
        joints=list(joints),
        actuators=list(actuators),
        sites=list(sites),
        bodies=list(bodies),
        sensors=list(sensors),
    )


def _split_tokens(name: str) -> list[str]:
    return [token for token in name.split("_") if token]


def _match_prefix(full_name: str, suffix: str) -> str | None:
    full_tokens = _split_tokens(full_name)
    suffix_tokens = _split_tokens(suffix)
    if not suffix_tokens:
        return None

    if len(full_tokens) < len(suffix_tokens):
        return None

    if full_tokens[-len(suffix_tokens):] != suffix_tokens:
        return None

    prefix_tokens = full_tokens[:-len(suffix_tokens)]
    return "_".join(prefix_tokens)


def _collect_matches_by_prefix(required_suffixes: list[str], available_names: set[str]) -> dict[str, dict[str, str]]:
    matches_by_prefix: dict[str, dict[str, str]] = defaultdict(dict)
    for suffix in required_suffixes:
        for full_name in available_names:
            prefix = _match_prefix(full_name, suffix)
            if prefix is None:
                continue
            matches_by_prefix[prefix][suffix] = full_name
    return dict(matches_by_prefix)


def _build_instance_match(
    prefix: str,
    category_matches: dict[str, dict[str, dict[str, str]]],
    template: SuffixTemplate,
) -> InstanceMatch:
    required_by_category = {
        "joints": template.joints,
        "actuators": template.actuators,
        "sites": template.sites,
        "bodies": template.bodies,
        "sensors": template.sensors,
    }
    missing_suffixes = {}
    matched_names = {}

    for category, required_suffixes in required_by_category.items():
        matched = category_matches.get(category, {}).get(prefix, {})
        matched_names[category] = matched
        missing_suffixes[category] = [suffix for suffix in required_suffixes if suffix not in matched]

    return InstanceMatch(
        prefix=prefix,
        matched_names=matched_names,
        missing_suffixes=missing_suffixes,
    )


def match_robot_instances(template: SuffixTemplate, scene_names: SceneModelNames) -> SceneScanReport:
    category_available_names = {
        "joints": scene_names.joints,
        "actuators": scene_names.actuators,
        "sites": scene_names.sites,
        "bodies": scene_names.bodies,
        "sensors": scene_names.sensors,
    }
    category_required_suffixes = {
        "joints": template.joints,
        "actuators": template.actuators,
        "sites": template.sites,
        "bodies": template.bodies,
        "sensors": template.sensors,
    }

    category_matches = {
        category: _collect_matches_by_prefix(required_suffixes, category_available_names[category])
        for category, required_suffixes in category_required_suffixes.items()
        if required_suffixes
    }

    candidate_prefixes = set()
    for matches_by_prefix in category_matches.values():
        candidate_prefixes.update(matches_by_prefix.keys())

    instance_matches = [
        _build_instance_match(prefix, category_matches, template)
        for prefix in sorted(candidate_prefixes)
    ]
    complete_matches = [match for match in instance_matches if match.is_complete]
    partial_matches = [match for match in instance_matches if not match.is_complete]

    return SceneScanReport(
        model_name=template.model_name,
        complete_matches=complete_matches,
        partial_matches=partial_matches,
        scene_names=scene_names,
    )


def log_scene_scan_report(report: SceneScanReport) -> None:
    if report.complete_matches:
        _logger.info(
            f"当前模型 {report.model_name} 完全匹配，找到 {len(report.complete_matches)} 台："
            f"{[match.agent_name for match in report.complete_matches]}"
        )
    else:
        _logger.warning(f"当前模型 {report.model_name} 没有完整匹配实例。")

    for match in report.partial_matches[:8]:
        missing_parts = []
        for category, missing_suffixes in match.missing_suffixes.items():
            if missing_suffixes:
                missing_parts.append(f"{category}={missing_suffixes[:6]}")
        if missing_parts:
            _logger.warning(
                f"候选实例 {match.prefix or '<root>'} 未完全匹配：{' ; '.join(missing_parts)}"
            )


def require_complete_matches(
    report: SceneScanReport,
    *,
    min_count: int = 1,
    max_count: int | None = None,
    allow_empty_prefix: bool = False,
) -> list[InstanceMatch]:
    log_scene_scan_report(report)

    if not report.complete_matches:
        if report.partial_matches:
            first_partial = report.partial_matches[0]
            detail = []
            for category, missing_suffixes in first_partial.missing_suffixes.items():
                if missing_suffixes:
                    detail.append(f"{category} 缺失 {missing_suffixes[:8]}")
            raise ValueError(
                f"当前模型 {report.model_name} 未完全匹配，{' ; '.join(detail)}，正在退出运行。"
            )

        raise ValueError(f"找不到对应的机器人型号：{report.model_name}，正在退出运行。")

    if len(report.complete_matches) < min_count:
        raise ValueError(
            f"当前模型 {report.model_name} 完整匹配数量不足，"
            f"需要至少 {min_count} 台，实际找到 {len(report.complete_matches)} 台，正在退出运行。"
        )

    if max_count is not None and len(report.complete_matches) > max_count:
        raise ValueError(
            f"当前模型 {report.model_name} 完整匹配数量过多，"
            f"允许最多 {max_count} 台，实际找到 {len(report.complete_matches)} 台："
            f"{[match.agent_name for match in report.complete_matches]}，正在退出运行。"
        )

    if not allow_empty_prefix:
        unnamed_matches = [match for match in report.complete_matches if not match.prefix]
        if unnamed_matches:
            raise ValueError(
                f"当前模型 {report.model_name} 检测到未命名空间化实例，"
                "脚本无法将其映射到运行时 agent_names，正在退出运行。"
            )

    return report.complete_matches


def ordered_match_names(match: InstanceMatch, category: str, suffixes: Iterable[str]) -> list[str]:
    category_matches = match.matched_names.get(category, {})
    return [category_matches[suffix] for suffix in suffixes if suffix in category_matches]
