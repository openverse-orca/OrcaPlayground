from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from envs.fluid.sim_env import SimEnv
from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()


@dataclass(frozen=True)
class AssetUiHint:
    display_name: str
    asset_search_name: str


ASSET_UI_HINTS: dict[str, AssetUiHint] = {
    "Lite3": AssetUiHint(
        display_name="Lite3 四足机器人",
        asset_search_name="lite3",
    ),
    "go2": AssetUiHint(
        display_name="Go2 四足机器人",
        asset_search_name="go2",
    ),
    "g1": AssetUiHint(
        display_name="G1 人形机器人",
        asset_search_name="g1",
    ),
    "G1": AssetUiHint(
        display_name="G1 人形机器人",
        asset_search_name="g1",
    ),
    "ZQSA01": AssetUiHint(
        display_name="ZQ SA01 人形机器人",
        asset_search_name="zq_sa01",
    ),
    "Character": AssetUiHint(
        display_name="Remy 角色",
        asset_search_name="Remy",
    ),
    "Ackerman": AssetUiHint(
        display_name="阿克曼底盘",
        asset_search_name="hummer",
    ),
    "WheeledChassis": AssetUiHint(
        display_name="差速底盘",
        asset_search_name="openloong",
    ),
    "XBot": AssetUiHint(
        display_name="XBot 机器人",
        asset_search_name="xbot",
    ),
}


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


def _build_ui_hint_message(
    report: SceneScanReport,
    *,
    problem_message: str,
    min_count: int,
    max_count: int | None,
) -> str:
    hint = ASSET_UI_HINTS.get(report.model_name)
    message_parts = [problem_message]
    if hint is not None:
        message_parts.append(
            f"缺少资产：{hint.asset_search_name}，请在资产搜索框内搜索并拖动到布局中。"
        )
        message_parts.append("如果搜索不到，请检查对应资产包是否已订阅。")

    if max_count == 1:
        message_parts.append("同一布局中仅允许保留 1 台该模型。")
    elif max_count is not None:
        message_parts.append(f"同一布局中最多允许保留 {max_count} 台该模型。")
    elif min_count > 1:
        message_parts.append(f"当前布局中至少需要 {min_count} 台完整匹配实例。")
    else:
        message_parts.append("请确认场景中至少存在 1 台完整匹配实例。")

    return " ".join(message_parts)


def _emit_terminal_hint(message: str) -> None:
    _logger.error(f"[场景绑定提示] {message}")


def require_complete_matches(
    report: SceneScanReport,
    *,
    min_count: int = 1,
    max_count: int | None = None,
    allow_empty_prefix: bool = False,
    orcagym_addr: str | None = None,
) -> list[InstanceMatch]:
    log_scene_scan_report(report)

    if not report.complete_matches:
        if report.partial_matches:
            first_partial = report.partial_matches[0]
            detail = []
            for category, missing_suffixes in first_partial.missing_suffixes.items():
                if missing_suffixes:
                    detail.append(f"{category} 缺失 {missing_suffixes[:8]}")
            error_message = (
                f"当前模型 {report.model_name} 未完全匹配，{' ; '.join(detail)}，正在退出运行。"
            )
            _emit_terminal_hint(
                _build_ui_hint_message(
                    report,
                    problem_message=f"{report.model_name} 关节或驱动器不匹配。",
                    min_count=min_count,
                    max_count=max_count,
                ),
            )
            raise ValueError(error_message)

        error_message = f"找不到对应的机器人型号：{report.model_name}，正在退出运行。"
        _emit_terminal_hint(
            _build_ui_hint_message(
                report,
                problem_message=f"当前布局中未找到 {report.model_name} 的完整匹配实例。",
                min_count=min_count,
                max_count=max_count,
            ),
        )
        raise ValueError(error_message)

    if len(report.complete_matches) < min_count:
        error_message = (
            f"当前模型 {report.model_name} 完整匹配数量不足，"
            f"需要至少 {min_count} 台，实际找到 {len(report.complete_matches)} 台，正在退出运行。"
        )
        _emit_terminal_hint(
            _build_ui_hint_message(
                report,
                problem_message=(
                    f"{report.model_name} 数量不足，当前只找到 {len(report.complete_matches)} 台。"
                ),
                min_count=min_count,
                max_count=max_count,
            ),
        )
        raise ValueError(error_message)

    if max_count is not None and len(report.complete_matches) > max_count:
        error_message = (
            f"当前模型 {report.model_name} 完整匹配数量过多，"
            f"允许最多 {max_count} 台，实际找到 {len(report.complete_matches)} 台："
            f"{[match.agent_name for match in report.complete_matches]}，正在退出运行。"
        )
        _emit_terminal_hint(
            _build_ui_hint_message(
                report,
                problem_message=(
                    f"{report.model_name} 数量过多，当前布局中找到 {len(report.complete_matches)} 台。"
                ),
                min_count=min_count,
                max_count=max_count,
            ),
        )
        raise ValueError(error_message)

    if not allow_empty_prefix:
        unnamed_matches = [match for match in report.complete_matches if not match.prefix]
        if unnamed_matches:
            error_message = (
                f"当前模型 {report.model_name} 检测到未命名空间化实例，"
                "脚本无法将其映射到运行时 agent_names，正在退出运行。"
            )
            _emit_terminal_hint(
                _build_ui_hint_message(
                    report,
                    problem_message=f"{report.model_name} 实例名称异常，无法绑定到运行时 agent。",
                    min_count=min_count,
                    max_count=max_count,
                ),
            )
            raise ValueError(error_message)

    return report.complete_matches


def ordered_match_names(match: InstanceMatch, category: str, suffixes: Iterable[str]) -> list[str]:
    category_matches = match.matched_names.get(category, {})
    return [category_matches[suffix] for suffix in suffixes if suffix in category_matches]
