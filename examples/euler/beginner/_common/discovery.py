"""名称发现助手 — 层 3 课「场景无关设计」的基础（FR-A08）。

层 3 课不硬编码物体位置，通过名称关键词发现用户拖入 / 自 spawn
的物体。发现失败的提示由各课给出（含 --default-scene 兜底指引）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 避免课程骨架在无 OrcaGym 环境时 import 失败
    from orca_gym.environment.euler import OrcaGymEulerEnv


def find_body(env: "OrcaGymEulerEnv", keyword: str) -> str | None:
    """按关键词查找 body 名称（不区分大小写，首个匹配）。

    前置：env 已完成 reset（模型已加载）。
    后置：返回首个名称含 keyword 的 body 名；无匹配返回 None。
    不变式：不修改 env 状态。

    TODO(P0)：body 名称枚举的公共 API 待实测确认（候选
    env.model 的名称列表接口）；确认后替换下方占位实现。
    """
    raise NotImplementedError(
        "P0 实测项：待确认 OrcaGymEulerEnv 的 body 名称枚举公共 API 后实现"
    )
