"""名称发现助手 — 层 3 课「场景无关设计」的基础（FR-A08）。

层 3 课不硬编码物体位置，通过名称关键词发现用户拖入 / 自 spawn
的物体。发现失败的提示由各课给出（含 --default-scene 兜底指引）。

实现依据：OrcaGymEulerEnv 在线模式连接后，model.get_body_names()
返回当前 Studio 场景的全部 body 名（dev 分支 scene_scanner 探针
同款机制）。P0 实测（tools/verify_spawn_env_link.py）进一步验证。
"""

from __future__ import annotations

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()


def probe_body_names(addr: str, time_step: float = 0.002) -> list[str]:
    """用探针 env 读取当前场景的全部 body 名称（只读，不改场景）。

    前置：OrcaLab 已运行。
    后置：返回 body 名列表（可能为空）；探针 env 已关闭。
    """
    probe = OrcaGymEulerEnv(
        frame_skip=1,
        orcagym_addr=addr,
        agent_names=["SceneProbe"],
        time_step=time_step,
    )
    try:
        return list(probe.model.get_body_names())
    finally:
        probe.close()


def find_body(env: OrcaGymEulerEnv, keyword: str) -> str | None:
    """在已连接的 env 中按关键词查找 body 名称（不区分大小写，首个匹配）。

    前置：env 已完成 reset（模型已加载）。
    后置：返回首个名称含 keyword 的 body 名；无匹配返回 None。
    不变式：不修改 env 状态。
    """
    names = list(env.model.get_body_names())
    for name in names:
        if keyword.lower() in name.lower():
            return name
    _logger.warning(f"find_body: 未找到含「{keyword}」的 body（共 {len(names)} 个 body）")
    return None
