"""P0 实测：spawn 资产能否被 EulerEnv 按名称驱动（缺口 ①）。

验证 REQUIREMENTS.md §7.2 / BeginnerLessonsSkeleton 02_接口契约 §3 的
关键链路：脚本 spawn 的 XML 实体 → OrcaGymEulerEnv 连接加载 →
按名称发现 → 步进驱动。

链路分三段验证（任何一段失败即给出定位）：
    [1] spawn：用真实 spawnable 资产（go2，OrcaPlaygroundAssets 包）
        按三步范式发布到 Studio
    [2] 发现：探针 env 连接，读取场景 body 名称表（model.get_body_names()），
        确认 spawn 实体的 body 可按名称发现，并解析 agent 前缀
    [3] 驱动：以解析出的前缀构造正式 env，reset → do_simulation →
        读取 body 位置，确认状态可读、步进可执行

用法（需 OrcaLab 已启动 + 已订阅 OrcaPlaygroundAssets，首次订阅后重启）:
    /path/to/conda/envs/orca/bin/python tools/verify_spawn_env_link.py

结论写回: .trae_history/BeginnerLessonsSkeleton/99_实施记录.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许从任意 cwd 运行：把 repo root 加入 sys.path（本文件位于 <root>/tools/）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.scene_recipe import (
    ActorSpec,
    clear_scene,
    spawn_recipe,
)

_logger = get_orca_logger()

# 真实 spawnable XML 机器人资产（dev 分支 scene_building 已验证可 spawn）
# 资产包：OrcaPlaygroundAssets
GO2_PATH = "assets/e071469a36d3c8aa/default_project/prefabs/go2_usda"
_INSTANCE_NAME = "go2_test"
_KEYWORD = "go2"

_TIME_STEP = 0.002
_FRAME_SKIP = 5
_STEP_ROUNDS = 10


def check_spawn(addr: str) -> object:
    """[1] spawn 阶段：发布 go2 到干净场景，返回存活的 scene。"""
    _logger.info("[1/3] spawn：清空场景并发布 go2 ...")
    clear_scene(addr)
    scene = spawn_recipe(
        addr,
        [ActorSpec(name=_INSTANCE_NAME, asset_path=GO2_PATH, position=(0.0, 0.0, 0.1))],
    )
    _logger.info("[1/3] PASS：go2 已发布到 Studio")
    return scene


def resolve_agent_prefix(addr: str) -> str:
    """[2] 发现阶段：探针 env 读 body 名表，解析 spawn 实体的 agent 前缀。

    返回：agent 前缀（body 名的顶层命名空间，如 "go2_usda"）。
    raise RuntimeError：若 body 表中找不到含关键词的名称。
    """
    _logger.info("[2/3] 发现：探针 env 连接，读取场景 body 名称表 ...")
    probe = OrcaGymEulerEnv(
        frame_skip=1,
        orcagym_addr=addr,
        agent_names=["SceneProbe"],
        time_step=_TIME_STEP,
    )
    try:
        body_names = list(probe.model.get_body_names())
        _logger.info(f"      场景共 {len(body_names)} 个 body")

        matches = sorted(n for n in body_names if _KEYWORD in n.lower())
        if not matches:
            raise RuntimeError(
                f"body 名称表中未发现含「{_KEYWORD}」的实体。全部 body：{body_names[:20]}..."
            )

        # agent 前缀 = body 名的顶层命名空间（"go2_usda/base" → "go2_usda"）
        prefixes = sorted({n.split("/", 1)[0] for n in matches})
        prefix = prefixes[0]
        _logger.info(
            f"      发现 {len(matches)} 个匹配 body（前缀 {prefixes}），"
            f"如 {matches[:3]}"
        )
        _logger.info(f"[2/3] PASS：spawn 实体可按名称发现，agent 前缀 = {prefix}")
        return prefix
    finally:
        probe.close()


def check_drive(addr: str, prefix: str) -> bool:
    """[3] 驱动阶段：正式 env 按前缀构造，reset → step → 读状态。

    返回：True 表示驱动链路可用。
    """
    _logger.info(f"[3/3] 驱动：以 agent_names=[{prefix}] 构造正式 env ...")
    env = OrcaGymEulerEnv(
        frame_skip=_FRAME_SKIP,
        orcagym_addr=addr,
        agent_names=[prefix],
        time_step=_TIME_STEP,
    )
    try:
        env.reset()
        _logger.info(f"      reset 成功：nq={env.model.nq}, nv={env.model.nv}, nu={env.model.nu}")

        # 按名称读取状态（层 3 场景无关设计依赖的能力）
        target_body = f"{prefix}/base" if f"{prefix}/base" in env.model.get_body_names() else prefix
        pos_before = np.asarray(env.data.body_xpos(target_body)).copy()
        _logger.info(f"      {target_body} 初始位置 = {pos_before.round(4)}")

        # 步进驱动：零控制量推 10 轮（FR-N07：步进+render 最小循环）
        ctrl = np.zeros(env.model.nu)
        for i in range(_STEP_ROUNDS):
            env.do_simulation(ctrl, _FRAME_SKIP)
        env.render()

        pos_after = np.asarray(env.data.body_xpos(target_body)).copy()
        delta = float(np.linalg.norm(pos_after - pos_before))
        _logger.info(
            f"      {_STEP_ROUNDS} 轮步进后位置 = {pos_after.round(4)}（位移 {delta:.4f} m）"
        )
        _logger.info("[3/3] PASS：状态可按名称读取，do_simulation 步进执行成功")
        # 位移可能为 0（零控制下机器人静止），不作为失败条件；只验证链路
        return True
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="P0 实测：spawn → EulerEnv 按名称驱动")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    args = parser.parse_args()

    _logger.info("=" * 60)
    _logger.info("P0 实测：spawn 资产能否被 EulerEnv 按名称驱动")
    _logger.info(f"  资产: {GO2_PATH}（OrcaPlaygroundAssets 包）")
    _logger.info("=" * 60)

    scene = None
    try:
        scene = check_spawn(addr=args.addr)  # [1] spawn
        prefix = resolve_agent_prefix(addr=args.addr)  # [2] 发现
        ok = check_drive(addr=args.addr, prefix=prefix)  # [3] 驱动

        _logger.info("=" * 60)
        _logger.info("结论：PASS — spawn 实体可被 EulerEnv 按名称发现与驱动" if ok else "结论：FAIL")
        _logger.info("回填：discovery.find_body 用 model.get_body_names() 实现；")
        _logger.info("      lesson_13.connect_env 用解析前缀构造 OrcaGymEulerEnv。")
        _logger.info("=" * 60)
        return 0 if ok else 1
    except Exception as exc:
        hint = ""
        if "Connection refused" in str(exc) or "code = 14" in str(exc):
            hint = "（OrcaLab 未运行？请先在 OrcaPlayground 根目录执行 `orcalab .` 启动）"
        _logger.error(f"结论：FAIL — {exc}{hint}")
        import traceback

        _logger.error(traceback.format_exc())
        return 1
    finally:
        if scene is not None:
            scene.close()
            _logger.info("场景连接已关闭")


if __name__ == "__main__":
    sys.exit(main())
