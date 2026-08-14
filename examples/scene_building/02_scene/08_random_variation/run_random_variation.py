"""2.2.2 (8) 入口脚本：场景随机变体（厨房家具域随机化）。

流程:
    1. OrcaGymScene add_actor + publish_scene —— spawn N 个厨房家具（1~3 倍清单）
    2. 完成（不创建 env，不运行仿真）

用法:
    # 默认全随机（每次运行结果不同）
    python examples/scene_building/02_scene/08_random_variation/run_random_variation.py

    # 指定 seed 复现某次布局
    python examples/scene_building/02_scene/08_random_variation/run_random_variation.py --seed 7

详见: 08_random_variation.md
"""

from __future__ import annotations

import argparse
import os
import sys

from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import OrcaGymScene

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from random_variation import build_random_variation  # noqa: E402

_logger = get_orca_logger()

# 需订阅的资产包名（运行前提醒用户检查）
_REQUIRED_ASSETS: list[str] = ["kitchen_night"]


def _log(msg: str) -> None:
    """双路输出（logger + print），确保终端可见。"""
    _logger.info(msg)
    print(msg, flush=True)


def _check_assets() -> None:
    """提醒用户检查所需资产是否已订阅。"""
    _log(f"[资产检查] 本示例需在 OrcaLab 资产库中订阅: {', '.join(_REQUIRED_ASSETS)}")
    _log(f"  若未订阅，请在 OrcaLab 资产库搜索并点击订阅")
    _log(f"  并在 OrcaLab 中加载空关卡")


def main() -> None:
    parser = argparse.ArgumentParser(description="场景随机变体（厨房家具域随机化）")
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（省略则全随机，指定值可复现）",
    )
    args = parser.parse_args()

    _check_assets()
    seed_desc = "全随机" if args.seed is None else f"seed={args.seed}"
    _log(f"\n搭建域随机化场景 @ {args.addr}（{seed_desc}）")

    # ── spawn 厨房家具到 OrcaLab ──
    _log("[1/1] spawn 厨房家具到 OrcaLab...")
    scene = OrcaGymScene(args.addr)
    try:
        collector = build_random_variation(scene, seed=args.seed)
        scene.publish_scene()
        _log(f"  spawn 完成：{len(collector.actors)} 个厨房家具已发布")
    finally:
        scene.close()

    _log("完成。请在 OrcaLab 视口查看场景。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        _logger.error(f"脚本异常退出: {exc}\n{tb}")
        print(f"[ERROR] 脚本异常退出: {exc}", file=sys.stderr, flush=True)
        print(tb, file=sys.stderr, flush=True)
        sys.exit(1)
