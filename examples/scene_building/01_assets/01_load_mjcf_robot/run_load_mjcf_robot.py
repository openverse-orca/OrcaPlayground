"""2.1.1 入口脚本：时序 spawn 三个机器人并保持场景。

时序 spawn g1_omnipicker → go2 → h1，以原点为中心并排，go2 居中。
spawn 完成后保持场景运行，按 Ctrl+C 退出。

用法:
    # 默认：间距 1.0m，间隔 5s
    python examples/scene_building/01_assets/run_load_mjcf_robot.py

    # 自定义间距与间隔
    python examples/scene_building/01_assets/run_load_mjcf_robot.py --spacing 1.5 --interval 3

    # 指定 Studio 地址
    python examples/scene_building/01_assets/run_load_mjcf_robot.py --addr 192.168.1.100:50051

前置条件:
    1. OrcaLab 已启动并监听 --addr
    2. 已订阅 OrcaPlaygroundAssets 资产包（含 g1_omnipicker / go2 / h1）
    3. 加载一个空关卡，点击运行

验证点:
    1. t=0s 视口出现 g1_omnipicker（左侧）
    2. t=5s 视口出现 g1_omnipicker + go2（中间）
    3. t=10s 视口出现 g1_omnipicker + go2 + h1（右侧）
    4. 三机器人并排，go2 居中
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import OrcaGymScene

# 添加脚本所在目录以支持同目录 import（数字开头目录不能作为包路径）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from load_mjcf_robot import (  # noqa: E402
    SPAWN_INTERVAL,
    SPACING,
    spawn_robot_sequence,
)

_logger = get_orca_logger()


def _log(msg: str) -> None:
    """统一日志输出（logger + print 兜底，确保终端可见）。"""
    _logger.info(msg)
    print(msg, flush=True)


def sceneinfo(addr: str, stage: str) -> None:
    """向 Studio 报告脚本运行阶段（rundata 记录）。

    参考 examples/replicator/run_actors.py 的 sceneinfo 逻辑：每次新建临时 scene，
    调用 get_rundata + set_image_enabled 后立即 close。

    Args:
        addr: OrcaLab gRPC 地址
        stage: "beginscene" | "endscene"
    """
    scene = OrcaGymScene(addr)
    try:
        script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
        scene.get_rundata(script_name, stage)
        if stage == "beginscene":
            _log("加载场景中")
        elif stage == "endscene":
            _log("加载完成")
        scene.set_image_enabled(1, True)
    finally:
        scene.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="时序 spawn 三个机器人（g1_omnipicker → go2 → h1），go2 居中"
    )
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument("--spacing", type=float, default=SPACING, help="机器人间距（米，沿 x 轴）")
    parser.add_argument(
        "--interval", type=float, default=SPAWN_INTERVAL, help="spawn 间隔（秒）"
    )
    args = parser.parse_args()

    _log(
        f"时序 spawn 三机器人 @ {args.addr}（间距 {args.spacing:.2f}m，间隔 {args.interval:.1f}s）"
    )

    # 1. Studio rundata：开始
    sceneinfo(args.addr, "beginscene")

    # 2. 时序 spawn（每个时序点独立发布周期：add 全部累积 + publish + sleep 3s + close）
    spawn_robot_sequence(
        addr=args.addr, spacing=args.spacing, interval=args.interval
    )

    # 3. Studio rundata：完成
    sceneinfo(args.addr, "endscene")

    # 4. 保持场景运行，等待用户观察
    _log("spawn 完成，保持场景运行，按 Ctrl+C 退出")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _log("用户中断")

    _log("退出")


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
