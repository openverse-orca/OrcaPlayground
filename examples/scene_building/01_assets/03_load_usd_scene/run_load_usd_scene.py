"""2.1.3 入口脚本：从场景 JSON 批量加载（d12.json）。

从 OrcaLab 场景导出的 JSON 文件批量加载多 actor 场景。
spawn 完成后保持场景运行，按 Ctrl+C 退出。

用法:
    # 默认：加载同目录 d12.json
    python examples/scene_building/01_assets/03_load_usd_scene/run_load_usd_scene.py

    # 指定 JSON 文件
    python examples/scene_building/01_assets/03_load_usd_scene/run_load_usd_scene.py --json /path/to/scene.json

    # 自定义间隔
    python examples/scene_building/01_assets/03_load_usd_scene/run_load_usd_scene.py --interval 2.0

    # 指定 Studio 地址
    python examples/scene_building/01_assets/03_load_usd_scene/run_load_usd_scene.py --addr 192.168.1.100:50051

前置条件:
    1. OrcaLab 已启动并监听 --addr
    2. 已订阅 d12_openpi_pick 和 d12_waist 资产包
    3. 加载一个空关卡，点击运行

验证点:
    1. JSON 正确解析出 5 个 AssetActor
    2. 5 个 actor 依次 spawn，前序不被销毁
    3. transform 的 position / rotation / scale 正确应用
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

from load_usd_scene import SPAWN_INTERVAL, load_usd_scene  # noqa: E402

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
        description="从 Studio 场景 JSON 文件批量加载 AssetActor"
    )
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="场景 JSON 文件路径（默认使用同目录 d12.json）",
    )
    parser.add_argument(
        "--interval", type=float, default=SPAWN_INTERVAL, help="spawn 间隔（秒）"
    )
    args = parser.parse_args()

    json_desc = args.json if args.json else "d12.json (默认)"
    _log(
        f"加载 USD 场景 @ {args.addr}（JSON: {json_desc}, 间隔 {args.interval:.1f}s）"
    )

    # 1. Studio rundata：开始
    sceneinfo(args.addr, "beginscene")

    # 2. 从 JSON 批量加载场景（append_scene 增量 spawn）
    load_usd_scene(addr=args.addr, json_path=args.json, interval=args.interval)

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
