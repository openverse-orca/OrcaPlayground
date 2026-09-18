"""实测：spawn 后运行时改色（set_material_info）是否对本地 pak 资产生效。

背景（.trae_history 第 54 号之后的决策）：第 10 课现状用"换资产变体"
教改色，但 OrcaGym 提供 scene.set_material_info(actor_name,
MaterialInfo(base_color=RGBA)) 运行时改色接口。本脚本实测该接口对
本地导入包（345a60e1cced）资产是否被 O3DE 渲染端正确呈现。

验证序列：spawn 灰色 cube → 2s 后改红 → 2s 后改蓝。
预期：视口中方块颜色随之变化（物理形状不变）。

用法:
    python -m tools.verify_runtime_recolor --addr localhost:50051

判定:
    - gRPC 全部 SUCCESS 且视口颜色变化 → 接口可用，第 10 课改写为
      set_material_info 主路径
    - gRPC SUCCESS 但视口不变 → 渲染端未消费，维持资产变体法并
      在 FAQ 记录
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import MaterialInfo

from examples.euler.beginner._common.scene_recipe import (
    FLOOR_Z_OFFSET,
    ActorSpec,
    clear_scene,
    setup_console_logging,
    spawn_recipe,
)

_logger = get_orca_logger()

# 与 18 课资产清单一致的本地包路径（cube 为中性灰色）
_CUBE_PATH = "assets/345a60e1cced/prefabs/cube_usda"

# 验证色序：红 → 蓝（RGBA，alpha=1 不透明）
_REDLIKE = np.array([0.9, 0.1, 0.1, 1.0])
_BLUEST = np.array([0.1, 0.2, 0.9, 1.0])


def main() -> int:
    parser = argparse.ArgumentParser(description="实测 spawn 后运行时改色")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    args = parser.parse_args()

    setup_console_logging()
    _logger.info("=" * 60)
    _logger.info("实测：set_material_info 运行时改色（本地 pak 资产）")
    _logger.info("=" * 60)

    clear_scene(args.addr)
    scene = spawn_recipe(args.addr, [
        ActorSpec(name="ground", asset_path="assets/345a60e1cced/prefabs/floor_usda",
                  position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="box_1", asset_path=_CUBE_PATH, position=(0.0, 0.0, 0.5)),
    ])
    _logger.info("[1/3] 灰色 cube 已 spawn（原始资产色），等待 2s 后改红")

    try:
        time.sleep(2.0)
        scene.set_material_info("box_1", MaterialInfo(base_color=_REDLIKE))
        _logger.info("[2/3] 已下发红色 RGBA (0.9, 0.1, 0.1, 1.0)，等待 2s 后改蓝")

        time.sleep(2.0)
        scene.set_material_info("box_1", MaterialInfo(base_color=_BLUEST))
        _logger.info("[3/3] 已下发蓝色 RGBA (0.1, 0.2, 0.9, 1.0)")

        _logger.info("请到视口确认：方块是否 红 → 蓝 变化（形状不变）")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        _logger.info("用户中断，正在退出")
    finally:
        scene.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        import traceback

        _logger.error(f"脚本异常退出: {exc}\n{traceback.format_exc()}")
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
