"""2.1.3 加载 USD 场景（从 Studio JSON 场景描述文件批量加载）。

从 OrcaLab 场景导出的 JSON 文件（GroupActor + AssetActor 树结构）批量加载场景。

JSON 格式参考（examples/d12/demo/d12.json）:
    {
      "type": "GroupActor",
      "children": [
        {
          "type": "AssetActor",
          "name": "xxx",
          "asset_path": "assets/...",
          "transform": {
            "position": "[x,y,z]",
            "rotation": "[w,x,y,z]",  # 四元数
            "scale": 1.0
          }
        },
        ...
      ]
    }

范式说明:
    使用 append_scene() 增量 spawn —— 仅 spawn m_addActorMap 中的新 Actor，
    不销毁已 spawn 的实体。每个 actor 添加后立即 append_scene。

模式：在线（需 OrcaLab）
资产来源：OrcaLab 资产库 https://simassets.orca3d.cn/

验证点:
    1. 从 JSON 文件批量加载所有 AssetActor
    2. 每个 actor 的 position/rotation/scale 正确应用
    3. append_scene 增量 spawn，前序 actor 不被销毁

参见:
    03_示例开发计划.md §2.1.3
    examples/d12/demo/d12.json （JSON 格式参考）
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orca_gym.utils.rotations as rotations
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import Actor, OrcaGymScene

_logger = get_orca_logger()


def _log(msg: str) -> None:
    """统一日志输出（logger + print 兜底，确保终端可见）。"""
    _logger.info(msg)
    print(msg, flush=True)


# spawn 间隔（秒）
SPAWN_INTERVAL: float = 1.0

# 清空后等待 Studio 响应的缓冲时间（秒）
_CLEAR_BUFFER: float = 0.5


@dataclass
class SceneActorSpec:
    """从 JSON 解析出的单个 actor 规格。"""

    name: str
    asset_path: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # 四元数 (w, x, y, z)
    scale: float


def _parse_vec(s: str, n: int) -> tuple[float, ...]:
    """解析 JSON 中的 "[a,b,c]" 字符串为 tuple。

    Args:
        s: 形如 "[1,2,3]" 或 "[1,2,3,4]" 的字符串
        n: 期望的元素个数（3 或 4）

    Returns:
        tuple[float, ...]: 解析后的元组
    """
    s = s.strip().strip("[]")
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != n:
        raise ValueError(f"期望 {n} 个元素，实际解析到 {len(parts)}: {s!r}")
    return tuple(parts)


def parse_scene_json(json_path: str | Path) -> list[SceneActorSpec]:
    """从 OrcaLab 场景 JSON 解析 AssetActor 列表。

    递归遍历 GroupActor 的 children，收集所有 AssetActor。

    Args:
        json_path: JSON 文件路径

    Returns:
        list[SceneActorSpec]: 所有 AssetActor 的规格列表
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"场景 JSON 文件不存在: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    specs: list[SceneActorSpec] = []
    _collect_asset_actors(data, specs)
    return specs


def _collect_asset_actors(node: dict[str, Any], out: list[SceneActorSpec]) -> None:
    """递归收集 AssetActor。

    Args:
        node: JSON 节点（dict）
        out: 输出列表
    """
    node_type = node.get("type", "")

    if node_type == "AssetActor":
        name = node["name"]
        asset_path = node["asset_path"]
        tf = node["transform"]
        pos = _parse_vec(tf["position"], 3)
        quat = _parse_vec(tf["rotation"], 4)  # (w, x, y, z)
        scale = float(tf.get("scale", 1.0))
        out.append(
            SceneActorSpec(
                name=name,
                asset_path=asset_path,
                position=pos,
                rotation=quat,
                scale=scale,
            )
        )

    # 递归处理 children（GroupActor）
    for child in node.get("children", []):
        _collect_asset_actors(child, out)


def _make_actor(spec: SceneActorSpec) -> Actor:
    """根据 SceneActorSpec 构造 Actor。

    JSON 中 rotation 为 (w, x, y, z)，需要转换为 rotations 期望的格式。
    """
    return Actor(
        name=spec.name,
        asset_path=spec.asset_path,
        position=np.array(spec.position, dtype=np.float64),
        rotation=np.array(spec.rotation, dtype=np.float64),  # (w, x, y, z)
        scale=spec.scale,
    )


def clear_scene(addr: str) -> None:
    """清空 Studio 场景（新 OrcaGymScene 实例 + publish 空）。"""
    _log("清空现有场景...")
    temp = OrcaGymScene(addr)
    temp.publish_scene()
    temp.close()
    time.sleep(_CLEAR_BUFFER)
    _log("场景已清空")


def load_usd_scene(
    addr: str = "localhost:50051",
    json_path: str | Path | None = None,
    interval: float = SPAWN_INTERVAL,
) -> OrcaGymScene:
    """从 JSON 场景描述文件批量加载场景。

    使用 append_scene() 增量 spawn：每个 actor 添加后立即 append_scene，
    前序 actor 不被销毁。

    Args:
        addr: OrcaLab gRPC 地址
        json_path: 场景 JSON 文件路径。None 则使用 d12.json 作为示例
        interval: actor 之间的 spawn 间隔（秒）

    Returns:
        OrcaGymScene 实例（保持运行，调用方负责 close）
    """
    if json_path is None:
        # 默认使用同目录下的 d12.json 作为示例
        json_path = Path(__file__).parent / "d12.json"

    all_specs = parse_scene_json(json_path)
    _log(f"从 JSON 加载到 {len(all_specs)} 个 AssetActor:")
    for s in all_specs:
        _log(f"  - {s.name} @ {s.position} scale={s.scale}")

    # 清空场景一次（后续用 append_scene 增量添加）
    clear_scene(addr)

    scene = OrcaGymScene(addr)

    for i, spec in enumerate(all_specs):
        _log(f"[{i + 1}/{len(all_specs)}] 开始 spawn: {spec.name}")
        scene.add_actor(_make_actor(spec))
        _log(f"已经添加 {spec.name} @ {spec.position}")
        scene.append_scene()
        _log(f"  本轮添加完毕，当前场景共 {i + 1} 个 actor")
        if i < len(all_specs) - 1:
            _log(f"  等待 {interval:.1f}s 后 spawn 下一个...")
            time.sleep(interval)

    _log("本次添加完毕所有模型，如需退出请在当前终端中断或者在OrcaLab退出运行时模式")
    return scene


def sceneinfo(addr: str, stage: str) -> None:
    """向 Studio 报告脚本运行阶段（rundata 记录）。"""
    scene = OrcaGymScene(addr)
    try:
        import os
        import sys
        script_name = os.path.basename(sys.argv[0]) if sys.argv else __file__
        scene.get_rundata(script_name, stage)
        if stage == "beginscene":
            _log("加载场景中")
        elif stage == "endscene":
            _log("加载完成")
        scene.set_image_enabled(1, True)
    finally:
        scene.close()


def main() -> None:
    import argparse

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
    parser.add_argument("--interval", type=float, default=SPAWN_INTERVAL, help="spawn 间隔（秒）")
    args = parser.parse_args()

    json_path = args.json if args.json else None
    _log(f"加载 USD 场景 @ {args.addr}（JSON: {json_path or 'd12.json (默认)'}, 间隔 {args.interval:.1f}s）")

    # 1. Studio rundata：开始
    sceneinfo(args.addr, "beginscene")

    # 2. 加载场景
    load_usd_scene(addr=args.addr, json_path=json_path, interval=args.interval)

    # 3. Studio rundata：完成
    sceneinfo(args.addr, "endscene")

    # 4. 保持场景运行
    _log("spawn 完成，保持场景运行，按 Ctrl+C 退出")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _log("用户中断")

    _log("退出")


if __name__ == "__main__":
    import sys

    try:
        main()
    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        _logger.error(f"脚本异常退出: {exc}\n{tb}")
        print(f"[ERROR] 脚本异常退出: {exc}", file=sys.stderr, flush=True)
        print(tb, file=sys.stderr, flush=True)
        sys.exit(1)
