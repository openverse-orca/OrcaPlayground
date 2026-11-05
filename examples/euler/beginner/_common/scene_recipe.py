"""场景配方助手 — 声明式 Actor 配方 + spawn + 资产竞态退避重试。

教学约定（FR-A07）：课程代码里的摆放用「位置(米) + 欧拉角(度)」表达，
四元数转换收口在本模块，避免每课重复处理 wxyz/xyzw 约定。

资产竞态背景（REQUIREMENTS.md §3.3）：引擎侧 spawnable 资产表为一次性
填充，云端订阅资产可能晚注册，导致 AddActor 首次失败——此处带退避
重试，耗尽后给出「重启 OrcaLab」的明确指引（FR-N06）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import orca_gym.utils.rotations as rotations
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import Actor, OrcaGymScene

_logger = get_orca_logger()

# 地板微抬偏移（米）：避免与 OrcaLab 自带地面重叠闪烁（z-fighting），
# 取值沿用 scene_building/indoor_room 的验证值
FLOOR_Z_OFFSET: float = 0.01


def setup_console_logging() -> None:
    """让 INFO 级日志进入终端（OrcaLog 默认 console_level=WARNING，仅写文件）。

    每课 main() 开头调用一次，保证新手能看见课程引导输出。
    """
    get_orca_logger().set_console_level("INFO")

# FR-N06：AddActor 失败退避重试参数
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_S = 2.0

# 清空场景后等待 Studio 响应的缓冲时间（秒）
_CLEAR_BUFFER_S = 0.5


@dataclass
class ActorSpec:
    """一个教学物体的声明式摆放规格。

    position 单位为米（世界坐标）；rotation_xyz_deg 单位为度，
    内部先 deg2rad 再经 euler2quat 转为 wxyz 四元数（与 MuJoCo
    约定一致），转换收口在 _to_actor。
    """

    name: str
    asset_path: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0


def _to_actor(spec: ActorSpec) -> Actor:
    """把教学规格转换为 OrcaGymScene 的 Actor（四元数转换收口处）。"""
    # euler2quat 约定输入为弧度（orca_gym.utils.rotations 模块级声明），
    # 教学配方用度表达，此处必须先 deg2rad——漏掉会让 45° 变成 45 rad
    quat = rotations.euler2quat(
        np.deg2rad(np.array(spec.rotation_xyz_deg, dtype=np.float64))
    )
    return Actor(
        name=spec.name,
        asset_path=spec.asset_path,
        position=np.array(spec.position, dtype=np.float64),
        rotation=quat,
        scale=float(spec.scale),
    )


def clear_scene(addr: str) -> None:
    """清空 Studio 场景（空 publish），用于「改配方→重建」的复位语义。

    前置：OrcaLab 已运行。
    后置：场景内既有实体被销毁（PublishScene 语义），等待缓冲后返回。
    """
    temp = OrcaGymScene(addr)
    try:
        temp.publish_scene()
    finally:
        temp.close()
    time.sleep(_CLEAR_BUFFER_S)


def spawn_recipe(addr: str, specs: list[ActorSpec]) -> OrcaGymScene:
    """按配方 spawn 全部物体并发布场景（三步范式封装）。

    前置：OrcaLab 已运行、资产包已订阅、specs 内 name 无重复。
    后置：返回存活的 OrcaGymScene（调用方负责 close）；
          任一 Actor 耗尽重试后 raise RuntimeError，不返回半成品。
    """
    scene = OrcaGymScene(addr)
    try:
        for i, spec in enumerate(specs):
            _add_actor_with_retry(scene, spec, attempt_label=f"[{i + 1}/{len(specs)}]")
        scene.publish_scene()
        _logger.info(f"场景发布完成：{len(specs)} 个物体")
    except Exception:
        # 失败时释放连接，避免半成品 scene 泄漏；close 自身的异常
        # 只记录不抛——不能让它遮蔽真正的失败原因
        try:
            scene.close()
        except Exception as close_exc:  # noqa: BLE001 — 清理路径，只记录
            _logger.warning(f"失败清理时关闭连接出错（已忽略）：{close_exc}")
        raise
    return scene


def _add_actor_with_retry(
    scene: OrcaGymScene, spec: ActorSpec, *, attempt_label: str = ""
) -> None:
    """单个 Actor 的添加，带 FR-N06 退避重试与中文指引。"""
    actor = _to_actor(spec)
    last_error: Exception | None = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            scene.add_actor(actor)
            return
        except Exception as exc:  # noqa: BLE001 — gRPC 错误类型跨版本不稳定，按消息分类
            last_error = exc
            _logger.warning(
                f"{attempt_label} 添加 {spec.name} 失败（第 {attempt}/{_RETRY_ATTEMPTS} 次）：{exc}"
            )
            if attempt < _RETRY_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_S * attempt)
    raise RuntimeError(
        f"物体「{spec.name}」添加失败（已重试 {_RETRY_ATTEMPTS} 次）。"
        f"常见原因：1) 资产包未订阅或订阅后未重启 OrcaLab；"
        f"2) asset_path 拼写错误：{spec.asset_path}。"
        f"请重启 OrcaLab 后重试。最后错误：{last_error}"
    ) from last_error
