"""仿真连接助手 — 层 3 课（13–18）共用的 EulerEnv 连接与驱动原语。

职责（高内聚）：
    1. connect_simulation_env：在线模式连接 OrcaLab 当前场景（XML 从
       Studio 拉取），带退避重试（publish 后引擎侧 MuJoCo 初始化存在
       竞态窗口，实测结论见 discovery.py 同款注释）；
    2. 零控制 buffer / 按名读高度 / 步进+渲染 最小原语。

设计依据（OrcaGym Euler 架构公共 API 契约）：
    - 步进走 do_simulation(ctrl, n_frames)，控制量必须整块传 buffer；
    - 循环内必须 render() 才能推送状态到 O3DE 视口（FR-N07）；
    - 全部经公共方法访问，不触碰 _mjModel / _mjData（SLF001）。
"""

from __future__ import annotations

import time

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()

# 13–18 课统一的基础参数；17 课专门把 time_step 拿出来做对比实验
TIME_STEP = 0.002  # 物理步长（秒）
FRAME_SKIP = 5  # 每次 do_simulation 推进的物理步数 → dt = 0.01 s

_CONNECT_ATTEMPTS = 5
_CONNECT_BACKOFF_S = 2.0


def connect_simulation_env(
    addr: str,
    *,
    time_step: float = TIME_STEP,
    frame_skip: int = FRAME_SKIP,
) -> OrcaGymEulerEnv:
    """连接 OrcaLab 当前场景并 reset，返回就绪的仿真环境。

    前置：OrcaLab 已运行且场景已就绪（spawn 过或用户拖拽过）。
    后置：env 已 reset（模型加载、初始状态就位）；调用方负责 close()。
    连接耗尽重试后 raise RuntimeError（指引见消息）。
    """
    last_error: Exception | None = None
    for attempt in range(1, _CONNECT_ATTEMPTS + 1):
        try:
            env = OrcaGymEulerEnv(
                frame_skip=frame_skip,
                orcagym_addr=addr,
                agent_names=["SceneProbe"],
                time_step=time_step,
                render_mode="human",
            )
        except Exception as exc:  # noqa: BLE001 — 引擎错误类型跨版本不稳定，按消息重试
            last_error = exc
            _logger.warning(f"环境连接失败（第 {attempt}/{_CONNECT_ATTEMPTS} 次）：{exc}")
            if attempt < _CONNECT_ATTEMPTS:
                time.sleep(_CONNECT_BACKOFF_S * attempt)
            continue
        reset_env(env)
        _logger.info(
            f"仿真环境就绪：nq={env.model.nq}, nv={env.model.nv}, "
            f"nu={env.model.nu}, dt={env.dt}s"
        )
        return env
    raise RuntimeError(
        f"环境连接失败（已重试 {_CONNECT_ATTEMPTS} 次）。"
        f"常见原因：1) 紧随场景发布后引擎侧 MuJoCo 尚未初始化完成，稍等数秒重跑；"
        f"2) OrcaLab 未运行或地址不可达：{addr}。最后错误：{last_error}"
    ) from last_error


def zero_ctrl(env: OrcaGymEulerEnv) -> np.ndarray:
    """零控制 buffer：教学场景通常不含执行器（nu=0），零向量即被动动力学。"""
    return np.zeros(env.model.nu)


def read_height(env: OrcaGymEulerEnv, body_name: str) -> float:
    """按名称读取 body 的世界坐标高度 z（copy 脱离 MuJoCo 视图）。"""
    return float(np.asarray(env.data.body_xpos(body_name)).copy()[2])


def read_linear_velocity(env: OrcaGymEulerEnv, body_name: str) -> np.ndarray:
    """按名称读取 body 线速度 (3,)（cvel 布局：[角速度(3), 线速度(3)]）。"""
    return np.asarray(env.data.body_cvel(body_name)).copy()[3:6]


def reset_env(env: OrcaGymEulerEnv) -> None:
    """复位整个场景到初始状态（qpos / qvel / 仿真时间）并推送渲染。

    用公共 API reset_simulation()，不走 Gymnasium 的 reset()——后者
    要求子类实现 reset_model（层 3 课无 obs/reward 语义，不需要）。
    mj_resetData 不重算运动学，补一次 mj_forward 否则复位后首帧
    读到的 body_xpos 是零值。
    """
    env.reset_simulation()
    env.mj_forward()
    env.render()


def advance(env: OrcaGymEulerEnv, ctrl: np.ndarray, n_frames: int = 1) -> None:
    """步进 n_frames 个物理步并推送渲染（全速，17 课墙钟对比实验专用）。"""
    env.do_simulation(ctrl, n_frames)
    env.render()


def pace(sim_elapsed: float, wall_start: float) -> None:
    """实时节拍：把墙钟对齐到已流逝的仿真时间（教学演示 1 秒仿真 ≈ 1 秒墙钟）。"""
    remaining = (wall_start + sim_elapsed) - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def advance_realtime(env: OrcaGymEulerEnv, ctrl: np.ndarray, n_frames: int = 1) -> None:
    """实时节拍步进：1 秒仿真 ≈ 1 秒墙钟，让视口看得见下落/绕圈过程。

    全速版是 advance()。纯 CPU 步进 1 秒仿真只需几毫秒，不加节拍
    的话演示"瞬间结束"，视口什么也看不到。
    """
    t0 = float(env.data.time)
    wall_start = time.perf_counter()
    env.do_simulation(ctrl, n_frames)
    env.render()
    pace(float(env.data.time) - t0, wall_start)


def resolve_free_joint(
    env: OrcaGymEulerEnv, body_name: str
) -> tuple[str | None, int]:
    """解析目标 body 挂载的自由关节（free joint），返回 (关节名, qpos 起始地址)。

    场景无关设计（FR-A08）：不假定 qpos 布局，经 model.get_joint_dict()
    的 BodyID / Type(0=mjJNT_FREE) / QposIdxStart 公共字段解析。
    无自由关节返回 (None, -1)——调用方据此给出指引（不假定任意 Actor
    都有自由关节）。
    """
    body_names = list(env.model.get_body_names())
    if body_name not in body_names:
        return None, -1
    body_id = body_names.index(body_name)
    for joint_name, info in env.model.get_joint_dict().items():
        if info["BodyID"] == body_id and info["Type"] == 0:  # mjJNT_FREE
            return joint_name, int(info["QposIdxStart"])
    return None, -1


class LandingDetector:
    """单拍骤停式触地检测（场景无关，13/16/18 课共用）。

    判据：上一拍还在快速下落（单拍位移 < -fall_per_tick），这一拍
    几乎停住（|单拍位移| < stop_per_tick），且累计下落超过 min_drop
    → 判定首次接触，当拍即触发。
    为什么不用高度阈值：不同尺寸物体静置高度不同（1m 方块 body 中心
    落地后约 0.51m，小方块约 0.05m），硬编码阈值会误判（FR-A08）。
    为什么不用连续停滞：接触后有慢速沉降（实测 0.5s 才完全静止），
    等停滞会晚报约 0.5s；单拍骤停在接触当拍就触发。
    为什么不用 body_cvel：cvel 的逐拍同步时机实测不稳定，做不了当拍判定；
    按名称读取数值做展示对照没问题（第 15 课的用法）。
    阈值按 dt=0.01s 标定：fall_per_tick=0.005 即 0.5m/s，
    自由落体末速约 3m/s（每拍 30mm），余量一个数量级。
    """

    def __init__(
        self,
        z0: float,
        *,
        min_drop: float = 0.05,
        fall_per_tick: float = 0.005,
        stop_per_tick: float = 0.002,
    ) -> None:
        self._z0 = z0
        self._min_drop = min_drop
        self._fall_per_tick = fall_per_tick
        self._stop_per_tick = stop_per_tick
        self._z_prev: float | None = None
        self._dz_prev: float | None = None
        self._latched = False

    def update(self, z_now: float) -> bool:
        """喂入当前高度，返回是否判定触地（首次触地后恒返回 False）。"""
        if self._z_prev is not None and not self._latched:
            dz = z_now - self._z_prev
            # 上一拍还在快速下落，这一拍骤停 → 接触当拍触发
            was_falling = self._dz_prev is not None and self._dz_prev < -self._fall_per_tick
            stopped = abs(dz) < self._stop_per_tick
            if was_falling and stopped and (self._z0 - z_now) > self._min_drop:
                self._latched = True
                return True
            self._dz_prev = dz
        self._z_prev = z_now
        return False
