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
    device: str = "cpu",
) -> OrcaGymEulerEnv:
    """连接 OrcaLab 当前场景并 reset，返回就绪的仿真环境。

    前置：OrcaLab 已运行且场景已就绪（spawn 过或用户拖拽过）。
    后置：env 已 reset（模型加载、初始状态就位）；调用方负责 close()。
    连接耗尽重试后 raise RuntimeError（指引见消息）。

    device（第 19 课引入）：
        "cpu"（默认）→ MuJoCo CPU 后端（1-18 课的路径）；
        "cuda:0" 等 → Euler GPU 后端（构造期切换，timestep/gravity
        随构造固化——运行中 setter 只读，见 OrcaGym SimConfig 契约）。
        GPU 构造失败（无 CUDA / 依赖缺失）不重试——重试无意义，
        直接抛 RuntimeError 并附排查指引。
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
                device=device,
            )
        except Exception as exc:  # noqa: BLE001 — 引擎错误类型跨版本不稳定，按消息重试
            last_error = exc
            msg = str(exc)
            transient = "not been initialized" in msg or "Try again later" in msg
            if device != "cpu" and not transient:
                # GPU 环境性失败（无 CUDA / Euler 依赖缺失 / sandbox 剥离
                # 进程能力）重试无意义——立即失败并给出排查路径；
                # 竞态类错误（publish 后引擎侧初始化未完成）与 CPU 同款重试
                raise RuntimeError(
                    f"GPU 后端环境构造失败（device={device}）：{exc}\n"
                    "排查：1) GPU 可用性（nvidia-smi）；2) 是否在 sandbox 内运行"
                    "（cuInit 报 CUDA_ERROR_304 时需白名单解释器直跑，见"
                    "DEVELOPER_GUIDE.md）；3) orca.euler / orca.flow 依赖已安装。"
                ) from exc
            _logger.warning(f"环境连接失败（第 {attempt}/{_CONNECT_ATTEMPTS} 次）：{exc}")
            if attempt < _CONNECT_ATTEMPTS:
                time.sleep(_CONNECT_BACKOFF_S * attempt)
            continue
        reset_env(env)
        _logger.info(
            f"仿真环境就绪（device={device}）：nq={env.model.nq}, nv={env.model.nv}, "
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
    joint_name, qadr, _ = resolve_joint(env, body_name, joint_type=0)
    return joint_name, qadr


def resolve_hinge_joint(
    env: OrcaGymEulerEnv, body_name: str
) -> tuple[str | None, int, int]:
    """解析目标 body 挂载的铰链关节（hinge），返回 (关节名, qpos 地址, dof 地址)。

    第 20 课引入：积木臂等关节体（ articulated Actor）的驱动入口。
    qpos 地址给 set_joint_qpos 用（弧度），dof 地址给 set_joint_qvel
    用（角速度 rad/s）。无铰链关节返回 (None, -1, -1)。
    Type=3 对应 mjJNT_HINGE（MuJoCo mjtJoint 枚举）。
    """
    return resolve_joint(env, body_name, joint_type=3)


def resolve_joint(
    env: OrcaGymEulerEnv, body_name: str, *, joint_type: int
) -> tuple[str | None, int, int]:
    """按 body 与关节类型解析关节，返回 (关节名, qpos 地址, dof 地址)。

    resolve_free_joint / resolve_hinge_joint 的公共实现。铰链关节
    1 qpos + 1 dof；自由关节 7 qpos + 6 dof（dof 地址可由 QposIdxStart
    推出，但 get_joint_dict 不直接给 dof 地址——自由关节 qpos 每关节
    占 7 槽而 dof 占 6 槽，无法从 qpos 地址推算，需按 dof 布局重算。
    对 freejoint 调用方只用 qpos 地址，dof 返回 -1 占位。
    """
    body_names = list(env.model.get_body_names())
    if body_name not in body_names:
        return None, -1, -1
    body_id = body_names.index(body_name)
    for joint_name, info in env.model.get_joint_dict().items():
        if info["BodyID"] == body_id and info["Type"] == joint_type:
            qadr = int(info["QposIdxStart"])
            if joint_type == 3:  # hinge：dof 地址 = 按 hinge 关节顺序累计
                dofadr = _hinge_dof_address(env, joint_name)
                return joint_name, qadr, dofadr
            return joint_name, qadr, -1
    return None, -1, -1


def free_joint_dof_address(env: OrcaGymEulerEnv, target_joint: str) -> int:
    """计算自由关节的 dof 地址（qvel 布局索引，前 3 槽为线速度）。

    第 23 课引入：给自由关节物体写初速度（set_joint_qvel）用。
    与铰链共用 qvel 布局规则：按关节出现顺序，freejoint 占 6 槽、
    hinge/slide 占 1 槽；get_joint_dict 只给 QposIdxStart，dof 地址
    需自行累计（见 mjtJoint 文档）。
    """
    return _dof_address(env, target_joint)


def _hinge_dof_address(env: OrcaGymEulerEnv, target_joint: str) -> int:
    """计算铰链关节的 dof 地址（qvel 布局索引）。

    qvel 布局按关节出现顺序排布：freejoint 占 6 槽、hinge/slide 占 1 槽。
    get_joint_dict 只给 QposIdxStart，dof 地址需按此规则自行累计——
    依据 MuJoCo 的 qpos/qvel 地址分配约定（见 mjtJoint 文档）。
    """
    return _dof_address(env, target_joint)


def _dof_address(env: OrcaGymEulerEnv, target_joint: str) -> int:
    """按 qvel 布局规则累计目标关节的 dof 地址（free/hinge 共用）。"""
    dof = 0
    for name, info in env.model.get_joint_dict().items():
        if name == target_joint:
            return dof
        dof += 6 if info["Type"] == 0 else 1
    return -1


def kick_body(env: OrcaGymEulerEnv, body_name: str, velocity: np.ndarray) -> None:
    """给自由关节物体写线速度初值（23/24 课共用）。

    set_joint_qvel 只收全量数组：复制当前 qvel，按自由关节 dof 地址
    （前 3 槽线速度）覆写目标分量后整块写回。相比 apply_body_force
    "踢一脚"，写状态可控——对照实验的起点条件才能完全相同。
    """
    joint_name, _ = resolve_free_joint(env, body_name)
    if joint_name is None:
        raise RuntimeError(f"{body_name} 没有自由关节，无法写初速度")
    dof = free_joint_dof_address(env, joint_name)
    qvel = np.asarray(env.data.qvel).copy()
    qvel[dof : dof + 3] = np.asarray(velocity, dtype=float)
    env.set_joint_qvel(qvel)


class LandingDetector:
    """单拍骤停式触地检测（场景无关，13/16/18/19 课共用）。

    判据：上一拍还在快速下落（单拍位移 < -fall_per_tick），这一拍
    不再快速下落（单拍位移 > -stop_per_tick，允许微小回弹），且
    累计下落超过 min_drop → 判定首次接触，当拍即触发。
    为什么不用高度阈值：不同尺寸物体静置高度不同（1m 方块 body 中心
    落地后约 0.51m，小方块约 0.05m），硬编码阈值会误判（FR-A08）。
    为什么不用连续停滞：接触后有慢速沉降（实测 0.5s 才完全静止），
    等停滞会晚报约 0.5s；单拍骤停在接触当拍就触发。
    为什么不用 body_cvel：cvel 的逐拍同步时机实测不稳定，做不了当拍判定；
    按名称读取数值做展示对照没问题（第 15 课的用法）。
    为什么"不再快速下落"而非"|位移|骤停"：地球重力末速约 7.7m/s 时
    软接触回弹猛烈，实测 dz 序列为 -0.043 → -0.0025 → +0.006——
    骤停窗口只有一拍且带残余位移，绝对值阈值会整窗错过（第 19 课
    实测踩坑）；"位移 > -阈值"既接得住骤停拍也接得住回弹拍。
    起点低速段（初速 0，前几拍 |dz| < 阈值）由 was_falling 前置
    条件挡住，不会误触发。
    阈值按 dt=0.01s 标定：fall/stop_per_tick=0.005 即 0.5m/s。
    """

    def __init__(
        self,
        z0: float,
        *,
        min_drop: float = 0.05,
        fall_per_tick: float = 0.005,
        stop_per_tick: float = 0.005,
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
            # 上一拍还在快速下落，这一拍不再快速下落（含微小回弹）→ 接触当拍触发
            was_falling = self._dz_prev is not None and self._dz_prev < -self._fall_per_tick
            stopped = dz > -self._stop_per_tick
            if was_falling and stopped and (self._z0 - z_now) > self._min_drop:
                self._latched = True
                return True
            self._dz_prev = dz
        self._z_prev = z_now
        return False
