"""G1BaseEnv — G1 人形机器人 Euler 环境基类（阶段四在线验证，公共库唯一版）。

为 Lesson 5–10 提供统一的运行框架（run_lesson）与钩子方法：
- 加载 Euler 专用 G1 模型（g1_29dof_camera.xml）
- 定义 G1 关节/执行器/传感器后缀常量（供 model_scanner 场景扫描用）
- run_lesson 框架：reset → before_loop → 循环(compute_ctrl/step/
  verify_step/observe_step/_draw_debug_viz/render) → after_loop → verify_final → report
- 子类通过重写钩子方法插入差异化验证逻辑

架构合规:
- 继承 OrcaGymEulerEnv，不触 _gym/_stub/_channel 等私有属性
- 状态访问通过 env.data / env.query_* 公共 API
- 仿真步进通过 env.step 公共方法（架构 §6.4 S5/S6：内部含 PD 闭环）
- 参见 docs/design/development/orca_gym_euler_phase4_online_validation_development.md §3.1

注意:
- 本基类实现 Gymnasium step 接口（内含 PD 闭环骨架，见架构 §6.4 S6），
  reset_model/_get_obs 提供最小默认实现。run_lesson 框架通过 step() 步进，
  保证与 Gymnasium 训练循环语义一致。RL 子类可复写 _pd_controller/
  _compute_reward/_is_terminated 等 hook 接入策略。
- agent_name 在线模式通过场景扫描得到；离线模式（skip_grpc_load）使用传入值。
- step() 按后端 + 控制需求分派三种模式（见 step() docstring 与
  orca_playground_euler_common_env_library_implementation_guide.md §3.2）：
  模式 1 host 批量（零控/直接力矩）、模式 2 host 逐子步闭环（CPU 行走回退）、
  模式 3 GPU-native PD（GPU 行走，由 _device_pd 接管）。
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import numpy as np
from common.online_verifier import OnlineVerifier
from common.scene_scanner import (
    build_suffix_template,
    require_complete_matches,
    scan_scene_for_template,
)
from common.perf_log import PerfTimer, print_rtf_report, rtf_log_enabled
from orca_gym.core.euler.sim_config import SimBackend
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

#: OrcaGym 单例日志器（info 级别）；供 step() 三模式分支在线诊断打印
_logger = get_orca_logger()

# --- G1 资源路径（指向 examples/euler/assets/g1/）---
# assets 在 examples/euler/assets/，__file__ 上溯 2 层到 examples/euler/
_EULER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSETS_DIR = os.path.join(_EULER_ROOT, "assets")
_ROBOTS_DIR = os.path.join(_ASSETS_DIR, "g1")
G1_MODEL_XML = os.path.join(_ROBOTS_DIR, "g1_29dof_camera.xml")
G1_CONFIG_YAML = os.path.join(_ROBOTS_DIR, "config", "g1_29dof_hist.yaml")
G1_LOCO_ONNX = os.path.join(_ROBOTS_DIR, "models", "dec_loco", "model_6600.onnx")

# --- 统一运行配置（§2.0.3） ---
# dt=0.005 + frame_skip=4：dt 与 Isaac Lab 训练 G1 locomotion 时一致
# （Isaac 训练物理步长即 0.005，策略在该 dt 下的动力学上收敛）；控制周期
# 0.005×4=20ms（50Hz，与 dec_loco 策略控制频率一致）。此前 dt=0.002+
# frame_skip=10（同样 20ms 控制周期）的 RTF 依据见技术报告
# Docs/Design/technical_report/single_world_gpu_rtf_analysis.md
# （OrcaEuler 仓）；dt 放大到 0.005 后每控制周期物理步从 10 减到 4，
# 单周期 GPU 计算量进一步下降，RTF 更宽裕。代价：device-PD 执行频率从
# 500Hz 降为 200Hz——若步态退化，可局部改回 dt=0.002/frame_skip=10。
G1_TIME_STEP = 0.005
G1_FRAME_SKIP = 4
G1_ORCAGYM_ADDR = "127.0.0.1:50051"

# --- G1 关节后缀（30 个：29 旋转 + 1 free base）---
# 与 run_g1_sim.py 的 G1_JOINT_SUFFIXES 一致，供 model_scanner 场景扫描用
G1_JOINT_SUFFIXES: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "floating_base_joint",
]

# --- G1 执行器后缀（29 个 motor）---
G1_ACTUATOR_SUFFIXES: list[str] = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

# --- G1 旋转关节后缀（29 个，不含 free base；用于 query_joint_qpos 等）---
G1_ROT_JOINT_SUFFIXES: list[str] = G1_JOINT_SUFFIXES[:-1]

# --- G1 传感器后缀（最小匹配集）---
G1_SENSOR_SUFFIXES: list[str] = ["imu_quat", "imu_gyro"]

# --- G1 关键 body 名称后缀 ---
G1_PELVIS_BODY = "pelvis"
G1_TORSO_BODY = "torso_link"
G1_IMU_SITE = "imu"
G1_CAMERA_HEAD_BODY = "camera_head"


def scoped_name(agent_name: str | None, suffix: str) -> str:
    """拼接 agent 前缀与后缀名，返回场景中的实际名称。

    在线模式场景名带命名空间前缀（如 ``g1_left_hip_pitch_joint``）；离线模式本地
    XML 无前缀（``left_hip_pitch_joint``），此时 ``agent_name`` 为空字符串，直接
    返回 ``suffix``。
    """
    return f"{agent_name}_{suffix}" if agent_name else suffix


def build_g1_template():
    """构建 G1 场景扫描后缀模板（供 model_scanner.scan_scene_for_template 使用）。"""
    return build_suffix_template(
        model_name="G1",
        joints=G1_JOINT_SUFFIXES,
        actuators=G1_ACTUATOR_SUFFIXES,
        sensors=G1_SENSOR_SUFFIXES,
    )


def resolve_g1_agent_name(orcagym_addr: str, time_step: float = G1_TIME_STEP) -> str:
    """扫描场景中的 G1 实例，返回 agent_name 前缀。

    在线模式专用：通过 gRPC 连接渲染端，扫描场景中匹配 G1 后缀模板的实例。
    要求场景中恰好存在 1 台完整匹配的 G1。

    Args:
        orcagym_addr: 渲染端 gRPC 地址。
        time_step: 物理时间步长（扫描探针 env 用）。

    Returns:
        agent_name 前缀（如 "g1" 或场景中实际命名）。

    Raises:
        ValueError: 场景中无完整匹配或数量不符。
    """
    template = build_g1_template()
    report = scan_scene_for_template(
        orcagym_addr=orcagym_addr,
        time_step=time_step,
        template=template,
    )
    match = require_complete_matches(
        report,
        min_count=1,
        max_count=1,
        allow_empty_prefix=False,
        orcagym_addr=orcagym_addr,
    )[0]
    return match.agent_name


def _format_array_stats(a: np.ndarray) -> str:
    """返回数组的 min/mean/max 与 NaN/Inf 计数，供 step() 三模式分支诊断日志用。"""
    a = np.asarray(a, dtype=np.float64)
    nan = int(np.isnan(a).sum())
    inf = int(np.isinf(a).sum())
    if a.size == 0:
        return "size=0"
    return f"min={a.min():.6g} mean={a.mean():.6g} max={a.max():.6g} nan={nan} inf={inf}"


def trace_step_enabled() -> bool:
    """是否启用 step() 逐子步详细日志（环境变量 ``ORCA_TRACE_STEP=1`` 时）。

    默认仅打印入口/出口一行；开启后额外打印三模式分支描述与逐子步 ctrl 统计。
    """
    return os.environ.get("ORCA_TRACE_STEP") in {"1", "true", "True", "yes"}


_TRACE_LOGGER_NAME = "orca_step_trace"
_trace_logger: logging.Logger | None = None


def _get_trace_logger() -> logging.Logger:
    """懒创建专用 trace 文件日志器（``propagate=False``，不串扰控制台/OrcaGym 主日志）。"""
    global _trace_logger
    if _trace_logger is None:
        path = os.environ.get("ORCA_TRACE_STEP_FILE", "orca_step_trace.log")
        _trace_logger = logging.getLogger(_TRACE_LOGGER_NAME)
        _trace_logger.setLevel(logging.DEBUG)
        _trace_logger.propagate = False
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
        _trace_logger.addHandler(handler)
    return _trace_logger


def trace_step_log(msg: str) -> None:
    """仅在 ``ORCA_TRACE_STEP=1`` 时把逐子步详情追加写入专用 trace 文件。

    关闭时零副作用（仅一次环境变量判定）；避免上万行刷屏控制台/污染主日志。
    """
    if trace_step_enabled():
        _get_trace_logger().info(msg)


class G1BaseEnv(OrcaGymEulerEnv):
    """G1 Euler 环境基类：提供 run_lesson 统一运行框架与钩子方法。

    子类通过重写以下钩子插入差异化逻辑：
        - compute_ctrl(step) → np.ndarray: 控制输入/动作（Lesson 5/6/7 零控，7/8/9 ONNX q_target）
        - _pd_controller(target) → np.ndarray: 单步 PD 控制（默认=target；Locomotion 复写为 tau）
        - _compute_reward(obs, action) → float: 奖励（默认=0.0，RL 子类复写）
        - _is_terminated(obs) → bool: 终止条件（默认=False，RL 子类复写）
        - before_loop(verifier): 循环前准备（Lesson 9 begin_save_video）
        - verify_step(step, verifier): 每周期数值判定
        - observe_step(step, verifier): 阶段性人工观察提示
        - _draw_debug_viz(step): 每周期 DebugMesh 可视化（render 前，默认空）
        - after_loop(verifier): 循环后收尾（Lesson 9 stop_save_video + mp4 检查）
        - verify_final(verifier): 结束最终判定

    使用契约:
        读取状态: env.data / env.query_* / env.get_body_xpos_xmat_xquat
        写入状态: env.set_joint_qpos / env.apply_body_force / env.set_mocap_pos_and_quat
        仿真步进: env.step(action)  # run_lesson 内部调用，含 PD 闭环（架构 §6.4 S6）
    """

    metadata = {"render_modes": ["human", "none"], "version": "0.0.1", "render_fps": 30}

    #: Gymnasium step 截断步数上限（RL 子类按需覆写；run_lesson 不依赖此值）
    MAX_EPISODE_STEPS: int = 10000

    #: host 逐子步闭环开关（step() 模式 2 触发器，架构 §3.2）。CPU 行走子类
    #: （LocomotionEnv / VideoCaptureEnv / BodyManipulationEnv）置 True，使其
    #: 每物理步重读 obs 重算 tau 保持稳定；05/06/07 零控场景保持 False（模式 1
    #: 批量）。GPU 行走时被模式 3（_device_pd 非空）优先接管。
    _per_substep_ctrl: bool = False

    #: device-side PD 开关（Phase D §8.2）。Locomotion 系子类（08/09/10）置 True，
    #: 使其在 GPU(Euler) 后端初始化时调用 register_pid_controller 构造 _device_pd，
    #: step() 走模式 3（GPU-native PD）。CPU 后端则跳过（回退模式 2），恒不构造。
    _requires_device_pd: bool = False

    def __init__(
        self,
        orcagym_addr: str = G1_ORCAGYM_ADDR,
        agent_names: list[str] | None = None,
        time_step: float = G1_TIME_STEP,
        frame_skip: int = G1_FRAME_SKIP,
        skip_grpc_load: bool = False,
        model_xml_path: str | None = None,
        **kwargs,
    ) -> None:
        """初始化 G1 Euler 环境基类。

        Args:
            orcagym_addr: 渲染端 gRPC 地址（在线模式用）。
            agent_names: agent 名称列表。在线模式若为 None，则在 initialize_simulation
                中通过场景扫描自动解析；离线模式本地 XML 无前缀，agent_name 置空。
            time_step: 物理时间步长（默认 0.005s，与 Isaac 训练 locomotion 一致）。
            frame_skip: 每个 control 周期的物理步数（默认 4，控制周期
                0.005×4=20ms 即控制频率 50Hz，与 dec_loco 策略训练频率一致）。
            skip_grpc_load: 是否跳过 gRPC 加载（离线模式 True）。
            model_xml_path: G1 模型 XML 路径（默认 Euler 专用 g1_29dof_camera.xml）。
            **kwargs: 透传 OrcaGymEulerEnv。
        """
        if model_xml_path is None:
            model_xml_path = G1_MODEL_XML
        if agent_names is None:
            agent_names = ["g1"]
        # agent_name 前缀默认值。离线模式本地 XML 无命名空间前缀，故置空字符串
        # （scoped_name 遇空前缀直接返回后缀名）；在线模式由 initialize_simulation
        # 通过场景扫描覆盖为实际前缀，如 "g1_29dof_old_usda"。
        # 必须在 super().__init__() 之前赋值，且此处不可在 super() 之后重新赋值，
        # 否则会覆盖 initialize_simulation 解析的结果。
        self.agent_name: str = "" if skip_grpc_load else agent_names[0]
        # GPU-native PD 控制器句柄占位：必须在 super().__init__() 之前置 None。
        # super().__init__() 内部会多态调用 self.initialize_simulation()，其末尾
        # _register_device_pd() 会写入本句柄；若放在 super() 之后赋值 None，会把
        # 已注册的句柄抹掉——而 solver 上的 pre-step PD kernel（q_target=0）仍
        # 挂载，导致每个物理步把所有关节往 0 拉（机器人僵硬摔倒）且无法更新目标。
        self._device_pd = None
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            model_xml_path=model_xml_path,
            skip_grpc_load=skip_grpc_load,
            **kwargs,
        )
        # Gymnasium step 循环计数（reset_model 中重置）
        self._step_count: int = 0

    # --- 场景扫描（在线模式自动解析 agent_name）---

    def initialize_simulation(self):
        """初始化仿真：在线模式先扫描场景解析 agent_name，再加载模型。

        覆写父类方法：在线模式（skip_grpc_load=False）在加载模型前先扫描场景，
        解析 G1 实例的 agent_name 前缀，更新 self._agent_names 与 self.agent_name。
        """
        if not self._skip_grpc_load:
            # 在线模式：扫描场景解析 agent_name
            resolved = resolve_g1_agent_name(self.orcagym_addr, self._time_step)
            self._agent_names = [resolved]
            self.agent_name = resolved
        return super().initialize_simulation()

    def _register_device_pd(self) -> None:
        """GPU(Euler) 后端下注册 device-side PD 控制器（Phase D §8.2）。

        仅当子类声明 ``_requires_device_pd=True``（Locomotion 系）且后端为 Euler 时
        调用真实注册，否则跳过（CPU 回退模式 2）。参数来自 ``self.locomotion``
        （G1Locomotion 的 kp/kd/motor_limits/joint_names），关节偏移由
        ``MuJoCoSimCoreEuler.register_pid_controller`` 内部解析，本层不感知
        ``flow.*`` / ``_pd_kernel`` / ``set_pre_step_kernel``。

        注意：必须在子类 ``initialize_simulation`` 末尾（G1Locomotion 创建之后）调用。
        """
        if not self._requires_device_pd:
            return
        if self.sim_config.backend != SimBackend.EULER:
            _logger.info(
                "register_pid_controller 跳过：后端非 EULER（CPU 回退模式 2）"
            )
            return
        loco = getattr(self, "locomotion", None)
        if loco is None:
            raise RuntimeError(
                "_requires_device_pd=True 但 self.locomotion 未创建；"
                "请在 initialize_simulation 中先构造 G1Locomotion 再调用本方法。"
            )
        _logger.info(
            "register_pid_controller: controller_type='pd' "
            f"nu={len(loco.joint_names)} agent={self.agent_name}"
        )
        self._device_pd = self.register_pid_controller(
            "pd",
            kp=loco.joint_kp,
            kd=loco.joint_kd,
            motor_limits=loco.motor_effort_limit,
            joint_names=loco.joint_names,
        )

    # --- run_lesson 统一运行框架（§3.1）---

    def run_lesson(
        self,
        num_steps: int,
        verifier: OnlineVerifier,
        real_time: bool = True,
    ) -> dict[str, Any]:
        """统一运行入口：子类通过重写钩子方法插入验证逻辑。

        流程:
            1. reset → 打印初始观察提示
            2. before_loop（循环前准备）
            3. 循环 num_steps 次:
                a. compute_ctrl(step) → action
                b. step(action)  # 架构 §6.4 S5/S6：内部含 PD 闭环
                c. verify_step(step, verifier)  # 数值判定
                d. observe_step(step, verifier)  # 人工观察提示
                e. _draw_debug_viz(step)  # DebugMesh 可视化（每周期，render 前）
                f. render()
                g. real_time 限速（RTF=1.0，按墙钟对齐 frame_skip*time_step）
            4. after_loop（循环后收尾）
            5. verify_final(verifier)  # 最终判定
            6. verifier.report()  # 输出报告

        Args:
            num_steps: 控制周期数（每个周期 frame_skip 个物理步）。
            verifier: 在线判定器实例。
            real_time: 是否按墙钟限速到 RTF=1.0（每个控制周期对齐
                ``frame_skip * time_step`` 真实秒）。默认 True，Lesson 5–10
                在线验证需 RTF=1.0 以便人工观察视口动作。

        Returns:
            判定报告字典。
        """
        self.reset()
        verifier.observe(
            "start",
            "请在渲染端视口观察 G1 初始姿态：应站立在地面上",
            step=0,
        )

        # 循环前钩子（Lesson 9 用于 begin_save_video）
        self.before_loop(verifier)

        # RTF=1.0 限速：每个控制周期目标墙钟时长 = frame_skip * time_step。
        # 采用每周期独立计时（而非绝对 deadline）：暂停只会拉长当前周期，
        # 恢复后后续周期仍按 RTF=1.0 对齐，避免暂停后全速追赶破坏实时性。
        cycle_target = self.frame_skip * self._time_step

        # RTF/性能诊断计时器（ORCA_RTF_LOG=1 时挂载，与 real_time 解耦：
        # --no-real-time 全速基准跑也应打印 step/verify/render 分段与真实 RTF）
        perf = PerfTimer() if rtf_log_enabled() else None
        if perf is not None:
            perf.start()

        for step in range(num_steps):
            cycle_start = time.perf_counter() if real_time else 0.0
            t0 = time.perf_counter() if perf is not None else 0.0
            action = self.compute_ctrl(step)
            # 走 Gymnasium step 接口（架构 §6.4 S5）：内部含 PD 闭环（§6.4 S6）
            # run_lesson 忽略 step 返回的五元组，verifier 直接读 env.data 判定
            self.step(action)
            t1 = time.perf_counter() if perf is not None else 0.0
            self.verify_step(step, verifier)
            self.observe_step(step, verifier)
            t2 = time.perf_counter() if perf is not None else 0.0
            self._draw_debug_viz(step)
            self.render()
            t3 = time.perf_counter() if perf is not None else 0.0

            if perf is not None:
                perf.add("step", t1 - t0)
                perf.add("verify", t2 - t1)
                perf.add("render", t3 - t2)

            # 墙钟对齐：若本周期提前完成，睡眠剩余时间以维持 RTF=1.0；
            # 超时计数与 real_time 解耦（--no-real-time 全速跑也按计算耗时统计）
            if real_time:
                cycle_elapsed = time.perf_counter() - cycle_start
            else:
                cycle_elapsed = t3 - t0
            remaining = cycle_target - cycle_elapsed
            if real_time and remaining > 0:
                time.sleep(remaining)
                if perf is not None:
                    perf.record_sleep(remaining)
            elif perf is not None and cycle_elapsed > cycle_target:
                perf.record_overrun()

        # 循环后钩子（Lesson 9 用于 stop_save_video + mp4 检查）
        self.after_loop(verifier)

        if perf is not None:
            print_rtf_report(num_steps, cycle_target, perf)

        # 最终判定
        self.verify_final(verifier)

        return verifier.report()

    # --- 人工观察暂停（子类在 verify_step/observe_step 中调用）---

    def wait_for_keypress(self, prompt: str, key: str = " ", timeout: float = 25.0) -> None:
        """暂停循环，等待用户在终端按键恢复（用于人工观察视口状态）。

        在 run_lesson 循环的 verify_step/observe_step 中调用，阻塞直到用户
        按下指定键（默认空格）或 ``timeout`` 秒后自动恢复。配合每周期独立计时的
        RTF 限速，暂停只会拉长当前周期，恢复后后续周期仍按 RTF=1.0 对齐，不会全速追赶。

        暂停期间以 ~30fps 轮询调用 render()，确保视口持续刷新到最新仿真状态
        （render 自身有 30fps 节流，物理仿真 1000Hz 远快于渲染，IK 修改 qpos 后
        需主动 render 才能提交到渲染端视口，否则画面停留在旧帧）。

        超时自动恢复：``timeout`` 秒内无按键则自动继续，防止 DebugMesh retained
        对象因 keepalive 过期而消失（keepalive=30s，timeout=25s 留 5s 余量）。

        Args:
            prompt: 显示给用户的提示文本。
            key: 触发恢复的按键（默认空格 ``" "``）；同时接受 Enter。
            timeout: 自动恢复超时（秒，默认 25）。<=0 表示永不超时。
        """
        print(f"  [PAUSE] {prompt}（按 Space 键继续）")
        sys.stdout.flush()
        # 非交互终端（stdin 非 tty，如管道/重定向/无输入）：input 回退或跳过
        if not sys.stdin.isatty():
            try:
                input()
            except EOFError:
                return
            return
        # 交互终端：非阻塞轮询按键，边等按键边 render 刷新视口
        render_interval = 1.0 / 30.0  # 30fps，与 render() 自身节流一致
        last_render = 0.0
        deadline = time.perf_counter() + timeout if timeout > 0 else None

        if sys.platform == "win32":
            # Windows: msvcrt 非阻塞按键检测
            import msvcrt

            while True:
                if msvcrt.kbhit():
                    ch = msvcrt.getch().decode(errors="ignore")
                    if ch in (key, "\r", "\n"):
                        break
                now = time.perf_counter()
                if now - last_render >= render_interval:
                    last_render = now
                    self.render()
                # 超时自动恢复（防止 retained 对象 keepalive 过期）
                if deadline is not None and now >= deadline:
                    print(f"\r\n  [TIMEOUT] {timeout:.0f}s 无按键，自动继续")
                    break
                time.sleep(0.01)  # 避免 CPU 空转
        else:
            # Unix: raw 模式 + select 非阻塞轮询
            import select
            import termios
            import tty

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                while True:
                    rlist, _, _ = select.select([fd], [], [], render_interval)
                    if rlist:
                        ch = os.read(fd, 1).decode(errors="ignore")
                        if ch in (key, "\r", "\n"):
                            break
                    now = time.perf_counter()
                    if now - last_render >= render_interval:
                        last_render = now
                        self.render()
                    # 超时自动恢复（防止 retained 对象 keepalive 过期）
                    if deadline is not None and now >= deadline:
                        print(f"\r\n  [TIMEOUT] {timeout:.0f}s 无按键，自动继续")
                        break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            sys.stdout.write("\r\n")
        sys.stdout.flush()

    # --- 钩子方法（子类重写）---

    def compute_ctrl(self, step: int) -> np.ndarray:
        """控制输入计算（子类重写）。

        - Lesson 5/6/7: 返回零控（验证状态查询/外力/雅可比，不需要行走）
        - Lesson 8/9/10: 返回 ONNX 策略输出（行走控制）

        Args:
            step: 当前控制周期索引（0-based）。

        Returns:
            控制输入数组，形状 (nu,)。
        """
        return np.zeros(self.model.nu, dtype=np.float64)

    def before_loop(self, verifier: OnlineVerifier) -> None:
        """循环前准备（子类重写，如 Lesson 9 begin_save_video）。"""
        return None

    def verify_step(self, step: int, verifier: OnlineVerifier) -> None:
        """每控制周期数值判定（子类重写）。"""
        return None

    def observe_step(self, step: int, verifier: OnlineVerifier) -> None:
        """阶段性人工观察提示（子类重写）。"""
        return None

    def _draw_debug_viz(self, step: int) -> None:
        """每控制周期 DebugMesh 可视化（子类重写，render 前调用）。

        默认空实现。Lesson 8 重写为调用 LocomotionDebugVisualizer.draw，
        绘制头顶指令箭头/接触球/足底力箭头。离线模式自动 no-op。
        """
        return None

    def after_loop(self, verifier: OnlineVerifier) -> None:
        """循环后收尾（子类重写，如 Lesson 9 stop_save_video + mp4 检查）。"""
        return None

    def verify_final(self, verifier: OnlineVerifier) -> None:
        """运行结束后最终判定（子类重写）。"""
        return None

    # --- Gymnasium 接口（架构 §6.4 S5/S6：step 是唯一对外步进入口，内含 PD 闭环；
    #     run_lesson 通过 step() 步进，保证与 Gymnasium 训练循环语义一致。
    #     reset_model/_get_obs 提供最小默认实现：G1 保持初始 keyframe 姿态。）---

    def step(self, action: np.ndarray) -> tuple:
        """Gymnasium 标准步进接口（含 PD 闭环骨架，架构 §6.4 S6）。

        按后端 + 控制需求分派三种模式（见
        orca_playground_euler_common_env_library_implementation_guide.md §3.2）：

        - 模式 1（默认）：host 批量。``self._pd_controller`` 一次算 ctrl，批量
          ``do_simulation(ctrl, frame_skip)``。适用于零控/直接力矩（Lesson 5/6/7），
          每周期仅 1 次 D2H。
        - 模式 2（``self._per_substep_ctrl=True``）：host 逐子步闭环。每物理步
          重读 obs 重算 ctrl，避免开环 20 步累积误差导致行走失稳（CPU 后端行走回退）。
        - 模式 3（``self._device_pd is not None``）：GPU-native PD。q_target 经
          H2D 下到 device，``do_simulation(ctrl, frame_skip)`` 内部由 device-side
          PD kernel 逐物理步覆盖 ctrl，graph 重放 `[pd_kernel + physics]×frame_skip`。

        Args:
            action: 控制输入。零控场景为 0；Locomotion 场景为 q_target（由
                compute_ctrl 生成，_pd_controller 转为 tau）。

        Returns:
            Gymnasium 五元组 (obs, reward, terminated, truncated, info)。
            run_lesson 忽略返回值，verifier 直接读 env.data 判定。
        """
        action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
        if self._device_pd is not None:
            mode_label = "3(GPU-PD)"
        elif self._per_substep_ctrl:
            mode_label = "2(host逐子步)"
        else:
            mode_label = "1(host批量)"
        entry_msg = (
            f"step() 入口: mode={mode_label} frame_skip={self.frame_skip} "
            f"time_step={self._time_step:.6g} nu={self.model.nu} "
            f"data.time={float(self.data.time):.6g} action[{_format_array_stats(action)}]"
        )
        _logger.info(entry_msg)
        trace_step_log(entry_msg)
        if self._device_pd is not None:
            # 模式 3：GPU-native PD（08/09/10 + 后端为 GPU）
            trace_step_log(
                f"step() 模式3(GPU-PD): update_target(1×H2D) + "
                f"do_simulation(ctrl=0, frame_skip={self.frame_skip}) graph重放"
            )
            self._device_pd.update_target(action)  # q_target_dev.assign (1 次 H2D)
            ctrl = np.zeros(self.model.nu, dtype=np.float32)
            self.do_simulation(ctrl, self.frame_skip)  # graph 重放 [pd_kernel + physics]×N
        elif self._per_substep_ctrl:
            # 模式 2：host 闭环 PD（CPU 行走回退，保持稳定）
            trace_step_log(
                f"step() 模式2(host逐子步): 循环 {self.frame_skip} 次 "
                f"(_pd_controller + do_simulation(ctrl,1))"
            )
            for i in range(self.frame_skip):
                ctrl = self._pd_controller(action)
                self.do_simulation(ctrl, 1)
                trace_step_log(
                    f"step() 模式2 substep[{i}/{self.frame_skip}]: "
                    f"ctrl[{_format_array_stats(ctrl)}] data.time={float(self.data.time):.6g}"
                )
        else:
            # 模式 1：host 批量（零控/直接力矩，05/06/07）
            trace_step_log(
                f"step() 模式1(host批量): _pd_controller + "
                f"do_simulation(ctrl, frame_skip={self.frame_skip})"
            )
            ctrl = self._pd_controller(action)
            trace_step_log(f"step() 模式1 ctrl[{_format_array_stats(ctrl)}]")
            self.do_simulation(ctrl, self.frame_skip)
        exit_msg = f"step() 出口: data.time={float(self.data.time):.6g}"
        _logger.info(exit_msg)
        trace_step_log(exit_msg)
        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        terminated = self._is_terminated(obs)
        self._step_count += 1
        truncated = self._step_count >= self.MAX_EPISODE_STEPS
        info: dict[str, float] = {"time": float(self.data.time)}
        return obs, reward, terminated, truncated, info

    def reset_model(self) -> tuple[dict, dict]:
        """重置模型状态：G1 保持初始 keyframe 姿态（不随机化）。

        run_lesson 框架在 reset() 中调用本方法。Lesson 5–10 不需要随机化，
        G1 直接使用 XML keyframe 定义的站立姿态。RL 子类可重写为随机化初始状态。
        """
        # reset_simulation 已将 MjData 重置到初始 keyframe，此处仅需同步视图并返回观测
        self._sync_view()
        self._step_count = 0
        return self._get_obs(), {}

    def _get_obs(self) -> dict:
        """返回最小观测字典（step 与 reset_model 共用）。

        包含 G1 基础状态：pelvis 位姿、关节 qpos/qvel。RL 子类可重写为 Box 观测。
        """
        agent = self.agent_name
        pelvis = self.get_body_xpos_xmat_xquat(
            [scoped_name(agent, "pelvis")]
        )[scoped_name(agent, "pelvis")]
        joint_names = [scoped_name(agent, s) for s in G1_ROT_JOINT_SUFFIXES]
        return {
            "pelvis_xpos": np.asarray(pelvis["xpos"], dtype=np.float64),
            "joint_qpos": self.query_joint_qpos(joint_names),
            "joint_qvel": self.query_joint_qvel(joint_names),
        }

    # --- step 的 PD 闭环 hook（子类按需复写）---

    def _pd_controller(self, target: np.ndarray) -> np.ndarray:
        """单步 PD 控制 hook（架构 §6.4 S6）。

        默认实现：直接返回 target 作为 ctrl（适用于零控/直接力矩场景）。
        Locomotion 场景子类复写为：重读 qpos/qvel → compute_tau(target, ...)。

        Args:
            target: 控制目标。零控场景为 0；Locomotion 场景为 q_target（位置）。

        Returns:
            ctrl 数组，形状 (nu,)，传入 do_simulation。
        """
        return target

    def _compute_reward(self, obs: dict, action: np.ndarray) -> float:
        """奖励 hook（默认 0.0，RL 子类复写）。"""
        return 0.0

    def _is_terminated(self, obs: dict) -> bool:
        """终止条件 hook（默认 False，RL 子类复写）。"""
        return False
