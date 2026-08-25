"""G1BaseEnv — G1 人形机器人 Euler 环境基类（阶段四在线验证）。

为 Lesson 5–9 提供统一的运行框架（run_lesson）与钩子方法：
- 加载 Euler 专用 G1 模型（g1_29dof_camera.xml）
- 定义 G1 关节/执行器/传感器后缀常量（供 model_scanner 场景扫描用）
- run_lesson 框架：reset → before_loop → 循环(compute_ctrl/step/
  verify_step/observe_step/render) → after_loop → verify_final → report
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
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np
from common.online_verifier import OnlineVerifier
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from common.scene_scanner import (
    build_suffix_template,
    require_complete_matches,
    scan_scene_for_template,
)

# --- G1 资源路径（指向 examples/euler/assets/g1/）---
# assets 在 examples/euler/assets/，__file__ 上溯 2 层到 examples/euler/
_EULER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSETS_DIR = os.path.join(_EULER_ROOT, "assets")
_ROBOTS_DIR = os.path.join(_ASSETS_DIR, "g1")
G1_MODEL_XML = os.path.join(_ROBOTS_DIR, "g1_29dof_camera.xml")
G1_CONFIG_YAML = os.path.join(_ROBOTS_DIR, "config", "g1_29dof_hist.yaml")
G1_LOCO_ONNX = os.path.join(_ROBOTS_DIR, "models", "dec_loco", "model_6600.onnx")

# --- 统一运行配置（§2.0.3） ---
G1_TIME_STEP = 0.001
G1_FRAME_SKIP = 20
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

    在线模式专用：通过 gRPC 连接 Studio，扫描场景中匹配 G1 后缀模板的实例。
    要求场景中恰好存在 1 台完整匹配的 G1。

    Args:
        orcagym_addr: OrcaStudio gRPC 地址。
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


class G1BaseEnv(OrcaGymEulerEnv):
    """G1 Euler 环境基类：提供 run_lesson 统一运行框架与钩子方法。

    子类通过重写以下钩子插入差异化逻辑：
        - compute_ctrl(step) → np.ndarray: 控制输入/动作（Lesson 5/6/7 零控，7/8/9 ONNX q_target）
        - _pd_controller(target) → np.ndarray: 单步 PD 控制（默认=target；Locomotion 复写为 tau）
        - _compute_reward(obs, action) → float: 奖励（默认=0.0，RL 子类复写）
        - _is_terminated(obs) → bool: 终止条件（默认=False，RL 子类复写）
        - before_loop(verifier): 循环前准备（Lesson 8 begin_save_video）
        - verify_step(step, verifier): 每周期数值判定
        - observe_step(step, verifier): 阶段性人工观察提示
        - after_loop(verifier): 循环后收尾（Lesson 8 stop_save_video + mp4 检查）
        - verify_final(verifier): 结束最终判定

    使用契约:
        读取状态: env.data / env.query_* / env.get_body_xpos_xmat_xquat
        写入状态: env.set_joint_qpos / env.apply_body_force / env.set_mocap_pos_and_quat
        仿真步进: env.step(action)  # run_lesson 内部调用，含 PD 闭环（架构 §6.4 S6）
    """

    metadata = {"render_modes": ["human", "none"], "version": "0.0.1", "render_fps": 30}

    #: Gymnasium step 截断步数上限（RL 子类按需覆写；run_lesson 不依赖此值）
    MAX_EPISODE_STEPS: int = 10000

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
            orcagym_addr: OrcaStudio gRPC 地址（在线模式用）。
            agent_names: agent 名称列表。在线模式若为 None，则在 initialize_simulation
                中通过场景扫描自动解析；离线模式用默认 ["g1"]。
            time_step: 物理时间步长（默认 0.001s）。
            frame_skip: 每个 control 周期的物理步数（默认 20，控制频率 50Hz）。
            skip_grpc_load: 是否跳过 gRPC 加载（离线模式 True）。
            model_xml_path: G1 模型 XML 路径（默认 Euler 专用 g1_29dof_camera.xml）。
            **kwargs: 透传 OrcaGymEulerEnv。
        """
        if model_xml_path is None:
            model_xml_path = G1_MODEL_XML
        if agent_names is None:
            agent_names = ["g1"]
        # agent_name 前缀默认值（离线模式使用；在线模式由 initialize_simulation
        # 通过场景扫描覆盖为实际前缀，如 "g1_29dof_old_usda"）。
        # 必须在 super().__init__() 之前赋值，且此处不可在 super() 之后重新赋值，
        # 否则会覆盖 initialize_simulation 解析的结果。
        self.agent_name: str = agent_names[0]
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
                e. render()
                f. real_time 限速（RTF=1.0，按墙钟对齐 frame_skip*time_step）
            4. after_loop（循环后收尾）
            5. verify_final(verifier)  # 最终判定
            6. verifier.report()  # 输出报告

        Args:
            num_steps: 控制周期数（每个周期 frame_skip 个物理步）。
            verifier: 在线判定器实例。
            real_time: 是否按墙钟限速到 RTF=1.0（每个控制周期对齐
                ``frame_skip * time_step`` 真实秒）。默认 True，Lesson 5–9
                在线验证需 RTF=1.0 以便人工观察视口动作。

        Returns:
            判定报告字典。
        """
        self.reset()
        verifier.observe(
            "start",
            "请在 Studio 视口观察 G1 初始姿态：应站立在地面上",
            step=0,
        )

        # 循环前钩子（Lesson 8 用于 begin_save_video）
        self.before_loop(verifier)

        # RTF=1.0 限速：每个控制周期目标墙钟时长 = frame_skip * time_step。
        # 采用每周期独立计时（而非绝对 deadline）：暂停只会拉长当前周期，
        # 恢复后后续周期仍按 RTF=1.0 对齐，避免暂停后全速追赶破坏实时性。
        cycle_target = self.frame_skip * self._time_step

        for step in range(num_steps):
            cycle_start = time.perf_counter() if real_time else 0.0
            action = self.compute_ctrl(step)
            # 走 Gymnasium step 接口（架构 §6.4 S5）：内部含 PD 闭环（§6.4 S6）
            # run_lesson 忽略 step 返回的五元组，verifier 直接读 env.data 判定
            self.step(action)
            self.verify_step(step, verifier)
            self.observe_step(step, verifier)
            self.render()

            # 墙钟对齐：若本周期提前完成，睡眠剩余时间以维持 RTF=1.0
            if real_time:
                remaining = cycle_target - (time.perf_counter() - cycle_start)
                if remaining > 0:
                    time.sleep(remaining)

        # 循环后钩子（Lesson 8 用于 stop_save_video + mp4 检查）
        self.after_loop(verifier)

        # 最终判定
        self.verify_final(verifier)

        return verifier.report()

    # --- 人工观察暂停（子类在 verify_step/observe_step 中调用）---

    def wait_for_keypress(self, prompt: str, key: str = " ") -> None:
        """暂停循环，等待用户在终端按键恢复（用于人工观察视口状态）。

        在 run_lesson 循环的 verify_step/observe_step 中调用，阻塞直到用户
        按下指定键（默认空格）。配合每周期独立计时的 RTF 限速，暂停只会拉长
        当前周期，恢复后后续周期仍按 RTF=1.0 对齐，不会全速追赶。

        暂停期间以 ~30fps 轮询调用 render()，确保视口持续刷新到最新仿真状态
        （render 自身有 30fps 节流，物理仿真 1000Hz 远快于渲染，IK 修改 qpos 后
        需主动 render 才能提交到 Studio 视口，否则画面停留在旧帧）。

        Args:
            prompt: 显示给用户的提示文本。
            key: 触发恢复的按键（默认空格 ``" "``）；同时接受 Enter。
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
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            sys.stdout.write("\r\n")
        sys.stdout.flush()

    # --- 钩子方法（子类重写）---

    def compute_ctrl(self, step: int) -> np.ndarray:
        """控制输入计算（子类重写）。

        - Lesson 5/6/7: 返回零控（验证状态查询/外力/雅可比，不需要行走）
        - Lesson 8/9: 返回 ONNX 策略输出（行走控制）

        Args:
            step: 当前控制周期索引（0-based）。

        Returns:
            控制输入数组，形状 (nu,)。
        """
        return np.zeros(self.model.nu, dtype=np.float64)

    def before_loop(self, verifier: OnlineVerifier) -> None:
        """循环前准备（子类重写，如 Lesson 8 begin_save_video）。"""
        return None

    def verify_step(self, step: int, verifier: OnlineVerifier) -> None:
        """每控制周期数值判定（子类重写）。"""
        return None

    def observe_step(self, step: int, verifier: OnlineVerifier) -> None:
        """阶段性人工观察提示（子类重写）。"""
        return None

    def after_loop(self, verifier: OnlineVerifier) -> None:
        """循环后收尾（子类重写，如 Lesson 8 stop_save_video + mp4 检查）。"""
        return None

    def verify_final(self, verifier: OnlineVerifier) -> None:
        """运行结束后最终判定（子类重写）。"""
        return None

    # --- Gymnasium 接口（架构 §6.4 S5/S6：step 是唯一对外步进入口，内含 PD 闭环；
    #     run_lesson 通过 step() 步进，保证与 Gymnasium 训练循环语义一致。
    #     reset_model/_get_obs 提供最小默认实现：G1 保持初始 keyframe 姿态。）---

    def step(self, action: np.ndarray) -> tuple:
        """Gymnasium 标准步进接口（含 PD 闭环骨架，架构 §6.4 S6）。

        以 frame_skip=1 多次调用 do_simulation 实现精细 PD 控制：
        每物理步通过 _pd_controller 重读 obs 重算 tau，避免开环累积误差导致失稳。

        Args:
            action: 控制输入。零控场景为 0；Locomotion 场景为 q_target（由
                compute_ctrl 生成，_pd_controller 转为 tau）。

        Returns:
            Gymnasium 五元组 (obs, reward, terminated, truncated, info)。
            run_lesson 忽略返回值，verifier 直接读 env.data 判定。
        """
        action = np.asarray(action, dtype=np.float32).reshape(self.model.nu)
        # PD 控制内循环：每物理步重读 obs 重算 ctrl（架构 §6.4 S6）
        # do_simulation(ctrl, 1) 内部：set_ctrl → mj_step(1) → sync_to_view
        for _ in range(self.frame_skip):
            ctrl = self._pd_controller(action)
            self.do_simulation(ctrl, 1)
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
        pelvis = self.get_body_xpos_xmat_xquat([f"{agent}_pelvis"])[f"{agent}_pelvis"]
        joint_names = [f"{agent}_{s}" for s in G1_ROT_JOINT_SUFFIXES]
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
