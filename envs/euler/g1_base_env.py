"""G1BaseEnv — G1 人形机器人 Euler 环境基类（阶段四在线验证）。

为 Lesson 4–8 提供统一的运行框架（run_lesson）与钩子方法：
- 加载 Euler 专用 G1 模型（g1_29dof_camera.xml）
- 定义 G1 关节/执行器/传感器后缀常量（供 model_scanner 场景扫描用）
- run_lesson 框架：reset → before_loop → 循环(compute_ctrl/do_simulation/
  verify_step/observe_step/render) → after_loop → verify_final → report
- 子类通过重写钩子方法插入差异化验证逻辑

架构合规:
- 继承 OrcaGymEulerEnv，不触 _gym/_stub/_channel 等私有属性
- 状态访问通过 env.data / env.query_* 公共 API
- 仿真步进通过 env.do_simulation 公共方法
- 参见 docs/design/development/orca_gym_euler_phase4_online_validation_development.md §3.1

注意:
- 本基类不实现 Gymnasium 的 step/reset_model/_get_obs（由各 Lesson 子类按需实现），
  因为 Lesson 4–8 使用 run_lesson 框架而非 Gymnasium 训练循环。
- agent_name 在线模式通过场景扫描得到；离线模式（skip_grpc_load）使用传入值。
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv

from envs.common.model_scanner import (
    build_suffix_template,
    require_complete_matches,
    scan_scene_for_template,
)
from envs.euler.online_verifier import OnlineVerifier

# --- G1 资源路径 ---
_ROBOTS_DIR = os.path.join(os.path.dirname(__file__), "robots")
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
        - compute_ctrl(step) → np.ndarray: 控制输入（Lesson 4/5/6 零控，7/8 ONNX）
        - before_loop(verifier): 循环前准备（Lesson 7 begin_save_video）
        - verify_step(step, verifier): 每周期数值判定
        - observe_step(step, verifier): 阶段性人工观察提示
        - after_loop(verifier): 循环后收尾（Lesson 7 stop_save_video + mp4 检查）
        - verify_final(verifier): 结束最终判定

    使用契约:
        读取状态: env.data / env.query_* / env.get_body_xpos_xmat_xquat
        写入状态: env.set_joint_qpos / env.apply_body_force / env.set_mocap_pos_and_quat
        仿真步进: env.do_simulation(ctrl, n_frames)
    """

    metadata = {"render_modes": ["human", "none"], "version": "0.0.1", "render_fps": 30}

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
                a. compute_ctrl(step) → ctrl
                b. do_simulation(ctrl, frame_skip)
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
                ``frame_skip * time_step`` 真实秒）。默认 True，Lesson 4–8
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

        # 循环前钩子（Lesson 7 用于 begin_save_video）
        self.before_loop(verifier)

        # RTF=1.0 限速：每个控制周期目标墙钟时长 = frame_skip * time_step
        cycle_target = self.frame_skip * self._time_step
        loop_start = time.perf_counter() if real_time else 0.0

        for step in range(num_steps):
            ctrl = self.compute_ctrl(step)
            self.do_simulation(ctrl, self.frame_skip)
            self.verify_step(step, verifier)
            self.observe_step(step, verifier)
            self.render()

            # 墙钟对齐：若本周期提前完成，睡眠剩余时间以维持 RTF=1.0
            if real_time:
                deadline = loop_start + (step + 1) * cycle_target
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)

        # 循环后钩子（Lesson 7 用于 stop_save_video + mp4 检查）
        self.after_loop(verifier)

        # 最终判定
        self.verify_final(verifier)

        return verifier.report()

    # --- 钩子方法（子类重写）---

    def compute_ctrl(self, step: int) -> np.ndarray:
        """控制输入计算（子类重写）。

        - Lesson 4/5/6: 返回零控（验证状态查询/外力/雅可比，不需要行走）
        - Lesson 7/8: 返回 ONNX 策略输出（行走控制）

        Args:
            step: 当前控制周期索引（0-based）。

        Returns:
            控制输入数组，形状 (nu,)。
        """
        return np.zeros(self.model.nu, dtype=np.float64)

    def before_loop(self, verifier: OnlineVerifier) -> None:
        """循环前准备（子类重写，如 Lesson 7 begin_save_video）。"""
        return None

    def verify_step(self, step: int, verifier: OnlineVerifier) -> None:
        """每控制周期数值判定（子类重写）。"""
        return None

    def observe_step(self, step: int, verifier: OnlineVerifier) -> None:
        """阶段性人工观察提示（子类重写）。"""
        return None

    def after_loop(self, verifier: OnlineVerifier) -> None:
        """循环后收尾（子类重写，如 Lesson 7 stop_save_video + mp4 检查）。"""
        return None

    def verify_final(self, verifier: OnlineVerifier) -> None:
        """运行结束后最终判定（子类重写）。"""
        return None

    # --- Gymnasium 接口（run_lesson 框架通过 reset() 间接调用 reset_model/_get_obs，
    #     此处提供最小默认实现：G1 保持初始 keyframe 姿态，不做随机化。
    #     RL 子类（如需训练）可重写为随机化初始状态 + Box 观测。）---

    def step(self, action):
        """Gymnasium step（run_lesson 框架不调用，RL 子类按需重写）。"""
        raise NotImplementedError("G1BaseEnv 使用 run_lesson 框架，step 由 RL 子类按需实现")

    def reset_model(self) -> tuple[dict, dict]:
        """重置模型状态：G1 保持初始 keyframe 姿态（不随机化）。

        run_lesson 框架在 reset() 中调用本方法。Lesson 4–8 不需要随机化，
        G1 直接使用 XML keyframe 定义的站立姿态。RL 子类可重写为随机化初始状态。
        """
        # reset_simulation 已将 MjData 重置到初始 keyframe，此处仅需同步视图并返回观测
        self._sync_view()
        return self._get_obs(), {}

    def _get_obs(self) -> dict:
        """返回最小观测字典（run_lesson 框架不使用，仅为 Gymnasium API 兼容）。

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
