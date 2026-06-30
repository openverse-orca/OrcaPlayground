"""G1Locomotion — ONNX 行走策略封装（阶段四 Lesson 7/8 复用）。

封装 model_6600.onnx（deepmimic_dec_loco_height 策略）的推理流程：
1. 从 env.data / env.query_* 公共 API 读取 G1 状态（base 位姿、角速度、关节角度/速度）
2. 组装策略观测（匹配 decoupled_locomotion_stand_height 的观测布局）
3. ONNX 推理 → lower-body action (12,)
4. 后处理：scaled = action * 0.25 → concat(ref_upper) → + default_dof_angles → clip

内部状态：history handler（滚动缓冲）、phase time、速度指令、last_policy_action。

架构合规:
- 状态读取通过 env.data.qpos / env.data.qvel / env.query_joint_qpos / env.query_joint_qvel
- 不触 _gym/_stub/_mjModel/_mjData 等私有属性
- 参见 docs/design/development/orca_gym_euler_phase4_online_validation_development.md §3.4

注意:
- 观测布局参考 envs/g1/rl_policy/decoupled_locomotion_stand_height.py
- history 使用 history_loco_height_config + obs_loco_dims（actions=12 lower body），
  与 locomotion 策略的 last_policy_action (1,12) 形状一致
- 实际运行时若 ONNX 输入维度不匹配，需根据 model_6600.onnx 的实际输入规范微调
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import onnxruntime
import yaml

from envs.euler.g1_base_env import (
    G1_CONFIG_YAML,
    G1_LOCO_ONNX,
    G1_ROT_JOINT_SUFFIXES,
)
from envs.g1.utils.history_handler import HistoryHandler


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """将世界系向量 v 用四元数 q 逆旋转到体系（q: [w,x,y,z]）。"""
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


class G1Locomotion:
    """G1 行走策略封装：ONNX 推理 + 观测组装 + 动作后处理。

    使用方式:
        loco = G1Locomotion(agent_name="g1")
        loco.reset()
        ctrl = loco.compute_action(env)  # 返回 (29,) q_target
        env.do_simulation(ctrl, frame_skip)
    """

    # 下半身关节数（12 = 29 - 17）
    NUM_LOWER_DOFS = 12
    # 上半身关节数
    NUM_UPPER_DOFS = 17
    # 全身旋转关节数
    NUM_DOFS = 29
    # 策略动作缩放
    POLICY_ACTION_SCALE = 0.25
    # 默认站立高度
    DEFAULT_BASE_HEIGHT = 0.78

    def __init__(
        self,
        agent_name: str,
        onnx_path: str | None = None,
        config_path: str | None = None,
    ) -> None:
        """初始化行走策略封装。

        Args:
            agent_name: G1 实例的 agent_name 前缀（场景扫描结果）。
            onnx_path: ONNX 模型路径（默认 G1_LOCO_ONNX）。
            config_path: 配置 YAML 路径（默认 G1_CONFIG_YAML）。
        """
        if onnx_path is None:
            onnx_path = G1_LOCO_ONNX
        if config_path is None:
            config_path = G1_CONFIG_YAML

        self.agent_name = agent_name
        self.onnx_path = onnx_path

        # 加载配置
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 关节名（agent_name 前缀 + 后缀）
        self.joint_names = [
            f"{agent_name}_{suffix}" for suffix in G1_ROT_JOINT_SUFFIXES
        ]

        # 默认关节角度（29,）
        self.default_dof_angles = np.array(
            self.config["DEFAULT_DOF_ANGLES"], dtype=np.float64
        ).reshape(1, -1)

        # 上半身参考位姿（17,）— locomotion 策略中为 0
        self.ref_upper_dof_pos = np.zeros((1, self.NUM_UPPER_DOFS), dtype=np.float64)

        # 关节限位
        self.motor_pos_lower = np.array(
            self.config["motor_pos_lower_limit_list"], dtype=np.float64
        )
        self.motor_pos_upper = np.array(
            self.config["motor_pos_upper_limit_list"], dtype=np.float64
        )

        # 观测缩放
        self.obs_scales = self.config["obs_scales"]

        # 步态周期
        self.gait_period = self.config["GAIT_PERIOD"]

        # 初始化 ONNX 会话
        self._session = onnxruntime.InferenceSession(onnx_path)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # history handler（使用 history_loco_height_config + obs_loco_dims）
        history_cfg = self.config["history_loco_height_config"]
        obs_dims = self.config["obs_loco_dims"]
        self.history_handler = HistoryHandler(history_cfg, obs_dims)

        # 内部状态（reset 中初始化）
        self.last_policy_action: np.ndarray
        self.phase_time: float
        self.lin_vel_command: np.ndarray
        self.ang_vel_command: np.ndarray
        self.stand_command: np.ndarray
        self.base_height_command: np.ndarray
        self._start_time: float
        self.reset()

    def reset(self) -> None:
        """重置内部状态（history、phase、指令、last_action）。"""
        self.last_policy_action = np.zeros((1, self.NUM_LOWER_DOFS), dtype=np.float64)
        self.phase_time = 0.0
        # 指令：线速度 (2,)、角速度 (1,)、站立开关 (1,)、基准高度 (1,)
        self.lin_vel_command = np.array([[0.0, 0.0]], dtype=np.float64)
        self.ang_vel_command = np.array([[0.0]], dtype=np.float64)
        self.stand_command = np.array([[1]], dtype=np.float64)  # 1=站立/行走
        self.base_height_command = np.array(
            [[self.DEFAULT_BASE_HEIGHT]], dtype=np.float64
        )
        self._start_time = time.time()
        if self.history_handler is not None:
            self.history_handler.reset([0])

    def set_commands(
        self,
        lin_vel: tuple[float, float] | None = None,
        ang_vel: float | None = None,
        base_height: float | None = None,
    ) -> None:
        """设置速度/高度指令（行走时调用）。

        Args:
            lin_vel: (forward, lateral) 线速度指令，None 不变。
            ang_vel: 偏航角速度指令，None 不变。
            base_height: 基准高度指令，None 不变。
        """
        if lin_vel is not None:
            self.lin_vel_command[0] = lin_vel
        if ang_vel is not None:
            self.ang_vel_command[0, 0] = ang_vel
        if base_height is not None:
            self.base_height_command[0, 0] = base_height

    def compute_action(self, env: Any) -> np.ndarray:
        """从 env 读取状态，运行 ONNX 推理，返回 q_target (29,)。

        状态读取（公共 API）:
        - env.data.qpos[3:7]: free joint 基座四元数 [w,x,y,z]
        - env.data.qvel[3:6]: free joint 基座角速度
        - env.query_joint_qpos(joint_names): 29 关节角度
        - env.query_joint_qvel(joint_names): 29 关节速度

        Args:
            env: OrcaGymEulerEnv 实例（G1BaseEnv 子类）。

        Returns:
            q_target (29,): 目标关节角度（已 clip 到限位）。
        """
        # 1. 读取状态
        base_quat = env.data.qpos[3:7].reshape(1, -1)  # (1, 4) [w,x,y,z]
        base_ang_vel = env.data.qvel[3:6].reshape(1, -1)  # (1, 3)

        joint_qpos = env.query_joint_qpos(self.joint_names)
        joint_qvel = env.query_joint_qvel(self.joint_names)
        dof_pos = np.array(
            [joint_qpos[name] for name in self.joint_names], dtype=np.float64
        ).reshape(1, -1)  # (1, 29)
        dof_vel = np.array(
            [joint_qvel[name] for name in self.joint_names], dtype=np.float64
        ).reshape(1, -1)  # (1, 29)

        # 2. 组装观测
        obs = self._build_obs(base_quat, base_ang_vel, dof_pos, dof_vel)

        # 3. ONNX 推理
        policy_action = self._session.run(
            [self._output_name], {self._input_name: obs.astype(np.float32)}
        )[0]  # (1, 12) lower body
        policy_action = np.clip(policy_action, -100, 100)

        # 4. 更新内部状态
        self.last_policy_action = policy_action.copy()
        self._update_history(base_quat, base_ang_vel, dof_pos, dof_vel)

        # 5. 后处理：scaled → concat(ref_upper) → + default → clip
        scaled = policy_action * self.POLICY_ACTION_SCALE  # (1, 12)
        combined = np.concatenate(
            [scaled, self.ref_upper_dof_pos], axis=1
        )  # (1, 29)
        q_target = combined + self.default_dof_angles  # (1, 29)
        q_target = np.clip(
            q_target[0], self.motor_pos_lower, self.motor_pos_upper
        )  # (29,)

        return q_target

    def _get_phase_time(self) -> np.ndarray:
        """计算当前步态相位时间 [0, 1)。"""
        elapsed = time.time() - self._start_time
        phase = (elapsed % self.gait_period) / self.gait_period
        return np.array([[phase]], dtype=np.float64)

    def _build_obs(
        self,
        base_quat: np.ndarray,
        base_ang_vel: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        """组装策略观测向量（匹配 decoupled_locomotion_stand_height 布局）。

        观测拼接顺序（use_history=True）:
            [last_policy_action(12), base_ang_vel*0.25(3), ang_vel_command(1),
             base_height_command*2(1), lin_vel_command(2), stand_command(1),
             cos_phase(1), dof_pos_minus_default(29), dof_vel*0.05(29),
             history(400), projected_gravity(3), ref_upper_dof_pos(17),
             sin_phase(1)]
        """
        dof_pos_minus_default = dof_pos - self.default_dof_angles

        # projected gravity: 体系下的重力方向（z=-1 逆旋转到体系）
        v = np.array([[0.0, 0.0, -1.0]])
        projected_gravity = _quat_rotate_inverse(base_quat[0], v[0]).reshape(1, -1)

        # 相位
        phase_time = self._get_phase_time()
        sin_phase = np.sin(2 * np.pi * phase_time)
        cos_phase = np.cos(2 * np.pi * phase_time)

        # history
        history = self._get_obs_history()

        obs = np.concatenate(
            [
                self.last_policy_action,                      # (1, 12)
                base_ang_vel * 0.25,                          # (1, 3)
                self.ang_vel_command,                         # (1, 1)
                self.base_height_command * 2.0,               # (1, 1)
                self.lin_vel_command,                         # (1, 2)
                self.stand_command,                           # (1, 1)
                cos_phase,                                    # (1, 1)
                dof_pos_minus_default,                        # (1, 29)
                dof_vel * 0.05,                               # (1, 29)
                history,                                      # (1, 400)
                projected_gravity,                            # (1, 3)
                self.ref_upper_dof_pos,                       # (1, 17)
                sin_phase,                                    # (1, 1)
            ],
            axis=1,
        )
        return obs

    def _get_obs_history(self) -> np.ndarray:
        """查询 history handler，按 sorted key 顺序拼接。"""
        history_cfg = self.config["history_loco_height_config"]
        history_list = []
        for key in sorted(history_cfg.keys()):
            history_length = history_cfg[key]
            history_array = self.history_handler.query(key)[:, :history_length]
            history_array = history_array.reshape(history_array.shape[0], -1)
            history_list.append(history_array)
        return np.concatenate(history_list, axis=1)

    def _update_history(
        self,
        base_quat: np.ndarray,
        base_ang_vel: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> None:
        """向 history handler 添加当前帧的 scaled 观测。"""
        v = np.array([[0.0, 0.0, -1.0]])
        projected_gravity = _quat_rotate_inverse(base_quat[0], v[0]).reshape(1, -1)
        dof_pos_minus_default = dof_pos - self.default_dof_angles
        phase_time = self._get_phase_time()
        sin_phase = np.sin(2 * np.pi * phase_time)
        cos_phase = np.cos(2 * np.pi * phase_time)

        s = self.obs_scales
        self.history_handler.add("base_ang_vel", base_ang_vel * s["base_ang_vel"])
        self.history_handler.add(
            "command_lin_vel", self.lin_vel_command * s["command_lin_vel"]
        )
        self.history_handler.add(
            "command_ang_vel", self.ang_vel_command * s["command_ang_vel"]
        )
        self.history_handler.add(
            "command_stand", self.stand_command * s["command_stand"]
        )
        self.history_handler.add(
            "command_base_height",
            self.base_height_command * s["command_base_height"],
        )
        self.history_handler.add("dof_pos", dof_pos_minus_default * s["dof_pos"])
        self.history_handler.add("dof_vel", dof_vel * s["dof_vel"])
        self.history_handler.add(
            "projected_gravity", projected_gravity * s["projected_gravity"]
        )
        self.history_handler.add(
            "ref_upper_dof_pos", self.ref_upper_dof_pos * s["ref_upper_dof_pos"]
        )
        self.history_handler.add(
            "actions", self.last_policy_action * s["actions"]
        )
        self.history_handler.add("sin_phase", sin_phase * s["sin_phase"])
        self.history_handler.add("cos_phase", cos_phase * s["cos_phase"])
