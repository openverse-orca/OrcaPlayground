# xbot_simple_env.py
"""
基于 standaloneMujoco.py 的简化 XBot 环境
直接移植standalone代码的控制逻辑到OrcaGym框架
"""

import numpy as np
import math
import mujoco
from collections import deque
from typing import Optional, Tuple, Dict
from gymnasium import spaces

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()



class XBotSimpleEnv(OrcaGymLocalEnv):
    """
    简化的XBot环境，直接移植standalone_mujoco_sim.py的逻辑
    """

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        frame_stack: int = 15,
        verbose: bool = True,  # 控制日志输出
        **kwargs
    ):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs
        )
        
        self.verbose = verbose  # 保存verbose标志

        # 模型信息
        self.nu = int(self.model.nu)
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        _logger.info(f"[XBotSimpleEnv] Model: nq={self.nq}, nv={self.nv}, nu={self.nu}")

        # 控制参数 - 恢复humanoid-gym的原始值
        # 因为我们会实现正确的decimation（10次内部循环）
        self.kps = np.array([200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15], dtype=np.float64)
        self.kds = np.array([10.0] * 12, dtype=np.float64)
        self.tau_limit = 200.0
        self.action_scale = 0.25
        
        # Decimation - 关键！匹配humanoid-gym
        self.decimation = 10  # 每次策略更新，内部循环10次物理步
        
        # 简单检查timestep
        if self.verbose:
            actual_timestep = self.gym._mjModel.opt.timestep
            _logger.performance(f"[XBotSimpleEnv] Physics timestep: {actual_timestep}s ({1.0/actual_timestep:.0f}Hz)")
        
        _logger.info(f"[XBotSimpleEnv] Using humanoid-gym PD gains: kp_max={np.max(self.kps)}, kd={self.kds[0]}, tau_limit={self.tau_limit}, action_scale={self.action_scale}, decimation={self.decimation}")

        # 观察历史
        self.frame_stack = frame_stack
        self.single_obs_dim = 47
        self.hist_obs = deque(maxlen=self.frame_stack)
        for _ in range(self.frame_stack):
            self.hist_obs.append(np.zeros(self.single_obs_dim, dtype=np.float32))

        # 命令 (初始不移动，先站稳)
        self.cmd_vx = 0.0  # 先站稳，不移动
        self.cmd_vy = 0.0
        self.cmd_dyaw = 0.0

        # 最后的动作
        self.last_action = np.zeros(12, dtype=np.float32)
        
        # 动作平滑 - 减缓脚步摆动频率
        # ⭐ 修改：关闭平滑，与standaloneMujoco一致
        self.use_action_filter = False  # 关闭动作平滑（standaloneMujoco无平滑）
        self.action_filter_alpha = 0.08  # 如果启用的话
        self.filtered_action = np.zeros(12, dtype=np.float32)

        # 控制数组
        self.ctrl = np.zeros(self.nu, dtype=np.float32)
        
        # Warmup阶段 - 渐进式启动避免摔倒
        # ⭐ 修改：关闭warmup，与standaloneMujoco一致
        self.step_count = 0
        self.warmup_steps = 500  # 如果启用的话
        self.use_warmup = False  # 关闭warmup（standaloneMujoco无warmup）
        
        # ⭐ 调试信息变量
        self.last_tau = np.zeros(12, dtype=np.float64)  # 最后的扭矩
        self.last_base_pos = np.zeros(3, dtype=np.float64)  # 最后的base位置
        
        # 基座名称（用于真实位置查询）
        # 从错误信息看，body名称格式是: XBot-L_usda_base_link
        # 尝试自动检测正确的base_link名称
        self.base_body_name = None
        try:
            all_bodies = self.model.get_body_names()
            # 尝试几种可能的命名模式
            candidates = [
                "XBot-L_usda_base_link",  # USDA导入格式
                f"{agent_names[0]}_base_link" if len(agent_names) > 0 else None,
                "base_link",
            ]
            for candidate in candidates:
                if candidate and candidate in all_bodies:
                    self.base_body_name = candidate
                    break
            
            if self.base_body_name is None:
                # 搜索包含"base"和"link"的body
                for body in all_bodies:
                    if "base" in body.lower() and "link" in body.lower():
                        self.base_body_name = body
                        break
        except:
            self.base_body_name = "base_link"  # 默认值
        
        _logger.info(f"[XBotSimpleEnv] Using base body name: {self.base_body_name}")
        
        # 上一次的基座位置（用于检测位移和估算角速度）
        self.last_base_pos = None
        self.last_base_euler = None
        self.last_base_quat = None

        # 定义动作和观察空间
        self.action_space = spaces.Box(
            low=-18.0, high=18.0, shape=(12,), dtype=np.float32
        )
        full_obs_dim = self.single_obs_dim * self.frame_stack
        self.observation_space = spaces.Box(
            low=-18.0, high=18.0, shape=(full_obs_dim,), dtype=np.float32
        )

        # 识别 XBot 的关节（匹配前缀 'XBot-L'，与 OrcaPlaygroundAssets 中 XBot-L_usda 导入后命名一致）
        all_joint_names = list(self.model.get_joint_dict().keys())
        all_actuator_names = list(self.model.get_actuator_dict().keys())
        xbot_prefix = "XBot-L"

        xbot_joint_base_names = [
            "left_leg_roll_joint", "left_leg_yaw_joint", "left_leg_pitch_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_leg_roll_joint", "right_leg_yaw_joint", "right_leg_pitch_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
        ]

        self.xbot_joint_names = []
        self.xbot_joint_indices = []
        self.xbot_qpos_indices = []
        self.xbot_qvel_indices = []
        self.xbot_actuator_indices = []

        for joint_base_name in xbot_joint_base_names:
            matching_joint = None
            for joint_name in all_joint_names:
                if joint_name.startswith(xbot_prefix) and joint_name.endswith("_" + joint_base_name):
                    matching_joint = joint_name
                    break

            if matching_joint:
                try:
                    joint_id = self.model.joint_name2id(matching_joint)
                    qpos_addr = self.gym.jnt_qposadr(matching_joint)
                    qvel_addr = self.gym.jnt_dofadr(matching_joint)

                    actuator_name = matching_joint
                    if actuator_name in all_actuator_names:
                        actuator_id = self.model.actuator_name2id(actuator_name)
                        self.xbot_joint_names.append(matching_joint)
                        self.xbot_joint_indices.append(joint_id)
                        self.xbot_qpos_indices.append(qpos_addr)
                        self.xbot_qvel_indices.append(qvel_addr)
                        self.xbot_actuator_indices.append(actuator_id)
                    else:
                        matching_actuator = None
                        for actuator_name_candidate in all_actuator_names:
                            if actuator_name_candidate.startswith(xbot_prefix) and actuator_name_candidate.endswith("_" + joint_base_name):
                                matching_actuator = actuator_name_candidate
                                break
                        if matching_actuator:
                            actuator_id = self.model.actuator_name2id(matching_actuator)
                            self.xbot_joint_names.append(matching_joint)
                            self.xbot_joint_indices.append(joint_id)
                            self.xbot_qpos_indices.append(qpos_addr)
                            self.xbot_qvel_indices.append(qvel_addr)
                            self.xbot_actuator_indices.append(actuator_id)
                except Exception as e:
                    _logger.error(f"[XBot] 匹配关节 '{joint_base_name}' 时出错: {e}")
            else:
                matching_actuator = None
                for actuator_name in all_actuator_names:
                    if actuator_name.startswith(xbot_prefix) and actuator_name.endswith("_" + joint_base_name):
                        matching_actuator = actuator_name
                        break

                if matching_actuator:
                    try:
                        joint_name = matching_actuator
                        joint_id = self.model.joint_name2id(joint_name)
                        actuator_id = self.model.actuator_name2id(matching_actuator)
                        qpos_addr = self.gym.jnt_qposadr(joint_name)
                        qvel_addr = self.gym.jnt_dofadr(joint_name)
                        self.xbot_joint_names.append(joint_name)
                        self.xbot_joint_indices.append(joint_id)
                        self.xbot_qpos_indices.append(qpos_addr)
                        self.xbot_qvel_indices.append(qvel_addr)
                        self.xbot_actuator_indices.append(actuator_id)
                    except Exception as e:
                        _logger.error(f"[XBot] 通过执行器匹配关节 '{joint_base_name}' 时出错: {e}")
        
        if len(self.xbot_qpos_indices) < 12:
            _logger.warning(f"[XBot] 只识别到 {len(self.xbot_qpos_indices)} 个关节，使用最后12个作为兜底")

            self.xbot_qpos_indices = list(range(max(0, self.nq - 12), self.nq))
            self.xbot_qvel_indices = list(range(max(0, self.nv - 12), self.nv))
            self.xbot_actuator_indices = list(range(max(0, self.nu - 12), self.nu))
            if len(all_joint_names) >= 12:
                self.xbot_joint_names = all_joint_names[-12:]
            else:
                self.xbot_joint_names = [f"joint_{i}" for i in range(12)]
        else:
            if self.verbose:
                _logger.info(f"[XBot] 成功识别到 {len(self.xbot_qpos_indices)} 个关节")

        _logger.info(f"[XBotSimpleEnv] Initialized with frame_stack={frame_stack}")
        _logger.info(f"[XBotSimpleEnv] Action space: {self.action_space.shape}")
        _logger.info(f"[XBotSimpleEnv] Observation space: {self.observation_space.shape}")

    def set_command(self, vx: float = 0.0, vy: float = 0.0, dyaw: float = 0.0):
        """设置运动命令"""
        self.cmd_vx = vx
        self.cmd_vy = vy
        self.cmd_dyaw = dyaw
        _logger.info(f"[Command] Set to vx={vx:.2f}, vy={vy:.2f}, dyaw={dyaw:.2f}")
    
    def set_smoothness(self, alpha: float = 0.15):
        """
        设置动作平滑度
        alpha: 0.0-1.0，越小越平滑
          - 0.1: 非常平滑，脚步摆动很慢
          - 0.3: 中等平滑
          - 1.0: 无平滑，完全响应
        """
        self.action_filter_alpha = np.clip(alpha, 0.0, 1.0)
        _logger.info(f"[Config] Action smoothness set to alpha={self.action_filter_alpha:.2f}")
    
    def enable_warmup(self, enabled: bool = True, steps: int = 300):
        """启用/禁用warmup阶段"""
        self.use_warmup = enabled
        self.warmup_steps = steps
        _logger.info(f"[Config] Warmup {'enabled' if enabled else 'disabled'} ({steps} steps)")

    def quaternion_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """
        四元数转欧拉角
        quat: 可能是 [x,y,z,w] 或 [w,x,y,z] 格式
        """
        # OrcaGym返回的四元数可能是[w,x,y,z]格式
        if len(quat) == 4:
            # 尝试判断格式：如果第一个元素接近±1，可能是w在前
            if abs(quat[0]) > 0.9 and abs(quat[1]) < 0.5 and abs(quat[2]) < 0.5 and abs(quat[3]) < 0.5:
                # 可能是 [w,x,y,z] 格式
                w, x, y, z = quat
            else:
                # 可能是 [x,y,z,w] 格式
                x, y, z, w = quat
        else:
            x, y, z, w = 0, 0, 0, 1
            
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _build_single_observation(self) -> np.ndarray:
        """
        构建单帧观察 (47维) - 完全按照 standalone 的逻辑
        """
        if hasattr(self, 'xbot_qpos_indices') and len(self.xbot_qpos_indices) == 12:
            q = self.data.qpos[self.xbot_qpos_indices].astype(np.float64)
            dq = self.data.qvel[self.xbot_qvel_indices].astype(np.float64)
        else:
            if self.step_count == 1:
                _logger.warning(f"[XBot] 使用兜底方案读取关节状态")
            q = self.data.qpos[-12:].astype(np.float64)
            dq = self.data.qvel[-12:].astype(np.float64)

        # 获取IMU传感器数据 - 尝试多种传感器名称格式
        sensor_read_success = False
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        omega = np.zeros(3, dtype=np.float64)
        
        # 尝试多种传感器名称格式
        sensor_name_candidates = [
            ("orientation", "angular-velocity"),  # 原始名称
            ("XBot_orientation", "XBot_angular-velocity"),  # 加agent前缀
            ("XBot-L_orientation", "XBot-L_angular-velocity"),  # 加模型名前缀
        ]
        
        for ori_name, gyro_name in sensor_name_candidates:
            try:
                sensor_dict = self.query_sensor_data([ori_name, gyro_name])
                quat_wxyz = np.array(sensor_dict[ori_name], dtype=np.float64)
                quat = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
                omega = np.array(sensor_dict[gyro_name], dtype=np.float64)
                sensor_read_success = True
                break
            except:
                continue
        
        # 如果所有尝试都失败，使用body查询获取姿态
        if not sensor_read_success and self.base_body_name:
            try:
                _, _, xquat = self.get_body_xpos_xmat_xquat([self.base_body_name])
                quat = xquat.copy()
                
                # 尝试估算角速度（从姿态变化）
                if self.last_base_quat is not None:
                    # 简单的差分估计: omega ≈ Δeuler / Δt
                    current_euler = self.quaternion_to_euler(quat)
                    if self.last_base_euler is not None:
                        delta_euler = current_euler - self.last_base_euler
                        dt = self.time_step * self.frame_skip  # 0.01s
                        omega = delta_euler / dt  # rad/s
                        # 限制范围避免异常值
                        # ⭐ 修改：放宽限制（standaloneMujoco无限制）
                        omega = np.clip(omega, -20.0, 20.0)
                
                sensor_read_success = True
            except:
                pass
        
        if not sensor_read_success and self.step_count == 1:
            _logger.warning(f"[XBot] 无法读取IMU传感器，使用默认值")

        # 转换欧拉角
        eu_ang = self.quaternion_to_euler(quat)

        # 构建观察向量 (完全按照standalone的顺序)
        obs = np.zeros(47, dtype=np.float32)
        obs[0] = math.sin(2 * math.pi * self.data.time / 0.64)  # 步态相位 sin
        obs[1] = math.cos(2 * math.pi * self.data.time / 0.64)  # 步态相位 cos
        obs[2] = self.cmd_vx * 2.0  # 线速度命令 x
        obs[3] = self.cmd_vy * 2.0  # 线速度命令 y
        obs[4] = self.cmd_dyaw * 1.0  # 角速度命令
        obs[5:17] = q * 1.0  # 关节位置 (12)
        obs[17:29] = dq * 0.05  # 关节速度 (12)
        obs[29:41] = self.last_action  # 历史动作 (12)
        obs[41:44] = omega  # 角速度 (3)
        obs[44:47] = eu_ang  # 欧拉角 (3)

        return np.clip(obs, -18.0, 18.0)

    def get_full_obs_vector(self) -> np.ndarray:
        """
        构建完整的观察向量 (47 * frame_stack)
        """
        single = self._build_single_observation()
        self.hist_obs.append(single)

        # 拼接历史观察
        stacked = np.zeros(self.single_obs_dim * self.frame_stack, dtype=np.float32)
        for i, h in enumerate(self.hist_obs):
            stacked[i * self.single_obs_dim:(i + 1) * self.single_obs_dim] = h

        return stacked

    def reset_model(self) -> Tuple[np.ndarray, Dict]:
        """
        重置环境
        """
        self.ctrl = np.zeros(self.nu, dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.filtered_action = np.zeros(12, dtype=np.float32)
        self.step_count = 0
        
        self.hist_obs.clear()
        for _ in range(self.frame_stack):
            self.hist_obs.append(np.zeros(self.single_obs_dim, dtype=np.float32))
        
        self.last_base_pos = None
        self.last_base_euler = None
        self.last_base_quat = None

        try:
            current_qpos = self.data.qpos.copy()
            CRAWL_HEIGHT = 0.05
            current_qpos[2] = CRAWL_HEIGHT
            import asyncio
            if hasattr(self, 'loop') and self.loop:
                self.loop.run_until_complete(self.gym.set_qpos(current_qpos))
                self.gym.mj_forward()
                self.gym.update_data()
        except Exception as e:
            if self.verbose:
                _logger.debug(f"[XBot] 设置初始高度失败: {e}")

        obs = self.get_full_obs_vector()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步 - 添加平滑和warmup来提高稳定性
        
        action: 策略输出 [-18, 18] 范围的12维动作
        """
        self.step_count += 1
        
        # 裁剪动作到有效范围
        action = np.clip(action, -18.0, 18.0).astype(np.float32)
        
        if self.use_warmup and self.step_count <= self.warmup_steps:
            warmup_progress = self.step_count / self.warmup_steps
            warmup_scale = 0.1 + 0.9 * warmup_progress
            action = action * warmup_scale
        
        if self.use_action_filter:
            # 指数移动平均: new = α*current + (1-α)*history
            self.filtered_action = (self.action_filter_alpha * action + 
                                   (1 - self.action_filter_alpha) * self.filtered_action)
            action_to_use = self.filtered_action
        else:
            action_to_use = action
        
        self.last_action = action_to_use.copy()

        target_q = action_to_use * self.action_scale
        target_dq = np.zeros(12, dtype=np.float64)

        for _ in range(self.decimation):
            if hasattr(self, 'xbot_qpos_indices') and len(self.xbot_qpos_indices) == 12:
                q = self.data.qpos[self.xbot_qpos_indices].astype(np.float64)
                dq = self.data.qvel[self.xbot_qvel_indices].astype(np.float64)
            else:
                if self.step_count == 1:
                    _logger.warning(f"[XBot] 使用兜底方案读取关节状态")
                q = self.data.qpos[-12:].astype(np.float64)
                dq = self.data.qvel[-12:].astype(np.float64)

            tau = (target_q - q) * self.kps + (target_dq - dq) * self.kds
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)

            if hasattr(self, 'xbot_actuator_indices') and len(self.xbot_actuator_indices) == 12:
                self.ctrl.fill(0.0)
                for i, actuator_idx in enumerate(self.xbot_actuator_indices):
                    if actuator_idx < len(self.ctrl):
                        self.ctrl[actuator_idx] = tau[i]
            else:
                if self.step_count == 1:
                    _logger.warning(f"[XBot] 使用兜底方案设置执行器控制")
                self.ctrl[-12:] = tau.astype(np.float32)

            self.do_simulation(self.ctrl, 1)
        
        self.last_tau = tau.copy()

        # 获取新的观察
        obs = self.get_full_obs_vector()

        real_base_z = 0.0
        real_euler = np.array([0.0, 0.0, 0.0])
        real_omega = np.array([0.0, 0.0, 0.0])
        position_valid = False
        xpos = None
        xquat = None
        
        # 保存上一次位置用于计算位移
        prev_base_pos = self.last_base_pos.copy() if self.last_base_pos is not None else None
        
        try:
            xpos, xmat, xquat = self.get_body_xpos_xmat_xquat([self.base_body_name])
            real_base_z = float(xpos[2])
            real_euler = self.quaternion_to_euler(xquat)
            position_valid = True
            try:
                sensor_dict = self.query_sensor_data(["angular-velocity"])
                real_omega = np.array(sensor_dict.get("angular-velocity", [0.0, 0.0, 0.0]))
            except:
                pass
        except Exception as e:
            if self.step_count == 1:
                _logger.warning(f"[XBot] 无法查询base body '{self.base_body_name}': {e}")
            try:
                real_base_z = float(self.data.qpos[2])
            except:
                real_base_z = 0.0
        
        is_fallen = False
        fall_reason = []
        roll_deg = 0.0
        pitch_deg = 0.0
        
        if position_valid:
            roll_deg = np.degrees(abs(real_euler[0]))
            pitch_deg = np.degrees(abs(real_euler[1]))
            
            if real_base_z < 0.02:
                is_fallen = True
                fall_reason.append(f"高度过低({real_base_z:.2f}m<0.02m)")
            
            if roll_deg > 30:
                is_fallen = True
                fall_reason.append(f"Roll过大({roll_deg:.1f}°>30°)")
            if pitch_deg > 30:
                is_fallen = True
                fall_reason.append(f"Pitch过大({pitch_deg:.1f}°>30°)")
        
        terminated = is_fallen
        truncated = False
        
        reward = 0.0
        if position_valid:
            height_error = abs(real_base_z - 0.9)
            height_reward = np.exp(-height_error * 10.0)
            reward += height_reward * 2.0
            
            orientation_error = (roll_deg + pitch_deg) / 180.0
            orientation_reward = np.exp(-orientation_error * 20.0)
            reward += orientation_reward * 3.0
            
            if not terminated:
                reward += 0.5
            
            if terminated:
                reward -= 5.0
        
        if position_valid:
            self.last_base_pos = xpos.copy()
            self.last_base_euler = real_euler.copy()
            self.last_base_quat = xquat.copy()
        
        info = {
            'base_z': real_base_z,
            'euler': real_euler,
            'is_fallen': is_fallen,
            'fall_reason': ' + '.join(fall_reason) if fall_reason else '',
            'reward': reward,
            'roll_deg': roll_deg,
            'pitch_deg': pitch_deg
        }
        
        if is_fallen and fall_reason and self.verbose:
            _logger.info(f"[XBot] 摔倒 Step={self.step_count} | {' + '.join(fall_reason)}")
        
        # 打印步态信息：前10步每步打印，之后每200步打印
        should_print_diagnostic = (self.step_count <= 10) or (self.step_count % 200 == 0)
        
        if should_print_diagnostic and position_valid and xpos is not None:
            try:
                displacement = ""
                if prev_base_pos is not None:
                    delta_pos = xpos - prev_base_pos
                    delta_interval = 1 if self.step_count <= 10 else 200
                    displacement = f"Δ{delta_interval}步=({delta_pos[0]:.3f},{delta_pos[1]:.3f},{delta_pos[2]:.3f})"
                
                roll_deg = np.degrees(real_euler[0])
                pitch_deg = np.degrees(real_euler[1])
                yaw_deg = np.degrees(real_euler[2])
                
                status = "✓"
                if abs(roll_deg) > 30 or abs(pitch_deg) > 30:
                    status = "⚠️"
                if abs(roll_deg) > 40 or abs(pitch_deg) > 40:
                    status = "❌"
                if real_base_z < 0.02:
                    status = "❌"
                
                print(f"[{status}] Step={self.step_count:5d} | "
                      f"Pos=({xpos[0]:6.3f},{xpos[1]:6.3f},{xpos[2]:6.3f})m | "
                      f"Roll={roll_deg:6.1f}° | Pitch={pitch_deg:6.1f}° | Yaw={yaw_deg:6.1f}° | "
                      f"Reward={reward:6.2f} | "
                      f"Action={np.linalg.norm(action):5.2f} | "
                      f"Filtered={np.linalg.norm(action_to_use):5.2f} | "
                      f"Tau_max={np.max(np.abs(tau)):6.1f} | "
                      f"{displacement}")
            except Exception as e:
                if self.verbose:
                    _logger.debug(f"[XBot] 打印步态信息失败: {e}")

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    # 简单测试
    _logger.info("XBotSimpleEnv module loaded successfully")

