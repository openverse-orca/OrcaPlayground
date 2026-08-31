"""LocomotionEnv — Lesson 8：G1 行走控制链路验证（ONNX 推理 + PD 控制器 + 稳定性）。

在阶段四在线模式下，连接 OrcaStudio 加载 G1 关卡，专注验证 G1 ONNX 行走控制链路
（不含视频采集，视频采集见 Lesson 9）。脚本通过 G1BaseEnv.run_lesson 框架步进 500 帧，
验证 G1 在 ONNX 策略 + PD 控制器驱动下能稳定站立/行走。

验证 API（行走控制层，公共 API）:
    - env.data.qpos / env.data.qvel（基座状态读取）
    - env.query_joint_qpos / env.query_joint_qvel（关节状态读取）
    - env.get_body_xpos_xmat_xquat（基座位姿读取，稳定性判定）
    - env.step（步进物理，架构 §6.4 S5/S6：内部含 PD 闭环）

验证点（5 项数值判定 + 2 项人工观察）:
    verify_step（每 50 步）:
    1. base_height_stable: 基座高度维持在合理范围（0.6-0.9m，DEFAULT_BASE_HEIGHT=0.78 附近）
    2. not_fallen: 基座俯仰/横滚角未超过阈值（< 0.8 rad，约 45°，判定未摔倒）
    3. joint_torque_within_limit: 关节力矩未持续触限（clip 比例 < 50%）
    before_loop / verify_step:
    4. standing_at_start: 前 50 步 G1 保持站立（基座高度 > 0.6m）
    verify_step:
    5. policy_action_finite: ONNX 输出无 NaN/Inf
    - g1_standing: Studio 视口 G1 应站立，不瘫倒
    - g1_walking_stable: Studio 视口 G1 行走应稳定，不乱踹（双腿交替迈步，非剧烈抖动）

行走控制链路:
    ONNX 策略输出位置目标 q_target (29,)
    → PD 控制器: tau = Kp*(q_target - q) + Kd*(0 - qd)
    → clip 到 motor_effort_limit
    → 传给 motor 执行器（G1 执行器是力矩控制）

参见 docs/design/development/orca_gym_euler_phase4_directory_restructure.md §3.6
"""

from __future__ import annotations

import numpy as np
from common.g1_base_env import G1BaseEnv, OnlineVerifier, scoped_name
from common.g1_locomotion import G1Locomotion
from locomotion_debug_viz import LocomotionDebugVisualizer

# 行走稳定性判定阈值
_BASE_HEIGHT_MIN = 0.6  # 站立最低高度（m）
_BASE_HEIGHT_MAX = 0.9  # 站立最高高度（m）
_FALL_ANGLE_THRESHOLD = 0.8  # 摔倒阈值（rad，约 45°）
_TORQUE_CLIP_RATIO_LIMIT = 0.5  # 力矩触限比例上限（50%）


class LocomotionEnv(G1BaseEnv):
    """Lesson 8 Env 子类：G1 行走控制链路验证 + DebugMesh 可视化。

    重写钩子:
        - initialize_simulation: 创建 G1Locomotion 实例（含 PD 控制器）+ LocomotionDebugVisualizer
        - compute_ctrl: 调用 G1Locomotion.compute_q_target 返回位置目标（29 维）
        - _pd_controller: 闭环 PD 单步（重读 obs 重算 tau，架构 §6.4 S6）
        - before_loop: 行走观察提示 + 初始化力矩触限计数
        - verify_step: 每 50 步检查基座高度/姿态/力矩触限/ONNX 输出有限性
        - _draw_debug_viz: 每周期绘制 DebugMesh 可视化（头顶指令箭头/接触球/足底力箭头）
        - observe_step: 行走稳定性观察提示
    """

    #: CPU 行走走 host 逐子步闭环（模式 2）；GPU 行走由 _device_pd（模式 3）接管。
    _per_substep_ctrl: bool = True

    #: GPU(Euler) 后端启用 device-side PD（Phase D §8.2）。
    _requires_device_pd: bool = True

    def initialize_simulation(self):
        """初始化仿真 + 创建 G1Locomotion 行走策略封装（含 PD 控制器）。"""
        super().initialize_simulation()
        self.locomotion = G1Locomotion(agent_name=self.agent_name)
        # GPU(Euler) 后端注册 device-side PD（Phase D §8.2）；CPU 后端内部跳过
        self._register_device_pd()
        # DebugMesh 可视化器（离线模式自动 no-op）
        self._viz = LocomotionDebugVisualizer(self.agent_name, self.locomotion)
        # 力矩触限统计（用于 joint_torque_within_limit 判定）
        self._torque_clip_count = 0
        self._torque_total_count = 0
        # ONNX 输出有限性统计
        self._policy_action_finite = True
        # q_target 缓存（compute_ctrl 写入，_pd_controller 读取）
        self._q_target: np.ndarray = np.zeros(self.model.nu, dtype=np.float64)

    def compute_ctrl(self, step: int) -> np.ndarray:
        """ONNX 策略推理 → 位置目标 q_target（29 维，rad）。

        本方法只做 ONNX 推理生成 q_target，不做 PD 转力矩。
        PD 转力矩在 _pd_controller 中每物理步闭环重算（与原版 g1_env.py
        每 mj_step 重读 obs 重算 PD 一致），避免开环 20 步累积误差导致失稳。

        q_target 存到 self._q_target 供 _pd_controller 使用。

        Returns:
            q_target (29,): 关节位置目标（rad）。
        """
        q_target = self.locomotion.compute_q_target(self)

        # 检查 ONNX 输出有限性
        if not np.all(np.isfinite(q_target)):
            self._policy_action_finite = False

        # 存储供 _pd_controller 闭环 PD 使用
        self._q_target = q_target

        return q_target

    def _pd_controller(self, target: np.ndarray) -> np.ndarray:
        """闭环 PD 单步 hook（架构 §6.4 S6）：重读 obs 重算 tau。

        原版 g1_env.step 在 frame_skip 循环内每步重读 qpos/qvel 重算 PD 力矩，
        否则单次 tau 跑 20 步会因开环累积误差导致 G1 失稳侧翻。
        本方法由父类 step() 在 frame_skip 循环内每物理步调用一次，
        返回 tau 后由父类 step() → do_simulation(tau, 1) 执行单步仿真。

        Args:
            target: q_target（29,）（由 compute_ctrl 返回），位置目标，非力矩。

        Returns:
            tau (29,): PD 力矩，供 do_simulation(tau, 1) 执行。
        """
        q_target = target  # compute_ctrl 返回的是 q_target，不是 tau
        dof_pos, dof_vel = self.locomotion.read_joint_state(self)
        tau = self.locomotion.compute_tau(q_target, dof_pos, dof_vel)
        # 统计触限比例（供 verify_step 判定）
        self._record_torque_stats(tau)
        return tau

    def _record_torque_stats(self, tau: np.ndarray) -> None:
        """统计力矩触限比例（供 verify_step 判定）。"""
        if not np.all(np.isfinite(tau)):
            self._policy_action_finite = False
        effort_limit = self.locomotion.motor_effort_limit
        clipped = np.sum(np.abs(tau) >= effort_limit - 0.1)
        self._torque_clip_count += int(clipped)
        self._torque_total_count += len(tau)

    def reset_model(self) -> tuple[dict, dict]:
        """重置到站立姿态：G1 关节 qpos = default_dof_angles（对齐 07 参考实现）。

        XML keyframe 姿态与策略的 DEFAULT_DOF_ANGLES 最大差 ~0.3 rad（膝盖），
        若直接从 keyframe 开始，PD 控制器第一步就会输出 ~60 Nm 的力矩把机器人
        蹬倒（参考 OrcaEuler 07_locomotion._standing_qpos：reset 后显式把
        关节 qpos 覆写为 DEFAULT_DOF_ANGLES，使初始 PD 误差为 0）。

        关节 qpos 起始地址通过 jnt_qposadr(首个关节名) 按名字解析——
        完整关卡场景中 G1 之前存在其他 free joint（Toys/ActorManipulator，
        各占 7 个 qpos），硬编码偏移 7 会把 default 角度写进玩具的 free joint，
        导致 G1 保持 keyframe 姿态、初始 PD 误差巨大而摔倒。
        """
        obs, info = super().reset_model()
        default = self.locomotion.default_dof_angles.reshape(-1)
        qpos = self.data.qpos.copy()
        first_joint = self.locomotion.joint_names[0]
        qpos_adr = self.jnt_qposadr(first_joint)
        qpos[qpos_adr:qpos_adr + len(default)] = default
        self.set_joint_qpos(qpos)
        self.set_joint_qvel(np.zeros_like(self.data.qvel))
        self._sync_view()
        return self._get_obs(), info

    def before_loop(self, verifier: OnlineVerifier) -> None:
        """循环前：行走观察提示 + 初始化统计。"""
        verifier.observe(
            "g1_start_walking",
            "Studio 视口：G1 即将在 ONNX 策略 + PD 控制器驱动下行走",
        )

    def verify_step(self, step: int, verifier: OnlineVerifier) -> None:
        """循环中：每 50 步检查行走稳定性。"""
        if step % 50 != 0:
            return

        # 读取基座状态（pelvis 位姿）
        agent = self.agent_name
        pelvis_name = scoped_name(agent, "pelvis")
        pelvis_data = self.get_body_xpos_xmat_xquat([pelvis_name])[pelvis_name]
        pelvis_xpos = pelvis_data["xpos"]  # (3,)
        pelvis_xmat = pelvis_data["xmat"]  # (3,3) 旋转矩阵

        # 1. 基座高度稳定（z 坐标）
        base_height = pelvis_xpos[2]
        verifier.check(
            f"base_height_stable_{step}",
            _BASE_HEIGHT_MIN <= base_height <= _BASE_HEIGHT_MAX,
            base_height,
            f"[{_BASE_HEIGHT_MIN}, {_BASE_HEIGHT_MAX}]",
            f"基座高度稳定（step={step}）",
        )

        # 2. 未摔倒（俯仰/横滚角 < 阈值）
        # 从旋转矩阵提取俯仰角（pitch）和横滚角（roll）
        # xmat 是行优先 3x3 旋转矩阵
        # pitch = arcsin(-xmat[2,0]); roll = arctan2(xmat[2,1], xmat[2,2])
        pitch = np.arcsin(np.clip(-pelvis_xmat[2, 0], -1.0, 1.0))
        roll = np.arctan2(pelvis_xmat[2, 1], pelvis_xmat[2, 2])
        max_tilt = max(abs(pitch), abs(roll))
        verifier.check(
            f"not_fallen_{step}",
            max_tilt < _FALL_ANGLE_THRESHOLD,
            max_tilt,
            f"<{_FALL_ANGLE_THRESHOLD}",
            f"未摔倒（pitch={pitch:.3f}, roll={roll:.3f}, step={step}）",
        )

        # 3. 力矩触限比例（仅在有统计后判定）
        if self._torque_total_count > 0:
            clip_ratio = self._torque_clip_count / self._torque_total_count
            verifier.check(
                f"joint_torque_within_limit_{step}",
                clip_ratio < _TORQUE_CLIP_RATIO_LIMIT,
                clip_ratio,
                f"<{_TORQUE_CLIP_RATIO_LIMIT}",
                f"力矩触限比例（step={step}）",
            )

        # 4. 前 50 步站立（基座高度 > 0.6m）
        if step <= 50:
            verifier.check(
                f"standing_at_start_{step}",
                base_height > _BASE_HEIGHT_MIN,
                base_height,
                f">{_BASE_HEIGHT_MIN}",
                f"起步站立（step={step}）",
            )

        # 5. ONNX 输出有限性（累计判定）
        verifier.check(
            f"policy_action_finite_{step}",
            self._policy_action_finite,
            self._policy_action_finite,
            True,
            f"ONNX 输出有限（step={step}）",
        )

    def _draw_debug_viz(self, step: int) -> None:
        """每控制周期绘制 DebugMesh 可视化（immediate 模式，render 前）。"""
        self._viz.draw(self)

    def close(self) -> None:
        """退出时主动销毁 DebugMesh retained 对象（箭头 + 接触球池）。"""
        if hasattr(self, "_viz"):
            self._viz.destroy(self)
        super().close()

    def observe_step(self, step: int, verifier: OnlineVerifier) -> None:
        """循环中：5 阶段动作演示 + 暂停解说。

        每阶段 200 步（4 秒），进入新阶段时暂停，解说当前指令及对应代码，
        用户按 Space 键继续后观察该阶段动作。

        阶段序列:
            step   0: 站立（stand=0）
            step 200: 前进（stand=1, lin_vel=(0.5,0)）
            step 400: 左转（ang_vel=0.5）
            step 600: 左移（lin_vel=(0,0.3), ang_vel=0）
            step 800: 停止（stand=0, lin_vel=0）
        """
        if step == 0:
            verifier.observe(
                "phase_1_stand",
                "阶段 1/5 站立：stand=0（相位固定不迈步），"
                "代码 self.locomotion.set_commands(stand=0)",
            )
            self.wait_for_keypress("观察 G1 原位站立（PD 维持默认姿态）")
        elif step == 200:
            self.locomotion.set_commands(stand=1, lin_vel=(0.5, 0.0), ang_vel=0.0)
            verifier.observe(
                "phase_2_forward",
                "阶段 2/5 前进：stand=1 + lin_vel=(0.5,0)（前进 0.5 m/s），"
                "代码 self.locomotion.set_commands(stand=1, lin_vel=(0.5,0.0), ang_vel=0.0)",
            )
            self.wait_for_keypress("观察 G1 直线前进（双腿交替迈步）")
        elif step == 400:
            self.locomotion.set_commands(stand=1, ang_vel=0.5)
            verifier.observe(
                "phase_3_turn",
                "阶段 3/5 左转：ang_vel=0.5（左转 0.5 rad/s），"
                "代码 self.locomotion.set_commands(stand=1, ang_vel=0.5)；"
                "ang_vel>0 左转、<0 右转",
            )
            self.wait_for_keypress("观察 G1 左转前进（保持 lin_vel=(0.5,0)）")
        elif step == 600:
            self.locomotion.set_commands(stand=1, lin_vel=(0.0, 0.3), ang_vel=0.0)
            verifier.observe(
                "phase_4_lateral",
                "阶段 4/5 左移：lin_vel=(0,0.3)（左移 0.3 m/s）+ ang_vel=0，"
                "代码 self.locomotion.set_commands(stand=1, lin_vel=(0.0,0.3), ang_vel=0.0)；"
                "lin_vel[1]>0 左移、<0 右移",
            )
            self.wait_for_keypress("观察 G1 侧向左移")
        elif step == 800:
            self.locomotion.set_commands(stand=0, lin_vel=(0.0, 0.0), ang_vel=0.0)
            verifier.observe(
                "phase_5_stop",
                "阶段 5/5 停止：stand=0 + lin_vel=(0,0) + ang_vel=0，"
                "代码 self.locomotion.set_commands(stand=0, lin_vel=(0.0,0.0), ang_vel=0.0)",
            )
            self.wait_for_keypress("观察 G1 减速回到站立")

    def after_loop(self, verifier: OnlineVerifier) -> None:
        """循环后：无视频采集操作（视频采集见 Lesson 9）。"""
        verifier.observe(
            "lesson7_done",
            "Lesson 8 行走控制验证结束，请在视口确认 G1 行走稳定性",
        )
