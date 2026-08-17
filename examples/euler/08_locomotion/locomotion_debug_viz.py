"""LocomotionDebugVisualizer — Lesson 8 行走控制 DebugMesh 可视化封装（retained 模式）。

三类绘制（retained 模式，跨帧持久，避免 immediate 闪烁）：
1. 头顶速度指令箭头：线速度（红）+ 转向角速度（蓝）
2. 接触点不透明球体：MuJoCo contact 位置（绿）
3. 足底外力箭头：cfrc_ext 线性力（橙）

尺度设计（G1 身高 1.3m，以 1.0m 为尺度单位）：
- 速度箭头长度：0.5m ~ 2.0m 映射速度 0.01 ~ 2.0 m/s（超出截断）
- 角速度箭头长度：0.5m ~ 2.0m 映射角速度 0.01 ~ 2.0 rad/s（超出截断）
- 力箭头长度：0.25m ~ 1.0m 映射力 10 ~ 500 N（超出截断，FORCE_SCALE=0.5 缩半）
- 箭头杆半径 = clamp(length * 0.03, 0.02, 0.06)，与长度成比例
- 接触球半径：0.05m

设计要点：
- retained 模式：首次 draw() 创建对象，后续 draw() 用 update_transforms 更新
- update_transforms 自动刷新 keepalive 计时器，每周期调用即保活
- 接触球使用预分配对象池（MAX_CONTACTS），按当前接触数激活/隐藏
- 离线模式（debug_draw.is_online == False）所有操作 no-op
- 每周期 3 次 RPC（3 次 update_transforms：头顶箭头 + 接触球 + 足底力箭头）

依赖：orca_gym.utils.orca_debug_draw（顶层工具模块，不依赖 env 内部）
"""

from __future__ import annotations

from typing import Any

import numpy as np
from orca_gym.utils.orca_debug_draw import (
    DebugMeshType,
    InstanceFlags,
    _make_arrow_instance,
    _make_instance,
)

# 箭头对象池索引（4 个箭头：线速度/转向/左足力/右足力）
_IDX_CMD_LIN = 0
_IDX_CMD_ANG = 1
_IDX_FORCE_L = 2
_IDX_FORCE_R = 3
N_ARROWS = 4
MAX_CONTACTS = 30  # 接触球对象池大小


class LocomotionDebugVisualizer:
    """08_locomotion DebugMesh 可视化封装（retained 模式）。

    使用方式（在 Env.verify_step 末尾、render 之前调用）:
        self._viz = LocomotionDebugVisualizer(self.agent_name, self.locomotion)
        # 每控制周期：
        self._viz.draw(self)
        # 退出时清理（可选，进程退出时 FP 销毁自动释放）：
        self._viz.destroy(self)
    """

    def __init__(self, agent_name: str, locomotion: Any) -> None:
        """初始化可视化器。

        Args:
            agent_name: G1 实例的 agent_name 前缀（场景扫描结果）。
            locomotion: G1Locomotion 实例（读取 lin_vel_command/ang_vel_command）。
        """
        self.agent = agent_name
        self.loco = locomotion
        self._head = f"{agent_name}_camera_head"
        self._pelvis = f"{agent_name}_pelvis"
        self._feet = [
            f"{agent_name}_left_ankle_roll_link",
            f"{agent_name}_right_ankle_roll_link",
        ]

        # --- 尺度参数（G1 身高 1.3m，以 1.0m 为尺度单位）---
        # 速度/角速度/力 → 箭头长度映射（线性，超出截断）
        self.VEL_MIN = 0.01    # m/s，低于此值隐藏箭头
        self.VEL_MAX = 2.0     # m/s，超出截断
        self.ANG_MIN = 0.01    # rad/s
        self.ANG_MAX = 2.0     # rad/s
        self.FORCE_MIN = 10.0  # N
        self.FORCE_MAX = 500.0  # N
        self.LEN_MIN = 0.5     # m，最小箭头长度
        self.LEN_MAX = 2.0     # m，最大箭头长度
        self.FORCE_SCALE = 0.5  # 足底力箭头尺度系数（长度/半径相对速度箭头减半）
        # 箭头杆半径与长度成比例（绝对米值；Arrow 网格 tip 半径自动按 3:1 缩放）
        self.SHAFT_RATIO = 0.03
        self.SHAFT_MIN = 0.02  # m
        self.SHAFT_MAX = 0.06  # m
        # 其他参数
        self.HEAD_OFFSET_Z = 0.30   # m，头顶箭头锚点 Z 偏移
        self.CONTACT_RADIUS = 0.05  # m，接触球半径
        self.ARROW_COLOR_LIN = [1.0, 0.0, 0.0, 1.0]    # 红（线速度）
        self.ARROW_COLOR_ANG = [0.0, 0.0, 1.0, 1.0]    # 蓝（转向）
        self.ARROW_COLOR_FORCE = [1.0, 0.5, 0.0, 1.0]  # 橙（足底力）

        # --- retained 句柄（保留全部含 invalid，保持索引对齐）---
        self._arrow_handles: list[dict] = []
        self._sphere_handles: list[dict] = []
        self._initialized = False

    # ============= 辅助 =============

    def _mag_to_length(self, mag: float, lo: float, hi: float) -> float:
        """幅值 → 箭头长度（线性映射 lo..hi → LEN_MIN..LEN_MAX，超出截断）。"""
        if mag < lo:
            return 0.0
        t = (min(mag, hi) - lo) / (hi - lo)
        return self.LEN_MIN + t * (self.LEN_MAX - self.LEN_MIN)

    def _shaft_radius(self, length: float) -> float:
        """箭头杆半径 = clamp(length * SHAFT_RATIO, SHAFT_MIN, SHAFT_MAX)。"""
        return max(self.SHAFT_MIN, min(self.SHAFT_MAX, length * self.SHAFT_RATIO))

    @staticmethod
    def _hidden_instance():
        """隐藏实例（不活跃的对象池成员）。"""
        return _make_instance(
            [0.0, 0.0, -10000.0], [0.0, 0.0, 0.0, 1.0],
            [0.001, 0.001, 0.001], [0.0, 0.0, 0.0, 0.01],
            InstanceFlags.NONE,
        )

    # ============= 生命周期 =============

    def draw(self, env: Any) -> None:
        """每控制周期调用：更新三类要素。离线模式自动 no-op。

        首次调用创建 retained 对象，后续调用仅更新 transforms（避免闪烁）。
        """
        dd = env.debug_draw()
        if not dd.is_online:
            return
        if not self._initialized:
            self._create_objects(env, dd)
            self._initialized = True
        self._update_command_arrows(env, dd)
        self._update_contact_spheres(env, dd)
        self._update_foot_force_arrows(env, dd)

    def destroy(self, env: Any) -> None:
        """销毁所有 retained 对象（可选，进程退出时 FP 销毁自动释放）。"""
        dd = env.debug_draw()
        if not dd.is_online or not self._initialized:
            return
        all_handles = self._arrow_handles + self._sphere_handles
        valid = [h for h in all_handles if h.get("valid")]
        if valid:
            env.loop.run_until_complete(dd.destroy_objects(valid))
        self._arrow_handles.clear()
        self._sphere_handles.clear()
        self._initialized = False

    def _create_objects(self, env: Any, dd: Any) -> None:
        """首次调用：创建所有 retained 对象（4 箭头 + MAX_CONTACTS 球，初始隐藏）。

        keepalive=30s：每控制周期 update_transforms 自动刷新保活计时器，
        可容忍 ≤30s 暂停（等待用户输入等）。超 30s 未刷新则自动销毁，
        避免手动 kill Python 进程后的永久残留。
        """
        arrow_insts = [self._hidden_instance() for _ in range(N_ARROWS)]
        hs = env.loop.run_until_complete(
            dd.create_objects(DebugMeshType.ARROW, arrow_insts, keepalive=30.0))
        self._arrow_handles = list(hs)  # 保留全部，保持索引对齐
        sphere_insts = [self._hidden_instance() for _ in range(MAX_CONTACTS)]
        hs = env.loop.run_until_complete(
            dd.create_objects(DebugMeshType.SPHERE, sphere_insts, keepalive=30.0))
        self._sphere_handles = list(hs)

    # ============= 要素 1：头顶速度指令箭头 =============

    def _update_command_arrows(self, env: Any, dd: Any) -> None:
        """更新线速度（红）+ 转向角速度（蓝）箭头。

        坐标系约定（MuJoCo Z-up 右手系，G1 体系）：
          - X=forward（前）, Y=left（左）, Z=up（上）
          - 线速度箭头（红）：合并 (forward, lateral) 为世界系矢量，
            沿 pelvis 前向（xmat 第 0 列 = X 轴）+ 左向（xmat 第 1 列 = Y 轴）合成，
            长度由 |lin_vel| 合成幅值映射，指向合成速度方向
          - 转向箭头（蓝）：沿 pelvis 左向（xmat 第 1 列 = Y 轴），长度由 |ang_vel| 映射
            （>0 左转指向左，<0 右转指向右）
        速度为 0 时隐藏对应箭头。
        """
        lin_vel_fwd = float(self.loco.lin_vel_command[0, 0])
        lin_vel_lat = float(self.loco.lin_vel_command[0, 1])
        ang_vel = float(self.loco.ang_vel_command[0, 0])

        body_data = env.get_body_xpos_xmat_xquat([self._head, self._pelvis])
        head_pos = body_data[self._head]["xpos"]
        pelvis_xmat = body_data[self._pelvis]["xmat"]  # (3,3) 行优先
        anchor = head_pos + np.array([0.0, 0.0, self.HEAD_OFFSET_Z])

        instances = []
        # 线速度箭头（红）：合并 forward(X) + lateral(Y) 为世界系矢量
        lin_mag = float(np.hypot(lin_vel_fwd, lin_vel_lat))
        lin_len = self._mag_to_length(lin_mag, self.VEL_MIN, self.VEL_MAX)
        if lin_len > 0:
            vel_world = pelvis_xmat[:, 0] * lin_vel_fwd + pelvis_xmat[:, 1] * lin_vel_lat
            direction = vel_world / lin_mag
            end = anchor + direction * lin_len
            instances.append(_make_arrow_instance(
                anchor.tolist(), end.tolist(), self._shaft_radius(lin_len),
                self.ARROW_COLOR_LIN, InstanceFlags.EDGE_HIGHLIGHT))
        else:
            instances.append(self._hidden_instance())

        # 转向箭头（>0 左转指向左，<0 右转指向右，长度代表角速度大小）
        ang_len = self._mag_to_length(ang_vel, self.ANG_MIN, self.ANG_MAX)
        if ang_len > 0:
            turn_dir = pelvis_xmat[:, 1] * np.sign(ang_vel)  # Y轴=左，>0 左转指左
            end = anchor + turn_dir * ang_len
            instances.append(_make_arrow_instance(
                anchor.tolist(), end.tolist(), self._shaft_radius(ang_len),
                self.ARROW_COLOR_ANG, InstanceFlags.EDGE_HIGHLIGHT))
        else:
            instances.append(self._hidden_instance())

        env.loop.run_until_complete(
            dd.update_transforms(self._arrow_handles[_IDX_CMD_LIN:_IDX_CMD_ANG + 1], instances))

    # ============= 要素 2：接触点不透明球体 =============

    def _update_contact_spheres(self, env: Any, dd: Any) -> None:
        """更新接触点球体（对象池，按当前接触数激活/隐藏）。

        接触点统一绿色不透明。超出池容量的接触不显示。
        """
        contacts = env.query_contact_simple()
        n = min(len(contacts), len(self._sphere_handles))
        contact_color = [0.0, 1.0, 0.0, 1.0]  # 纯绿
        instances = []
        for i in range(len(self._sphere_handles)):
            if i < n:
                c = contacts[i]
                pos = np.asarray(c["pos"], dtype=np.float32)
                instances.append(_make_instance(
                    pos.tolist(), [0.0, 0.0, 0.0, 1.0],
                    [self.CONTACT_RADIUS] * 3, contact_color, InstanceFlags.NONE))
            else:
                instances.append(self._hidden_instance())
        env.loop.run_until_complete(
            dd.update_transforms(self._sphere_handles, instances))

    # ============= 要素 3：足底外力箭头 =============

    def _update_foot_force_arrows(self, env: Any, dd: Any) -> None:
        """更新足底外力箭头（左/右足各一个）。

        力在世界系，箭头从 body xpos 出发，指向力方向。
        力 < FORCE_MIN 隐藏对应箭头（抬腿相无受力）。

        注意：MuJoCo cfrc_ext 的 6 分量布局为 [torque(3), force(3)]，
        线性力在后 3 个分量 (cfrc[3:6] = fx, fy, fz)。
        """
        cfrc = env.get_cfrc_ext()  # (nbody, 6), 布局 [torque(3), force(3)]
        foot_data = env.get_body_xpos_xmat_xquat(self._feet)

        instances = []
        for name in self._feet:
            bid = env.model.body_name2id(name)
            force = cfrc[bid, 3:]  # 世界系线性力 (fx, fy, fz)
            mag = float(np.linalg.norm(force))
            f_len = self._mag_to_length(mag, self.FORCE_MIN, self.FORCE_MAX) * self.FORCE_SCALE
            if f_len > 0:
                origin = np.asarray(foot_data[name]["xpos"], dtype=np.float64)
                direction = force / mag
                end = origin + direction * f_len
                instances.append(_make_arrow_instance(
                    origin.tolist(), end.tolist(), self._shaft_radius(f_len),
                    self.ARROW_COLOR_FORCE, InstanceFlags.EDGE_HIGHLIGHT))
            else:
                instances.append(self._hidden_instance())
        env.loop.run_until_complete(
            dd.update_transforms(self._arrow_handles[_IDX_FORCE_L:_IDX_FORCE_R + 1], instances))
