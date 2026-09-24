"""SlamPolicyEnv — 12_slam 策略服务 Env：订阅 /cmd_vel 驱动 G1 行走，发布 /odom。

脚本 A（策略服务）的 Euler env 子类，基于 08_locomotion 的 LocomotionEnv
PD 闭环模式改造：

- 订阅 ROS2 topic /cmd_vel (geometry_msgs/Twist)，把速度指令映射到
  G1Locomotion.set_commands（linear.x=前进、linear.y=左移、angular.z=左转）
- 发布 ROS2 topic /odom (nav_msgs/Odometry)，内容为 pelvis 的 x/y/yaw
  （供 SLAM 脚本作里程计输入）
- ONNX 推理 + PD 闭环控制与 LocomotionEnv 完全一致

stand 约定（Twist 无 stand 字段）:
    - 非零速度指令 → stand=1（行走）
    - 连续 1s 全零指令 → stand=0（站立减速）
    - 超时 2s 未收到指令 → stand=0（安全停止）

架构合规:
- 状态读取通过 env.data / env.query_* / env.get_body_xpos_xmat_xquat 公共 API
- 不触 _gym/_stub/_mjModel/_mjData 等私有属性

用法（由 run_policy_service.py 入口调用，本模块不直接运行）:
    import ros2_bootstrap  # 必须先于 rclpy 导入
    env = SlamPolicyEnv(orcagym_addr="127.0.0.1:50051")
    env.run_lesson(num_steps=..., verifier=OnlineVerifier(...))
"""

from __future__ import annotations

import math
import time

import numpy as np

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from g1_base_env import G1BaseEnv, OnlineVerifier
from g1_locomotion import G1Locomotion

# 默认目标点 site 名（场景中的标记点，MuJoCo 世界坐标由 query_site 查询）
_DEFAULT_GOAL_SITE = "site1_site1"

# 行走稳定性判定阈值（沿用 locomotion_env.py）
_BASE_HEIGHT_MIN = 0.6  # 站立最低高度（m）
_BASE_HEIGHT_MAX = 0.9  # 站立最高高度（m）
_FALL_ANGLE_THRESHOLD = 0.8  # 摔倒阈值（rad，约 45°）

# 命令安全参数
_CMD_TIMEOUT_S = 2.0  # 命令超时（超过则安全停止）
_ZERO_TO_STAND_S = 1.0  # 全零指令持续多久后切站立

# 指令死区（低于此值视为零，避免 rclpy 浮点噪声触发行走）
_CMD_DEADBAND = 1e-3


class SlamPolicyEnv(G1BaseEnv):
    """12_slam 策略服务 Env：/cmd_vel 订阅 → ONNX 行走策略 → /odom、/goal 发布。

    重写钩子:
        - initialize_simulation: 创建 G1Locomotion + rclpy 节点（订阅/发布）
        - compute_ctrl: 处理 /cmd_vel 回调 → set_commands → ONNX 推理 q_target
        - _pd_controller: 闭环 PD 单步（重读 obs 重算 tau，架构 §6.4 S6）
        - verify_step: 每 50 步检查基座高度/姿态（稳定性）
        - close: rclpy 收尾
    """

    def __init__(self, *args, goal_site: str = _DEFAULT_GOAL_SITE, **kwargs):
        """goal_site: 场景中的目标标记点 site 名（MuJoCo 世界坐标通过
        query_site_pos_and_mat 查询，发布到 /goal 供 SLAM 导航脚本使用）。"""
        self.goal_site = goal_site
        super().__init__(*args, **kwargs)

    def initialize_simulation(self):
        """初始化仿真 + G1Locomotion 行走策略 + ROS2 节点。"""
        super().initialize_simulation()
        self.locomotion = G1Locomotion(agent_name=self.agent_name)

        # ROS2 节点：订阅 /cmd_vel，发布 /odom 与 /goal
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("g1_policy_service")
        self._node.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, 10)
        self._odom_pub = self._node.create_publisher(Odometry, "/odom", 10)
        # /goal 使用 TRANSIENT_LOCAL（latched）：SLAM 脚本晚启动也能收到
        _goal_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self._goal_pub = self._node.create_publisher(
            PointStamped, "/goal", _goal_qos
        )

        # 命令缓存（回调写入，compute_ctrl 读取，同线程 spin_once 无需锁）
        self._last_cmd: tuple[float, float, float] | None = None
        self._last_cmd_time: float = 0.0
        self._zero_cmd_since: float | None = None

        # 目标点尚未发布：site/body 世界位置在首次 forward()/step 前是
        # 未初始化数据（如 0,0,0），必须在至少一次步进后再查询发布
        self._goal_published = False

    # --- ROS2 回调（spin_once 中执行）---

    def _cmd_vel_cb(self, msg: Twist) -> None:
        """缓存最新 /cmd_vel 指令（linear.x, linear.y, angular.z）。"""
        self._last_cmd = (msg.linear.x, msg.linear.y, msg.angular.z)
        self._last_cmd_time = time.time()

    def _publish_goal(self) -> None:
        """查询目标标记点 site 的 MuJoCo 世界坐标，发布到 /goal。

        通过公共 API query_site_pos_and_mat 查询（不触 _mjModel/_mjData）。
        site 的 xpos 即其世界坐标（米），SLAM 脚本收到后结合首帧 /odom
        初始位姿转换到 SLAM 地图坐标。
        """
        try:
            site = self.query_site_pos_and_mat([self.goal_site])[self.goal_site]
        except Exception as e:
            print(
                f"[WARN] 查询目标 site '{self.goal_site}' 失败: {e}；"
                f"SLAM 脚本将无法自动获取 /goal，可用 lidar_slam_nav.py "
                f"--goal 手动指定（MuJoCo 世界坐标）"
            )
            return
        pos = np.asarray(site["xpos"], dtype=np.float64)

        msg = PointStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.point.x = float(pos[0])
        msg.point.y = float(pos[1])
        msg.point.z = float(pos[2])
        self._goal_pub.publish(msg)
        print(
            f"[INFO] 目标点 /goal 已发布: site '{self.goal_site}' "
            f"world=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m"
        )

    def _publish_odom(self) -> None:
        """发布 pelvis 位姿（x, y, yaw）到 /odom，作为 SLAM 里程计输入。"""
        agent = self.agent_name
        pelvis = self.get_body_xpos_xmat_xquat([f"{agent}_pelvis"])[
            f"{agent}_pelvis"
        ]
        x = float(pelvis["xpos"][0])
        y = float(pelvis["xpos"][1])
        # 行优先 3x3 旋转矩阵 → yaw
        xmat = pelvis["xmat"]
        yaw = math.atan2(xmat[1, 0], xmat[0, 0])

        msg = Odometry()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        # yaw 绕 z 轴旋转的四元数
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self._odom_pub.publish(msg)

    # --- run_lesson 钩子 ---

    def compute_ctrl(self, step: int) -> np.ndarray:
        """处理 /cmd_vel → set_commands → ONNX 推理 → q_target (29,)。"""
        # 1. 非阻塞处理 ROS2 回调（同线程，无锁）
        rclpy.spin_once(self._node, timeout_sec=0.0)

        # 2. 首次步进后查询目标 site 世界坐标并发布 /goal（site/body
        #    世界位置在首次 forward()/step 前是未初始化数据）
        if not self._goal_published and step >= 1:
            self._publish_goal()
            self._goal_published = True

        # 3. 命令映射 + stand 约定
        self._apply_cmd()

        # 4. 发布里程计
        self._publish_odom()

        # 5. ONNX 推理
        q_target = self.locomotion.compute_q_target(self)
        return q_target

    def _apply_cmd(self) -> None:
        """按 stand 约定把缓存的 /cmd_vel 映射到 locomotion 指令。"""
        now = time.time()
        if self._last_cmd is not None and now - self._last_cmd_time <= _CMD_TIMEOUT_S:
            vx, vy, w = self._last_cmd
            moving = (
                abs(vx) > _CMD_DEADBAND
                or abs(vy) > _CMD_DEADBAND
                or abs(w) > _CMD_DEADBAND
            )
            if moving:
                self._zero_cmd_since = None
                self.locomotion.set_commands(
                    stand=1, lin_vel=(vx, vy), ang_vel=w
                )
            else:
                # 全零指令：持续 1s 后切站立（给减速留时间）
                if self._zero_cmd_since is None:
                    self._zero_cmd_since = now
                if now - self._zero_cmd_since >= _ZERO_TO_STAND_S:
                    self.locomotion.set_commands(
                        stand=0, lin_vel=(0.0, 0.0), ang_vel=0.0
                    )
        else:
            # 未收到过命令或超时：安全站立
            self.locomotion.set_commands(stand=0, lin_vel=(0.0, 0.0), ang_vel=0.0)

    def _pd_controller(self, target: np.ndarray) -> np.ndarray:
        """闭环 PD 单步 hook（架构 §6.4 S6）：重读 obs 重算 tau。"""
        dof_pos, dof_vel = self.locomotion.read_joint_state(self)
        return self.locomotion.compute_tau(target, dof_pos, dof_vel)

    def before_loop(self, verifier: OnlineVerifier) -> None:
        """循环前：服务就绪提示。"""
        verifier.observe(
            "policy_service_ready",
            "策略服务已启动：订阅 /cmd_vel，发布 /odom 与 /goal；"
            "等待 SLAM 导航脚本（lidar_slam_nav.py）下发指令",
        )

    def verify_step(self, step: int, verifier: OnlineVerifier) -> None:
        """循环中：每 50 步检查行走稳定性（高度/姿态）。"""
        if step % 50 != 0:
            return

        agent = self.agent_name
        pelvis_data = self.get_body_xpos_xmat_xquat([f"{agent}_pelvis"])[
            f"{agent}_pelvis"
        ]
        pelvis_xpos = pelvis_data["xpos"]
        pelvis_xmat = pelvis_data["xmat"]

        base_height = pelvis_xpos[2]
        verifier.check(
            f"base_height_stable_{step}",
            _BASE_HEIGHT_MIN <= base_height <= _BASE_HEIGHT_MAX,
            base_height,
            f"[{_BASE_HEIGHT_MIN}, {_BASE_HEIGHT_MAX}]",
            f"基座高度稳定（step={step}）",
        )

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

    def close(self) -> None:
        """退出时销毁 ROS2 节点。"""
        if hasattr(self, "_node"):
            self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        super().close()
