"""12_slam 脚本 A：G1 策略服务 — 订阅 /cmd_vel 驱动 ONNX 行走，发布 /odom。

策略服务进程：Euler env（本地 MuJoCo 步进 + render 同步 OrcaStudio）+
G1Locomotion ONNX 行走策略。作为 ROS2 节点 "g1_policy_service"：
- 订阅 /cmd_vel (geometry_msgs/Twist)：SLAM 导航脚本（lidar_slam_nav.py）下发的
  速度指令（linear.x=前进 m/s、linear.y=左移 m/s、angular.z=左转 rad/s）
- 发布 /odom (nav_msgs/Odometry)：pelvis 位姿（x, y, yaw），作 SLAM 里程计

运行环境: ros2_bridge conda 环境（Python 3.10，与 Humble rclpy ABI 匹配；
Euler env / onnxruntime / mujoco 全栈已验证可用）。ros2_bootstrap 自动注入
/opt/ros/humble 路径，无需手动 source。

用法:
    # 1. 先启动 OrcaStudio 并加载含 1 个 G1（+ LiDAR 实体）的关卡，点击运行
    # 2. 运行策略服务（先于 SLAM 导航脚本启动）
    /home/orca/miniconda3/envs/ros2_bridge/bin/python \\
        examples/euler/12_slam/run_policy_service.py

    # 指定 Studio 地址 / 运行时长
    ... run_policy_service.py --addr 192.168.1.100:50051 --num-steps 5000

验证点（每 50 步数值判定）:
    1. base_height_stable: 基座高度 0.6-0.9m
    2. not_fallen: 俯仰/横滚角 < 0.8 rad

参见同目录 lidar_slam_nav.py（脚本 B：SLAM + 导航）。
"""

from __future__ import annotations

import argparse
import sys

import ros2_bootstrap  # noqa: F401  ROS2 环境注入（必须先于 rclpy 导入）


from g1_base_env import (
    G1_FRAME_SKIP,
    G1_MODEL_XML,
    G1_ORCAGYM_ADDR,
    G1_TIME_STEP,
)
from online_verifier import OnlineVerifier
from slam_policy_env import SlamPolicyEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="12_slam 脚本 A: G1 策略服务（订阅 /cmd_vel，发布 /odom）"
    )
    parser.add_argument(
        "--addr",
        default=G1_ORCAGYM_ADDR,
        help=f"OrcaStudio gRPC 地址（默认 {G1_ORCAGYM_ADDR}）",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100000,
        help=f"控制周期数（默认 100000，每周期 {G1_FRAME_SKIP} 物理步约 0.02s，"
        f"合计约 33 分钟；Ctrl+C 随时退出）",
    )
    parser.add_argument(
        "--goal-site",
        default="goal_goal_site",
        help="目标标记点 site 名（通过 query_site 查询其 MuJoCo 世界坐标，"
        "发布到 /goal 供 SLAM 导航脚本作终点）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = SlamPolicyEnv(
        frame_skip=G1_FRAME_SKIP,
        goal_site=args.goal_site,
        orcagym_addr=args.addr,
        agent_names=["g1"],  # 在线模式由场景扫描覆盖为实际 agent_name
        time_step=G1_TIME_STEP,
        model_xml_path=G1_MODEL_XML,
    )

    verifier = OnlineVerifier("12_slam: 策略服务")
    try:
        report = env.run_lesson(num_steps=args.num_steps, verifier=verifier)
    finally:
        env.close()

    if not report["summary"]["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
