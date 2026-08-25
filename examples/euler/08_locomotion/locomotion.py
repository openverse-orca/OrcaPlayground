"""第 8 课：G1 行走控制链路 — ONNX 推理 + PD 控制器 + 稳定性验证

在阶段四在线模式下，连接 OrcaStudio 加载 G1 关卡，专注验证 G1 ONNX 行走控制链路
（不含视频采集，视频采集见 Lesson 9）。脚本通过 G1BaseEnv.run_lesson 框架步进 500 帧
（10 秒仿真），验证 G1 在 ONNX 策略 + PD 控制器驱动下能稳定站立/行走。

行走控制链路:
    ONNX 策略输出位置目标 q_target (29,)
    → PD 控制器: tau = Kp*(q_target - q) + Kd*(0 - qd)
    → clip 到 motor_effort_limit
    → 传给 motor 执行器（G1 执行器是力矩控制，ctrlrange 为 N·m）

验证 API（行走控制层，公共 API）:
    - env.data.qpos / env.data.qvel（基座状态读取）
    - env.query_joint_qpos / env.query_joint_qvel（关节状态读取）
    - env.get_body_xpos_xmat_xquat（基座位姿读取，稳定性判定）
    - env.do_simulation（步进物理）

用法:
    # 1. 先启动 OrcaStudio 并加载含 1 个 G1 的关卡，点击运行
    # 2. 运行脚本
    python examples/euler/08_locomotion/locomotion.py

    # Euler GPU 后端
    python examples/euler/08_locomotion/locomotion.py --device cuda:0

    # 指定 Studio 地址
    python examples/euler/08_locomotion/locomotion.py --addr 192.168.1.100:50051

验证点（5 项数值判定 + 2 项人工观察）:
    verify_step（每 50 步）:
    1. base_height_stable: 基座高度维持在合理范围（0.6-0.9m）
    2. not_fallen: 基座俯仰/横滚角未超过阈值（< 0.8 rad）
    3. joint_torque_within_limit: 关节力矩未持续触限（clip 比例 < 50%）
    4. standing_at_start: 前 50 步 G1 保持站立（基座高度 > 0.6m）
    5. policy_action_finite: ONNX 输出无 NaN/Inf
    - g1_standing / g1_walking_stable: Studio 视口观察 G1 行走稳定性（人工）

参见 docs/design/development/orca_gym_euler_phase4_directory_restructure.md §3.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_base_env import (
    G1_FRAME_SKIP,
    G1_MODEL_XML,
    G1_ORCAGYM_ADDR,
    G1_TIME_STEP,
)
from locomotion_env import LocomotionEnv
from common.online_verifier import OnlineVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lesson 8: G1 行走控制链路验证（ONNX + PD 控制器 + 稳定性）"
    )
    parser.add_argument(
        "--addr",
        default=G1_ORCAGYM_ADDR,
        help=f"OrcaStudio gRPC 地址（默认 {G1_ORCAGYM_ADDR}）",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help=f"控制周期数（默认 1000，每周期 {G1_FRAME_SKIP} 物理步，共 20 秒；"
        f"含 5 阶段动作演示：站立/前进/转弯/侧向/停止，每阶段 200 步）",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="后端：cpu=CPU MuJoCo（默认），cuda:0=Euler GPU",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = LocomotionEnv(
        frame_skip=G1_FRAME_SKIP,
        orcagym_addr=args.addr,
        agent_names=["g1"],  # 在线模式由场景扫描覆盖为实际 agent_name
        time_step=G1_TIME_STEP,
        model_xml_path=G1_MODEL_XML,
        device=args.device,
    )

    verifier = OnlineVerifier("Lesson 8: 行走控制")
    try:
        report = env.run_lesson(num_steps=args.num_steps, verifier=verifier)
    finally:
        env.close()

    if not report["summary"]["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
