"""第 5 课：状态查询 API — G1 全套查询 API 在线验证

在阶段四在线模式下，连接 OrcaStudio 加载 G1 关卡，验证 G1 全套状态查询 API
在真实仿真中运行正确。脚本通过 G1BaseEnv.run_lesson 框架自动步进 100 帧，
在 step 0 集中执行 9 项查询验证，并通过 OnlineVerifier 输出判定报告。

验证 API:
    - query_joint_qpos / query_joint_qvel / query_joint_qacc（关节状态）
    - get_body_xpos_xmat_xquat（Body 位姿）
    - query_site_pos_and_mat（Site 查询）
    - query_sensor_data（传感器查询）
    - query_actuator_torques（执行器力矩）
    - query_contact_simple（接触查询）
    - body_subtree_mass（质量查询）
    - query_position_body_B（基座坐标系变换）

用法:
    # 1. 先启动 OrcaStudio 并加载含 1 个 G1 的关卡，点击运行
    # 2. 运行脚本
    python examples/euler/05_query_api/query_api.py

    # Euler GPU 后端
    python examples/euler/05_query_api/query_api.py --device cuda:0

    # 指定 Studio 地址
    python examples/euler/05_query_api/query_api.py --addr 192.168.1.100:50051

验证点（11 项数值判定 + 3 项人工观察）:
    阶段一 step 0（初始直立）:
    1. joint_qpos_dim: 29 个 hinge joint qpos
    2. joint_qpos_vs_data: query_joint_qpos 与 data.qpos 按关节地址切片一致
    3. pelvis_initial_height: G1 站立初始高度 [0.70, 0.95]
    4. imu_quat_dim: imu_quat sensor 维度 = 4
    5. torso_subtree_mass_positive: torso 子树质量 > 0
    6. torso_rel_pelvis_z: 躯干在骨盆上方（基座系 z ∈ [0.0, 0.2]）
    7. actuator_torque_dim: 29 个 motor 力矩
    8. site_pos_dim: imu site xpos 维度 = 3
    9. contact_count: G1 站立时与地面接触数 ≥ 1
    阶段二 step 50（瘫倒验证，零控下力控 motor 无法保持站立）:
    10. g1_collapsed_pelvis_drop: pelvis 高度较初始下降 > 0.1m
    11. g1_collapsed_torso_drop: torso 高度较初始下降 > 0.05m
    - g1_standing: Studio 视口 G1 初始站立地面（人工观察）
    - g1_collapsed: Studio 视口 G1 瘫倒在地（人工观察）

参见 docs/design/development/orca_gym_euler_phase4_online_validation_development.md §4.3.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.g1_base_env import (
    G1_FRAME_SKIP,
    G1_MODEL_XML,
    G1_ORCAGYM_ADDR,
    G1_TIME_STEP,
)
from common.online_verifier import OnlineVerifier
from query_api_env import QueryApiEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lesson 5: G1 状态查询 API 在线验证"
    )
    parser.add_argument(
        "--addr",
        default=G1_ORCAGYM_ADDR,
        help=f"OrcaStudio gRPC 地址（默认 {G1_ORCAGYM_ADDR}）",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="控制周期数（默认 100，每周期 {G1_FRAME_SKIP} 物理步）",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="后端：cpu=CPU MuJoCo（默认），cuda:0=Euler GPU",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = QueryApiEnv(
        frame_skip=G1_FRAME_SKIP,
        orcagym_addr=args.addr,
        agent_names=["g1"],  # 在线模式由场景扫描覆盖为实际 agent_name
        time_step=G1_TIME_STEP,
        model_xml_path=G1_MODEL_XML,
        device=args.device,
    )

    verifier = OnlineVerifier("Lesson 5: 状态查询 API")
    try:
        report = env.run_lesson(num_steps=args.num_steps, verifier=verifier)
    finally:
        env.close()

    # 退出码反映验证结果
    if not report["summary"]["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
