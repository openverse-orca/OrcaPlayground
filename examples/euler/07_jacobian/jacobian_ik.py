"""第 7 课：雅可比与 IK — G1 雅可比计算与阻尼最小二乘 IK 在线验证

在阶段四在线模式下，连接 OrcaStudio 加载 G1 关卡，验证 G1 雅可比计算与 IK 迭代
API 在真实仿真中运行正确。脚本通过 G1BaseEnv.run_lesson 框架自动步进 100 帧，
在 step 0 集中执行 3 项验证，并通过 OnlineVerifier 输出判定报告。

验证 API:
    - mj_jacBody（body 平移/旋转雅可比）
    - mj_jacSite（site 雅可比）
    - query_site_xvalp_xvalr（site 速度查询）
    - set_joint_qpos + mj_forward（合规状态写入 + 前向更新）

用法:
    # 1. 先启动 OrcaStudio 并加载含 1 个 G1 的关卡，点击运行
    # 2. 运行脚本
    python examples/euler/07_jacobian/jacobian_ik.py

    # Euler GPU 后端
    python examples/euler/07_jacobian/jacobian_ik.py --device cuda:0

    # 指定 Studio 地址
    python examples/euler/07_jacobian/jacobian_ik.py --addr 192.168.1.100:50051

验证点（3 项数值判定 + 1 项人工观察）:
    step 0（雅可比与 IK 集中验证）:
    1. jac_shape: pelvis 雅可比形状 (3, nv)，nv ≥ 35
    2. site_vel_vs_jac: imu site 速度 = jacp_site @ qvel（atol=1e-4）
    3. ik_foot_target: IK 迭代 80 次后左脚到达目标位置（atol=0.02）
    - ik_foot_movement: Studio 视口左脚抬高约 10cm（人工观察）

参见 docs/design/development/orca_gym_euler_phase4_online_validation_development.md §4.3.3
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
from jacobian_env import JacobianEnv
from common.online_verifier import OnlineVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lesson 7: G1 雅可比与 IK 在线验证"
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
        help=f"控制周期数（默认 100，每周期 {G1_FRAME_SKIP} 物理步）",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="后端：cpu=CPU MuJoCo（默认），cuda:0=Euler GPU",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = JacobianEnv(
        frame_skip=G1_FRAME_SKIP,
        orcagym_addr=args.addr,
        agent_names=["g1"],  # 在线模式由场景扫描覆盖为实际 agent_name
        time_step=G1_TIME_STEP,
        model_xml_path=G1_MODEL_XML,
        device=args.device,
    )

    verifier = OnlineVerifier("Lesson 7: 雅可比 IK")
    try:
        report = env.run_lesson(num_steps=args.num_steps, verifier=verifier)
    finally:
        env.close()

    if not report["summary"]["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
