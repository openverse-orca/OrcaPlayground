"""第 6 课：外力应用与状态设置 — G1 推力/摩擦/接触力/mocap 位姿写入在线验证

在阶段四在线模式下，连接 OrcaStudio 加载 G1 关节，验证 G1 外力应用与状态设置
R 类 API 在真实仿真中运行正确。脚本通过 G1BaseEnv.run_lesson 框架自动步进 100 帧，
分阶段执行验证，并通过 OnlineVerifier 输出判定报告。

验证 API:
    - apply_body_force / clear_body_force / clear_all_forces（外力应用与清除）
    - set_geom_friction（geom 摩擦系数设置）
    - query_contact_force（接触力查询）
    - set_mocap_pos_and_quat（mocap 位姿写入，经 weld 约束驱动 manipulation_box）

前置条件:
    场景需导入 g1_29dof_camera.xml（含 TestMocapAnchor mocap body、
    manipulation_box 及 anchor_box_weld 等约束）。

用法:
    # 1. 先启动 OrcaStudio 并加载含 1 个 G1 的关卡（导入 g1_29dof_camera.xml），点击运行
    # 2. 运行脚本
    python examples/euler/06_force_apply/force_apply.py

    # 指定 Studio 地址
    python examples/euler/06_force_apply/force_apply.py --addr 192.168.1.100:50051

验证点（9 项数值判定 + 3 项人工观察）:
    阶段一 step 0（初始直立，接触力查询）:
    1. contact_exists: G1 站立时与地面有接触
    2. contact_normal_force: query_contact_force 返回显著法向力 (> 50N)
    阶段二 step 10–35（外力应用与清除）:
    3. force_lift_pelvis: 施力后 pelvis 上升 > 1cm
    4. xfrc_recorded: xfrc_applied 记录了施加的力
    5. xfrc_cleared: clear_body_force 后 xfrc 归零
    阶段三 step 50（全清力 + 摩擦设置）:
    6. clear_all_forces: clear_all_forces 清除全部 body 外力
    7. set_geom_friction_ok: set_geom_friction 调用成功
    阶段四 step 70–90（mocap 位姿写入与 weld 驱动）:
    8. mocap_pos_readback / mocap_quat_readback: 写入回读一致
    9. mocap_drives_box_via_weld: weld 约束驱动 box 跟随 mocap
    - force_applied: Studio 视口 G1 被向上抬起（人工观察）
    - force_cleared: Studio 视口 清力后 G1 自由落体回落（人工观察）
    - mocap_box_follow: Studio 视口 manipulation_box 跟随 mocap（人工观察）

参见 docs/design/development/orca_gym_euler_phase4_online_validation_development.md §4.3.2
"""

from __future__ import annotations

import argparse
import sys

from force_apply_env import ForceApplyEnv
from g1_base_env import (
    G1_FRAME_SKIP,
    G1_MODEL_XML,
    G1_ORCAGYM_ADDR,
    G1_TIME_STEP,
)
from online_verifier import OnlineVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lesson 6: G1 外力应用与状态设置 API 在线验证"
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = ForceApplyEnv(
        frame_skip=G1_FRAME_SKIP,
        orcagym_addr=args.addr,
        agent_names=["g1"],  # 在线模式由场景扫描覆盖为实际 agent_name
        time_step=G1_TIME_STEP,
        model_xml_path=G1_MODEL_XML,
    )

    verifier = OnlineVerifier("Lesson 6: 外力应用")
    try:
        report = env.run_lesson(num_steps=args.num_steps, verifier=verifier)
    finally:
        env.close()

    if not report["summary"]["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
