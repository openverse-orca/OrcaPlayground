"""第 10 课：体操作与交互式驱动 — 锚定/mocap 驱动/equality

交互式菜单驱动模式：机器人在无绑定状态下向前行走 3 秒，然后暂停询问用户选择
操作（1/2/3/4/5），根据选择绑定 mocap 并在 3 秒内周期性移动机器人，或取消
绑定继续自主行走，或退出。

验证 API（体操作层，公共原语，消费者自管编排）:
    - env.equality_find_slot_by_body / equality_constraint（定位槽位 + 保存快照）
    - env.equality_update（写入/恢复约束，绑定/释放均走此原语）
    - env.set_mocap_pos_and_quat（对齐 mocap 位姿 + 驱动 mocap body）
    - env.get_body_xpos_xmat_xquat（读取位姿）

用法:
    # 1. 先启动 OrcaStudio 并加载含 1 个 G1 的关卡（需含 mocap body），点击运行
    # 2. 运行脚本
    python examples/euler/10_body_manipulation/body_manipulation.py

    # 指定 Studio 地址
    python examples/euler/10_body_manipulation/body_manipulation.py --addr 192.168.1.100:50051

交互流程:
    1. G1 无绑定向前行走 3 秒
    2. 暂停，显示菜单，等待用户输入：
       1: 绑定 mocap，提升 0.5 米
       2: 绑定 mocap，向前移动 1 米
       3: 绑定 mocap，向左移动 1 米
       4: 取消绑定，继续自主向前移动 3 秒
       5: 结束退出
    3. 执行操作（绑定模式：3 秒内周期性移动到位；取消模式：自主行走 3 秒）
    4. 回到步骤 1，循环

验证点:
    - 每次暂停前：pelvis 位姿有限性检查
    - 绑定移动后：位移到位检查（atol=0.1m）

参见 docs/design/development/orca_gym_euler_phase4_directory_restructure.md §4.6
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
from body_manipulation_env import BodyManipulationEnv
from common.online_verifier import OnlineVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lesson 10: 体操作与交互式驱动（锚定/mocap 驱动/equality）"
    )
    parser.add_argument(
        "--addr",
        default=G1_ORCAGYM_ADDR,
        help=f"OrcaStudio gRPC 地址（默认 {G1_ORCAGYM_ADDR}）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = BodyManipulationEnv(
        frame_skip=G1_FRAME_SKIP,
        orcagym_addr=args.addr,
        agent_names=["g1"],  # 在线模式由场景扫描覆盖为实际 agent_name
        time_step=G1_TIME_STEP,
        model_xml_path=G1_MODEL_XML,
    )

    verifier = OnlineVerifier("Lesson 10: 体操作")
    try:
        report = env.run_interactive(verifier)
    finally:
        env.close()

    if not report["summary"]["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
