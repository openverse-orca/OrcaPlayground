"""第 7 课：Studio 视频录制与截帧 — G1 行走录制在线验证

在阶段四在线模式下，连接 OrcaStudio 加载 G1 关卡，在 G1Locomotion ONNX 策略
驱动行走的同时，验证 Studio 视频/帧/时间戳采集 API 运行正确。脚本通过
G1BaseEnv.run_lesson 框架步进 500 帧（10 秒仿真），录制完整行走过程。

验证 API（Studio 交互层）:
    - begin_save_video / stop_save_video（视频录制）
    - get_current_frame / get_next_frame（帧索引查询）
    - get_frame_png（PNG 截帧）
    - get_camera_time_stamp（相机时间戳）

用法:
    # 1. 先启动 OrcaStudio 并加载含 1 个 G1 的关卡，点击运行
    # 2. 运行脚本
    python examples/euler/07_studio_capture/studio_capture.py

    # 指定 Studio 地址
    python examples/euler/07_studio_capture/studio_capture.py --addr 192.168.1.100:50051

验证点（5 项数值判定 + 2 项人工观察）:
    before_loop:
    1. camera_enabled: 摄像头使能（frame_idx >= 0）
    verify_step（每 50 步）:
    2. frame_index_increasing_{step}: 帧索引递增
    after_loop:
    3. png_file_generated: PNG 截帧文件生成（size > 100）
    4. timestamp_returned: 时间戳查询返回 camera_head 键
    5. mp4_file_generated: 录制完成后 mp4 文件生成
    - g1_walking / walking_stable: Studio 视口观察 G1 行走（人工）

参见 docs/design/development/orca_gym_euler_phase4_online_validation_development.md §4.3.4
"""

from __future__ import annotations

import argparse
import sys

from g1_base_env import (
    G1_FRAME_SKIP,
    G1_MODEL_XML,
    G1_ORCAGYM_ADDR,
    G1_TIME_STEP,
)
from online_verifier import OnlineVerifier
from studio_capture_env import StudioCaptureEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lesson 7: G1 行走录制与 Studio 采集在线验证"
    )
    parser.add_argument(
        "--addr",
        default=G1_ORCAGYM_ADDR,
        help=f"OrcaStudio gRPC 地址（默认 {G1_ORCAGYM_ADDR}）",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=500,
        help=f"控制周期数（默认 500，每周期 {G1_FRAME_SKIP} 物理步，共 10 秒）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = StudioCaptureEnv(
        frame_skip=G1_FRAME_SKIP,
        orcagym_addr=args.addr,
        agent_names=["g1"],  # 在线模式由场景扫描覆盖为实际 agent_name
        time_step=G1_TIME_STEP,
        model_xml_path=G1_MODEL_XML,
    )

    verifier = OnlineVerifier("Lesson 7: 视频录制")
    try:
        report = env.run_lesson(num_steps=args.num_steps, verifier=verifier)
    finally:
        env.close()

    if not report["summary"]["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
