"""第 8 课：视频输出验证 — 截帧 + 录制

验证视频输出功能:

1. 截帧功能: get_frame_png 保存 PNG，校验格式（PNG magic + PIL 解码）
2. 录制功能: begin_save_video 启动录制，运动过程中帧索引递增，
   stop_save_video 后生成 mp4 文件
3. 时间戳查询: get_camera_time_stamp 返回 camera_head_* 键

阶段序列（num_steps=450，共 9 秒仿真）:
    阶段 1（脚本启动）: OrcaGymScene.add_actor 加载 G1 spawnable 到场景
    阶段 2（EulerEnv）:
        before_loop:  激活摄像头流 + 使能检查 + 开始录制 + 启动前进
        steps 0–149:  G1 前进 3 秒（lin_vel=0.3）+ 周期截帧
        steps 150–299: G1 转弯 3 秒（ang_vel=0.5）+ 周期截帧
        steps 300–449: G1 横移 3 秒（lin_vel=(0,0.3)）+ 周期截帧
        after_loop:   停止录制 + mp4 检查 + 时间戳 + 提示查看文件

> **前置依赖**：本课依赖 Lesson 7 行走控制已验证（复用 ``g1_locomotion.py`` 驱动行走）。

> **摄像头激活原理**：Euler 体系走 LoadLocalEnv 路径，不填充 Studio 端
> ``m_spawnedEntities``，导致 ``SetCameraSensorInfo`` 找不到 actor。本课通过
> OrcaGymScene 的 ``add_actor`` + ``publish_scene`` 走 AddActor 路径加载 G1，
> spawn 时填充 ``m_spawnedEntities`` 并激活 ``CameraCaptureComponent``，随后
> EulerEnv 的 ``LoadLocalEnv`` 从场景生成 MJCF 用于仿真控制。

用法:
    # 1. 先启动 OrcaStudio 并加载一个**空关卡**（无 G1），点击运行
    # 2. 在 OrcaStudio 中导入障碍物 mjcf 文件，生成 spawnable actor：
    #      assets/scenes/obstacle_box.xml
    #      assets/scenes/obstacle_capsule.xml
    #      assets/scenes/obstacle_cylinder.xml
    #      assets/scenes/obstacle_sphere.xml
    #    （导入后默认生成在 prefabs 目录，文件名加 _usda 后缀，
    #     即 assets/prefabs/obstacle_<type>_usda，与 G1 路径风格一致）
    # 3. 运行脚本（脚本会自动通过 add_actor 加载 G1 + 50 个障碍物）
    python examples/euler/08_video_capture/video_capture.py

    # 指定 Studio 地址
    python examples/euler/08_video_capture/video_capture.py --addr 192.168.1.100:50051

    # 不加载障碍物（仅 G1，空场景）
    python examples/euler/08_video_capture/video_capture.py --no-obstacles

验证点:
    before_loop:
    1. camera_enabled: 摄像头使能（frame_idx >= 0）
    observe_step（step 0）:
    2. png_file_valid_format: PNG 截帧文件格式校验（magic + PIL 解码）
    verify_step（每 50 步）:
    3. frame_index_increasing_{step}: 帧索引递增
    after_loop:
    4. timestamp_returned: 时间戳查询返回 camera_head_* 键
    5. mp4_file_generated: 录制完成后 mp4 文件生成
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from g1_base_env import (
    G1_FRAME_SKIP,
    G1_MODEL_XML,
    G1_ORCAGYM_ADDR,
    G1_TIME_STEP,
)
from online_verifier import OnlineVerifier
from video_capture_env import VideoCaptureEnv

# G1 spawnable 配置
# Studio 资产缓存路径（去掉 .spawnable 扩展名），对应文件：
#   Cache/linux/assets/prefabs/g1_29dof_camera_usda.spawnable
G1_SPAWNABLE_PATH = "assets/prefabs/g1_29dof_camera_usda"
# AddActor name 与 spawnable name 一致，确保与 EulerEnv 场景扫描的 agent_name 匹配
G1_ACTOR_NAME = "g1_29dof_camera_usda"


def _wait_for_keypress(prompt: str) -> None:
    """阻塞等待用户按 Space/Enter 继续（无 render，用于 spawn 阶段）。

    与 G1BaseEnv.wait_for_keypress 不同：此函数不依赖 env，
    用于 spawn_g1_actor 阶段——此时用户观察的是 Studio 编辑器视口，
    而非 Python 端 render 出来的画面，故无需轮询 render。
    """
    print(f"  [PAUSE] {prompt}（按 Space 键继续）")
    sys.stdout.flush()
    if not sys.stdin.isatty():
        try:
            input()
        except EOFError:
            return
        return
    if sys.platform == "win32":
        # Windows: msvcrt 非阻塞按键检测
        import msvcrt

        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getch().decode(errors="ignore")
                if ch in (" ", "\r", "\n"):
                    break
            time.sleep(0.01)
    else:
        # Unix: raw 模式 + select 非阻塞轮询
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                rlist, _, _ = select.select([fd], [], [], 1.0)
                if rlist:
                    ch = os.read(fd, 1).decode(errors="ignore")
                    if ch in (" ", "\r", "\n"):
                        break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def spawn_scene(orcagym_addr: str, spawn_obstacles: bool = True) -> None:
    """用 OrcaGymScene 将 G1 + 障碍物加载到 Studio 场景。

    走 AddActor + PublishScene 路径激活 CameraCaptureComponent：
    1. publish_scene() 清空场景（销毁已有 spawned entity）
    2. add_actor() 注册 G1 + 障碍物到 m_addActorMap
    3. publish_scene() 触发 spawn，填充 m_spawnedEntities
    4. set_material_info() 为障碍物设置随机颜色（spawn 后 actor 已存在）

    spawn 后 Studio 端：
    - m_spawnedEntities[G1_ACTOR_NAME] = containerEntityId
    - CameraCaptureComponent::Activate → InitCameraSensor → RegisterCameraComponent
    - 随后 SetCameraSensorInfo 能找到 actor 并激活 RGB/depth 流

    publish_scene 会触发 Studio 端 MuJoCo grpc server 重启，立即创建 env
    会因 server 未就绪而失败（LoadLocalEnv: MuJoCo has not been initialized）。
    因此 publish 后等待 3 秒让 server 完成重启，并暂停引导用户确认场景已出现在
    Studio 视口中，再继续创建 EulerEnv。
    """
    import time

    from orca_gym.scene.orca_gym_scene import Actor, OrcaGymScene

    scene = OrcaGymScene(grpc_addr=orcagym_addr)
    try:
        scene.publish_scene()

        # G1
        g1_actor = Actor(
            name=G1_ACTOR_NAME,
            asset_path=G1_SPAWNABLE_PATH,
            position=np.array([0.0, 0.0, 0.0]),
            rotation=np.array([0.0, 0.0, 0.0, 1.0]),
            scale=1.0,
        )
        scene.add_actor(g1_actor)

        # 障碍物（50 个静态几何体，环形分布在 5-10m 区域）
        obstacle_specs = None
        if spawn_obstacles:
            from obstacle_spawner import (
                generate_obstacle_layout,
                spawn_obstacles,
                set_obstacle_colors,
            )
            obstacle_specs = generate_obstacle_layout()
            spawn_obstacles(scene, obstacle_specs)
            print(f"[INFO] 已注册 G1 + {len(obstacle_specs)} 个障碍物到场景")

        # 触发 spawn
        scene.publish_scene()

        # spawn 后为障碍物设置颜色（actor 已存在于 m_spawnedEntities）
        if obstacle_specs:
            set_obstacle_colors(scene, obstacle_specs)
            print(f"[INFO] 已为 {len(obstacle_specs)} 个障碍物设置随机颜色")
    finally:
        scene.close()

    # publish_scene 触发 Studio 端 MuJoCo grpc server 重启，等待其完成初始化
    print("[INFO] 等待 3 秒，让 Studio 端 MuJoCo grpc server 完成重启...")
    time.sleep(3)
    _wait_for_keypress(
        "请在 OrcaStudio 视口中确认 G1 机器人 + 障碍物已出现在场景中"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lesson 8: 视频输出验证（截帧 + 录制）"
    )
    parser.add_argument(
        "--addr",
        default=G1_ORCAGYM_ADDR,
        help=f"OrcaStudio gRPC 地址（默认 {G1_ORCAGYM_ADDR}）",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=450,
        help=(
            f"控制周期数（默认 450，每周期 {G1_FRAME_SKIP} 物理步，共 9 秒仿真；"
            "前进 3 秒 + 转弯 3 秒 + 横移 3 秒，全程自动录制 + 周期截帧）"
        ),
    )
    parser.add_argument(
        "--no-obstacles",
        action="store_true",
        help="不加载障碍物（仅 G1，空场景，用于快速测试）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 阶段 1：用 OrcaGymScene 加载 G1 + 障碍物（激活 CameraCaptureComponent）
    spawn_scene(args.addr, spawn_obstacles=not args.no_obstacles)

    # 阶段 2：创建 EulerEnv 控制仿真 + 视频捕获
    env = VideoCaptureEnv(
        frame_skip=G1_FRAME_SKIP,
        orcagym_addr=args.addr,
        agent_names=["g1"],  # 在线模式由场景扫描覆盖为实际 agent_name
        time_step=G1_TIME_STEP,
        model_xml_path=G1_MODEL_XML,
    )

    verifier = OnlineVerifier("Lesson 8: 视频输出")
    try:
        report = env.run_lesson(num_steps=args.num_steps, verifier=verifier)
    finally:
        env.close()

    if not report["summary"]["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
