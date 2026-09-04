"""第 9 课：视频输出验证 — 截帧 + 录制

验证视频输出功能（OrcaGym 新相机 API，客户端 PyAV remux 录制）:

1. 截帧功能: get_frame_png 保存 PNG，校验格式（PNG magic + PIL 解码）
2. 推流与录制: start_streaming 启动 RGB+Depth 推流；render(simulate_index=...)
   驱动帧对齐；save_streaming 将区间 H.264 流客户端 remux 为 mp4
3. 时间戳查询: save_streaming 返回的 RemuxResult.timestamps_ns 携带每帧
   纳秒时间戳

阶段序列（num_steps=450，共 9 秒仿真）:
    阶段 1（脚本启动）: OrcaGymScene.add_actor 加载 G1 spawnable 到场景
    阶段 2（EulerEnv）:
        before_loop:  枚举相机 + 启动 RGB/Depth 推流 + 启动前进
        steps 0–149:  G1 前进 3 秒（lin_vel=0.3）+ 周期截帧
        steps 150–299: G1 转弯 3 秒（ang_vel=0.5）+ 周期截帧
        steps 300–449: G1 横移 3 秒（lin_vel=(0,0.3)）+ 周期截帧
        after_loop:   save_streaming 生成 color/depth mp4 + 时间戳 + 提示查看文件

> **前置依赖**：本课依赖 Lesson 8 行走控制已验证（复用 ``g1_locomotion.py`` 驱动行走）。

> **摄像头激活原理**：Euler 体系走 LoadLocalEnv 路径，不填充 Studio 端
> ``m_spawnedEntities``，导致相机注册与截帧 RPC 找不到 actor。本课通过
> OrcaGymScene 的 ``add_actor`` + ``publish_scene`` 走 AddActor 路径加载 G1，
> spawn 时填充 ``m_spawnedEntities`` 并激活 ``CameraCaptureComponent``，随后
> EulerEnv 的 ``LoadLocalEnv`` 从场景生成 MJCF 用于仿真控制。

用法:
    # 1. 先启动 OrcaStudio/OrcaLab 并加载一个**空关卡**（无 G1），点击运行
    # 2. 在 OrcaStudio/OrcaLab 中订阅 Euler_asset 资产包（含 G1 与障碍物 spawnable）
    #    脚本自动按顺序尝试两种 spawnable 路径，任一可用即可：
    #      - OrcaStudio 缓存：assets/prefabs/<name>_usda
    #      - OrcaLab Euler_asset：assets/e071469a36d3c8aa/default_project/prefabs/<name>_usda
    #    两者均不可用时脚本抛出错误提醒，提示订阅 Euler_asset 资产包
    # 3. 运行脚本（脚本会自动通过 add_actor 加载 G1 + 50 个障碍物）
    python examples/euler/09_video_capture/video_capture.py

    # 指定 Studio 地址
    python examples/euler/09_video_capture/video_capture.py --addr 192.168.1.100:50051

    # 不加载障碍物（仅 G1，空场景）
    python examples/euler/09_video_capture/video_capture.py --no-obstacles

验证点:
    before_loop:
    1. camera_registered: 相机已注册（get_camera_names 非空）
    2. streaming_started: start_streaming RGB+Depth 推流启动
    observe_step（step 0）:
    3. png_file_valid_format: PNG 截帧文件格式校验（magic + PIL 解码）
    verify_step（每 50 步）:
    4. frame_index_increasing_{step}: 已到达帧 simulate_index 递增
    after_loop:
    5. mp4_file_generated: save_streaming 客户端 remux 生成 mp4 文件
    6. timestamp_returned: RemuxResult.timestamps_ns 非空
"""

from __future__ import annotations

import argparse
import logging
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
# spawnable 资产路径候选（去掉 .spawnable 扩展名），按顺序尝试，任一可用即可：
#   [0] OrcaStudio 缓存路径（assets/prefabs/g1_29dof_camera_usda）
#   [1] OrcaLab Euler_asset 路径
#       （assets/e071469a36d3c8aa/default_project/prefabs/g1_29dof_camera_usda）
# 均失败时 spawn_scene 抛出错误提醒，提示订阅 Euler_asset 资产包
G1_SPAWNABLE_PATHS: list[str] = [
    "assets/prefabs/g1_29dof_camera_usda",
    "assets/e071469a36d3c8aa/default_project/prefabs/g1_29dof_camera_usda",
]
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

    spawnable 路径双候选回退（OrcaStudio 缓存优先，OrcaLab Euler_asset 兜底）：
    按候选路径索引顺序尝试。每个索引对应一次独立的 OrcaGymScene 连接：先 publish_scene
    清空，再 add_actor 注册 G1 + 障碍物，最后 publish_scene 触发 spawn。任一索引完整
    成功即返回；所有索引均失败时抛出 RuntimeError，提示订阅 Euler_asset 资产包。

    publish_scene 会触发 Studio 端 MuJoCo grpc server 重启，立即创建 env
    会因 server 未就绪而失败（LoadLocalEnv: MuJoCo has not been initialized）。
    因此 spawn 后等待 3 秒让 server 完成重启，并暂停引导用户确认场景已出现在
    Studio 视口中，再继续创建 EulerEnv。
    """
    from orca_gym.scene.orca_gym_scene import Actor, OrcaGymScene
    from obstacle_spawner import (
        EULER_ASSET_SUBSCRIBE_HINT,
        generate_obstacle_layout,
        set_obstacle_colors,
        spawn_obstacles,
    )

    # 上游 OrcaGym 的 _add_actor 在失败时会用 ERROR 级别 logger 打印大段诊断信息。
    # 双候选回退时单个候选失败属于正常切换流程，不应刷屏；仅在所有候选均失败时
    # 统一抛错并恢复日志级别。这里临时把上游 logger（单例，名为 "OrcaGym"）及其
    # console handler 调到 CRITICAL 静音。
    _og_scene_logger = logging.getLogger("OrcaGym")
    _orig_logger_level = _og_scene_logger.level
    _og_scene_logger.setLevel(logging.CRITICAL)
    _og_handler_levels: list[tuple[logging.Handler, int]] = []
    for _h in _og_scene_logger.handlers:
        _og_handler_levels.append((_h, _h.level))
        _h.setLevel(logging.CRITICAL)

    obstacle_specs = generate_obstacle_layout() if spawn_obstacles else None
    num_candidates = len(G1_SPAWNABLE_PATHS)
    errors: list[str] = []
    connection_failed = False  # gRPC 连接类失败（与资产包无关，单独提示）
    spawned = False

    for idx in range(num_candidates):
        g1_path = G1_SPAWNABLE_PATHS[idx]
        scene = OrcaGymScene(grpc_addr=orcagym_addr)
        try:
            # 清空已有 spawned entity
            scene.publish_scene()

            # G1
            g1_actor = Actor(
                name=G1_ACTOR_NAME,
                asset_path=g1_path,
                position=np.array([0.0, 0.0, 0.0]),
                rotation=np.array([0.0, 0.0, 0.0, 1.0]),
                scale=1.0,
            )
            scene.add_actor(g1_actor)

            # 障碍物（50 个静态几何体，环形分布在 5-10m 区域）
            if obstacle_specs:
                spawn_obstacles(scene, obstacle_specs, path_index=idx)

            # 触发 spawn
            scene.publish_scene()

            # spawn 后为障碍物设置颜色（actor 已存在于 m_spawnedEntities）
            if obstacle_specs:
                set_obstacle_colors(scene, obstacle_specs)
            print(f"[INFO] 已加载 G1"
                  + (f" + {len(obstacle_specs)} 个障碍物" if obstacle_specs else "")
                  + f"（候选路径 #{idx}: {g1_path}）")
            if obstacle_specs:
                print(f"[INFO] 已为 {len(obstacle_specs)} 个障碍物设置随机颜色")
            spawned = True
            break
        except Exception as e:
            # 单个候选失败时静默记录，仅在所有候选均失败时统一报错。
            # 区分两类失败：
            #   - 连接类（gRPC UNAVAILABLE / Connection refused）：OrcaStudio/OrcaLab
            #     未启动或端口不通，与资产包无关，只标记一次，不记录资产路径字眼。
            #   - 其他（如 Spawnable name not found）：资产问题，提示订阅资产包。
            err_str = str(e)
            is_conn_err = (
                "Connection refused" in err_str
                or "failed to connect to all addresses" in err_str
                or "grpc_status:14" in err_str
            )
            if is_conn_err:
                connection_failed = True
            else:
                errors.append(f"候选 #{idx} '{g1_path}': {err_str}")
        finally:
            try:
                scene.close()
            except Exception:
                pass

    # 恢复上游 logger 及 handler 级别；若所有候选均失败，恢复后再抛错以便用户看到完整 traceback
    _og_scene_logger.setLevel(_orig_logger_level)
    for _h, _lvl in _og_handler_levels:
        _h.setLevel(_lvl)

    if not spawned:
        target = "G1 与障碍物" if spawn_obstacles else "G1"
        # 连接失败：不提及资产，只提示启动 OrcaStudio/OrcaLab
        if connection_failed:
            raise RuntimeError(
                f"无法连接 OrcaStudio/OrcaLab gRPC 服务（{orcagym_addr}）"
                "请确认 OrcaStudio/OrcaLab 已启动、已点击运行，且 gRPC 端口监听正常。"
            )
        # 仅资产类失败才提示订阅 Euler_asset
        raise RuntimeError(
            f"所有 spawnable 候选路径均失败，无法加载 {target}。\n"
            "已尝试:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n"
            + EULER_ASSET_SUBSCRIBE_HINT
        )

    # publish_scene 触发 Studio 端 MuJoCo grpc server 重启，等待其完成初始化
    print("[INFO] 等待 3 秒，让 Studio 端 MuJoCo grpc server 完成重启...")
    time.sleep(3)
    if spawn_obstacles:
        prompt = "请在 OrcaStudio 视口中确认 G1 机器人 + 障碍物已出现在场景中"
    else:
        prompt = "请在 OrcaStudio 视口中确认 G1 机器人已出现在场景中"
    _wait_for_keypress(prompt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lesson 9: 视频输出验证（截帧 + 录制）"
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

    verifier = OnlineVerifier("Lesson 9: 视频输出")
    try:
        report = env.run_lesson(num_steps=args.num_steps, verifier=verifier)
    finally:
        env.close()

    if not report["summary"]["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
