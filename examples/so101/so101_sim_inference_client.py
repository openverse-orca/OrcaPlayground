#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.so101 import _config
from examples.so101._camera import setup_cameras, stop_cameras, wait_for_cameras
from examples.so101._env import close_env, create_env, start_video_stream, stop_video_stream
from examples.so101._policy_client import connect_policy_server, prepare_image
from examples.so101._preflight import require_open_port


MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@lru_cache(maxsize=1)
def is_headless() -> bool:
    try:
        import pynput  # noqa: F401

        return False
    except Exception:
        return True


def init_keyboard_listener():
    events = {
        "exit_early": False,
        "stop_recording": False,
    }
    if is_headless():
        return None, events

    from pynput import keyboard

    def on_press(key):
        if key == keyboard.Key.page_down:
            events["exit_early"] = True
        elif key == keyboard.Key.esc:
            events["stop_recording"] = True
            events["exit_early"] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def busy_wait(seconds: float) -> None:
    if seconds <= 0:
        return
    if platform.system() in ("Darwin", "Windows"):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            pass
    else:
        time.sleep(seconds)


def extract_state(obs: dict) -> np.ndarray:
    arm_qpos = obs[f"{_config.AGENT_NAME}_arm_joint_qpos"].flatten()
    grasp = obs[f"{_config.AGENT_NAME}_grasp_value"].flatten()
    return np.concatenate([arm_qpos, grasp]).astype(np.float32)


def policy_to_env_action(policy_action_6: np.ndarray) -> np.ndarray:
    full = np.zeros(_config.ACTION_DIM, dtype=np.float32)
    full[_config.ARM_OFFSET : _config.ARM_OFFSET + 5] = policy_action_6[:5]
    full[_config.GRIPPER_INDEX] = policy_action_6[5]
    return full


def resolve_task(default_task: str, interactive: bool) -> str:
    if not interactive:
        print(f'任务已确认: "{default_task}"')
        return default_task

    try:
        value = input(f'任务描述[{default_task}]: ').strip()
        task = value or default_task
    except EOFError:
        task = default_task

    print(f'任务已确认: "{task}"')
    return task


def format_joint_state(obs: dict) -> str:
    state = extract_state(obs)
    parts = [f"{name[:3]}={state[idx]:+.3f}" for idx, name in enumerate(MOTOR_NAMES)]
    return "  ".join(parts)


def print_runtime_step(chunk_idx: int, sub_step: int, total_actions: int, global_step: int, obs: dict) -> None:
    print(
        f"[执行] chunk={chunk_idx:03d}  sub={sub_step:02d}/{total_actions:02d}  "
        f"step={global_step:04d}  {format_joint_state(obs)}",
        flush=True,
    )


def run_inference(args: argparse.Namespace) -> None:
    log_every = max(1, args.log_every)
    orcagym_host, orcagym_port = args.orcagym_addr.split(":")
    require_open_port(orcagym_host, int(orcagym_port), "OrcaGym gRPC")
    require_open_port(args.host, int(args.port), "策略服务")
    for camera_port in args.ports:
        require_open_port(
            "127.0.0.1",
            int(camera_port),
            f"相机端口 {camera_port}",
            "请确认 `--ports` 与 OrcaStudio 中的 websocket 端口一致；"
            "如果没有看到该端口，请打开对应相机的 `is_recording` 开关。",
        )

    client = connect_policy_server(args.host, args.port)
    env = create_env(
        orcagym_addr=args.orcagym_addr,
        fps=args.fps,
        xml_path=args.xml_path,
    )
    start_video_stream(env, args.dump_dir)
    cameras = setup_cameras(args.ports)
    if len(cameras) < 2:
        raise RuntimeError("至少需要两路相机端口")
    wait_for_cameras(cameras)

    listener, events = init_keyboard_listener()
    task = resolve_task(args.task, args.interactive_task)
    obs, _ = env.reset()
    for _ in range(5):
        obs, _, _, _, _ = env.step(np.zeros(_config.ACTION_DIM, dtype=np.float32))
    env.render()

    step = 0
    chunk_idx = 0
    success = False
    stop_reason = "用户中止"
    try:
        while not events["stop_recording"]:
            if args.max_steps > 0 and step >= args.max_steps:
                stop_reason = f"达到最大步数 {args.max_steps}"
                break

            print(f"[推理] 开始请求 chunk={chunk_idx:03d}  step={step:04d}", flush=True)
            infer_started = time.perf_counter()
            state = extract_state(obs)
            front_img = cameras["camera_global"].get_frame(format="rgb24")[0]
            wrist_img = cameras["camera_wrist"].get_frame(format="rgb24")[0]
            result = client.infer(
                {
                    "observation/image": prepare_image(front_img),
                    "observation/wrist_image": prepare_image(wrist_img),
                    "observation/state": state,
                    "prompt": task,
                }
            )
            actions = result["actions"]
            infer_elapsed_ms = (time.perf_counter() - infer_started) * 1000.0
            print(
                f"[推理] chunk={chunk_idx:03d} 返回 {len(actions)} 步动作，耗时 {infer_elapsed_ms:.0f} ms",
                flush=True,
            )
            for sub_step, action_vec in enumerate(actions, start=1):
                if events["exit_early"]:
                    events["exit_early"] = False
                    stop_reason = f"用户提前结束 chunk {chunk_idx}"
                    break
                if events["stop_recording"]:
                    stop_reason = "用户按下 ESC"
                    break
                if args.max_steps > 0 and step >= args.max_steps:
                    stop_reason = f"达到最大步数 {args.max_steps}"
                    break

                t0 = time.perf_counter()
                env_action = policy_to_env_action(action_vec[: len(MOTOR_NAMES)])
                obs, _reward, terminated, _truncated, _info = env.step(env_action)
                env.render()
                step += 1
                if step == 1 or step % log_every == 0:
                    print_runtime_step(chunk_idx, sub_step, len(actions), step, obs)
                if terminated:
                    success = True
                    stop_reason = "任务成功"
                    print(f"任务已完成，共执行 {step} 步。", flush=True)
                    events["stop_recording"] = True
                    break
                busy_wait((1.0 / args.fps) - (time.perf_counter() - t0))
            chunk_idx += 1
    finally:
        if not success and step > 0 and stop_reason == "用户中止":
            stop_reason = "推理循环结束"
        print(f"推理结束: {stop_reason}。累计执行 {step} 步，完成 {chunk_idx} 个 chunk。", flush=True)
        if listener is not None and not is_headless():
            listener.stop()
        stop_cameras(cameras)
        stop_video_stream(env)
        close_env(env)


def main() -> None:
    parser = argparse.ArgumentParser(description="SO101 + pi0.5 仿真推理客户端")
    parser.add_argument("--host", default=_config.DEFAULT_POLICY_HOST)
    parser.add_argument("--port", type=int, default=_config.DEFAULT_POLICY_PORT)
    parser.add_argument("--task", default="Pick up the blue block")
    parser.add_argument("--orcagym_addr", default=_config.DEFAULT_ORCAGYM_ADDR)
    parser.add_argument("--xml_path", default=str(_config.DEFAULT_XML_PATH))
    parser.add_argument("--ports", type=int, nargs="+", default=_config.DEFAULT_MONITOR_PORTS.copy())
    parser.add_argument("--fps", type=int, default=_config.DEFAULT_FPS)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--interactive-task", action="store_true")
    parser.add_argument("--dump-dir", default="/tmp/so101_infer_stream")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
