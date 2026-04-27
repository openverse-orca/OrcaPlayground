#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from examples.so101 import _config
from examples.so101._camera import port_map, setup_cameras, stop_cameras, wait_for_cameras
from examples.so101._env import close_env, create_env, start_video_stream, stop_video_stream
from examples.so101._preflight import require_open_port


def run_monitor(args: argparse.Namespace) -> None:
    stop_event = threading.Event()

    def _sigint_handler(sig, frame):
        del sig, frame
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    host, port = args.orcagym_addr.split(":")
    require_open_port(host, int(port), "OrcaGym gRPC")
    for camera_port in args.ports:
        require_open_port(
            "127.0.0.1",
            int(camera_port),
            f"相机端口 {camera_port}",
            "请确认 `--ports` 与 OrcaStudio 中的 websocket 端口一致；"
            "如果没有看到该端口，请打开对应相机的 `is_recording` 开关。",
        )

    windows = {}
    camera_ports = port_map(args.ports)
    for name, port in camera_ports.items():
        title = f"{name} (port {port}) | q/ESC exit"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, np.zeros((480, 640, 3), np.uint8))
        windows[name] = title
    cv2.waitKey(1)

    env = create_env(
        orcagym_addr=args.orcagym_addr,
        fps=max(1, int(args.fps)),
        xml_path=args.xml_path,
    )
    obs, _ = env.reset()
    del obs
    start_video_stream(env, args.dump_dir)
    cameras = setup_cameras(args.ports)
    wait_for_cameras(cameras)

    interval_ms = max(1, int(1000.0 / args.fps))
    try:
        while not stop_event.is_set():
            for name, cam in cameras.items():
                frame, idx = cam.get_frame(format="bgr24")
                img = frame.copy()
                label = f"{name} | port:{camera_ports[name]} | frame#{idx}"
                cv2.putText(img, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                if args.scale != 1.0:
                    h, w = img.shape[:2]
                    img = cv2.resize(img, (int(w * args.scale), int(h * args.scale)))
                cv2.imshow(windows[name], img)

            key = cv2.waitKey(interval_ms) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
        stop_cameras(cameras)
        stop_video_stream(env)
        close_env(env)


def main() -> None:
    parser = argparse.ArgumentParser(description="SO101 相机实时监控")
    parser.add_argument("--orcagym_addr", default=_config.DEFAULT_ORCAGYM_ADDR)
    parser.add_argument("--xml_path", default=str(_config.DEFAULT_XML_PATH))
    parser.add_argument("--ports", type=int, nargs="+", default=_config.DEFAULT_MONITOR_PORTS.copy())
    parser.add_argument("--fps", type=float, default=float(_config.DEFAULT_FPS))
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--dump-dir", default="/tmp/so101_camera_monitor_stream")
    args = parser.parse_args()
    run_monitor(args)


if __name__ == "__main__":
    main()
