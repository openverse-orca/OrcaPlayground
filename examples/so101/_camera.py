from __future__ import annotations

import asyncio
import io
import threading
import time

import av
import cv2
import numpy as np
import websockets


class CameraWrapper:
    def __init__(self, name: str, port: int):
        self._name = name
        self.port = port
        self.image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        self.thread = None
        self.received_first_frame = False
        self.image_index = 0
        self.running = False
        self.last_error: Exception | None = None

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self) -> None:
        try:
            asyncio.run(self._read_stream())
        except Exception as exc:
            self.last_error = exc

    def is_first_frame_received(self) -> bool:
        return self.received_first_frame

    async def _read_stream(self) -> None:
        uri = f"ws://localhost:{self.port}"
        async with websockets.connect(uri) as websocket:
            cur_pos = 0
            raw_data = io.BytesIO()
            container = None
            while self.running:
                data = await websocket.recv()
                data = data[8:]
                raw_data.write(data)
                raw_data.seek(cur_pos)
                if cur_pos == 0:
                    container = av.open(raw_data, mode="r")
                for packet in container.demux():
                    if packet.size == 0:
                        continue
                    for frame in packet.decode():
                        self.image = frame.to_ndarray(format="bgr24")
                        self.image_index += 1
                        self.received_first_frame = True
                cur_pos += len(data)

    def get_frame(self, format: str = "bgr24", size: tuple[int, int] | None = None) -> tuple[np.ndarray, int]:
        if format == "rgb24":
            frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            frame = self.image
        if size is not None:
            frame = cv2.resize(frame, size)
        return frame, self.image_index


def port_map(ports: list[int]) -> dict[str, int]:
    if len(ports) == 2:
        return {
            "camera_global": ports[0],
            "camera_wrist": ports[1],
        }
    return {f"camera_{port}": port for port in ports}


def setup_cameras(ports: list[int]) -> dict[str, CameraWrapper]:
    cameras = {}
    for name, port in port_map(ports).items():
        cam = CameraWrapper(name=name, port=port)
        cam.start()
        cameras[name] = cam
    return cameras


def wait_for_cameras(cameras: dict[str, CameraWrapper], timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    pending = list(cameras.keys())
    while time.time() < deadline:
        errored = {name: cam.last_error for name, cam in cameras.items() if cam.last_error is not None}
        if errored:
            details = "\n".join(
                f"- {name}: {type(exc).__name__}: {exc}" for name, exc in errored.items() if exc is not None
            )
            raise RuntimeError(
                "相机流连接失败。\n"
                f"{details}\n"
                "请确认 `--ports` 与 OrcaStudio 中暴露的 websocket 端口一致；"
                "如果没有暴露端口，请打开对应相机的 `is_recording` 开关。"
            )
        pending = [name for name, cam in cameras.items() if not cam.is_first_frame_received()]
        if not pending:
            return
        time.sleep(1.0)
    pending_text = ", ".join(pending)
    raise RuntimeError(
        f"等待相机首帧超时，仍未收到画面的相机: {pending_text}\n"
        "请确认场景正在运行，`--ports` 配置正确；"
        "如果 OrcaStudio 中没有开始推流，请打开对应相机的 `is_recording` 开关。"
    )


def stop_cameras(cameras: dict[str, CameraWrapper]) -> None:
    for cam in cameras.values():
        cam.running = False
    for cam in cameras.values():
        if cam.thread is not None and cam.thread.is_alive():
            cam.thread.join(timeout=2.5)
