from __future__ import annotations

import time

import numpy as np

from envs.so101.openpi_client import image_tools
from envs.so101.openpi_client.websocket_client_policy import WebsocketClientPolicy


IMG_SIZE = 224


def prepare_image(img: np.ndarray) -> np.ndarray:
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, IMG_SIZE, IMG_SIZE))


def connect_policy_server(host: str, port: int, retry: int = 10, interval: float = 3.0) -> WebsocketClientPolicy:
    last_error = None
    for _ in range(retry):
        try:
            return WebsocketClientPolicy(host=host, port=port)
        except Exception as exc:
            last_error = exc
            time.sleep(interval)
    raise RuntimeError(f"无法连接策略服务器 ws://{host}:{port}") from last_error
