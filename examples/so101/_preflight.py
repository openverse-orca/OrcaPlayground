from __future__ import annotations

import socket


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def require_open_port(host: str, port: int, label: str, extra_hint: str | None = None) -> None:
    if not is_port_open(host, port):
        message = f"{label} 未启动：{host}:{port}"
        if extra_hint:
            message += f"\n{extra_hint}"
        raise RuntimeError(message)
