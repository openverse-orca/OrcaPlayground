"""
ClothRobot test20260508 抓取回放（兼容 re-export）。

权威实现：``cloth_robot_scene_layout``（MJCF 扫描 + B 系位移 → Pico JSON）。
"""

from __future__ import annotations

from typing import Any

from .cloth_robot_scene_layout import (
    CLOSE_T0,
    CLOSE_T1,
    DURATION_SEC,
    OPEN_T0,
    OPEN_T1,
    grip_cmd_to_pico_trigger,
)


def cloth_center_from_session(session: dict[str, Any]) -> tuple[float, float, float]:
    """从 session 读取 ``cloth.center_yup``。"""
    cloth = session.get("cloth") or {}
    center = cloth.get("center_yup")
    if center and len(center) >= 3:
        return float(center[0]), float(center[1]), float(center[2])
    discovered = cloth.get("discovered_cloths") or []
    if discovered:
        c2 = discovered[0].get("center_yup")
        if c2 and len(c2) >= 3:
            return float(c2[0]), float(c2[1]), float(c2[2])
    raise ValueError("session JSON missing cloth.center_yup")


def load_cloth_center_from_session_path(path) -> tuple[float, float, float]:
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return cloth_center_from_session(data)


def trajectory_duration() -> float:
    return DURATION_SEC
