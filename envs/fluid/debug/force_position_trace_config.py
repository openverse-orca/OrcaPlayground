"""force_position 调试采集开关：仅 build_mode=debug 且配置 enabled 时开启。"""

from __future__ import annotations

import os
from typing import Any, Dict


ORCA_FP_ENV_KEYS = (
    "ORCA_FP_DEBUG_DIR",
    "ORCA_FP_DEBUG_SUBSTEPS",
    "ORCA_FP_SUBSTEP_TRACE",
    "ORCA_FP_SUBSTEP_MAX_MACROS",
    "ORCA_FP_DEBUG_ALIASES",
)


def clear_force_position_debug_env() -> None:
    """清除 SPH 侧 CP2–CP4 / 子步 trace 依赖的环境变量（release 或未启用时调用）。"""
    for key in ORCA_FP_ENV_KEYS:
        os.environ.pop(key, None)


def is_force_position_trace_enabled(config: Dict[str, Any]) -> bool:
    """
    判断是否启用 force_position 宏步/子步调试采集。

    须同时满足：
    - `build_mode` 为 ``debug``（``release`` 时强制关闭，忽略 trace.enabled）；
    - `debug.force_position_trace.enabled` 为 true。
    """
    build_mode = str(config.get("build_mode", "debug")).strip().lower()
    if build_mode != "debug":
        return False
    return bool(config.get("debug", {}).get("force_position_trace", {}).get("enabled", False))
