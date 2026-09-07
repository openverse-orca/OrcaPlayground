"""
程序化长方形布在 Y-up 下的边长与短轴角点（与 XPBD phys_world_add_sheet + X 轴 π/2 一致）。

sheet 局部：i∈[0,nx) 沿长轴 → 世界 X；j∈[0,ny) 沿短轴 → 世界 Z（j=0 为 Z_min）。
"""

from __future__ import annotations

CLOTH_SPACING_M = 0.0618


def sheet_axis_extents_m(
    nx: int, ny: int, spacing: float = CLOTH_SPACING_M
) -> tuple[float, float, float, float]:
    """返回 (x_neg_long, x_pos_long, z_short_min, z_short_max) 顶点列/行中心近似掌位。"""
    x_neg = -0.5 * nx * spacing
    x_pos = (nx - 1 - 0.5 * nx) * spacing
    z_min = 0.0
    z_max = (ny - 1) * spacing
    return x_neg, x_pos, z_min, z_max


def sheet_size_m(nx: int, ny: int, spacing: float = CLOTH_SPACING_M) -> tuple[float, float]:
    """长边 (nx-1)*spacing、短边 (ny-1)*spacing。"""
    return (nx - 1) * spacing, (ny - 1) * spacing
