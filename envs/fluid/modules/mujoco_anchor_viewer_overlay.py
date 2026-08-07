"""
MuJoCo 被动查看器（launch_passive）叠加层：用近黑色几何体标出 SPH 锚点世界坐标，
与 MJCF 中默认显示的 SITE 框架区分。（rgba 近黑并带少量 emission，否则默认光照下纯黑球不可见。）
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def update_viewer_sph_anchor_markers(
    viewer_handle: Any,
    positions_xyz: Optional[Sequence[np.ndarray]],
    *,
    radius: float = 0.014,
) -> None:
    """
    将 ``viewer_handle.user_scn`` 中的用户几何体重置为若干黑色球体（SPH 锚点）。

    :param viewer_handle: ``mujoco.viewer.launch_passive`` 返回的 Handle
    :param positions_xyz: 世界系 (x,y,z) 列表；None 或空则清空叠加层
    :param radius: 球半径（米），略大于典型 SITE 线框以便辨认
    """
    if viewer_handle is None:
        return
    user_scn = getattr(viewer_handle, "user_scn", None)
    if user_scn is None:
        return

    import mujoco

    user_scn.ngeom = 0
    if not positions_xyz:
        return

    # 纯黑 rgba 在默认 headlight 下几乎不反射，球体「看不见」；略加 emission 仍保持黑色观感
    rgba = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float32)
    mat9 = np.eye(3, dtype=np.float64).reshape(9, 1)
    r = float(radius)
    size = np.array([[r], [0.0], [0.0]], dtype=np.float64)

    maxg = int(user_scn.maxgeom)
    for i, raw in enumerate(positions_xyz):
        if i >= maxg:
            break
        pos = np.asarray(raw, dtype=np.float64).reshape(3, 1)
        geom = user_scn.geoms[i]
        mujoco.mjv_initGeom(
            geom,
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            size,
            pos,
            mat9,
            rgba,
        )
        geom.emission = 0.35
        user_scn.ngeom += 1
