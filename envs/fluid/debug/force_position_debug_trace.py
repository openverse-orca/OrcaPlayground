from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Set
import csv
import time

import numpy as np


@dataclass
class ForcePositionDebugTrace:
    session_dir: Path
    body_aliases: Dict[str, str] = field(default_factory=dict)
    surface_sites: Dict[str, Dict] = field(default_factory=dict)
    _logged_cp1_seq: Set[int] = field(default_factory=set)
    _logged_cp5_seq: Set[int] = field(default_factory=set)
    _logged_cp6_macro: Set[int] = field(default_factory=set)
    _cp1_csv_path: Optional[Path] = None
    _cp5_csv_path: Optional[Path] = None
    _cp6_csv_path: Optional[Path] = None
    _surface_csv_path: Optional[Path] = None

    def should_log_cp1(self, publish_seq: int) -> bool:
        return publish_seq not in self._logged_cp1_seq

    def should_log_cp5(self, force_seq: int) -> bool:
        return force_seq not in self._logged_cp5_seq

    def should_log_cp6(self, macro_step: int) -> bool:
        return macro_step not in self._logged_cp6_macro

    def _ensure_surface_csv(self) -> None:
        if self._surface_csv_path is None:
            self._surface_csv_path = self.session_dir / "cp_surface_world.csv"
            if not self._surface_csv_path.exists():
                with self._surface_csv_path.open("w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [
                            "macro_step",
                            "object_id",
                            "alias",
                            "source",
                            "site_name",
                            "coord_frame",
                            "px",
                            "py",
                            "pz",
                            "time_s",
                        ]
                    )

    def log_surface_row(
        self,
        macro_step: int,
        object_id: str,
        source: str,
        site_name: str,
        coord_frame: str,
        pos: np.ndarray,
        time_s: float,
    ) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_surface_csv()
        alias = self.body_aliases.get(object_id, object_id)
        with self._surface_csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    macro_step,
                    object_id,
                    alias,
                    source,
                    site_name,
                    coord_frame,
                    f"{float(pos[0]):.4f}",
                    f"{float(pos[1]):.4f}",
                    f"{float(pos[2]):.4f}",
                    f"{time_s:.6f}",
                ]
            )

    def log_cp1_pre_publish(self, publish_seq: int, positions: Iterable, env=None) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self._cp1_csv_path is None:
            self._cp1_csv_path = self.session_dir / "cp1_pre_publish.csv"
            if not self._cp1_csv_path.exists():
                with self._cp1_csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "publish_seq",
                            "object_id",
                            "alias",
                            "px",
                            "py",
                            "pz",
                            "vx",
                            "vy",
                            "vz",
                            "qw",
                            "qx",
                            "qy",
                            "qz",
                            "t1",
                        ]
                    )

        t1 = time.time()
        with self._cp1_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for p in positions:
                object_id = getattr(p, "object_id", "")
                alias = self.body_aliases.get(object_id, object_id)
                pos = getattr(p, "position", [0.0, 0.0, 0.0])
                vel = getattr(p, "linear_velocity", [0.0, 0.0, 0.0])
                quat = getattr(p, "rotation", [1.0, 0.0, 0.0, 0.0])
                writer.writerow(
                    [
                        publish_seq,
                        object_id,
                        alias,
                        f"{float(pos[0]):.4f}",
                        f"{float(pos[1]):.4f}",
                        f"{float(pos[2]):.4f}",
                        f"{float(vel[0]):.4f}",
                        f"{float(vel[1]):.4f}",
                        f"{float(vel[2]):.4f}",
                        f"{float(quat[0]):.4f}",
                        f"{float(quat[1]):.4f}",
                        f"{float(quat[2]):.4f}",
                        f"{float(quat[3]):.4f}",
                        f"{t1:.6f}",
                    ]
                )

                if env is not None and object_id in self.surface_sites:
                    from .surface_site_registry import (
                        query_site_world_mjc,
                        surface_world_from_body_pose,
                    )

                    spec = self.surface_sites[object_id]
                    site_name = spec["site_name"]
                    surf = query_site_world_mjc(env, site_name)
                    if surf is None:
                        surf = surface_world_from_body_pose(
                            np.asarray(pos, dtype=float),
                            np.asarray(quat, dtype=float),
                            np.asarray(spec["local_pos_mjc"], dtype=float),
                        )
                    self.log_surface_row(
                        publish_seq,
                        object_id,
                        "mjc",
                        site_name,
                        "z_up",
                        surf,
                        t1,
                    )

    def mark_cp1_logged(self, publish_seq: int) -> None:
        self._logged_cp1_seq.add(publish_seq)

    def log_cp5_after_force_apply(
        self,
        force_seq: int,
        macro_step: int,
        forces: Iterable,
        applied_flags: Dict[str, bool],
    ) -> None:
        """
        CP5：SPH 经 OrcaLink 下发的 FORCE 帧被 MuJoCo 订阅并尝试写入动力学内存之后。

        记录每个刚体的 force/torque（Z-up）及 ``applied``（是否成功调用 apply API）。
        """
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self._cp5_csv_path is None:
            self._cp5_csv_path = self.session_dir / "cp5_after_force_apply.csv"
            if not self._cp5_csv_path.exists():
                with self._cp5_csv_path.open("w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [
                            "force_seq",
                            "macro_step",
                            "object_id",
                            "alias",
                            "fx",
                            "fy",
                            "fz",
                            "tx",
                            "ty",
                            "tz",
                            "applied",
                            "t5",
                        ]
                    )

        t5 = time.time()
        with self._cp5_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for force_data in forces:
                object_id = getattr(force_data, "object_id", "")
                alias = self.body_aliases.get(object_id, object_id)
                force = getattr(force_data, "force", [0.0, 0.0, 0.0])
                torque = getattr(force_data, "torque", [0.0, 0.0, 0.0])
                applied = 1 if applied_flags.get(object_id, False) else 0
                writer.writerow(
                    [
                        force_seq,
                        macro_step,
                        object_id,
                        alias,
                        f"{float(force[0]):.4f}",
                        f"{float(force[1]):.4f}",
                        f"{float(force[2]):.4f}",
                        f"{float(torque[0]):.4f}",
                        f"{float(torque[1]):.4f}",
                        f"{float(torque[2]):.4f}",
                        applied,
                        f"{t5:.6f}",
                    ]
                )
        self._logged_cp5_seq.add(force_seq)

    def log_cp6_pre_next_publish(
        self,
        macro_step: int,
        next_publish_seq: int,
        positions: Iterable,
    ) -> None:
        """
        CP6：本宏步 mj_step×N 完成后、下一轮 publish_positions（CP1）之前。

        采集即将发出的 POSITION 包内容（Z-up）；``next_publish_seq`` 应对齐下一轮 CP1 的 ``publish_seq``。
        """
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self._cp6_csv_path is None:
            self._cp6_csv_path = self.session_dir / "cp6_pre_next_publish.csv"
            if not self._cp6_csv_path.exists():
                with self._cp6_csv_path.open("w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [
                            "macro_step",
                            "next_publish_seq",
                            "object_id",
                            "alias",
                            "px",
                            "py",
                            "pz",
                            "vx",
                            "vy",
                            "vz",
                            "qw",
                            "qx",
                            "qy",
                            "qz",
                            "t6",
                        ]
                    )

        t6 = time.time()
        with self._cp6_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for p in positions:
                object_id = getattr(p, "object_id", "")
                alias = self.body_aliases.get(object_id, object_id)
                pos = getattr(p, "position", [0.0, 0.0, 0.0])
                vel = getattr(p, "linear_velocity", [0.0, 0.0, 0.0])
                quat = getattr(p, "rotation", [1.0, 0.0, 0.0, 0.0])
                writer.writerow(
                    [
                        macro_step,
                        next_publish_seq,
                        object_id,
                        alias,
                        f"{float(pos[0]):.4f}",
                        f"{float(pos[1]):.4f}",
                        f"{float(pos[2]):.4f}",
                        f"{float(vel[0]):.4f}",
                        f"{float(vel[1]):.4f}",
                        f"{float(vel[2]):.4f}",
                        f"{float(quat[0]):.4f}",
                        f"{float(quat[1]):.4f}",
                        f"{float(quat[2]):.4f}",
                        f"{float(quat[3]):.4f}",
                        f"{t6:.6f}",
                    ]
                )
        self._logged_cp6_macro.add(macro_step)


_ACTIVE_TRACE: Optional[ForcePositionDebugTrace] = None


def init_session(
    session_dir: Path,
    body_aliases: Dict[str, str],
    surface_sites: Optional[Dict[str, Dict]] = None,
) -> ForcePositionDebugTrace:
    global _ACTIVE_TRACE
    _ACTIVE_TRACE = ForcePositionDebugTrace(
        session_dir=Path(session_dir),
        body_aliases=dict(body_aliases or {}),
        surface_sites=dict(surface_sites or {}),
    )
    return _ACTIVE_TRACE


def get_active_trace() -> Optional[ForcePositionDebugTrace]:
    return _ACTIVE_TRACE
