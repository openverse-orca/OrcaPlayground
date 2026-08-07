"""
通用刚体「与 steer 同列」的 MuJoCo / PBD(sph_monitor) 跟踪。

- 耦合表：列与 steer_mjc_sph_couple.csv 相同（见 steer_coupling_recorder.CSV_HEADERS）。
  - 若提供 ``joint_name``：与方向盘相同，用关节轴与 qvel/qacc。
  - 若 ``joint_name`` 为空：用世界系 +Z 为 ``n_hat``，``mjc_omega_axis = ω_world·n_hat``，
    ``mjc_alpha_qacc`` 置 nan（``mjc_alpha_fd`` 仍由角速度差分得到）。
- 力矩平衡表：列与 steering_torque_balance.csv 相同（见 steering_torque_recorder.CSV_HEADERS）。
  - 无关节/执行器时：mjc_theta / mjc_ctrl / mjc_tau_motor / mjc_tau_damp / mjc_tau_fluid_recon /
    mjc_residual_A 等为 nan 或 0；I_eff 为刚体惯性绕 n 的投影（无 armature）。

启用（cup 示例）::

    export ORCA_CUP_COUPLE_CSV=.../cup_mjc_sph_couple.csv
    export ORCA_CUP_TORQUE_CSV=.../cup_torque_balance.csv
    # 可选：ORCA_CUP_TRACK_BODY（默认 cup_cup）、ORCA_CUP_TRACK_JOINT（默认空）
"""

from __future__ import annotations

import csv
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.fluid.csv_numeric import fmt_coupling_dict_row, fmt_f
from envs.fluid.launch.steer_coupling_recorder import (
    CSV_HEADERS as COUPLE_HEADERS,
    load_sph_monitor_by_cycle,
    _force_application_module,
    _joint_axis_world,
    _sph_quat_to_mjc_quat_wxyz,
    _sph_yup_to_mjc_zup,
    _tau_fluid_site_sum,
)
from envs.fluid.launch.steering_torque_recorder import (
    CSV_HEADERS as BALANCE_HEADERS,
    EPS_T,
    _format_balance_row_cells,
    _force_application_module as _fam_balance,
    _joint_axis_world as _joint_axis_world_b,
    _tau_fluid_site_sum as _tau_fluid_site_sum_b,
)

logger = logging.getLogger(__name__)


def _I_proj_world(mj: Any, d: Any, body_id: int, n_hat: np.ndarray) -> float:
    import mujoco

    q = mj.body_iquat[body_id].astype(np.float64).copy()
    r9 = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(r9, q)
    R_pb = r9.reshape(3, 3)
    diag = np.diag(mj.body_inertia[body_id].astype(np.float64))
    I_body = R_pb @ diag @ R_pb.T
    R_wb = d.xmat[body_id].reshape(3, 3)
    I_world = R_wb @ I_body @ R_wb.T
    return float(n_hat.T @ I_world @ n_hat)


class GenericCouplingRecorder:
    """与 SteerCouplingRecorder 同列；按 body + 可选关节记录。"""

    def __init__(self, path: str, body_name: str, joint_name: Optional[str], sph_body_name: str) -> None:
        self._path = path
        self.body_name = body_name
        self.joint_name = (joint_name or "").strip() or None
        self.sph_body_name = sph_body_name
        self._rows: List[Dict[str, Any]] = []
        self._prev_omega: Optional[float] = None
        self._prev_sim_t: Optional[float] = None

    @classmethod
    def from_env_cup(cls) -> Optional["GenericCouplingRecorder"]:
        if "ORCA_CUP_COUPLE_CSV" not in os.environ:
            return None
        raw = os.environ.get("ORCA_CUP_COUPLE_CSV")
        path = raw if raw else "/tmp/cup_mjc_sph_couple.csv"
        body = os.environ.get("ORCA_CUP_TRACK_BODY", "cup_cup").strip() or "cup_cup"
        joint = os.environ.get("ORCA_CUP_TRACK_JOINT", "").strip() or None
        sph = os.environ.get("ORCA_CUP_SPH_BODY", body).strip() or body
        return cls(path, body, joint, sph)

    def _sph_monitor_path(self) -> str:
        for key in ("ORCA_SPH_MONITOR_CSV", "ORCA_MONITOR_CSV"):
            p = os.environ.get(key, "").strip()
            if p and os.path.isfile(p):
                return p
        side = os.path.join(os.path.dirname(os.path.abspath(self._path)), "sph_monitor.csv")
        if os.path.isfile(side):
            return side
        return os.environ.get("ORCA_MONITOR_CSV", "/tmp/orca_sph_monitor.csv")

    def record_row(self, ctx: Any, env: Any, coupling_cycle: int) -> None:
        import mujoco

        uw = env.unwrapped
        mj = uw.gym._mjModel
        d = uw.gym._mjData
        mujoco.mj_forward(mj, d)

        body_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        if body_id < 0:
            logger.warning("generic couple CSV: missing body %s", self.body_name)
            return

        t_sim = float(d.time)
        pivot = d.xpos[body_id].astype(np.float64).copy()

        if self.joint_name:
            joint_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, self.joint_name)
            if joint_id < 0:
                logger.warning("generic couple CSV: missing joint %s", self.joint_name)
                return
            dof_adr = int(mj.jnt_dofadr[joint_id])
            n_hat = _joint_axis_world(mj, d, joint_id, body_id)
            omega = float(d.qvel[dof_adr])
            alpha_qacc = float(d.qacc[dof_adr])
        else:
            n_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            w = np.asarray(d.cvel[body_id], dtype=np.float64).reshape(6)[:3]
            omega = float(np.dot(w, n_hat))
            alpha_qacc = float("nan")

        dt_fd = float("nan")
        alpha_fd = float("nan")
        if self._prev_omega is not None and self._prev_sim_t is not None:
            dt = t_sim - self._prev_sim_t
            if dt > 1e-9:
                dt_fd = dt
                alpha_fd = (omega - self._prev_omega) / dt
        self._prev_omega = omega
        self._prev_sim_t = t_sim

        quat = np.asarray(d.xquat[body_id], dtype=np.float64).reshape(4).copy()

        xfrc = np.zeros(6, dtype=np.float64)
        if d.xfrc_applied is not None and body_id < mj.nbody:
            xfrc = np.asarray(d.xfrc_applied[body_id], dtype=np.float64).reshape(6).copy()

        fam = _force_application_module(ctx)
        site_forces: List[Tuple[str, np.ndarray]] = []
        if fam is not None and getattr(fam, "last_applied_site_forces", None):
            site_forces = list(fam.last_applied_site_forces)
        tau_fluid, fsum = _tau_fluid_site_sum(mj, d, body_id, n_hat, pivot, site_forces)

        row: Dict[str, Any] = {h: float("nan") for h in COUPLE_HEADERS}
        row["coupling_cycle"] = int(coupling_cycle)
        row["sim_time_mj"] = t_sim
        row["sph_cycle_matched"] = int(coupling_cycle)
        row["sph_sim_time"] = float("nan")
        row["valid_cycle_pair"] = 0
        row["dt_fd"] = dt_fd
        row["mjc_omega_axis"] = omega
        row["mjc_alpha_qacc"] = alpha_qacc
        row["mjc_alpha_fd"] = alpha_fd
        row["mjc_quat_w"] = float(quat[0])
        row["mjc_quat_x"] = float(quat[1])
        row["mjc_quat_y"] = float(quat[2])
        row["mjc_quat_z"] = float(quat[3])
        row["mjc_xfrc_fx"] = float(xfrc[0])
        row["mjc_xfrc_fy"] = float(xfrc[1])
        row["mjc_xfrc_fz"] = float(xfrc[2])
        row["mjc_xfrc_tx"] = float(xfrc[3])
        row["mjc_xfrc_ty"] = float(xfrc[4])
        row["mjc_xfrc_tz"] = float(xfrc[5])
        row["mjc_fluid_fx"] = float(fsum[0])
        row["mjc_fluid_fy"] = float(fsum[1])
        row["mjc_fluid_fz"] = float(fsum[2])
        row["mjc_tau_fluid_axis"] = float(tau_fluid)
        row["mjc_n_hat_x"] = float(n_hat[0])
        row["mjc_n_hat_y"] = float(n_hat[1])
        row["mjc_n_hat_z"] = float(n_hat[2])
        self._rows.append(row)

    def close(self) -> None:
        pass

    def post_merge(self) -> None:
        sph_path = self._sph_monitor_path()
        idx = load_sph_monitor_by_cycle(sph_path, self.sph_body_name)
        if not idx:
            logger.warning("generic couple post_merge: 无 SPH 索引 (%s) body=%s", sph_path, self.sph_body_name)

        def _f(row: Dict[str, str], key: str) -> float:
            try:
                return float(row.get(key, "nan"))
            except (TypeError, ValueError):
                return float("nan")

        prev_sph_omega: Optional[float] = None
        prev_sph_t: Optional[float] = None

        for row in self._rows:
            cyc = int(row["coupling_cycle"])
            anchors = idx.get(cyc)
            if not anchors or len(anchors) < 1:
                row["valid_cycle_pair"] = 0
                continue

            row["valid_cycle_pair"] = 1
            a0 = anchors.get(0)
            if a0 is None:
                a0 = next(iter(anchors.values()))
            row["sph_sim_time"] = _f(a0, "sim_time")

            n_hat = np.array(
                [
                    float(row["mjc_n_hat_x"]),
                    float(row["mjc_n_hat_y"]),
                    float(row["mjc_n_hat_z"]),
                ],
                dtype=np.float64,
            )
            nn = float(np.linalg.norm(n_hat))
            if nn > 1e-12:
                n_hat = n_hat / nn

            ang = np.array(
                [_f(a0, "pbd_angvel_x"), _f(a0, "pbd_angvel_y"), _f(a0, "pbd_angvel_z")]
            )
            sph_w = float("nan")
            if not np.any(np.isnan(ang)):
                ang_mj = _sph_yup_to_mjc_zup(ang)
                sph_w = float(np.dot(ang_mj, n_hat))
                row["sph_omega_axis"] = sph_w

            sph_t = _f(a0, "sim_time")
            if prev_sph_omega is not None and prev_sph_t is not None and not math.isnan(sph_w):
                ds = sph_t - prev_sph_t
                if ds > 1e-9:
                    row["sph_alpha_fd"] = (sph_w - prev_sph_omega) / ds
                    if math.isnan(float(row.get("dt_fd", float("nan")))) or float(row["dt_fd"]) <= 0:
                        row["dt_fd"] = ds
            if not math.isnan(sph_w):
                prev_sph_omega = sph_w
            prev_sph_t = sph_t

            bf = np.array(
                [_f(a0, "sph_body_force_x"), _f(a0, "sph_body_force_y"), _f(a0, "sph_body_force_z")]
            )
            if not np.any(np.isnan(bf)):
                bf_mj = _sph_yup_to_mjc_zup(bf)
                row["sph_body_fx"] = float(bf_mj[0])
                row["sph_body_fy"] = float(bf_mj[1])
                row["sph_body_fz"] = float(bf_mj[2])

            bt = np.array(
                [_f(a0, "sph_body_torque_x"), _f(a0, "sph_body_torque_y"), _f(a0, "sph_body_torque_z")]
            )
            if not np.any(np.isnan(bt)):
                bt_mj = _sph_yup_to_mjc_zup(bt)
                row["sph_body_tx"] = float(bt_mj[0])
                row["sph_body_ty"] = float(bt_mj[1])
                row["sph_body_tz"] = float(bt_mj[2])
                row["sph_body_torque_axis"] = float(np.dot(bt_mj, n_hat))

            ff = np.array(
                [_f(a0, "sph_fluid_force_x"), _f(a0, "sph_fluid_force_y"), _f(a0, "sph_fluid_force_z")]
            )
            if not np.any(np.isnan(ff)):
                ff_mj = _sph_yup_to_mjc_zup(ff)
                row["sph_fluid_fx"] = float(ff_mj[0])
                row["sph_fluid_fy"] = float(ff_mj[1])
                row["sph_fluid_fz"] = float(ff_mj[2])

            for aid in range(4):
                ar = anchors.get(aid)
                if ar is None:
                    continue
                sf = np.array(
                    [
                        _f(ar, "sph_spring_force_x"),
                        _f(ar, "sph_spring_force_y"),
                        _f(ar, "sph_spring_force_z"),
                    ]
                )
                if not np.any(np.isnan(sf)):
                    sf_mj = _sph_yup_to_mjc_zup(sf)
                    row[f"sph_pd_f{aid}_x"] = float(sf_mj[0])
                    row[f"sph_pd_f{aid}_y"] = float(sf_mj[1])
                    row[f"sph_pd_f{aid}_z"] = float(sf_mj[2])
                if ar.get("pd_torque_com_x") not in (None, ""):
                    pt = np.array(
                        [
                            _f(ar, "pd_torque_com_x"),
                            _f(ar, "pd_torque_com_y"),
                            _f(ar, "pd_torque_com_z"),
                        ]
                    )
                    if not np.all(np.isnan(pt)):
                        pt_mj = _sph_yup_to_mjc_zup(pt)
                        row[f"sph_pd_tau{aid}_x"] = float(pt_mj[0])
                        row[f"sph_pd_tau{aid}_y"] = float(pt_mj[1])
                        row[f"sph_pd_tau{aid}_z"] = float(pt_mj[2])

            qw, qx, qy, qz = _f(a0, "pbd_quat_w"), _f(a0, "pbd_quat_x"), _f(a0, "pbd_quat_y"), _f(
                a0, "pbd_quat_z"
            )
            if not any(math.isnan(v) for v in (qw, qx, qy, qz)):
                row["sph_quat_w"] = qw
                row["sph_quat_x"] = qx
                row["sph_quat_y"] = qy
                row["sph_quat_z"] = qz
                wm, xm, ym, zm = _sph_quat_to_mjc_quat_wxyz(qw, qx, qy, qz)
                row["sph_quat_mjc_w"] = wm
                row["sph_quat_mjc_x"] = xm
                row["sph_quat_mjc_y"] = ym
                row["sph_quat_mjc_z"] = zm

        absp = os.path.abspath(self._path)
        os.makedirs(os.path.dirname(absp) or ".", exist_ok=True)
        with open(self._path, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=COUPLE_HEADERS)
            w.writeheader()
            for r in self._rows:
                w.writerow(fmt_coupling_dict_row(r, COUPLE_HEADERS))
        valid = sum(1 for r in self._rows if int(r.get("valid_cycle_pair", 0)) == 1)
        logger.info(
            "generic couple CSV (%s): %s rows=%d valid_cycle=%d sph=%s",
            self.body_name,
            absp,
            len(self._rows),
            valid,
            sph_path,
        )


class _SphMonitorReaderByBody:
    def __init__(self, sph_monitor_path: Optional[str], sph_body_name: str) -> None:
        self._path = sph_monitor_path
        self.sph_body_name = sph_body_name
        self._last_offset: int = 0
        self._header_read: bool = False
        self._header: List[str] = []
        self._cache: Dict[float, Dict[str, Any]] = {}
        self._max_cache: int = 600

    def _try_read_new_rows(self) -> None:
        if self._path is None or not os.path.isfile(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                if not self._header_read:
                    header_line = f.readline()
                    if not header_line:
                        return
                    self._header = header_line.strip().split(",")
                    self._last_offset = f.tell()
                    self._header_read = True
                f.seek(self._last_offset)
                new_content = f.read()
                self._last_offset = f.tell()
            if not new_content.strip():
                return
            for line in new_content.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != len(self._header):
                    continue
                row = dict(zip(self._header, parts))
                try:
                    body_name = row.get("body_name", "")
                    if body_name != self.sph_body_name:
                        continue
                    anchor_idx = int(float(row.get("anchor_idx", "-1")))
                    if anchor_idx != 0:
                        continue
                    sim_time = float(row.get("sim_time", "nan"))
                    if np.isnan(sim_time):
                        continue
                    self._cache[sim_time] = row
                    if len(self._cache) > self._max_cache:
                        oldest = min(self._cache.keys())
                        del self._cache[oldest]
                except (ValueError, KeyError):
                    continue
        except OSError:
            pass

    def get_sph_data(self, t_sim: float) -> Optional[Dict[str, Any]]:
        self._try_read_new_rows()
        best_t: Optional[float] = None
        best_err = float("inf")
        for t_cached in self._cache:
            err = abs(t_cached - t_sim)
            if err < best_err:
                best_err = err
                best_t = t_cached
        if best_t is not None and best_err <= 0.03:
            return self._cache[best_t]
        return None

    def read_all(self) -> Dict[float, Dict[str, Any]]:
        self._cache.clear()
        self._header_read = False
        self._last_offset = 0
        self._try_read_new_rows()
        return dict(self._cache)


class GenericTorqueBalanceRecorder:
    """与 SteeringTorqueRecorder 同列；可选关节+执行器。"""

    def __init__(
        self,
        path: str,
        body_name: str,
        joint_name: Optional[str],
        actuator_name: Optional[str],
        sph_body_name: str,
    ) -> None:
        self._path = path
        self.body_name = body_name
        self.joint_name = (joint_name or "").strip() or None
        self.actuator_name = (actuator_name or "").strip() or None
        self.sph_body_name = sph_body_name
        self._fp: Optional[Any] = None
        self._writer: Optional[Any] = None
        self._opened = False
        sph_monitor_path = os.environ.get("ORCA_SPH_MONITOR_CSV", "")
        if not sph_monitor_path:
            sph_monitor_path = os.path.join(os.path.dirname(path), "sph_monitor.csv")
            if not os.path.isfile(sph_monitor_path):
                sph_monitor_path = "/tmp/orca_sph_monitor.csv"
        self._sph_reader = _SphMonitorReaderByBody(sph_monitor_path, sph_body_name)
        self._prev_sph_omega: Optional[float] = None
        self._prev_sph_t: Optional[float] = None
        self._prev_mjc_omega: Optional[float] = None
        self._prev_mjc_t: Optional[float] = None

    @classmethod
    def from_env_cup(cls) -> Optional["GenericTorqueBalanceRecorder"]:
        if "ORCA_CUP_TORQUE_CSV" not in os.environ:
            return None
        raw = os.environ.get("ORCA_CUP_TORQUE_CSV")
        path = raw if raw else "/tmp/cup_torque_balance.csv"
        body = os.environ.get("ORCA_CUP_TRACK_BODY", "cup_cup").strip() or "cup_cup"
        joint = os.environ.get("ORCA_CUP_TRACK_JOINT", "").strip() or None
        act = os.environ.get("ORCA_CUP_TRACK_ACTUATOR", "").strip() or None
        sph = os.environ.get("ORCA_CUP_SPH_BODY", body).strip() or body
        return cls(path, body, joint, act, sph)

    def _ensure_open(self) -> None:
        if self._opened:
            return
        absp = os.path.abspath(self._path)
        dirp = os.path.dirname(absp)
        if dirp:
            os.makedirs(dirp, exist_ok=True)
        self._fp = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fp)
        self._writer.writerow(BALANCE_HEADERS)
        self._fp.flush()
        self._opened = True
        logger.info("generic torque CSV (%s): %s", self.body_name, os.path.abspath(self._path))

    def close(self) -> None:
        if self._fp is not None:
            try:
                self._fp.flush()
                self._fp.close()
            except OSError:
                pass
            self._fp = None
            self._writer = None

    def _extract_sph_data(self, t_sim: float, n_hat_mjc: np.ndarray) -> Dict[str, float]:
        result = {
            "sph_omega": float("nan"),
            "sph_alpha": float("nan"),
            "sph_tau_fluid": float("nan"),
            "sph_tau_spring": float("nan"),
            "sph_residual_B": float("nan"),
            "delta_C": float("nan"),
            "sph_ffx": float("nan"),
            "sph_ffy": float("nan"),
            "sph_ffz": float("nan"),
        }
        sph_row = self._sph_reader.get_sph_data(t_sim)
        if sph_row is None:
            return result

        try:
            angvel_sph = np.array([
                float(sph_row.get("pbd_angvel_x", "nan")),
                float(sph_row.get("pbd_angvel_y", "nan")),
                float(sph_row.get("pbd_angvel_z", "nan")),
            ])
            if not np.any(np.isnan(angvel_sph)):
                angvel_mjc = _sph_yup_to_mjc_zup(angvel_sph)
                result["sph_omega"] = float(np.dot(angvel_mjc, n_hat_mjc))

            sph_t = float(sph_row.get("sim_time", "nan"))
            if self._prev_sph_omega is not None and self._prev_sph_t is not None:
                dt = sph_t - self._prev_sph_t
                if dt > 1e-9:
                    result["sph_alpha"] = (result["sph_omega"] - self._prev_sph_omega) / dt
            self._prev_sph_omega = result["sph_omega"]
            self._prev_sph_t = sph_t

            body_torque_sph = np.array([
                float(sph_row.get("sph_body_torque_x", "nan")),
                float(sph_row.get("sph_body_torque_y", "nan")),
                float(sph_row.get("sph_body_torque_z", "nan")),
            ])
            if not np.any(np.isnan(body_torque_sph)):
                body_torque_mjc = _sph_yup_to_mjc_zup(body_torque_sph)
                result["sph_tau_fluid"] = float(np.dot(body_torque_mjc, n_hat_mjc))

            fluid_force_sph = np.array([
                float(sph_row.get("sph_fluid_force_x", "nan")),
                float(sph_row.get("sph_fluid_force_y", "nan")),
                float(sph_row.get("sph_fluid_force_z", "nan")),
            ])
            if not np.any(np.isnan(fluid_force_sph)):
                fluid_force_mjc = _sph_yup_to_mjc_zup(fluid_force_sph)
                result["sph_ffx"] = float(fluid_force_mjc[0])
                result["sph_ffy"] = float(fluid_force_mjc[1])
                result["sph_ffz"] = float(fluid_force_mjc[2])

            spring_force_sph = np.array([
                float(sph_row.get("sph_spring_force_x", "nan")),
                float(sph_row.get("sph_spring_force_y", "nan")),
                float(sph_row.get("sph_spring_force_z", "nan")),
            ])
            if not np.any(np.isnan(spring_force_sph)):
                result["sph_tau_spring"] = float("nan")

        except (ValueError, KeyError):
            pass

        return result

    def record_row(self, ctx: Any, env: Any, mj_cycle_idx: int) -> None:
        import mujoco

        self._ensure_open()
        assert self._writer is not None

        uw = env.unwrapped
        mj = uw.gym._mjModel
        d = uw.gym._mjData

        mujoco.mj_forward(mj, d)

        body_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        if body_id < 0:
            logger.warning("generic torque CSV: missing body %s", self.body_name)
            return

        pivot = d.xpos[body_id].astype(np.float64).copy()
        t_sim = float(d.time)

        if self.joint_name and self.actuator_name:
            joint_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, self.joint_name)
            aid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, self.actuator_name)
            if joint_id < 0 or aid < 0:
                logger.warning("generic torque: missing joint/actuator")
                return
            dof_adr = int(mj.jnt_dofadr[joint_id])
            qpos_adr = int(mj.jnt_qposadr[joint_id])
            n_hat = _joint_axis_world_b(mj, d, joint_id, body_id)
            theta = float(d.qpos[qpos_adr])
            omega = float(d.qvel[dof_adr])
            alpha = float(d.qacc[dof_adr])
            ctrl = float(d.ctrl[aid])
            damp_c = float(mj.dof_damping[dof_adr])
            tau_damp = -damp_c * omega
            tau_motor = float(d.actuator_force[aid])
            from envs.fluid.launch.steering_torque_recorder import _I_eff_about_axis

            I_eff = _I_eff_about_axis(mj, d, body_id, dof_adr, n_hat)
        else:
            n_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            theta = float("nan")
            w = np.asarray(d.cvel[body_id], dtype=np.float64).reshape(6)[:3]
            omega = float(np.dot(w, n_hat))
            alpha = float("nan")
            if self._prev_mjc_omega is not None and self._prev_mjc_t is not None:
                dt = t_sim - self._prev_mjc_t
                if dt > 1e-9:
                    alpha = (omega - self._prev_mjc_omega) / dt
            self._prev_mjc_omega = omega
            self._prev_mjc_t = t_sim
            ctrl = float("nan")
            tau_damp = 0.0
            tau_motor = float("nan")
            I_eff = _I_proj_world(mj, d, body_id, n_hat)

        fam = _fam_balance(ctx)
        site_forces: List[Tuple[str, np.ndarray]] = []
        if fam is not None and getattr(fam, "last_applied_site_forces", None):
            site_forces = list(fam.last_applied_site_forces)

        tau_fluid, fsum = _tau_fluid_site_sum_b(mj, d, body_id, n_hat, pivot, site_forces)

        if self.joint_name and self.actuator_name:
            tau_recon = I_eff * alpha - tau_motor - tau_damp
            resid_a = I_eff * alpha - (tau_motor + tau_damp + tau_fluid)
        else:
            tau_recon = float("nan")
            resid_a = float("nan")

        t_sph = float("nan")
        t_err = float("nan")
        valid = 0
        olc = getattr(getattr(ctx, "sph_wrapper", None), "orcalink_client", None)
        if olc is not None and hasattr(olc, "_last_received_sph_sim_time"):
            try:
                t_sph = float(olc._last_received_sph_sim_time)
                t_err = t_sim - t_sph
                if abs(t_err) <= EPS_T:
                    valid = 1
            except (TypeError, ValueError):
                pass

        sph_data = self._extract_sph_data(t_sim, n_hat)
        sph_omega = sph_data["sph_omega"]
        sph_alpha = sph_data["sph_alpha"]
        sph_tau_fluid = sph_data["sph_tau_fluid"]
        sph_tau_spring = sph_data["sph_tau_spring"]
        sph_ffx = sph_data["sph_ffx"]
        sph_ffy = sph_data["sph_ffy"]
        sph_ffz = sph_data["sph_ffz"]

        sph_resid_b = float("nan")
        if not np.isnan(sph_alpha) and not np.isnan(sph_tau_fluid) and not np.isnan(sph_tau_spring):
            sph_resid_b = I_eff * sph_alpha - (sph_tau_fluid + sph_tau_spring)

        delta_c = float("nan")
        if not np.isnan(sph_tau_fluid) and not np.isnan(tau_fluid):
            delta_c = sph_tau_fluid - tau_fluid

        row = [
            t_sim,
            t_sph,
            t_err,
            valid,
            mj_cycle_idx,
            theta,
            omega,
            alpha,
            ctrl,
            tau_motor,
            tau_damp,
            I_eff,
            tau_fluid,
            tau_recon,
            resid_a,
            float(n_hat[0]),
            float(n_hat[1]),
            float(n_hat[2]),
            float(fsum[0]),
            float(fsum[1]),
            float(fsum[2]),
            sph_omega,
            sph_alpha,
            sph_tau_fluid,
            sph_tau_spring,
            sph_resid_b,
            delta_c,
            sph_ffx,
            sph_ffy,
            sph_ffz,
        ]
        self._writer.writerow(_format_balance_row_cells(row))
        if mj_cycle_idx % 25 == 0:
            self._fp.flush()

    def post_merge(self) -> None:
        """与 SteeringTorqueRecorder.post_merge 相同逻辑，按本跟踪 body 重读 sph_monitor。"""
        self.close()

        sph_all = self._sph_reader.read_all()
        if not sph_all:
            logger.warning("generic post_merge: sph_monitor 无有效数据 body=%s", self.sph_body_name)
            return

        mj_rows: List[Dict[str, str]] = []
        with open(self._path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                mj_rows.append(r)

        if not mj_rows:
            return

        n_hat = np.array([
            float(mj_rows[-1]["mjc_n_hat_x"]),
            float(mj_rows[-1]["mjc_n_hat_y"]),
            float(mj_rows[-1]["mjc_n_hat_z"]),
        ])

        sph_times = sorted(sph_all.keys())

        def _find_closest_sph(t_mj: float) -> Optional[Dict[str, Any]]:
            best_t = None
            best_err = float("inf")
            for t_s in sph_times:
                err = abs(t_s - t_mj)
                if err < best_err:
                    best_err = err
                    best_t = t_s
            if best_t is not None and best_err <= 0.03:
                return sph_all[best_t]
            return None

        prev_omega: Optional[float] = None
        prev_t: Optional[float] = None
        updated_rows: List[List[str]] = []

        sph_idx_map = {
            "sph_omega": 21,
            "sph_alpha": 22,
            "sph_tau_fluid": 23,
            "sph_tau_spring": 24,
            "sph_residual_B": 25,
            "delta_C": 26,
            "sph_ffx": 27,
            "sph_ffy": 28,
            "sph_ffz": 29,
        }

        for r in mj_rows:
            t_mj = float(r["sim_time"])
            sph_row = _find_closest_sph(t_mj)

            sph_omega = float("nan")
            sph_alpha = float("nan")
            sph_tau_fluid = float("nan")
            sph_tau_spring = float("nan")
            sph_ffx = float("nan")
            sph_ffy = float("nan")
            sph_ffz = float("nan")

            if sph_row is not None:
                try:
                    angvel_sph = np.array([
                        float(sph_row.get("pbd_angvel_x", "nan")),
                        float(sph_row.get("pbd_angvel_y", "nan")),
                        float(sph_row.get("pbd_angvel_z", "nan")),
                    ])
                    if not np.any(np.isnan(angvel_sph)):
                        angvel_mjc = _sph_yup_to_mjc_zup(angvel_sph)
                        sph_omega = float(np.dot(angvel_mjc, n_hat))

                    sph_t = float(sph_row.get("sim_time", "nan"))
                    if prev_omega is not None and prev_t is not None:
                        dt = sph_t - prev_t
                        if dt > 1e-9:
                            sph_alpha = (sph_omega - prev_omega) / dt
                    prev_omega = sph_omega
                    prev_t = sph_t

                    body_torque_sph = np.array([
                        float(sph_row.get("sph_body_torque_x", "nan")),
                        float(sph_row.get("sph_body_torque_y", "nan")),
                        float(sph_row.get("sph_body_torque_z", "nan")),
                    ])
                    if not np.any(np.isnan(body_torque_sph)):
                        body_torque_mjc = _sph_yup_to_mjc_zup(body_torque_sph)
                        sph_tau_fluid = float(np.dot(body_torque_mjc, n_hat))

                    fluid_force_sph = np.array([
                        float(sph_row.get("sph_fluid_force_x", "nan")),
                        float(sph_row.get("sph_fluid_force_y", "nan")),
                        float(sph_row.get("sph_fluid_force_z", "nan")),
                    ])
                    if not np.any(np.isnan(fluid_force_sph)):
                        fluid_force_mjc = _sph_yup_to_mjc_zup(fluid_force_sph)
                        sph_ffx = float(fluid_force_mjc[0])
                        sph_ffy = float(fluid_force_mjc[1])
                        sph_ffz = float(fluid_force_mjc[2])

                    spring_force_sph = np.array([
                        float(sph_row.get("sph_spring_force_x", "nan")),
                        float(sph_row.get("sph_spring_force_y", "nan")),
                        float(sph_row.get("sph_spring_force_z", "nan")),
                    ])
                    if not np.any(np.isnan(spring_force_sph)):
                        sph_tau_spring = float("nan")
                except (ValueError, KeyError):
                    pass

            I_eff = float(r["mjc_I_eff"])
            mjc_tau_fluid = float(r["mjc_tau_fluid_axis"])

            sph_resid_b = float("nan")
            if not np.isnan(sph_alpha) and not np.isnan(sph_tau_fluid) and not np.isnan(sph_tau_spring):
                sph_resid_b = I_eff * sph_alpha - (sph_tau_fluid + sph_tau_spring)

            delta_c = float("nan")
            if not np.isnan(sph_tau_fluid) and not np.isnan(mjc_tau_fluid):
                delta_c = sph_tau_fluid - mjc_tau_fluid

            row_list = list(r.values())
            row_list[sph_idx_map["sph_omega"]] = str(sph_omega)
            row_list[sph_idx_map["sph_alpha"]] = str(sph_alpha)
            row_list[sph_idx_map["sph_tau_fluid"]] = str(sph_tau_fluid)
            row_list[sph_idx_map["sph_tau_spring"]] = str(sph_tau_spring)
            row_list[sph_idx_map["sph_residual_B"]] = str(sph_resid_b)
            row_list[sph_idx_map["delta_C"]] = str(delta_c)
            row_list[sph_idx_map["sph_ffx"]] = str(sph_ffx)
            row_list[sph_idx_map["sph_ffy"]] = str(sph_ffy)
            row_list[sph_idx_map["sph_ffz"]] = str(sph_ffz)
            updated_rows.append(row_list)

        with open(self._path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(BALANCE_HEADERS)
            writer.writerows([_format_balance_row_cells(r) for r in updated_rows])

        sph_valid = sum(1 for row in updated_rows if row[sph_idx_map["sph_omega"]] != "nan")
        logger.info(
            "generic post_merge (%s): 合并完成，%d/%d 行有 SPH 数据",
            self.body_name,
            sph_valid,
            len(updated_rows),
        )
