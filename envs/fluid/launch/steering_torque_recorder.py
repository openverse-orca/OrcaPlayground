"""
steering_box 力矩平衡 CSV

- 每成功执行一次 env.step（frame_skip=20, time_step=0.001 → Δt_cycle=0.02s）写一行。
- sim_time 使用 mjData.time；与 SPH 时间戳对齐见 valid_time_pair。
- SPH 侧数据：仿真期间尝试实时尾读 sph_monitor.csv；仿真结束后调用 post_merge() 做后处理合并。
- 坐标转换：SPH Y-up [x,y,z] → MuJoCo Z-up [x,-z,y]

启用：设置环境变量 ORCA_STEERING_TORQUE_CSV（可为空字符串，表示默认路径
/tmp/steering_torque_balance.csv）。
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from envs.fluid.csv_numeric import fmt_f

logger = logging.getLogger(__name__)

JOINT_NAME = "steering__joint_1_3"
BODY_NAME = "steering_box_body_3"
ACTUATOR_NAME = "steering_box_3_motor"
SPH_BODY_NAME = "steering_box_body_3"
EPS_T = 1e-6

_INT_BALANCE_HEADERS = frozenset({"valid_time_pair", "cycle_idx"})


def _format_balance_row_cells(row: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for h, cell in zip(CSV_HEADERS, row):
        if h in _INT_BALANCE_HEADERS:
            try:
                out.append(str(int(float(cell))))
            except (TypeError, ValueError):
                out.append("" if cell is None else str(cell))
            continue
        if cell is None or cell == "":
            out.append("")
            continue
        if isinstance(cell, str) and cell.strip().lower() == "nan":
            out.append("nan")
            continue
        out.append(fmt_f(cell))
    return out


CSV_HEADERS: List[str] = [
    "sim_time",
    "t_sph",
    "t_err",
    "valid_time_pair",
    "cycle_idx",
    "mjc_theta",
    "mjc_omega",
    "mjc_alpha",
    "mjc_ctrl",
    "mjc_tau_motor",
    "mjc_tau_damp",
    "mjc_I_eff",
    "mjc_tau_fluid_axis",
    "mjc_tau_fluid_recon",
    "mjc_residual_A",
    "mjc_n_hat_x",
    "mjc_n_hat_y",
    "mjc_n_hat_z",
    "mjc_fluid_fx",
    "mjc_fluid_fy",
    "mjc_fluid_fz",
    "sph_omega",
    "sph_alpha",
    "sph_tau_fluid",
    "sph_tau_spring",
    "sph_residual_B",
    "delta_C",
    "sph_ffx",
    "sph_ffy",
    "sph_ffz",
]


def _force_application_module(ctx: Any) -> Any:
    w = getattr(ctx, "sph_wrapper", None)
    if w is None:
        return None
    mode = getattr(w, "current_mode", None)
    if mode is None:
        return None
    return getattr(mode, "force_application_module", None)


def _joint_axis_world(mj: Any, d: Any, joint_id: int, body_id: int) -> np.ndarray:
    axis_b = mj.jnt_axis[joint_id].astype(np.float64).copy()
    R = d.xmat[body_id].reshape(3, 3)
    n = R @ axis_b
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return n / norm


def _I_eff_about_axis(mj: Any, d: Any, body_id: int, dof_adr: int, n_hat: np.ndarray) -> float:
    import mujoco

    q = mj.body_iquat[body_id].astype(np.float64).copy()
    r9 = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(r9, q)
    R_pb = r9.reshape(3, 3)
    diag = np.diag(mj.body_inertia[body_id].astype(np.float64))
    I_body = R_pb @ diag @ R_pb.T
    R_wb = d.xmat[body_id].reshape(3, 3)
    I_world = R_wb @ I_body @ R_wb.T
    Ip = float(n_hat.T @ I_world @ n_hat)
    arm = float(mj.dof_armature[dof_adr])
    return Ip + arm


def _tau_fluid_site_sum(
    mj: Any,
    d: Any,
    body_id: int,
    n_hat: np.ndarray,
    pivot: np.ndarray,
    site_forces: Sequence[Tuple[str, np.ndarray]],
) -> Tuple[float, np.ndarray]:
    import mujoco

    tau = 0.0
    fsum = np.zeros(3, dtype=np.float64)
    for site_name, f in site_forces:
        sid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            continue
        r = d.site_xpos[sid].astype(np.float64).copy()
        fv = np.asarray(f, dtype=np.float64).reshape(3)
        if np.any(np.abs(fv) > 1e6):
            continue
        fsum += fv
        tau += float(np.dot(np.cross(r - pivot, fv), n_hat))
    return tau, fsum


def _sph_yup_to_mjc_zup(v_sph: np.ndarray) -> np.ndarray:
    return np.array([v_sph[0], -v_sph[2], v_sph[1]], dtype=np.float64)


class SphMonitorReader:
    """从 sph_monitor.csv 尾读最新 SPH 数据，按 sim_time 对齐。"""

    def __init__(self, sph_monitor_path: Optional[str] = None) -> None:
        self._path = sph_monitor_path
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
                    if body_name != SPH_BODY_NAME:
                        continue
                    anchor_idx = int(row.get("anchor_idx", "-1"))
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


class SteeringTorqueRecorder:
    """每圈 env.step 后追加一行（由主循环调用）；仿真结束后调用 post_merge() 合并 SPH 数据。"""

    @classmethod
    def from_env(cls) -> Optional["SteeringTorqueRecorder"]:
        if "ORCA_STEERING_TORQUE_CSV" not in os.environ:
            return None
        path = os.environ.get("ORCA_STEERING_TORQUE_CSV") or "/tmp/steering_torque_balance.csv"
        return cls(path)

    def __init__(self, path: str) -> None:
        self._path = path
        self._fp: Optional[Any] = None
        self._writer: Optional[Any] = None
        self._opened = False
        sph_monitor_path = os.environ.get("ORCA_SPH_MONITOR_CSV", "")
        if not sph_monitor_path:
            sph_monitor_path = os.path.join(
                os.path.dirname(path), "sph_monitor.csv"
            )
            if not os.path.isfile(sph_monitor_path):
                sph_monitor_path = "/tmp/orca_sph_monitor.csv"
        self._sph_reader = SphMonitorReader(sph_monitor_path)
        self._prev_sph_omega: Optional[float] = None
        self._prev_sph_t: Optional[float] = None

    def _ensure_open(self) -> None:
        if self._opened:
            return
        absp = os.path.abspath(self._path)
        dirp = os.path.dirname(absp)
        if dirp:
            os.makedirs(dirp, exist_ok=True)
        self._fp = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fp)
        self._writer.writerow(CSV_HEADERS)
        self._fp.flush()
        self._opened = True
        logger.info("steering torque CSV: %s", os.path.abspath(self._path))
        logger.info("SPH monitor reader path: %s", self._sph_reader._path)

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
                spring_force_mjc = _sph_yup_to_mjc_zup(spring_force_sph)
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

        joint_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)
        body_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, BODY_NAME)
        aid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, ACTUATOR_NAME)
        if joint_id < 0 or body_id < 0 or aid < 0:
            logger.warning(
                "steering torque CSV: missing joint/body/actuator (%s,%s,%s)",
                joint_id,
                body_id,
                aid,
            )
            return

        dof_adr = int(mj.jnt_dofadr[joint_id])
        qpos_adr = int(mj.jnt_qposadr[joint_id])

        t_sim = float(d.time)
        n_hat = _joint_axis_world(mj, d, joint_id, body_id)
        pivot = d.xpos[body_id].astype(np.float64).copy()

        theta = float(d.qpos[qpos_adr])
        omega = float(d.qvel[dof_adr])
        alpha = float(d.qacc[dof_adr])
        ctrl = float(d.ctrl[aid])

        damp_c = float(mj.dof_damping[dof_adr])
        tau_damp = -damp_c * omega
        tau_motor = float(d.actuator_force[aid])

        I_eff = _I_eff_about_axis(mj, d, body_id, dof_adr, n_hat)

        fam = _force_application_module(ctx)
        site_forces: List[Tuple[str, np.ndarray]] = []
        if fam is not None and getattr(fam, "last_applied_site_forces", None):
            site_forces = list(fam.last_applied_site_forces)

        tau_fluid, fsum = _tau_fluid_site_sum(mj, d, body_id, n_hat, pivot, site_forces)
        tau_recon = I_eff * alpha - tau_motor - tau_damp
        resid_a = I_eff * alpha - (tau_motor + tau_damp + tau_fluid)

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
        """仿真结束后，读取完整的 sph_monitor.csv，重新合并 SPH 数据到 CSV。"""
        self.close()

        sph_all = self._sph_reader.read_all()
        if not sph_all:
            logger.warning("post_merge: sph_monitor.csv 无有效数据，跳过合并")
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
            writer.writerow(CSV_HEADERS)
            writer.writerows([_format_balance_row_cells(r) for r in updated_rows])

        sph_valid = sum(1 for row in updated_rows if row[sph_idx_map["sph_omega"]] != "nan")
        logger.info("post_merge: 合并完成，%d/%d 行有 SPH 数据", sph_valid, len(updated_rows))
