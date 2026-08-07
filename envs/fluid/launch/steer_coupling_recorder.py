"""
steering_box：MuJoCo 与 SPH 在同一 OrcaLink 周期（coupling_cycle）上的耦合监测 CSV。

- 采样：仅在一次成功的 env.step 之后（与 REALTIME_STEP=0.02s 一致）。
- 角加速度：同时记录 mjc_alpha_qacc（MuJoCo qacc）与 mjc_alpha_fd / sph_alpha_fd（对轴角速度差分）。
- SPH 行：仿真结束后按 sph_monitor.csv 的 cycle 与 coupling_cycle 精确匹配合并（不用 0.03s 最近邻）。

启用：设置环境变量 ORCA_STEER_COUPLE_CSV（可为空字符串，默认 /tmp/steer_mjc_sph_couple.csv）。
SPH 监测路径：ORCA_SPH_MONITOR_CSV 或 ORCA_MONITOR_CSV，否则与输出同目录的 sph_monitor.csv。
"""

from __future__ import annotations

import csv
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from envs.fluid.csv_numeric import fmt_coupling_dict_row, fmt_f

logger = logging.getLogger(__name__)

JOINT_NAME = "steering__joint_1_3"
BODY_NAME = "steering_box_body_3"
SPH_BODY_NAME = "steering_box_body_3"

CSV_HEADERS: List[str] = [
    "coupling_cycle",
    "sim_time_mj",
    "sph_cycle_matched",
    "sph_sim_time",
    "valid_cycle_pair",
    "dt_fd",
    "mjc_omega_axis",
    "mjc_alpha_qacc",
    "mjc_alpha_fd",
    "mjc_quat_w",
    "mjc_quat_x",
    "mjc_quat_y",
    "mjc_quat_z",
    "mjc_xfrc_fx",
    "mjc_xfrc_fy",
    "mjc_xfrc_fz",
    "mjc_xfrc_tx",
    "mjc_xfrc_ty",
    "mjc_xfrc_tz",
    "mjc_fluid_fx",
    "mjc_fluid_fy",
    "mjc_fluid_fz",
    "mjc_tau_fluid_axis",
    "mjc_n_hat_x",
    "mjc_n_hat_y",
    "mjc_n_hat_z",
    "sph_omega_axis",
    "sph_alpha_fd",
    "sph_body_fx",
    "sph_body_fy",
    "sph_body_fz",
    "sph_body_tx",
    "sph_body_ty",
    "sph_body_tz",
    "sph_body_torque_axis",
    "sph_fluid_fx",
    "sph_fluid_fy",
    "sph_fluid_fz",
    "sph_pd_f0_x",
    "sph_pd_f0_y",
    "sph_pd_f0_z",
    "sph_pd_f1_x",
    "sph_pd_f1_y",
    "sph_pd_f1_z",
    "sph_pd_f2_x",
    "sph_pd_f2_y",
    "sph_pd_f2_z",
    "sph_pd_f3_x",
    "sph_pd_f3_y",
    "sph_pd_f3_z",
    "sph_pd_tau0_x",
    "sph_pd_tau0_y",
    "sph_pd_tau0_z",
    "sph_pd_tau1_x",
    "sph_pd_tau1_y",
    "sph_pd_tau1_z",
    "sph_pd_tau2_x",
    "sph_pd_tau2_y",
    "sph_pd_tau2_z",
    "sph_pd_tau3_x",
    "sph_pd_tau3_y",
    "sph_pd_tau3_z",
    "sph_quat_w",
    "sph_quat_x",
    "sph_quat_y",
    "sph_quat_z",
    "sph_quat_mjc_w",
    "sph_quat_mjc_x",
    "sph_quat_mjc_y",
    "sph_quat_mjc_z",
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


def _sph_quat_to_mjc_quat_wxyz(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float, float]:
    """SPH Y-up 世界系四元数 → MuJoCo Z-up 世界系（与线速度变换同一固定基变换）。"""
    try:
        from scipy.spatial.transform import Rotation as R

        T = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        R_T = R.from_matrix(T)
        R_sph = R.from_quat([qx, qy, qz, qw])  # scipy: xyzw
        R_mj = R_T * R_sph * R_T.inv()
        x, y, z, w = R_mj.as_quat()
        return float(w), float(x), float(y), float(z)
    except Exception:
        return (float("nan"),) * 4


def load_sph_monitor_by_cycle(
    path: str, body_name: str
) -> Dict[int, Dict[int, Dict[str, str]]]:
    """cycle -> anchor_idx -> row dict（原始 CSV 字符串值）。"""
    out: Dict[int, Dict[int, Dict[str, str]]] = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("body_name") != body_name:
                continue
            try:
                cyc = int(float(row["cycle"]))
                aid = int(float(row["anchor_idx"]))
            except (KeyError, ValueError):
                continue
            out.setdefault(cyc, {})[aid] = row
    return out


class SteerCouplingRecorder:
    """每圈 env.step 后追加一行 MJ 数据；post_merge 按 cycle 合并 sph_monitor。"""

    @classmethod
    def from_env(cls) -> Optional["SteerCouplingRecorder"]:
        if "ORCA_STEER_COUPLE_CSV" not in os.environ:
            return None
        raw = os.environ.get("ORCA_STEER_COUPLE_CSV")
        path = raw if raw else "/tmp/steer_mjc_sph_couple.csv"
        return cls(path)

    def __init__(self, path: str) -> None:
        self._path = path
        self._rows: List[Dict[str, Any]] = []
        self._prev_omega: Optional[float] = None
        self._prev_sim_t: Optional[float] = None

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

        joint_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)
        body_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, BODY_NAME)
        if joint_id < 0 or body_id < 0:
            logger.warning("steer couple CSV: missing joint/body")
            return

        dof_adr = int(mj.jnt_dofadr[joint_id])
        qpos_adr = int(mj.jnt_qposadr[joint_id])
        t_sim = float(d.time)
        n_hat = _joint_axis_world(mj, d, joint_id, body_id)
        pivot = d.xpos[body_id].astype(np.float64).copy()

        omega = float(d.qvel[dof_adr])
        alpha_qacc = float(d.qacc[dof_adr])

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

        row: Dict[str, Any] = {h: float("nan") for h in CSV_HEADERS}
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
        idx = load_sph_monitor_by_cycle(sph_path, SPH_BODY_NAME)
        if not idx:
            logger.warning("steer couple post_merge: 无 SPH 索引 (%s)", sph_path)

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
            w = csv.DictWriter(fp, fieldnames=CSV_HEADERS)
            w.writeheader()
            for r in self._rows:
                w.writerow(fmt_coupling_dict_row(r, CSV_HEADERS))
        valid = sum(1 for r in self._rows if int(r.get("valid_cycle_pair", 0)) == 1)
        logger.info(
            "steer couple CSV: %s rows=%d valid_cycle=%d sph=%s",
            absp,
            len(self._rows),
            valid,
            sph_path,
        )
