#!/usr/bin/env python3

import argparse
from datetime import datetime
import os
import sys
import time

import gymnasium as gym
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.common.model_scanner import build_suffix_template, require_complete_matches, scan_scene_for_template
from envs.drone.drone_aero_config import (
    DEFAULT_DRONE_MODEL,
    get_drone_model_profile,
)
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import OrcaGymScene

_logger = get_orca_logger(console_level="WARNING", file_level="INFO", force_reinit=True)
ENV_ENTRY_POINT = "envs.drone.drone_orca_env:DroneOrcaEnv"

DEFAULT_TIME_STEP = 1.0 / 120.0
DEFAULT_FRAME_SKIP = 1
# 竖直模式的键盘基准 T/mg；仅在显式切到 vertical_z_only 时使用
_DEFAULT_DRONE_PROFILE = get_drone_model_profile(DEFAULT_DRONE_MODEL)
DEFAULT_VERTICAL_KEYBOARD_BASE_TMG = float(_DEFAULT_DRONE_PROFILE.vertical_keyboard_baseline_tmg)
DEFAULT_VERTICAL_XY_FORCE_FACTOR = float(_DEFAULT_DRONE_PROFILE.vertical_xy_force_factor)

DRONE_JOINT_SUFFIXES = [
    "drone_free",
    "FL_joint",
    "FR_joint",
    "BL_joint",
    "BR_joint",
]
# 桨关节由环境直接写 qpos/qvel 做动画；模型中不再挂 position 执行器，避免与脚本驱动冲突
DRONE_ACTUATOR_SUFFIXES: list[str] = []
DRONE_BODY_SUFFIXES = ["drone_frame", "Drone"]
DRONE_SITE_SUFFIXES = [
    "drone_body_center_site",
    "rotor_fl_site",
    "rotor_fr_site",
    "rotor_bl_site",
    "rotor_br_site",
]


def resolve_drone_scene_binding(orcagym_addr: str, time_step: float) -> tuple[list[str], dict]:
    template = build_suffix_template(
        model_name="Drone",
        joints=DRONE_JOINT_SUFFIXES,
        actuators=DRONE_ACTUATOR_SUFFIXES,
        bodies=DRONE_BODY_SUFFIXES,
        sites=DRONE_SITE_SUFFIXES,
    )
    report = scan_scene_for_template(
        orcagym_addr=orcagym_addr,
        time_step=time_step,
        template=template,
    )
    match = require_complete_matches(
        report,
        min_count=1,
        max_count=1,
        allow_empty_prefix=True,
        orcagym_addr=orcagym_addr,
    )[0]
    agent_names = [match.agent_name] if match.agent_name else []
    scene_binding = {
        "joints_by_suffix": dict(match.matched_names.get("joints", {})),
        "actuators_by_suffix": dict(match.matched_names.get("actuators", {})),
        "bodies_by_suffix": dict(match.matched_names.get("bodies", {})),
        "sites_by_suffix": dict(match.matched_names.get("sites", {})),
    }
    return agent_names, scene_binding


def sceneinfo(scene, stage: str, orcagym_address: str):
    toclose = False
    if scene is None:
        toclose = True
        scene = OrcaGymScene(orcagym_address)
    try:
        script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
        scene.get_rundata(script_name, stage)
        if stage == "beginscene":
            _logger.info("开始仿真：默认 full 四旋翼模式；W/S 前后、A/D 左右、R/F 升降、Q/E 偏航，Space 重置")
            _logger.info(
                "若需切回竖直 Z-only 调试链，请用 --vertical-z-only；"
                "W/S、A/D 在竖直模式下可驱动世界系水平力（见 --vertical-xy-force-factor / --vertical-pure-z）"
            )
        elif stage == "loadscene":
            _logger.info("加载模型中")
        scene.set_image_enabled(1, True)
    finally:
        if toclose:
            scene.close()


def register_env(
    orcagym_addr: str,
    env_index: int,
    agent_names: list[str],
    scene_binding: dict,
    time_step: float,
    frame_skip: int,
    autoplay: bool,
    max_episode_steps: int,
    vertical_z_only_physics: bool = False,
    vertical_thrust_ramp: bool = False,
    vertical_ramp_t0_factor: float = 0.65,
    vertical_ramp_t1_factor: float = 2.05,
    vertical_ramp_duration_s: float = 25.0,
    vertical_lock_quat_world_up: bool = True,
    vertical_fixed_thrust_over_hover: float = -1.0,
    vertical_keyboard_baseline_tmg: float = DEFAULT_VERTICAL_KEYBOARD_BASE_TMG,
    vertical_keyboard_xy_force_factor: float = DEFAULT_VERTICAL_XY_FORCE_FACTOR,
    reset_height_offset_m: float = 0.25,
    fullmode_reset_thrust_ramp_s: float = 0.8,
    fullmode_reset_thrust_start_factor: float = 0.2,
    fullmode_reset_minimal_stab_s: float = 0.35,
    drone_model: str = DEFAULT_DRONE_MODEL,
    diag_logs_enabled: bool = True,
    diag_every_env_steps: int = 0,
) -> tuple[str, dict]:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = "DroneOrca-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    kwargs = {
        "frame_skip": frame_skip,
        "orcagym_addr": orcagym_addr,
        "agent_names": agent_names,
        "time_step": time_step,
        "scene_binding": scene_binding,
        "autoplay": autoplay,
        "vertical_z_only_physics": vertical_z_only_physics,
        "vertical_thrust_ramp": vertical_thrust_ramp,
        "vertical_ramp_t0_factor": vertical_ramp_t0_factor,
        "vertical_ramp_t1_factor": vertical_ramp_t1_factor,
        "vertical_ramp_duration_s": vertical_ramp_duration_s,
        "vertical_lock_quat_world_up": vertical_lock_quat_world_up,
        "vertical_fixed_thrust_over_hover": vertical_fixed_thrust_over_hover,
        "vertical_keyboard_baseline_tmg": vertical_keyboard_baseline_tmg,
        "vertical_keyboard_xy_force_factor": float(vertical_keyboard_xy_force_factor),
        "reset_height_offset_m": float(reset_height_offset_m),
        "fullmode_reset_thrust_ramp_s": float(fullmode_reset_thrust_ramp_s),
        "fullmode_reset_thrust_start_factor": float(fullmode_reset_thrust_start_factor),
        "fullmode_reset_minimal_stab_s": float(fullmode_reset_minimal_stab_s),
        "drone_model": drone_model,
        "diag_logs_enabled": bool(diag_logs_enabled),
        "diag_every_env_steps": int(diag_every_env_steps),
    }
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT,
        kwargs=kwargs,
        max_episode_steps=max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs


def run_simulation(
    orcagym_addr: str,
    time_step: float,
    frame_skip: int,
    autoplay: bool,
    vertical_z_only_physics: bool = False,
    vertical_thrust_ramp: bool = False,
    vertical_ramp_t0_factor: float = 0.65,
    vertical_ramp_t1_factor: float = 2.05,
    vertical_ramp_duration_s: float = 25.0,
    vertical_lock_quat_world_up: bool = True,
    vertical_fixed_thrust_over_hover: float = -1.0,
    vertical_keyboard_baseline_tmg: float = DEFAULT_VERTICAL_KEYBOARD_BASE_TMG,
    vertical_keyboard_xy_force_factor: float = DEFAULT_VERTICAL_XY_FORCE_FACTOR,
    reset_height_offset_m: float = 0.25,
    fullmode_reset_thrust_ramp_s: float = 0.8,
    fullmode_reset_thrust_start_factor: float = 0.2,
    fullmode_reset_minimal_stab_s: float = 0.35,
    drone_model: str = DEFAULT_DRONE_MODEL,
    diag_logs_enabled: bool = True,
    diag_every_env_steps: int = 0,
) -> None:
    env = None
    try:
        realtime_step = time_step * frame_skip
        profile = get_drone_model_profile(drone_model)
        _logger.info(f"simulation running... , orcagym_addr: {orcagym_addr}")
        _logger.info(f"无人机参数配置: {profile.display_name} ({profile.key})")
        sceneinfo(None, "loadscene", orcagym_addr)

        agent_names, scene_binding = resolve_drone_scene_binding(orcagym_addr, time_step)
        resolved_name = agent_names[0] if agent_names else "<root>"
        _logger.info(f"检测到场景中的 Drone 实例: {resolved_name}")
        env_id, _ = register_env(
            orcagym_addr=orcagym_addr,
            env_index=0,
            agent_names=agent_names,
            scene_binding=scene_binding,
            time_step=time_step,
            frame_skip=frame_skip,
            autoplay=autoplay,
            max_episode_steps=sys.maxsize,
            vertical_z_only_physics=vertical_z_only_physics,
            vertical_thrust_ramp=vertical_thrust_ramp,
            vertical_ramp_t0_factor=vertical_ramp_t0_factor,
            vertical_ramp_t1_factor=vertical_ramp_t1_factor,
            vertical_ramp_duration_s=vertical_ramp_duration_s,
            vertical_lock_quat_world_up=vertical_lock_quat_world_up,
            vertical_fixed_thrust_over_hover=vertical_fixed_thrust_over_hover,
            vertical_keyboard_baseline_tmg=vertical_keyboard_baseline_tmg,
            vertical_keyboard_xy_force_factor=vertical_keyboard_xy_force_factor,
            reset_height_offset_m=reset_height_offset_m,
            fullmode_reset_thrust_ramp_s=fullmode_reset_thrust_ramp_s,
            fullmode_reset_thrust_start_factor=fullmode_reset_thrust_start_factor,
            fullmode_reset_minimal_stab_s=fullmode_reset_minimal_stab_s,
            drone_model=profile.key,
            diag_logs_enabled=diag_logs_enabled,
            diag_every_env_steps=diag_every_env_steps,
        )
        env = gym.make(env_id)
        obs, info = env.reset()
        sceneinfo(None, "beginscene", orcagym_addr)
        print(f"orcagym_addr: {orcagym_addr}")
        if reset_height_offset_m > 0.0:
            _logger.info(f"已启用 reset 安全抬高：dz={float(reset_height_offset_m):.4f}m")
        if not vertical_z_only_physics:
            _logger.info(
                "full 模式启动隔离参数："
                f" thrust_ramp={float(fullmode_reset_thrust_ramp_s):.3f}s"
                f" start={float(fullmode_reset_thrust_start_factor):.2f}·hover"
                f" minimal_stab={float(fullmode_reset_minimal_stab_s):.3f}s"
            )
            if int(diag_every_env_steps) > 0:
                _logger.info(f"已启用 periodic 动力学定位日志：every={int(diag_every_env_steps)} env steps")
        if autoplay:
            _logger.info("已启用 autoplay：无人机将持续执行前进、横移、升降和偏航扰动，便于反复调试")
        if vertical_z_only_physics:
            xy_k = float(vertical_keyboard_xy_force_factor)
            planar = (
                "姿态每步锁定；WASD 世界系水平力（系数 k_xy>0 时保留 vx,vy）。"
                if xy_k > 1e-12
                else "姿态与水平速度每步锁定（纯 Z）。"
            )
            _logger.info(
                "竖直 Z 模式：世界 +Z 推力与 vz 阻尼；" + planar
                + (
                    " 推力爬升：约每秒 ramp 进度；满足 Δz∧vz 持续时间则打「持续起飞临界(精估)」。"
                    if vertical_thrust_ramp
                    else ""
                )
                + (
                    " 未开 ramp/固定推力时可用 R/F 调升降；可用 --vertical-keyboard-base-tmg 设杆量零时的 T/mg。"
                    if not vertical_thrust_ramp and vertical_fixed_thrust_over_hover < 0
                    else ""
                )
                + (f" k_xy={xy_k:.4g}。" if xy_k > 1e-12 else "")
            )

        dummy_action = np.zeros(env.action_space.shape, dtype=np.float32)

        while True:
            start_time = datetime.now()
            obs, reward, terminated, truncated, info = env.step(dummy_action)
            if info.get("reset_requested"):
                obs, info = env.reset()
            env.render()

            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < realtime_step:
                time.sleep(realtime_step - elapsed_time.total_seconds())

    except KeyboardInterrupt:
        print("Simulation stopped")
    except ValueError:
        _logger.error("仿真出错")
    finally:
        if env is not None:
            env.close()


def run_takeoff_bisection(
    orcagym_addr: str,
    time_step: float,
    frame_skip: int,
    bisect_lo: float,
    bisect_hi: float,
    bisect_iters: int,
    bisect_hold_s: float,
    bisect_dz_m: float,
    vertical_lock_quat_world_up: bool,
    reset_height_offset_m: float = 0.0,
    drone_model: str = DEFAULT_DRONE_MODEL,
    diag_logs_enabled: bool = True,
) -> None:
    """竖直模式下对固定 T/(mg) 做二分：假设 lo 不能持续离地、hi 能（见 env 内判据）。"""
    env = None
    try:
        profile = get_drone_model_profile(drone_model)
        _logger.info(f"takeoff bisection, orcagym_addr: {orcagym_addr}")
        _logger.info(f"无人机参数配置: {profile.display_name} ({profile.key})")
        sceneinfo(None, "loadscene", orcagym_addr)
        agent_names, scene_binding = resolve_drone_scene_binding(orcagym_addr, time_step)
        env_id, _ = register_env(
            orcagym_addr=orcagym_addr,
            env_index=0,
            agent_names=agent_names,
            scene_binding=scene_binding,
            time_step=time_step,
            frame_skip=frame_skip,
            autoplay=False,
            max_episode_steps=sys.maxsize,
            vertical_z_only_physics=True,
            vertical_thrust_ramp=False,
            vertical_fixed_thrust_over_hover=-1.0,
            vertical_lock_quat_world_up=vertical_lock_quat_world_up,
            vertical_keyboard_xy_force_factor=0.0,
            reset_height_offset_m=reset_height_offset_m,
            drone_model=profile.key,
            diag_logs_enabled=diag_logs_enabled,
        )
        env = gym.make(env_id)
        raw = env.unwrapped
        raw.set_vertical_quiet_diag_logs(True)
        dummy_action = np.zeros(env.action_space.shape, dtype=np.float32)

        def climbed_at(ratio: float) -> bool:
            raw.set_vertical_fixed_thrust_over_hover(ratio)
            env.reset()
            z0 = raw.get_vertical_takeoff_z_reference()
            while float(raw.gym._mjData.time) < bisect_hold_s:
                env.step(dummy_action)
                env.render()
            z1 = float(raw.gym._mjData.xpos[raw._frame_body_id, 2])
            return (z1 - z0) >= bisect_dz_m

        _logger.warning(
            f"[run_drone_orca] 起飞二分：hold={bisect_hold_s}s Δz阈={bisect_dz_m}m 迭代={bisect_iters} 初区间[{bisect_lo},{bisect_hi}]"
        )
        if climbed_at(bisect_lo):
            _logger.warning("[run_drone_orca] bisect_lo 已能离地，请降低 --bisect-lo")
        if not climbed_at(bisect_hi):
            _logger.warning("[run_drone_orca] bisect_hi 仍不能离地，请提高 --bisect-hi；中止二分。")
            return

        lo, hi = float(bisect_lo), float(bisect_hi)
        for it in range(int(bisect_iters)):
            mid = 0.5 * (lo + hi)
            if climbed_at(mid):
                hi = mid
            else:
                lo = mid
            _logger.warning(f"[run_drone_orca] bisect it={it} mid={mid:.6f} → 区间[{lo:.6f},{hi:.6f}]")
        est = 0.5 * (lo + hi)
        _logger.warning(
            f"[run_drone_orca] 起飞临界 T/(mg) 二分估计 ≈ {est:.6f}（区间 [{lo:.6f},{hi:.6f}]）；"
            f"判据为 hold={bisect_hold_s}s 内 Δz≥{bisect_dz_m}m（与 env 内「持续起飞」vz+Δz+时间判据可略有差异）"
        )
    except KeyboardInterrupt:
        print("Bisection stopped")
    except ValueError:
        _logger.error("仿真出错")
    finally:
        if env is not None:
            try:
                env.unwrapped.set_vertical_quiet_diag_logs(False)
            except Exception:
                pass
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Run drone orca communication demo",
        description=(
            "默认：full 四旋翼模式。W/S、A/D 控制前后/左右平移（伴随机身轻微倾斜），"
            "R/F 控制集体升降，Q/E 控制偏航。"
        ),
    )
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--time_step", type=float, default=DEFAULT_TIME_STEP)
    parser.add_argument("--frame_skip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument(
        "--drone-model",
        type=str,
        default=DEFAULT_DRONE_MODEL,
        help="无人机参数配置，默认 Drone_ver_1.0，可传 x2 / skydio_x2 等别名",
    )
    parser.add_argument(
        "--diag-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否输出环境侧诊断 WARNING；可用 --no-diag-logs 关闭",
    )
    parser.add_argument(
        "--diag-every-env-steps",
        type=int,
        default=0,
        help="恢复之前用于定位 full 模式乱飘的 periodic 长日志；0=关闭，正整数=每 N 个 env step 输出一次",
    )
    parser.add_argument(
        "--vertical-z-only",
        action="store_true",
        help="显式切回竖直 Z-only 调试模式：世界 +Z 推力与 vz 阻尼，姿态可锁",
    )
    parser.add_argument(
        "--vertical-thrust-ramp",
        action="store_true",
        help="在竖直模式下线性爬升 T/(mg)，便于从日志读取 vz 过阈时的临界推力",
    )
    parser.add_argument("--vertical-ramp-t0", type=float, default=0.65, help="爬升起始 T/mg 系数")
    parser.add_argument("--vertical-ramp-t1", type=float, default=2.05, help="爬升结束 T/mg 系数（需明显高于 1 才易持续离地）")
    parser.add_argument("--vertical-ramp-duration", type=float, default=25.0, help="爬升持续时间 (s)")
    parser.add_argument(
        "--vertical-use-scene-quat",
        action="store_true",
        help="竖直模式下每步锁姿态为场景初始四元数（默认锁世界朝上，使 +Z 推力沿机体竖轴）",
    )
    parser.add_argument(
        "--vertical-fixed-tmg",
        type=float,
        default=-1.0,
        help="竖直模式固定 T/(mg)（>=0 时启用，与 ramp/键盘基准互斥，键盘 R/F 无效）",
    )
    parser.add_argument(
        "--vertical-keyboard-base-tmg",
        type=float,
        default=None,
        help="竖直模式且无固定、无 ramp 时杆量零对应的 T/(mg)，默认取所选无人机 profile",
    )
    parser.add_argument(
        "--vertical-xy-force-factor",
        type=float,
        default=None,
        help=(
            "竖直模式下 WASD 世界系水平力系数 k_xy（f∝k_xy·mg·杆量）；"
            "默认取所选无人机 profile；与 --vertical-pure-z 互斥"
        ),
    )
    parser.add_argument(
        "--vertical-pure-z",
        action="store_true",
        help="竖直模式关闭水平力（k_xy=0），每步清零 vx,vy，与二分标定一致",
    )
    parser.add_argument(
        "--vertical-takeoff-bisect",
        action="store_true",
        help="竖直模式下二分搜索起飞临界 T/(mg)（需 bisect_lo 不能离地、bisect_hi 能离地）",
    )
    parser.add_argument("--bisect-lo", type=float, default=0.98, help="二分下界 T/mg")
    parser.add_argument("--bisect-hi", type=float, default=1.12, help="二分上界 T/mg")
    parser.add_argument("--bisect-iters", type=int, default=14, help="二分迭代次数")
    parser.add_argument("--bisect-hold-s", type=float, default=3.0, help="每档试验持有的仿真时长 (s)")
    parser.add_argument("--bisect-dz", type=float, default=0.06, help="判定离地的 Δz (m)")
    parser.add_argument(
        "--reset-height-offset",
        type=float,
        default=None,
        help="reset 时给无人机初始 z 额外抬高的距离 (m)，默认取所选无人机 profile",
    )
    parser.add_argument(
        "--fullmode-reset-thrust-ramp",
        type=float,
        default=None,
        help="full 模式 reset 后集体推力渐入时间 (s)，默认取所选无人机 profile",
    )
    parser.add_argument(
        "--fullmode-reset-thrust-start-factor",
        type=float,
        default=None,
        help="full 模式 reset 推力渐入起点，占 hover thrust 的比例，默认取所选无人机 profile",
    )
    parser.add_argument(
        "--fullmode-reset-minimal-stab",
        type=float,
        default=None,
        help="full 模式 reset 后的最小稳定窗口时长 (s)，默认取所选无人机 profile",
    )
    args = parser.parse_args()
    profile = get_drone_model_profile(args.drone_model)
    lock_world_up = not bool(args.vertical_use_scene_quat)
    fix_tmg = float(args.vertical_fixed_tmg)
    use_fixed = fix_tmg >= 0.0
    use_vertical_mode = bool(args.vertical_z_only) or bool(
        args.vertical_thrust_ramp or args.vertical_takeoff_bisect or use_fixed
    )
    vz_only = use_vertical_mode
    vertical_keyboard_base_tmg = (
        float(profile.vertical_keyboard_baseline_tmg)
        if args.vertical_keyboard_base_tmg is None
        else float(args.vertical_keyboard_base_tmg)
    )
    vertical_xy_force_factor = (
        float(profile.vertical_xy_force_factor)
        if args.vertical_xy_force_factor is None
        else float(args.vertical_xy_force_factor)
    )
    reset_height_offset = (
        float(profile.full_mode.reset_height_offset_m)
        if args.reset_height_offset is None
        else float(args.reset_height_offset)
    )
    fullmode_reset_thrust_ramp = (
        float(profile.full_mode.fullmode_reset_thrust_ramp_s)
        if args.fullmode_reset_thrust_ramp is None
        else float(args.fullmode_reset_thrust_ramp)
    )
    fullmode_reset_thrust_start_factor = (
        float(profile.full_mode.fullmode_reset_thrust_start_factor)
        if args.fullmode_reset_thrust_start_factor is None
        else float(args.fullmode_reset_thrust_start_factor)
    )
    fullmode_reset_minimal_stab = (
        float(profile.full_mode.fullmode_reset_minimal_stab_s)
        if args.fullmode_reset_minimal_stab is None
        else float(args.fullmode_reset_minimal_stab)
    )
    xy_k = 0.0 if bool(args.vertical_pure_z) else float(vertical_xy_force_factor)
    if args.vertical_takeoff_bisect:
        run_takeoff_bisection(
            orcagym_addr=args.orcagym_addr,
            time_step=args.time_step,
            frame_skip=args.frame_skip,
            bisect_lo=float(args.bisect_lo),
            bisect_hi=float(args.bisect_hi),
            bisect_iters=int(args.bisect_iters),
            bisect_hold_s=float(args.bisect_hold_s),
            bisect_dz_m=float(args.bisect_dz),
            vertical_lock_quat_world_up=lock_world_up,
            reset_height_offset_m=reset_height_offset,
            drone_model=profile.key,
            diag_logs_enabled=bool(args.diag_logs),
        )
    else:
        run_simulation(
            orcagym_addr=args.orcagym_addr,
            time_step=args.time_step,
            frame_skip=args.frame_skip,
            autoplay=args.autoplay,
            vertical_z_only_physics=vz_only,
            vertical_thrust_ramp=bool(args.vertical_thrust_ramp) and not use_fixed,
            vertical_ramp_t0_factor=float(args.vertical_ramp_t0),
            vertical_ramp_t1_factor=float(args.vertical_ramp_t1),
            vertical_ramp_duration_s=float(args.vertical_ramp_duration),
            vertical_lock_quat_world_up=lock_world_up,
            vertical_fixed_thrust_over_hover=fix_tmg if use_fixed else -1.0,
            vertical_keyboard_baseline_tmg=vertical_keyboard_base_tmg,
            vertical_keyboard_xy_force_factor=xy_k,
            reset_height_offset_m=reset_height_offset,
            fullmode_reset_thrust_ramp_s=fullmode_reset_thrust_ramp,
            fullmode_reset_thrust_start_factor=fullmode_reset_thrust_start_factor,
            fullmode_reset_minimal_stab_s=fullmode_reset_minimal_stab,
            drone_model=profile.key,
            diag_logs_enabled=bool(args.diag_logs),
            diag_every_env_steps=max(0, int(args.diag_every_env_steps)),
        )
