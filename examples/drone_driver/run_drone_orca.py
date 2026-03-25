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
from envs.drone.drone_aero_config import DEFAULT_DRONE_AERO_CONFIG
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import OrcaGymScene

_logger = get_orca_logger()

ENV_ENTRY_POINT = {
    "DroneOrca": "envs.drone.drone_orca_env:DroneOrcaEnv",
}

DEFAULT_TIME_STEP = 1.0 / 120.0
DEFAULT_FRAME_SKIP = 1
# 默认与当前竖直标定一致：无参运行时 world-Z 推力 + 姿态锁；杆量零时 T/mg≈二分临界
DEFAULT_VERTICAL_KEYBOARD_BASE_TMG = 1.0022
DEFAULT_VERTICAL_XY_FORCE_FACTOR = float(
    DEFAULT_DRONE_AERO_CONFIG.vertical_z_only.keyboard_world_xy_force_factor
)

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
            _logger.info("开始仿真：默认竖直 Z-only（drone_frame 推力、世界朝上锁姿）；R/F 升降，Space 重置")
            _logger.info(
                "W/S、A/D 在竖直模式下可驱动世界系水平力（见 --vertical-xy-force-factor / --vertical-pure-z）；"
                "四旋翼杆量（俯仰/滚转力矩 + 机体系升力）请用 --full-6dof-thrust 或 --quad-wasd-torque"
            )
        elif stage == "loadscene":
            _logger.info("加载模型中")
        scene.set_image_enabled(1, True)
    finally:
        if toclose:
            scene.close()


def register_env(
    orcagym_addr: str,
    env_name: str,
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
) -> tuple[str, dict]:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
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
    }
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps=max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs


def run_simulation(
    orcagym_addr: str,
    env_name: str,
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
) -> None:
    env = None
    try:
        realtime_step = time_step * frame_skip
        _logger.info(f"simulation running... , orcagym_addr: {orcagym_addr}")
        sceneinfo(None, "loadscene", orcagym_addr)

        agent_names, scene_binding = resolve_drone_scene_binding(orcagym_addr, time_step)
        resolved_name = agent_names[0] if agent_names else "<root>"
        _logger.info(f"检测到场景中的 Drone 实例: {resolved_name}")
        env_id, _ = register_env(
            orcagym_addr=orcagym_addr,
            env_name=env_name,
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
        )
        env = gym.make(env_id)
        obs, info = env.reset()
        sceneinfo(None, "beginscene", orcagym_addr)
        print(f"orcagym_addr: {orcagym_addr}")
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
    finally:
        if env is not None:
            env.close()


def run_takeoff_bisection(
    orcagym_addr: str,
    env_name: str,
    time_step: float,
    frame_skip: int,
    bisect_lo: float,
    bisect_hi: float,
    bisect_iters: int,
    bisect_hold_s: float,
    bisect_dz_m: float,
    vertical_lock_quat_world_up: bool,
) -> None:
    """竖直模式下对固定 T/(mg) 做二分：假设 lo 不能持续离地、hi 能（见 env 内判据）。"""
    env = None
    try:
        _logger.info(f"takeoff bisection, orcagym_addr: {orcagym_addr}")
        sceneinfo(None, "loadscene", orcagym_addr)
        agent_names, scene_binding = resolve_drone_scene_binding(orcagym_addr, time_step)
        env_id, _ = register_env(
            orcagym_addr=orcagym_addr,
            env_name=env_name,
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
            f"默认：竖直 Z-only + 键盘基准 T/mg≈{DEFAULT_VERTICAL_KEYBOARD_BASE_TMG}（R/F 微调）。"
            "四旋翼式：W/A/S/D→滚转/俯仰力矩 + 机体系升力用 --full-6dof-thrust 或 --quad-wasd-torque。"
        ),
    )
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--env_name", type=str, default="DroneOrca")
    parser.add_argument("--time_step", type=float, default=DEFAULT_TIME_STEP)
    parser.add_argument("--frame_skip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument(
        "--full-6dof-thrust",
        action="store_true",
        help="关闭竖直 Z-only：W/S、A/D→俯仰/滚转力矩，R/F→集体升力，推力沿 drone_frame +Z（与 free 同体；倾斜即有水平分力），Q/E 偏航",
    )
    parser.add_argument(
        "--quad-wasd-torque",
        action="store_true",
        help="同 --full-6dof-thrust（四旋翼杆量语义别名）",
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
        default=DEFAULT_VERTICAL_KEYBOARD_BASE_TMG,
        help=f"竖直模式且无固定、无 ramp 时杆量零对应的 T/(mg)，默认 {DEFAULT_VERTICAL_KEYBOARD_BASE_TMG}",
    )
    parser.add_argument(
        "--vertical-xy-force-factor",
        type=float,
        default=DEFAULT_VERTICAL_XY_FORCE_FACTOR,
        help=(
            "竖直模式下 WASD 世界系水平力系数 k_xy（f∝k_xy·mg·杆量）；"
            f"默认 {DEFAULT_VERTICAL_XY_FORCE_FACTOR}；与 --vertical-pure-z 互斥"
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
    args = parser.parse_args()
    lock_world_up = not bool(args.vertical_use_scene_quat)
    fix_tmg = float(args.vertical_fixed_tmg)
    use_fixed = fix_tmg >= 0.0
    use_quad_flight = bool(args.full_6dof_thrust) or bool(args.quad_wasd_torque)
    vz_only = (not use_quad_flight) or bool(
        args.vertical_thrust_ramp or args.vertical_takeoff_bisect or use_fixed
    )
    xy_k = 0.0 if bool(args.vertical_pure_z) else float(args.vertical_xy_force_factor)
    if args.vertical_takeoff_bisect:
        run_takeoff_bisection(
            orcagym_addr=args.orcagym_addr,
            env_name=args.env_name,
            time_step=args.time_step,
            frame_skip=args.frame_skip,
            bisect_lo=float(args.bisect_lo),
            bisect_hi=float(args.bisect_hi),
            bisect_iters=int(args.bisect_iters),
            bisect_hold_s=float(args.bisect_hold_s),
            bisect_dz_m=float(args.bisect_dz),
            vertical_lock_quat_world_up=lock_world_up,
        )
    else:
        run_simulation(
            orcagym_addr=args.orcagym_addr,
            env_name=args.env_name,
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
            vertical_keyboard_baseline_tmg=float(args.vertical_keyboard_base_tmg),
            vertical_keyboard_xy_force_factor=xy_k,
        )
