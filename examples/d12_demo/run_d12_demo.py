"""
D12 脚本轨迹演示 — 入口脚本

直接使用 D12Env(OrcaGymLocalEnv) + OSC 控制器，不依赖 DualArmEnv（规避 robomimic 缺失问题）。

用法:
  python run_d12_demo.py --mode fixed
  python run_d12_demo.py --mode mp
  python run_d12_demo.py --mode rand-low
  python run_d12_demo.py --mode rand
  python run_d12_demo.py --mode dump --dump_keywords C12C Barcode Cardboardbox
  python run_d12_demo.py --pose_file pose_mp.yaml --rand_file rand_low.yaml --episodes 5
"""
import argparse
import os
import sys
import time
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.d12.scripts.pose_resolver import (
    build_segmented_trajectory,
    dump_object_poses,
    load_pose_spec_from_file,
    resolve_pose_spec_for_current_scene,
)
from envs.d12.scripts.object_randomizer import (
    advance_rand_spec_seed,
    apply_object_randomization,
    load_rand_spec_from_file,
)
from envs.d12.configs.d12_robot_config import d12_robot_config

from envs.d12.d12_env import D12Env
from orca_gym.log.orca_log import get_orca_logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

logger = get_orca_logger(
    name="D12Demo",
    log_file="d12_demo.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

MODE_PRESETS = {
    "fixed": {"pose_file": "pose_scripted.yaml", "rand_file": None, "episodes": 1},
    "mp": {"pose_file": "pose_mp.yaml", "rand_file": None, "episodes": 1},
    "rand-low": {"pose_file": "pose_mp.yaml", "rand_file": "rand_low.yaml", "episodes": 5},
    "rand": {"pose_file": "pose_mp.yaml", "rand_file": "rand.yaml", "episodes": 5},
    "dump": {"pose_file": "pose_mp.yaml", "rand_file": None, "episodes": 1},
}


def _detect_prefix(env) -> str:
    try:
        act_dict = env.model.get_actuator_dict()
        for name in act_dict:
            if "M_arm_l_01" in name:
                return name.replace("M_arm_l_01", "")
    except Exception:
        pass
    return ""


def _apply_prefix(cfg: dict, prefix: str) -> dict:
    result = {}
    for key, val in cfg.items():
        if isinstance(val, str):
            result[key] = prefix + val
        elif isinstance(val, list):
            result[key] = [prefix + v if isinstance(v, str) else v for v in val]
        elif isinstance(val, tuple):
            result[key] = tuple(prefix + v if isinstance(v, str) else v for v in val)
        else:
            result[key] = val
    return result


def build_prefixed_config(env) -> dict:
    prefix = _detect_prefix(env)
    conf = d12_robot_config
    return {
        "l_arm": _apply_prefix(conf["left_arm"], prefix),
        "r_arm": _apply_prefix(conf["right_arm"], prefix),
        "gripper_l": _apply_prefix(conf["gripper_l"], prefix),
        "gripper_r": _apply_prefix(conf["gripper_r"], prefix),
        "base_body": prefix + conf["base"]["base_body_name"],
        "_prefix": prefix,
    }


def create_env(orcagym_addr: str, agent_name: str) -> D12Env:
    env = D12Env(
        frame_skip=20,
        orcagym_addr=orcagym_addr,
        agent_names=[agent_name],
        time_step=0.001,
    )
    return env


def setup_osc_controllers(env, agent_conf: dict):
    from orca_gym.adapters.robosuite.controllers.controller_factory import controller_factory
    import orca_gym.adapters.robosuite.controllers.controller_config as controller_config

    l_ctrl_range = []
    for name in agent_conf["l_arm"]["motor_names"]:
        aid = env.model.actuator_name2id(name)
        l_ctrl_range.append(env.model.get_actuator_ctrlrange()[aid])

    r_ctrl_range = []
    for name in agent_conf["r_arm"]["motor_names"]:
        aid = env.model.actuator_name2id(name)
        r_ctrl_range.append(env.model.get_actuator_ctrlrange()[aid])

    l_config = controller_config.load_config("osc_pose")
    l_config["robot_name"] = "d12"
    l_config["sim"] = env.gym
    l_config["eef_name"] = agent_conf["l_arm"]["ee_center_site_name"]
    qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(agent_conf["l_arm"]["joint_names"])
    l_config["joint_indexes"] = {"joints": agent_conf["l_arm"]["joint_names"], "qpos": qpos_offsets, "qvel": qvel_offsets}
    l_config["actuator_range"] = l_ctrl_range
    l_config["policy_freq"] = 1.0 / env.dt
    l_config["ndim"] = len(agent_conf["l_arm"]["joint_names"])
    l_config["control_delta"] = False
    l_controller = controller_factory(l_config["type"], l_config)
    l_controller.update_initial_joints(np.array(agent_conf["l_arm"]["neutral_joint_values"]))

    r_config = controller_config.load_config("osc_pose")
    r_config["robot_name"] = "d12"
    r_config["sim"] = env.gym
    r_config["eef_name"] = agent_conf["r_arm"]["ee_center_site_name"]
    qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(agent_conf["r_arm"]["joint_names"])
    r_config["joint_indexes"] = {"joints": agent_conf["r_arm"]["joint_names"], "qpos": qpos_offsets, "qvel": qvel_offsets}
    r_config["actuator_range"] = r_ctrl_range
    r_config["policy_freq"] = 1.0 / env.dt
    r_config["ndim"] = len(agent_conf["r_arm"]["joint_names"])
    r_config["control_delta"] = False
    r_controller = controller_factory(r_config["type"], r_config)
    r_controller.update_initial_joints(np.array(agent_conf["r_arm"]["neutral_joint_values"]))

    return l_controller, r_controller


def b_to_global(env, base_body: str, pos_b: np.ndarray, quat_xyzw_b: np.ndarray):
    base_xpos, _, base_xquat = env.get_body_xpos_xmat_xquat([base_body])
    base_pos = base_xpos.reshape(3)
    base_quat_wxyz = base_xquat.reshape(4)
    base_quat_xyzw = base_quat_wxyz[[1, 2, 3, 0]]
    base_rot = R.from_quat(base_quat_xyzw)
    pos_global = base_pos + base_rot.apply(pos_b)
    quat_global_xyzw = (base_rot * R.from_quat(quat_xyzw_b)).as_quat()
    from orca_gym.adapters.robosuite.utils import transform_utils
    axisangle = transform_utils.quat2axisangle(quat_global_xyzw)
    return np.concatenate([pos_global, axisangle], dtype=np.float32)


def disable_actuator_group(env, agent_conf: dict, group: int):
    dummy_joint_name = agent_conf.get("base", {}).get("dummy_joint_name", "dummy_joint")
    try:
        dummy_id = env.model.joint_name2id(dummy_joint_name)
    except (KeyError, Exception):
        dummy_id = None

    if group == 0:
        names = agent_conf["l_arm"]["motor_names"] + agent_conf["r_arm"]["motor_names"]
    else:
        names = agent_conf["l_arm"]["position_names"] + agent_conf["r_arm"]["position_names"]

    for name in names:
        try:
            act_id = env.model.actuator_name2id(name)
            if dummy_id is not None:
                env.model.set_actuator_joint(act_id, dummy_id)
        except (KeyError, Exception):
            pass


class ScriptedDriver:
    """逐帧将 B 系末端位姿 + 夹爪写入 OSC 控制器。"""

    def __init__(
        self,
        env: D12Env,
        agent_conf: dict,
        l_controller,
        r_controller,
        l_pos: np.ndarray,
        l_quat_xyzw: np.ndarray,
        r_pos: np.ndarray,
        r_quat_xyzw: np.ndarray,
        l_grip_motor: np.ndarray,
        r_grip_motor: np.ndarray,
    ):
        self.env = env
        self.agent_conf = agent_conf
        self.l_controller = l_controller
        self.r_controller = r_controller
        self.l_pos = l_pos
        self.l_quat_xyzw = l_quat_xyzw
        self.r_pos = r_pos
        self.r_quat_xyzw = r_quat_xyzw
        self.l_grip_motor = l_grip_motor
        self.r_grip_motor = r_grip_motor
        self.t = 0
        self.total = len(l_pos)

        self.l_arm_actuator_ids = [
            env.model.actuator_name2id(n) for n in agent_conf["l_arm"]["motor_names"]
        ]
        self.r_arm_actuator_ids = [
            env.model.actuator_name2id(n) for n in agent_conf["r_arm"]["motor_names"]
        ]
        self.l_grip_actuator_ids = [
            env.model.actuator_name2id(n) for n in agent_conf["gripper_l"]["actuator_names"]
        ]
        self.r_grip_actuator_ids = [
            env.model.actuator_name2id(n) for n in agent_conf["gripper_r"]["actuator_names"]
        ]
        self.all_ctrlrange = env.model.get_actuator_ctrlrange()

    def step(self) -> bool:
        if self.t >= self.total:
            return False

        base_body = self.agent_conf["base_body"]
        l_action_global = b_to_global(
            self.env, base_body, self.l_pos[self.t], self.l_quat_xyzw[self.t]
        )
        r_action_global = b_to_global(
            self.env, base_body, self.r_pos[self.t], self.r_quat_xyzw[self.t]
        )

        self.l_controller.set_goal(l_action_global)
        l_ctrl = self.l_controller.run_controller()
        if l_ctrl is None or np.any(np.isnan(l_ctrl)):
            l_ctrl = np.array(self.agent_conf["l_arm"]["neutral_joint_values"], dtype=np.float32)

        self.r_controller.set_goal(r_action_global)
        r_ctrl = self.r_controller.run_controller()
        if r_ctrl is None or np.any(np.isnan(r_ctrl)):
            r_ctrl = np.array(self.agent_conf["r_arm"]["neutral_joint_values"], dtype=np.float32)

        for i, act_id in enumerate(self.l_arm_actuator_ids):
            self.env.ctrl[act_id] = l_ctrl[i]
        for i, act_id in enumerate(self.r_arm_actuator_ids):
            self.env.ctrl[act_id] = r_ctrl[i]

        l_grip = float(self.l_grip_motor[self.t])
        r_grip = float(self.r_grip_motor[self.t])
        for act_id in self.l_grip_actuator_ids:
            lo, hi = self.all_ctrlrange[act_id]
            self.env.ctrl[act_id] = np.clip(l_grip, lo, hi)
        for act_id in self.r_grip_actuator_ids:
            lo, hi = self.all_ctrlrange[act_id]
            self.env.ctrl[act_id] = np.clip(r_grip, lo, hi)

        self.t += 1
        return True


def init_env_state(env, agent_conf: dict):
    default_qpos = {}
    for name, val in zip(agent_conf["l_arm"]["joint_names"], agent_conf["l_arm"]["neutral_joint_values"]):
        default_qpos[name] = np.array([val], dtype=np.float32)
    for name, val in zip(agent_conf["r_arm"]["joint_names"], agent_conf["r_arm"]["neutral_joint_values"]):
        default_qpos[name] = np.array([val], dtype=np.float32)
    env.set_joint_qpos(default_qpos)

    for name, val in zip(agent_conf["l_arm"]["position_names"], agent_conf["l_arm"]["positions_init_ctrl"]):
        try:
            act_id = env.model.actuator_name2id(name)
            env.ctrl[act_id] = val
        except (KeyError, Exception):
            pass
    for name, val in zip(agent_conf["r_arm"]["position_names"], agent_conf["r_arm"]["positions_init_ctrl"]):
        try:
            act_id = env.model.actuator_name2id(name)
            env.ctrl[act_id] = val
        except (KeyError, Exception):
            pass
    for name, val in zip(agent_conf["gripper_l"]["actuator_names"], agent_conf["gripper_l"]["init_ctrl"]):
        try:
            act_id = env.model.actuator_name2id(name)
            env.ctrl[act_id] = val
        except (KeyError, Exception):
            pass
    for name, val in zip(agent_conf["gripper_r"]["actuator_names"], agent_conf["gripper_r"]["init_ctrl"]):
        try:
            act_id = env.model.actuator_name2id(name)
            env.ctrl[act_id] = val
        except (KeyError, Exception):
            pass

    env.mj_forward()


def resolve_path(base_dir: str, filename: str) -> str:
    expanded = os.path.expanduser(filename)
    if os.path.isabs(expanded):
        return expanded
    candidate = os.path.join(base_dir, expanded)
    if os.path.exists(candidate):
        return candidate
    return candidate


def main():
    parser = argparse.ArgumentParser(
        description="D12 scripted trajectory demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  fixed      Fixed scene + base-frame trajectory
  mp         Object-frame trajectory, no randomization
  rand-low   Object-frame + low randomization
  rand       Object-frame + medium randomization
  dump       Print object poses in base frame

Examples:
  python run_d12_demo.py --mode fixed
  python run_d12_demo.py --mode mp
  python run_d12_demo.py --mode rand-low --episodes 10
  python run_d12_demo.py --mode dump --dump_keywords C12C Barcode
""",
    )
    parser.add_argument("--mode", type=str, default=None,
                        choices=list(MODE_PRESETS.keys()))
    parser.add_argument("--pose_file", type=str, default=None)
    parser.add_argument("--rand_file", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--agent_name", type=str, default="d12_waist_usda_1")
    parser.add_argument("--dump_keywords", type=str, nargs="+", default=None)
    args = parser.parse_args()

    if args.mode is not None:
        preset = MODE_PRESETS[args.mode]
        if args.pose_file is None:
            args.pose_file = preset["pose_file"]
        if args.rand_file is None:
            args.rand_file = preset["rand_file"]
        if args.episodes is None:
            args.episodes = preset["episodes"]
    else:
        if args.pose_file is None:
            parser.error("Either --mode or --pose_file is required")
        if args.episodes is None:
            args.episodes = 1

    pose_path = resolve_path(BASE_DIR, args.pose_file)
    rand_path = resolve_path(BASE_DIR, args.rand_file) if args.rand_file else None

    spec = load_pose_spec_from_file(pose_path)
    rand_spec = load_rand_spec_from_file(rand_path) if rand_path else {}

    g_open = float(spec.get("gripper_open", 0.0))
    g_close = float(spec.get("gripper_close", 220.0))

    logger.info("=" * 50)
    logger.info("D12 Scripted Trajectory Demo")
    logger.info(f"  Pose: {pose_path}")
    logger.info(f"  Rand: {rand_path or 'none'}")
    logger.info(f"  Episodes: {args.episodes}")
    logger.info("=" * 50)

    logger.info("Connecting to OrcaGym...")
    env = create_env(args.orcagym_addr, args.agent_name)

    env.reset()
    env.mj_forward()

    agent_conf = build_prefixed_config(env)
    logger.info(f"  Prefix: '{agent_conf['_prefix']}'")
    logger.info(f"  Base body: {agent_conf['base_body']}")
    logger.info(f"  EE site L: {agent_conf['l_arm']['ee_center_site_name']}")
    logger.info(f"  EE site R: {agent_conf['r_arm']['ee_center_site_name']}")

    disable_actuator_group(env, agent_conf, group=1)

    l_controller, r_controller = setup_osc_controllers(env, agent_conf)
    logger.info("OSC controllers initialized")

    base_body = agent_conf["base_body"]
    ee_site_l = agent_conf["l_arm"]["ee_center_site_name"]
    ee_site_r = agent_conf["r_arm"]["ee_center_site_name"]

    try:
        for ep in range(args.episodes):
            logger.info(f"=== Episode {ep + 1}/{args.episodes} ===")

            env.reset()
            init_env_state(env, agent_conf)
            l_controller.reset_goal()
            r_controller.reset_goal()

            episode_rand = advance_rand_spec_seed(rand_spec, ep)
            if episode_rand:
                logger.info("Applying object randomization...")
                apply_object_randomization(env, episode_rand)

            if args.mode == "dump" or args.dump_keywords:
                keywords = args.dump_keywords or ["C12C", "Barcode", "Cardboardbox"]
                print(f"\nObject poses relative to {base_body}:")
                dump_object_poses(env, base_body, keywords)
                continue

            resolved_spec = resolve_pose_spec_for_current_scene(env, base_body, spec)

            if not resolved_spec.get("segments"):
                logger.error("No segments found in pose spec")
                continue

            l_pos, l_quat, r_pos, r_quat, l_grip, r_grip = build_segmented_trajectory(
                env, base_body, ee_site_l, ee_site_r,
                resolved_spec["segments"], g_open, g_close,
            )
            logger.info(f"Trajectory: {len(l_pos)} steps")

            driver = ScriptedDriver(
                env, agent_conf, l_controller, r_controller,
                l_pos, l_quat, r_pos, r_quat, l_grip, r_grip,
            )

            step_count = 0
            while driver.step():
                start_time = time.time()
                env.do_simulation(env.ctrl, env.frame_skip)
                env.render()
                step_count += 1
                if step_count % 100 == 0:
                    logger.info(f"  step {step_count}/{driver.total}")
                remain = env.realtime_step - (time.time() - start_time)
                if remain > 0:
                    time.sleep(remain)

            logger.info(f"Episode {ep + 1} done, {step_count} steps")

    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
    finally:
        env.close()
        logger.info("Closed")


if __name__ == "__main__":
    main()
