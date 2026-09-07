"""
D12 ACT 模型推理 — 入口脚本

沿用 D12Env + OSC 控制器架构，将脚本轨迹驱动替换为 ACT 策略推理。
支持 ACT-Lite（无视觉，闭环）和 ACT-V（带视觉，可选参考轨迹注入）。

用法:
  # ACT-Lite (无视觉, 闭环推理)
  python run_d12_act.py \
      --checkpoint /path/to/act_lite/best_model.pt \
      --max_steps 3000

  # ACT-V (带视觉)
  python run_d12_act.py \
      --checkpoint /path/to/act_vision/best_model.pt \
      --max_steps 6300 \
      --capture_images

  # ACT-V + 参考轨迹注入 (开环状态)
  python run_d12_act.py \
      --checkpoint /path/to/act_vision/best_model.pt \
      --max_steps 6300 \
      --ref_trajectory /path/to/proprio_stats.hdf5 \
      --capture_images
"""
import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from scipy.spatial.transform import Rotation as R

from examples.embodied.d12.act_policy import ACTPolicy
from examples.embodied.d12.configs.d12_robot_config import d12_robot_config
from examples.embodied.d12.d12_env import D12Env

from orca_gym.log.orca_log import get_orca_logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

logger = get_orca_logger(
    name="D12ACT",
    log_file="d12_act.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


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


def create_env(orcagym_addr: str, agent_name: str, frame_skip: int = 5) -> D12Env:
    return D12Env(
        frame_skip=frame_skip,
        orcagym_addr=orcagym_addr,
        agent_names=[agent_name],
        time_step=0.001,
    )


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
    l_config["sim"] = env
    l_config["eef_name"] = agent_conf["l_arm"]["ee_center_site_name"]
    qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(agent_conf["l_arm"]["joint_names"])
    l_config["joint_indexes"] = {"joints": agent_conf["l_arm"]["joint_names"], "qpos": qpos_offsets, "qvel": qvel_offsets}
    l_config["actuator_range"] = l_ctrl_range
    l_config["policy_freq"] = 1.0 / env.dt
    l_config["ndim"] = len(agent_conf["l_arm"]["joint_names"])
    l_config["control_delta"] = False
    l_config["kp"] = 600
    l_config["damping_ratio"] = 0.5
    l_config["ramp_ratio"] = 1.0
    l_config["output_max"] = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    l_config["output_min"] = [-1.0, -1.0, -1.0, -2.0, -2.0, -2.0]
    l_controller = controller_factory(l_config["type"], l_config)
    l_controller.update_initial_joints(np.array(agent_conf["l_arm"]["neutral_joint_values"]))

    r_config = controller_config.load_config("osc_pose")
    r_config["robot_name"] = "d12"
    r_config["sim"] = env
    r_config["eef_name"] = agent_conf["r_arm"]["ee_center_site_name"]
    qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(agent_conf["r_arm"]["joint_names"])
    r_config["joint_indexes"] = {"joints": agent_conf["r_arm"]["joint_names"], "qpos": qpos_offsets, "qvel": qvel_offsets}
    r_config["actuator_range"] = r_ctrl_range
    r_config["policy_freq"] = 1.0 / env.dt
    r_config["ndim"] = len(agent_conf["r_arm"]["joint_names"])
    r_config["control_delta"] = False
    r_config["kp"] = 600
    r_config["damping_ratio"] = 0.5
    r_config["ramp_ratio"] = 1.0
    r_config["output_max"] = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    r_config["output_min"] = [-1.0, -1.0, -1.0, -2.0, -2.0, -2.0]
    r_controller = controller_factory(r_config["type"], r_config)
    r_controller.update_initial_joints(np.array(agent_conf["r_arm"]["neutral_joint_values"]))

    return l_controller, r_controller


def b_to_global(env, base_body: str, pos_b: np.ndarray, quat_xyzw_b: np.ndarray):
    _base_pose = env.get_body_xpos_xmat_xquat([base_body])[base_body]
    base_xpos = _base_pose["xpos"]
    base_xquat = _base_pose["xquat"]
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


def init_env_state(env, agent_conf: dict):
    default_qpos = {}
    for name, val in zip(agent_conf["l_arm"]["joint_names"], agent_conf["l_arm"]["neutral_joint_values"]):
        default_qpos[name] = np.array([val], dtype=np.float32)
    for name, val in zip(agent_conf["r_arm"]["joint_names"], agent_conf["r_arm"]["neutral_joint_values"]):
        default_qpos[name] = np.array([val], dtype=np.float32)
    env.apply_joint_qpos_dict(default_qpos)

    # Euler 体系 ctrl property getter 返回 actuator_force（只读），
    # 索引赋值不写入 _mjData.ctrl。用本地缓冲区收集后一次性 set_ctrl。
    ctrl_buf = np.zeros(env.nu, dtype=np.float32)
    for name, val in zip(agent_conf["l_arm"]["position_names"], agent_conf["l_arm"]["positions_init_ctrl"]):
        try:
            act_id = env.model.actuator_name2id(name)
            ctrl_buf[act_id] = val
        except (KeyError, Exception):
            pass
    for name, val in zip(agent_conf["r_arm"]["position_names"], agent_conf["r_arm"]["positions_init_ctrl"]):
        try:
            act_id = env.model.actuator_name2id(name)
            ctrl_buf[act_id] = val
        except (KeyError, Exception):
            pass
    for name, val in zip(agent_conf["gripper_l"]["actuator_names"], agent_conf["gripper_l"]["init_ctrl"]):
        try:
            act_id = env.model.actuator_name2id(name)
            ctrl_buf[act_id] = val
        except (KeyError, Exception):
            pass
    for name, val in zip(agent_conf["gripper_r"]["actuator_names"], agent_conf["gripper_r"]["init_ctrl"]):
        try:
            act_id = env.model.actuator_name2id(name)
            ctrl_buf[act_id] = val
        except (KeyError, Exception):
            pass
    env.set_ctrl(ctrl_buf)

    env.mj_forward()


def get_ee_pos_quat_b(env, agent_conf: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ee_site_l = agent_conf["l_arm"]["ee_center_site_name"]
    ee_site_r = agent_conf["r_arm"]["ee_center_site_name"]
    base_body = agent_conf["base_body"]

    ee_data = env.query_site_pos_and_quat_B([ee_site_l, ee_site_r], [base_body])
    l_pos = ee_data[ee_site_l]["xpos"].flatten()
    l_quat_wxyz = ee_data[ee_site_l]["xquat"].flatten()
    r_pos = ee_data[ee_site_r]["xpos"].flatten()
    r_quat_wxyz = ee_data[ee_site_r]["xquat"].flatten()

    l_quat_xyzw = l_quat_wxyz[[1, 2, 3, 0]]
    r_quat_xyzw = r_quat_wxyz[[1, 2, 3, 0]]

    return l_pos, l_quat_xyzw, r_pos, r_quat_xyzw


class ACTDriver:
    """逐帧将 ACT 策略输出写入 OSC 控制器。"""

    def __init__(
        self,
        env: D12Env,
        agent_conf: dict,
        l_controller,
        r_controller,
        policy: ACTPolicy,
        capture_images: bool = False,
        capture_every_n: int = 50,
        action_scale: float = 1.0,
    ):
        self.env = env
        self.agent_conf = agent_conf
        self.l_controller = l_controller
        self.r_controller = r_controller
        self.action_scale = action_scale
        self.policy = policy
        self.capture_images = capture_images
        self.capture_every_n = capture_every_n

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
        # Euler 体系 ctrl property 的 getter 返回 actuator_force（只读），
        # 索引赋值 self.env.ctrl[act_id] = val 不写入 _mjData.ctrl。
        # 维护本地 ctrl 缓冲区，在主循环 do_simulation 前一次性传入。
        self._ctrl_buf = np.zeros(env.nu, dtype=np.float32)

        self._tmp_dir = None
        if capture_images and policy.model_type == "act_vision":
            shm_base = "/dev/shm"
            if os.path.isdir(shm_base) and os.access(shm_base, os.W_OK):
                self._tmp_dir = tempfile.mkdtemp(prefix="orca_act_", dir=shm_base)
            else:
                self._tmp_dir = tempfile.mkdtemp(prefix="orca_act_")

    def _get_camera_images(self) -> np.ndarray | None:
        if not self.capture_images or self.policy.model_type != "act_vision":
            return None
        if self.policy.t % self.capture_every_n != 0:
            return None

        try:
            import cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False

        H = W = self.policy.img_size
        N = self.policy.num_cameras

        if self._tmp_dir is not None:
            frame_dir = os.path.join(self._tmp_dir, f"frame_{self.policy.t & 1}")
            color_dir = os.path.join(frame_dir, "color")
            if os.path.isdir(color_dir):
                shutil.rmtree(color_dir, ignore_errors=True)
            os.makedirs(frame_dir, exist_ok=True)

            try:
                camera_info = self.env.get_frame_png(frame_dir)
                available_cams = list(camera_info.keys()) if camera_info else []
                cams_to_use = [c for c in self.policy.camera_names if c in available_cams] if self.policy.camera_names else available_cams[:N]

                max_wait = 0.5
                poll_interval = 0.005
                t_start = time.monotonic()
                png_files = []
                while time.monotonic() - t_start < max_wait:
                    if os.path.isdir(color_dir):
                        png_files = [f for f in os.listdir(color_dir) if f.endswith(".png")]
                        if png_files:
                            fpath = os.path.join(color_dir, png_files[0])
                            try:
                                s1 = os.path.getsize(fpath)
                                if s1 > 0:
                                    time.sleep(0.005)
                                    s2 = os.path.getsize(fpath)
                                    if s2 == s1:
                                        break
                            except OSError:
                                pass
                    time.sleep(poll_interval)

                images = []
                if png_files:
                    for cam_name in cams_to_use[:N]:
                        cam_file = f"{cam_name}_color_0.png"
                        png_path = os.path.join(color_dir, cam_file)
                        if not os.path.isfile(png_path):
                            png_path = os.path.join(color_dir, png_files[0])
                        if has_cv2:
                            bgr = cv2.imread(png_path, cv2.IMREAD_COLOR)
                            if bgr is not None:
                                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
                                images.append(rgb.astype(np.float32).transpose(2, 0, 1) / 255.0)
                                continue
                        try:
                            from PIL import Image
                            img = Image.open(png_path).convert("RGB")
                            img = img.resize((W, H), Image.BILINEAR)
                            img_arr = np.array(img, dtype=np.float32) / 255.0
                            images.append(img_arr.transpose(2, 0, 1))
                        except Exception:
                            images.append(np.zeros((3, H, W), dtype=np.float32))

                while len(images) < N:
                    images.append(np.zeros((3, H, W), dtype=np.float32))
                return np.stack(images[:N], axis=0)

            except Exception as e:
                if self.policy.t < 5:
                    logger.warning(f"get_frame_png failed: {e}")

        return np.zeros((N, 3, H, W), dtype=np.float32) if self.policy.model_type == "act_vision" else None

    def step(self) -> bool:
        if self.policy.t >= self.policy.max_steps:
            return False

        state = self.policy.get_state(self.env, self.agent_conf)
        images = self._get_camera_images()
        action = self.policy.predict(state, images)
        parsed = ACTPolicy.parse_action(action)

        base_body = self.agent_conf["base_body"]
        l_action_global = b_to_global(self.env, base_body, parsed["l_pos"], parsed["l_quat_xyzw"])
        r_action_global = b_to_global(self.env, base_body, parsed["r_pos"], parsed["r_quat_xyzw"])

        if self.action_scale != 1.0:
            cur_l_pos, _, cur_r_pos, _ = get_ee_pos_quat_b(self.env, self.agent_conf)
            cur_l_pos = cur_l_pos.flatten()
            cur_r_pos = cur_r_pos.flatten()
            l_delta_pos = l_action_global[:3] - cur_l_pos
            r_delta_pos = r_action_global[:3] - cur_r_pos
            l_action_global[:3] = cur_l_pos + l_delta_pos * self.action_scale
            r_action_global[:3] = cur_r_pos + r_delta_pos * self.action_scale

        self.l_controller.set_goal(l_action_global)
        l_ctrl = self.l_controller.run_controller()
        if l_ctrl is None or np.any(np.isnan(l_ctrl)):
            l_ctrl = np.array(self.agent_conf["l_arm"]["neutral_joint_values"], dtype=np.float32)

        self.r_controller.set_goal(r_action_global)
        r_ctrl = self.r_controller.run_controller()
        if r_ctrl is None or np.any(np.isnan(r_ctrl)):
            r_ctrl = np.array(self.agent_conf["r_arm"]["neutral_joint_values"], dtype=np.float32)

        for i, act_id in enumerate(self.l_arm_actuator_ids):
            self._ctrl_buf[act_id] = l_ctrl[i]
        for i, act_id in enumerate(self.r_arm_actuator_ids):
            self._ctrl_buf[act_id] = r_ctrl[i]

        for act_id in self.l_grip_actuator_ids:
            lo, hi = self.all_ctrlrange[act_id]
            self._ctrl_buf[act_id] = np.clip(parsed["l_grip"], lo, hi)
        for act_id in self.r_grip_actuator_ids:
            lo, hi = self.all_ctrlrange[act_id]
            self._ctrl_buf[act_id] = np.clip(parsed["r_grip"], lo, hi)

        return True

    def cleanup(self):
        if self._tmp_dir and os.path.isdir(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="D12 ACT model inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument("--norm_stats", type=str, default=None, help="归一化统计量路径")
    parser.add_argument("--max_steps", type=int, default=3000, help="每个 episode 最大步数")
    parser.add_argument("--num_episodes", type=int, default=1, help="运行 episode 数")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--ema_alpha", type=float, default=0.9, help="EMA 平滑系数 (竞赛代码用0.9)")
    parser.add_argument("--ensemble_lambda", type=float, default=0.1, help="temporal ensemble 衰减系数")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--agent_name", type=str, default="d12_waist_usda_1")
    parser.add_argument("--frame_skip", type=int, default=5,
                        help="frame_skip (训练数据用5, 脚本轨迹demo用20)")
    parser.add_argument("--ref_trajectory", type=str, default=None, help="参考轨迹 HDF5 (ACT-V 开环注入)")
    parser.add_argument("--capture_images", action="store_true", help="启用相机图像采集 (ACT-V)")
    parser.add_argument("--capture_every_n", type=int, default=50, help="每 N 步采集一次图像")
    parser.add_argument("--exec_mode", type=str, default="ensemble",
                        choices=["ensemble", "chunk"],
                        help="执行模式: ensemble=每步推理+时间集成, chunk=每K步推理一次")
    parser.add_argument("--use_phase", action="store_true",
                        help="为 ACT-Lite 启用 phase 条件化 (追加时间步维度)")
    parser.add_argument("--no_sleep", action="store_true", help="不 sleep 到实时 (加速测试)")
    parser.add_argument("--action_scale", type=float, default=1.0,
                        help="动作放大系数 (>1.0 放大末端位置差值, 闭环时建议2.0~5.0)")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("D12 ACT Model Inference")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Device: {args.device}")
    logger.info("=" * 50)

    logger.info("Connecting to OrcaGym...")
    env = create_env(args.orcagym_addr, args.agent_name, args.frame_skip)
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

    policy = ACTPolicy(
        checkpoint_path=args.checkpoint,
        norm_stats_path=args.norm_stats,
        device_str=args.device,
        max_steps=args.max_steps,
        ema_alpha=args.ema_alpha,
        ensemble_lambda=args.ensemble_lambda,
        ref_trajectory_path=args.ref_trajectory,
        exec_mode=args.exec_mode,
        use_phase=args.use_phase,
    )
    logger.info(
        f"Policy loaded: type={policy.model_type} state_dim={policy.state_dim} "
        f"action_dim={policy.action_dim} chunk_size={policy.chunk_size}"
    )

    try:
        for ep in range(args.num_episodes):
            logger.info(f"=== Episode {ep + 1}/{args.num_episodes} ===")

            env.reset()
            disable_actuator_group(env, agent_conf, group=1)
            env.mj_forward()
            l_controller.reset_goal()
            r_controller.reset_goal()
            policy.reset()

            driver = ACTDriver(
                env, agent_conf, l_controller, r_controller, policy,
                capture_images=args.capture_images,
                capture_every_n=args.capture_every_n,
                action_scale=args.action_scale,
            )

            step_count = 0
            while driver.step():
                start_time = time.time()
                env.do_simulation(driver._ctrl_buf, env.frame_skip)
                env.render()
                step_count += 1

                if step_count % 100 == 0:
                    l_pos, l_quat, r_pos, r_quat = get_ee_pos_quat_b(env, agent_conf)
                    logger.info(
                        f"  step {step_count}/{args.max_steps}  "
                        f"L_ee=[{l_pos[0]:.3f},{l_pos[1]:.3f},{l_pos[2]:.3f}]  "
                        f"R_ee=[{r_pos[0]:.3f},{r_pos[1]:.3f},{r_pos[2]:.3f}]"
                    )

                if not args.no_sleep:
                    remain = env.realtime_step - (time.time() - start_time)
                    if remain > 0:
                        time.sleep(remain)

            logger.info(f"Episode {ep + 1} done, {step_count} steps")
            driver.cleanup()

    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
    finally:
        env.close()
        logger.info("Closed")


if __name__ == "__main__":
    main()
