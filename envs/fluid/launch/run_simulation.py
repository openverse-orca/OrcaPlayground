"""流体仿真启动编排：Gym、scene、OrcaLink/OrcaSPH、主循环与清理。"""
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import gymnasium as gym

from ..orcalink_bridge import OrcaLinkBridge
from ..trajectory import TrajectoryRecorder, TrajectoryPlayer
from ..paths import FLUID_PACKAGE_DIR, ORCA_PLAYGROUND_ROOT
from ..utils.scene_generator import SceneGenerator
from ..utils.merge_particle_mujoco_h5 import merge_particle_mujoco_sidecar_into_particle_h5
from ..utils.mujoco_qpos_sidecar_recorder import (
    maybe_open_sidecar_for_record_config,
    mujoco_qpos_sidecar_tmp_path,
)
from .fluid_session import (
    _fluid_atexit_state,
    _fluid_send_end_simulation_from_config,
    _fluid_sync_initial_viewport_to_engine,
    _terminate_stats_plot_proc,
    _try_start_record_stats_plot_viewer,
    resolve_record_stats_orcasph_log_path,
)
from .orcasph_log_utils import log_fluid_particle_count_to_terminal
from .water_jug_trajectory_hook import apply_water_jug_trajectory, setup_water_jug_trajectory
from .process_utils import ProcessManager, is_tcp_port_accepting_connections
from .sph_config import generate_orcasph_config, setup_python_logging
from .steering_torque_recorder import SteeringTorqueRecorder
from .steer_coupling_recorder import SteerCouplingRecorder
from .generic_body_track_recorder import GenericCouplingRecorder, GenericTorqueBalanceRecorder
from ..csv_numeric import fmt_f

logger = logging.getLogger(__name__)

# 主循环控制周期（秒）；与 mujoco_trajectory 录制 meta 等一致。
REALTIME_STEP = 0.02


def _log_sph_scene_clock_fields(scene_data: Dict[str, Any], scene_path: Path) -> None:
    """排查用：记录 scene.json 中与仿真停止/暂停相关的 Configuration 字段。"""
    cfg = scene_data.get("Configuration")
    if not isinstance(cfg, dict):
        logger.info("SPH 排查: scene 无 Configuration 字典，path=%s", scene_path)
        return
    keys = ("stopAt", "pauseAt")
    present = {k: cfg[k] for k in keys if k in cfg}
    if present:
        logger.info("SPH 排查: scene Configuration 时间控制项 %s（path=%s）", present, scene_path)
    else:
        logger.info(
            "SPH 排查: scene Configuration 未含 %s（SPlisHSPlasH 缺省 stopAt<0 表示不按该时间停止）。"
            "若 sph_monitor 中 sim_time 仍在约 5s 截断，请查本会话 orcasph_*.log 是否出现 stopAt/--stopAt，"
            "并勿将 time_sync.csv 的 sph_sim_time（sequence/Hz）与 monitor 的物理 sim_time 混读。",
            keys,
        )


@dataclass
class FluidSimulationContext:
    """单次流体会话的可变状态（SIGTERM、主循环、finally 共享）。"""

    config: Dict
    session_timestamp: str
    cpu_affinity: Optional[str]
    orcagym_tmp_dir: Path
    process_manager: ProcessManager
    shutdown_event: threading.Event = field(default_factory=threading.Event)
    # >0 时主循环仅执行这么多步后正常退出（无 GUI 自动跑通、回归检查用；0/None=无限）
    max_steps: int = 0
    max_sim_time: float = 0.0

    env: Any = None
    sph_wrapper: Any = None
    mujoco_viewer: Any = None
    traj_rec: Any = None
    traj_player: Any = None
    traj_stats_log_f: Any = None
    mujoco_qpos_sidecar: Any = None
    scene_output_path: Optional[Path] = None
    particle_render_override: Any = None
    sphscale: float = 1.0
    prev_sigterm_handler: Any = None
    water_jug_driver: Any = None
    # water_jug_trajectory：轨迹期间不向 MuJoCo 水壶写 SPH 力；step 后是否再对齐 qpos
    water_jug_skip_sph_forces_on_mujoco: bool = False
    water_jug_reapply_after_step: bool = False
    water_jug_clear_external_forces: bool = False


def _resolve_cli_binary(command_name: str, pip_install_hint: str) -> Path:
    """在当前 Python 环境或 PATH 中解析可执行文件路径。"""
    python_bin_dir = Path(sys.executable).parent
    bin_path = python_bin_dir / command_name
    if not bin_path.exists():
        which_path = shutil.which(command_name)
        if which_path:
            bin_path = Path(which_path)
        else:
            raise FileNotFoundError(
                f"{command_name} command not found. "
                f"Searched: {python_bin_dir / command_name}, PATH. "
                f"{pip_install_hint}"
            )
    return bin_path


def _make_sigterm_cleanup_handler(
    ctx: FluidSimulationContext, shutdown_event: threading.Event
):
    """
    SIGTERM handler：同步完成全量清理后退出，保证在 OrcaLab kill 前完成。
    读 ctx 上当前 sph_wrapper / process_manager / config。
    """

    def _handler(_signum, _frame):
        logger.info("\n⏹️  收到 SIGTERM，开始同步清理（OrcaLab 停止）...")
        shutdown_event.set()
        if not _fluid_atexit_state.get("viewport_reset_done"):
            _owns = _fluid_atexit_state.get("owns_shared_services")
            _env = _fluid_atexit_state.get("env_ref")

            _terminate_stats_plot_proc()

            # 1. 断开 OrcaLink Bridge（停止推位置给 OrcaSPH）
            if ctx.sph_wrapper is not None:
                try:
                    ctx.sph_wrapper.close()
                except Exception:
                    pass

            # 2. 告知 ParticleRender 结束仿真（停止接收粒子帧）
            if _owns:
                try:
                    _fluid_send_end_simulation_from_config(ctx.config)
                except Exception:
                    pass

            # 3. 终止 OrcaSPH / OrcaLink 子进程
            ctx.process_manager.cleanup_all()

            # 4. 等待在途粒子帧被丢弃
            if _owns:
                time.sleep(0.2)

            # 5. 重置刚体位姿并推给 OrcaSim
            if _env is not None and _owns:
                try:
                    _fluid_sync_initial_viewport_to_engine(_env)
                except Exception as _e:
                    logger.warning(f"SIGTERM 清理：同步视口失败: {_e}")
                try:
                    _env.close()
                except Exception:
                    pass
            elif _env is not None:
                try:
                    _env.close()
                except Exception:
                    pass

            _fluid_atexit_state["viewport_reset_done"] = True
            _fluid_atexit_state["session_active"] = False
        logger.info("✅ SIGTERM 清理完成，退出")
        os._exit(0)

    return _handler


def _preflight_session(
    config: Dict, session_timestamp: Optional[str]
) -> Tuple[str, Path]:
    """
    时间戳、日志、临时目录、OrcaLink 端口检查。
    粒子 HDF5 playback 由 ``run_fluid_sim`` 在调用本函数之前单独处理，不经过本路径。
    """
    if session_timestamp is None:
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    setup_python_logging(config)

    orcagym_tmp_dir = Path.home() / ".orcagym" / "tmp"
    orcagym_tmp_dir.mkdir(parents=True, exist_ok=True)

    # 在注册 SIGTERM / ProcessManager / try-finally 之前做端口检查，避免第二实例走到 finally
    # 时仍调用 EndSimulation，误伤已占用该端口的第一个流体会话的粒子渲染。
    orcalink_cfg = config.get("orcalink", {})
    if orcalink_cfg.get("enabled", True) and orcalink_cfg.get("auto_start", True):
        link_host = orcalink_cfg.get("host", "localhost")
        if link_host in ("0.0.0.0", "::", ""):
            link_host = "127.0.0.1"
        link_port = int(orcalink_cfg.get("port", 50351))
        if is_tcp_port_accepting_connections(link_host, link_port):
            logger.error(
                "❌ OrcaLink 端口 %s:%s 已被占用，本脚本无法在此端口再启动 orcalink。\n"
                "   请先结束占用该端口的进程（例如已在运行的流体仿真），或改用其它端口；\n"
                "   若由 OrcaLab 等已提供 OrcaLink，请将配置中 orcalink.auto_start 设为 false 并用手动/外部方式启动。",
                link_host,
                link_port,
            )
            sys.exit(1)

    return session_timestamp, orcagym_tmp_dir


def _init_atexit_state_for_session(config: Dict) -> None:
    _fluid_atexit_state["session_active"] = True
    _fluid_atexit_state["viewport_reset_done"] = False
    _fluid_atexit_state["env_ref"] = None
    _fluid_atexit_state["config_ref"] = config
    _fluid_atexit_state["owns_shared_services"] = False


def _create_and_reset_gym_env(config: Dict) -> Any:
    logger.info("\n📦 步骤 1: 创建 MuJoCo 环境...")
    orcagym_cfg = config["orcagym"]
    env_id = f"{orcagym_cfg['env_name']}-OrcaGym-{orcagym_cfg['address'].replace(':', '-')}-000"

    print(
        "[PRINT-DEBUG] run_simulation.py - About to register gymnasium env",
        file=sys.stderr,
        flush=True,
    )
    gym.register(
        id=env_id,
        entry_point="envs.fluid.sim_env:SimEnv",
        kwargs={
            "frame_skip": 20,
            "orcagym_addr": orcagym_cfg["address"],
            "agent_names": [orcagym_cfg["agent_name"]],
            "time_step": 0.001,
        },
        max_episode_steps=sys.maxsize,
    )
    print(
        "[PRINT-DEBUG] run_simulation.py - Gymnasium env registered",
        file=sys.stderr,
        flush=True,
    )

    print(
        "[PRINT-DEBUG] run_simulation.py - About to call gym.make()",
        file=sys.stderr,
        flush=True,
    )
    # SimEnv.step(None) 表示无外围动作；需关闭环境检查器，否则 None 无法通过 Box 校验
    env = gym.make(env_id, disable_env_checker=True)
    _fluid_atexit_state["env_ref"] = env
    _fluid_atexit_state["owns_shared_services"] = True
    print(
        "[PRINT-DEBUG] run_simulation.py - gym.make() completed",
        file=sys.stderr,
        flush=True,
    )

    print(
        "[PRINT-DEBUG] run_simulation.py - About to call env.reset()",
        file=sys.stderr,
        flush=True,
    )
    env.reset()
    print(
        "[PRINT-DEBUG] run_simulation.py - env.reset() completed",
        file=sys.stderr,
        flush=True,
    )
    logger.info("✅ MuJoCo 环境创建成功\n")
    return env


def _maybe_launch_mujoco_viewer(ctx: FluidSimulationContext) -> None:
    if not ctx.config.get("mujoco_gui", False):
        return
    try:
        import mujoco
        import mujoco.viewer
        import numpy as np

        env = ctx.env
        unwrapped = env.unwrapped
        mj_model = unwrapped.gym._mjModel
        mj_data = unwrapped.gym._mjData

        mujoco.mj_forward(mj_model, mj_data)

        logger.info(
            "🖥️  MuJoCo 模型信息: ngeom=%d, nbody=%d, nmesh=%d, nlight=%d",
            mj_model.ngeom, mj_model.nbody, mj_model.nmesh, mj_model.nlight,
        )

        valid_positions = []
        for i in range(mj_model.nbody):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if "dummy" not in name.lower() and "manipulator" not in name.lower():
                valid_positions.append(mj_data.xpos[i].copy())
        if valid_positions:
            vp = np.array(valid_positions)
            scene_center = vp.mean(axis=0)
            scene_extent = float(np.linalg.norm(vp.max(axis=0) - vp.min(axis=0)))
        else:
            scene_center = mj_model.stat.center[:].copy()
            scene_extent = float(mj_model.stat.extent)

        logger.info(
            "🖥️  场景中心(排除dummy): %s, 范围: %.2f",
            scene_center, scene_extent,
        )

        if mj_model.nmat == 0:
            logger.info("🖥️  模型无材质定义，增强光照参数并分配 geom 颜色")
            mj_model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
            mj_model.vis.headlight.diffuse[:] = [0.9, 0.9, 0.9]
            mj_model.vis.headlight.specular[:] = [0.5, 0.5, 0.5]
            mj_model.vis.global_.offwidth = 1920
            mj_model.vis.global_.offheight = 1080

            _palette = np.array([
                [0.85, 0.33, 0.10, 1.0],
                [0.10, 0.60, 0.85, 1.0],
                [0.20, 0.75, 0.30, 1.0],
                [0.90, 0.75, 0.10, 1.0],
                [0.70, 0.20, 0.70, 1.0],
                [0.10, 0.75, 0.70, 1.0],
                [0.85, 0.55, 0.10, 1.0],
                [0.40, 0.40, 0.85, 1.0],
                [0.85, 0.20, 0.40, 1.0],
                [0.55, 0.80, 0.20, 1.0],
            ], dtype=np.float32)
            body_color_idx = {}
            for gi in range(mj_model.ngeom):
                bid = mj_model.geom_bodyid[gi]
                if bid not in body_color_idx:
                    body_color_idx[bid] = len(body_color_idx) % len(_palette)
                mj_model.geom_rgba[gi] = _palette[body_color_idx[bid]]

        mj_model.vis.map.znear = 0.0005
        mj_model.vis.map.zfar = 100.0
        mj_model.stat.extent = max(scene_extent, 1.0)
        mj_model.stat.center[:] = scene_center

        ctx.mujoco_viewer = mujoco.viewer.launch_passive(
            mj_model, mj_data, show_left_ui=True, show_right_ui=True
        )

        with ctx.mujoco_viewer.lock():
            ctx.mujoco_viewer.cam.azimuth = 135.0
            ctx.mujoco_viewer.cam.elevation = -25.0
            ctx.mujoco_viewer.cam.distance = max(5.0, scene_extent * 1.5)
            ctx.mujoco_viewer.cam.lookat[:] = scene_center
            ctx.mujoco_viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = 1
            ctx.mujoco_viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0

        logger.info("🖥️  MuJoCo 原生查看器已启动")
    except Exception as e:
        logger.warning(f"MuJoCo 原生查看器启动失败: {e}")
        ctx.mujoco_viewer = None


def _maybe_generate_sph_scene(ctx: FluidSimulationContext) -> None:
    ctx.particle_render_override = None
    config = ctx.config
    if not (config["orcasph"]["enabled"] and config["orcasph"]["scene_auto_generate"]):
        return

    logger.info("📝 步骤 2: 生成 SPH scene.json...")
    scene_uuid = str(uuid4()).replace("-", "_")
    ctx.scene_output_path = ctx.orcagym_tmp_dir / f"sph_scene_{scene_uuid}.json"

    scene_config_path = (
        ORCA_PLAYGROUND_ROOT / "examples" / "fluid" / config["sph"]["scene_config"]
    )
    if not scene_config_path.exists():
        scene_config_path = FLUID_PACKAGE_DIR / config["sph"]["scene_config"]

    sph_config_template_path = (
        ORCA_PLAYGROUND_ROOT / "examples" / "fluid" / config["orcasph"]["config_template"]
    )
    if not sph_config_template_path.exists():
        sph_config_template_path = FLUID_PACKAGE_DIR / config["orcasph"]["config_template"]

    if sph_config_template_path.exists():
        with open(sph_config_template_path, "r", encoding="utf-8") as f:
            sph_config = json.load(f)
        logger.info(f"✅ 加载 SPH 配置模板用于场景生成: {sph_config_template_path}")
    else:
        raise FileNotFoundError(
            f"SPH 配置模板未找到: {config['orcasph']['config_template']}\n"
            f"场景生成需要从该文件读取弹簧参数等配置。\n"
            f"尝试的路径: {sph_config_template_path}"
        )

    scene_generator = SceneGenerator(
        ctx.env.unwrapped,
        config_path=str(scene_config_path),
        runtime_config=sph_config,
    )
    scene_data = scene_generator.generate_complete_scene(
        output_path=str(ctx.scene_output_path),
        include_fluid_blocks=config["sph"]["include_fluid_blocks"],
        include_wall=config["sph"]["include_wall"],
    )
    logger.info(f"✅ scene.json 已生成: {ctx.scene_output_path}")
    logger.info(f"   - RigidBodies: {len(scene_data.get('RigidBodies', []))} 个\n")
    _log_sph_scene_clock_fields(scene_data, ctx.scene_output_path)

    ctx.particle_render_override = scene_generator.generate_particle_render_config(
        sph_config
    )
    ctx.sphscale = scene_generator.sphscale


def _start_orcalink_if_configured(ctx: FluidSimulationContext) -> None:
    config = ctx.config
    if not (config["orcalink"]["enabled"] and config["orcalink"]["auto_start"]):
        return

    logger.info("🚀 步骤 3: 启动 OrcaLink Server...")

    orcalink_bin = _resolve_cli_binary(
        "orcalink",
        "Please ensure orca-link is installed: pip install -e /path/to/OrcaLink",
    )

    orcalink_port = config["orcalink"].get("port", 50351)
    orcalink_args = ["--port", str(orcalink_port)]

    if "args" in config["orcalink"]:
        for arg in config["orcalink"]["args"]:
            if arg not in ["--port", str(orcalink_port)]:
                orcalink_args.append(arg)

    logger.info(f"启动 OrcaLink，端口: {orcalink_port}")
    log_file = ctx.orcagym_tmp_dir / f"orcalink_{ctx.session_timestamp}.log"
    ctx.process_manager.start_process(
        "OrcaLink",
        str(orcalink_bin),
        orcalink_args,
        log_file,
    )

    startup_delay = config["orcalink"].get("startup_delay", 5)
    logger.info(f"⏳ 等待 OrcaLink 启动完成（{startup_delay} 秒）...")
    time.sleep(startup_delay)
    logger.info(f"✅ OrcaLink Server 已就绪\n")


def _start_orcasph_if_configured(ctx: FluidSimulationContext) -> None:
    config = ctx.config
    if not (config["orcasph"]["enabled"] and config["orcasph"]["auto_start"]):
        return

    if ctx.scene_output_path is None:
        logger.error("❌ 无法启动 OrcaSPH：scene.json 未生成")
        config["orcasph"]["enabled"] = False
        return

    logger.info("🚀 步骤 4: 启动 OrcaSPH...")

    orcasph_bin = _resolve_cli_binary(
        "orcasph",
        "Please ensure orca-sph is installed: pip install -e /path/to/SPlisHSPlasH",
    )

    orcasph_config_path = (
        ctx.orcagym_tmp_dir / f"orcasph_config_{ctx.session_timestamp}.json"
    )
    orcasph_config_path, verbose_logging = generate_orcasph_config(
        config,
        orcasph_config_path,
        particle_render_override=ctx.particle_render_override,
        sphscale=ctx.sphscale,
    )

    orcasph_args = config["orcasph"]["args"].copy()
    orcasph_args.extend(["--config", str(orcasph_config_path)])
    orcasph_args.extend(["--scene", str(ctx.scene_output_path)])

    max_t = float(ctx.max_sim_time or 0.0)
    if max_t > 0.0 and "--stopAt" not in orcasph_args:
        orcasph_args.extend(["--stopAt", f"{max_t:.6g}"])
        logger.info(
            "SPH 排查: 已将主循环 max_sim_time=%.6g s 映射为 OrcaSPH 追加参数 --stopAt（与引擎提前退出条件一致）",
            max_t,
        )
    elif max_t > 0.0 and "--stopAt" in orcasph_args:
        logger.info(
            "SPH 排查: orcasph.args 已含 --stopAt，未自动覆盖；主循环仍按 max_sim_time=%.6g s 退出",
            max_t,
        )

    if verbose_logging:
        orcasph_args.extend(["--log-level", "DEBUG"])
        logger.info("🔍 启用 DEBUG 日志级别 (verbose_logging=true)")
    else:
        logger.info("ℹ️  使用默认 INFO 日志级别 (verbose_logging=false)")

    if ctx.cpu_affinity:
        logger.info(f"📌 OrcaSPH CPU 亲和性: 核心 {ctx.cpu_affinity}")
        orcasph_cmd = "taskset"
        orcasph_args = ["-c", ctx.cpu_affinity, str(orcasph_bin)] + orcasph_args
    else:
        orcasph_cmd = str(orcasph_bin)

    log_file = ctx.orcagym_tmp_dir / f"orcasph_{ctx.session_timestamp}.log"
    ctx.process_manager.start_process(
        "OrcaSPH",
        orcasph_cmd,
        orcasph_args,
        log_file,
    )
    logger.info("⏳ 等待 OrcaSPH 初始化并读取流体粒子数量...")
    log_fluid_particle_count_to_terminal(log_file, logger, timeout_sec=30.0)
    logger.info("✅ OrcaSPH 已启动\n")


def _connect_sph_bridge_if_enabled(ctx: FluidSimulationContext) -> None:
    config = ctx.config
    if not config["orcasph"]["enabled"]:
        logger.warning("⚠️  OrcaLink 未启用，SPH 集成已禁用")
        return

    logger.info("🔗 步骤 5: 初始化 OrcaLinkBridge...")
    logger.debug("[DEBUG] Creating OrcaLinkBridge instance...")
    print(
        "[PRINT-DEBUG] run_simulation.py - Creating OrcaLinkBridge instance...",
        file=sys.stderr,
        flush=True,
    )
    ctx.sph_wrapper = OrcaLinkBridge(ctx.env.unwrapped, config=config)
    logger.debug("[DEBUG] OrcaLinkBridge instance created")
    print(
        "[PRINT-DEBUG] run_simulation.py - OrcaLinkBridge instance created...",
        file=sys.stderr,
        flush=True,
    )

    logger.info("🔗 连接到 OrcaLink...")
    logger.debug("[DEBUG] Calling sph_wrapper.connect()...")
    sys.stdout.flush()
    sys.stderr.flush()
    print(
        "[PRINT-DEBUG] run_simulation.py - Calling sph_wrapper.connect()...",
        file=sys.stderr,
        flush=True,
    )
    connect_result = ctx.sph_wrapper.connect()
    print(
        f"[PRINT-DEBUG] run_simulation.py - sph_wrapper.connect() returned: {connect_result}",
        file=sys.stderr,
        flush=True,
    )
    logger.debug(f"[DEBUG] sph_wrapper.connect() RETURNED: {connect_result}")
    sys.stdout.flush()
    sys.stderr.flush()

    if not connect_result:
        logger.warning("⚠️  无法连接到 OrcaLink，SPH 集成已禁用")
        config["orcasph"]["enabled"] = False
    else:
        logger.info("✅ OrcaLink 连接成功\n")
        logger.debug("[DEBUG] After OrcaLink connection success message")


def _setup_main_loop_recorders(ctx: FluidSimulationContext) -> None:
    """
    人类轨迹 HDF5、统计日志、MuJoCo qpos 。
    人类操作控制帧：DESIGN_mujoco_human_trajectory_hdf5.md §6。
    仅在 env.step 之后采样：DESIGN_particle_record_mujoco_qpos_coupled_playback.md §3.3。
    """
    config = ctx.config
    env = ctx.env

    # SIGHUP（关终端/父 shell 退出）：协作退出主循环，由 finally 做清理
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, lambda *_: ctx.shutdown_event.set())

    traj_cfg = config.get("mujoco_trajectory") or {}
    sph_mocap_names = frozenset()
    if ctx.sph_wrapper is not None:
        sph_mocap_names = ctx.sph_wrapper.sph_coupling_mocap_names
    if traj_cfg.get("enabled") and ctx.session_timestamp:
        out_p = Path(traj_cfg["output_path"])
        ctx.traj_rec = TrajectoryRecorder(
            out_p,
            env.unwrapped,
            ctx.session_timestamp,
            REALTIME_STEP,
            sph_mocap_names,
        )
        logger.info("MuJoCo trajectory recording enabled: %s", out_p.resolve())
    pb = traj_cfg.get("playback_path")
    if pb:
        ctx.traj_player = TrajectoryPlayer(Path(pb), env.unwrapped)
        logger.info(
            "MuJoCo trajectory playback: %s (%s frames)",
            Path(pb).resolve(),
            ctx.traj_player.num_frames,
        )

    traj_stats_path = resolve_record_stats_orcasph_log_path(
        config, ctx.session_timestamp, ctx.orcagym_tmp_dir
    )
    if ctx.traj_player is not None and traj_stats_path is not None:
        try:
            traj_stats_path.parent.mkdir(parents=True, exist_ok=True)
            ctx.traj_stats_log_f = open(
                traj_stats_path, "a", encoding="utf-8", buffering=1
            )
            ctx.traj_stats_log_f.write(
                "[TRAJECTORY_RECORD_STATS] "
                f"frame_index=0 num_frames={ctx.traj_player.num_frames}\n"
            )
            ctx.traj_stats_log_f.flush()
        except OSError as e:
            logger.warning("trajectory stats log open (%s): %s", traj_stats_path, e)
            ctx.traj_stats_log_f = None

    ctx.mujoco_qpos_sidecar = maybe_open_sidecar_for_record_config(config, env)
    if ctx.mujoco_qpos_sidecar is not None:
        logger.info(
            "MuJoCo qpos sidecar recording: %s",
            ctx.mujoco_qpos_sidecar.path.resolve(),
        )


def _run_cooperative_main_loop(ctx: FluidSimulationContext) -> None:
    """协作式主循环（SIGTERM 由 handler 同步退出；此处响应 SIGHUP / 轨迹耗尽）。"""
    config = ctx.config
    env = ctx.env
    shutdown_event = ctx.shutdown_event

    logger.debug("[DEBUG] Entering main loop (cooperative shutdown on SIGTERM/SIGHUP)...")
    step_count = 0

    _steer_rec = SteeringTorqueRecorder.from_env()
    _steer_couple = SteerCouplingRecorder.from_env()
    _cup_couple = GenericCouplingRecorder.from_env_cup()
    _cup_torque = GenericTorqueBalanceRecorder.from_env_cup()
    _steer_mj_cycle = 0

    _mj_state_csv = None
    _mj_state_header_written = False
    _mj_monitor_bodies = []
    _time_sync_csv = None
    _time_sync_header_written = False
    if config["orcasph"]["enabled"] and ctx.sph_wrapper is not None:
        try:
            for rb_config in ctx.sph_wrapper.rigid_bodies.values():
                _mj_monitor_bodies.append(rb_config.mujoco_body)
        except Exception:
            pass

    while not shutdown_event.is_set():
        start_time = datetime.now()

        if ctx.traj_player is not None and ctx.traj_player.exhausted:
            logger.info(
                "MuJoCo trajectory playback finished (%s frames).",
                ctx.traj_player.num_frames,
            )
            break

        if step_count == 0:
            logger.debug("[DEBUG] First iteration - before SPH sync")

        apply_water_jug_trajectory(ctx)

        should_step = True
        if config["orcasph"]["enabled"] and ctx.sph_wrapper is not None:
            try:
                if step_count == 0:
                    logger.debug("[DEBUG] Calling sph_wrapper.step()...")
                should_step = ctx.sph_wrapper.step()
                if step_count == 0:
                    logger.debug(f"[DEBUG] sph_wrapper.step() returned: {should_step}")
            except Exception as e:
                logger.error(f"SPH 同步失败: {e}")
                config["orcasph"]["enabled"] = False

        if step_count == 0:
            logger.debug(f"[DEBUG] Before MuJoCo step, should_step={should_step}")

        # MuJoCo step（轨迹回放：bridge.step 已更新 SPH mocap，此处仅叠加人类操作）
        if should_step:
            if ctx.traj_player is not None:
                ctx.traj_player.push_pending_to_env()
                env.step(None)
                ctx.traj_player.advance_cursor()
                if ctx.traj_stats_log_f is not None:
                    try:
                        ctx.traj_stats_log_f.write(
                            "[TRAJECTORY_RECORD_STATS] "
                            f"frame_index={ctx.traj_player.frame_index} "
                            f"num_frames={ctx.traj_player.num_frames}\n"
                        )
                        ctx.traj_stats_log_f.flush()
                    except OSError as e:
                        logger.warning("trajectory stats log write: %s", e)
            else:
                env.step(None)
            if ctx.water_jug_reapply_after_step:
                apply_water_jug_trajectory(ctx)
            # §3.3：仅在执行 env.step 之后追加行（与 traj_rec 同控制帧）
            if ctx.mujoco_qpos_sidecar is not None:
                ctx.mujoco_qpos_sidecar.append_row(env, step_count)
            if ctx.traj_rec is not None:
                ctx.traj_rec.append_frame()

            _steer_mj_cycle += 1
            if _steer_rec is not None:
                try:
                    _steer_rec.record_row(ctx, env, _steer_mj_cycle)
                except Exception as e:
                    logger.debug("steering torque CSV (方案甲): %s", e, exc_info=True)
            if _steer_couple is not None:
                try:
                    _steer_couple.record_row(ctx, env, _steer_mj_cycle)
                except Exception as e:
                    logger.debug("steer couple CSV: %s", e, exc_info=True)
            if _cup_couple is not None:
                try:
                    _cup_couple.record_row(ctx, env, _steer_mj_cycle)
                except Exception as e:
                    logger.debug("cup couple CSV: %s", e, exc_info=True)
            if _cup_torque is not None:
                try:
                    _cup_torque.record_row(ctx, env, _steer_mj_cycle)
                except Exception as e:
                    logger.debug("cup torque CSV: %s", e, exc_info=True)

            # [MONITOR] Record MuJoCo body state for comparison with SPH
            if _mj_monitor_bodies and step_count >= 0:
                try:
                    env.unwrapped.mj_forward()
                    xpos_flat, _, xquat_flat = env.unwrapped.get_body_xpos_xmat_xquat(_mj_monitor_bodies)
                    n = len(_mj_monitor_bodies)
                    xpos_arr = xpos_flat.reshape(n, 3)
                    xquat_arr = xquat_flat.reshape(n, 4)
                    if _mj_state_csv is None:
                        import os
                        _csv_path = os.environ.get('ORCA_MJ_STATE_CSV', '/tmp/orca_mj_state_monitor.csv')
                        _mj_state_csv = open(_csv_path, 'a')
                    if not _mj_state_header_written:
                        _mj_state_csv.write("wall_time,cycle,sim_time,body_name,mj_xpos_x,mj_xpos_y,mj_xpos_z,mj_xquat_w,mj_xquat_x,mj_xquat_y,mj_xquat_z\n")
                        _mj_state_header_written = True
                    import time as time_module
                    wall_ts = time_module.time()
                    sim_time = step_count * 0.02
                    for i, bname in enumerate(_mj_monitor_bodies):
                        p = xpos_arr[i]
                        q = xquat_arr[i]
                        _mj_state_csv.write(
                            f"{fmt_f(wall_ts)},{step_count},{fmt_f(sim_time)},{bname},"
                            f"{fmt_f(p[0])},{fmt_f(p[1])},{fmt_f(p[2])},"
                            f"{fmt_f(q[0])},{fmt_f(q[1])},{fmt_f(q[2])},{fmt_f(q[3])}\n"
                        )
                    if step_count % 25 == 0:
                        _mj_state_csv.flush()
                except Exception as e:
                    logger.debug(f"MuJoCo state monitor error: {e}")
            
            env.render()
        else:
            env.render()

        if ctx.mujoco_viewer is not None:
            try:
                ctx.mujoco_viewer.sync()
            except Exception as e:
                logger.debug(f"MuJoCo viewer sync error: {e}")

        if _time_sync_csv is None and config["orcasph"]["enabled"]:
            import os as _os_for_sync
            _sync_csv_path = _os_for_sync.environ.get(
                'ORCA_TIME_SYNC_CSV',
                '/home/hjadmin/OrcaApr24/monitor_data/time_sync.csv'
            )
            _time_sync_csv = open(_sync_csv_path, 'a')
        if _time_sync_csv is not None and not _time_sync_header_written:
            _time_sync_csv.write(
                "mj_wall_time,step_count,should_step,mj_sim_time_engine,mj_sim_time_counter,"
                "sph_sim_time,sph_wall_time,orcalink_cycle\n"
            )
            _time_sync_header_written = True
        if _time_sync_csv is not None:
            try:
                import time as _t_for_sync
                _mj_wall_ts = _t_for_sync.time()
                _mj_sim_engine = env.unwrapped.gym._mjData.time
                _mj_sim_counter = step_count * 0.02
                _sph_sim_time = ""
                _sph_wall_time = ""
                _orcalink_cycle = ""
                if ctx.sph_wrapper is not None and ctx.sph_wrapper.orcalink_client is not None:
                    _olc = ctx.sph_wrapper.orcalink_client
                    if hasattr(_olc, '_last_received_sph_sim_time'):
                        _sph_sim_time = fmt_f(_olc._last_received_sph_sim_time)
                    if hasattr(_olc, '_last_received_sph_wall_time'):
                        _sph_wall_time = fmt_f(_olc._last_received_sph_wall_time)
                    if hasattr(_olc, 'subscribe_sequence'):
                        _orcalink_cycle = str(_olc.subscribe_sequence)
                _time_sync_csv.write(
                    f"{fmt_f(_mj_wall_ts)},{step_count},{1 if should_step else 0},"
                    f"{fmt_f(_mj_sim_engine)},{fmt_f(_mj_sim_counter)},"
                    f"{_sph_sim_time},{_sph_wall_time},{_orcalink_cycle}\n"
                )
                if step_count % 25 == 0:
                    _time_sync_csv.flush()
            except Exception as e:
                logger.debug(f"Time sync CSV error: {e}")

        if step_count == 0:
            logger.debug("[DEBUG] After render")

        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed < REALTIME_STEP:
            remaining = REALTIME_STEP - elapsed
            if shutdown_event.wait(timeout=remaining):
                break

        step_count += 1
        if step_count == 1:
            logger.debug("[DEBUG] Completed first iteration successfully")
        if step_count % 100 == 0:
            logger.info(f"仿真步数: {step_count}")

        if ctx.max_steps and step_count >= ctx.max_steps:
            logger.info("⏹️  已达 max_steps=%s，正常结束主循环", ctx.max_steps)
            break

        if ctx.max_sim_time > 0:
            current_sim_time = env.unwrapped.data.time
            if current_sim_time >= ctx.max_sim_time:
                logger.info("⏹️  已达 max_sim_time=%.2fs (sim_time=%.2fs)，正常结束主循环", ctx.max_sim_time, current_sim_time)
                break

    if shutdown_event.is_set():
        logger.info("\n⏹️  收到停止信号（SIGTERM/SIGHUP），协作退出主循环")

    if _mj_state_csv is not None:
        _mj_state_csv.flush()
        _mj_state_csv.close()
    if _time_sync_csv is not None:
        _time_sync_csv.flush()
        _time_sync_csv.close()
    if _steer_rec is not None:
        _steer_rec.close()
        _steer_rec.post_merge()
    if _steer_couple is not None:
        _steer_couple.close()
        _steer_couple.post_merge()
    if _cup_couple is not None:
        _cup_couple.close()
        _cup_couple.post_merge()
    if _cup_torque is not None:
        _cup_torque.close()
        _cup_torque.post_merge()
    from ..modules.force_application import ForceApplicationModule
    ForceApplicationModule.close_force_csv()


def _finalize_simulation_session(ctx: FluidSimulationContext) -> None:
    """
    原 finally：恢复 SIGTERM、关闭录制、EndSimulation、子进程、合并 HDF5（§5.1）、视口与 env。
    """
    config = ctx.config
    try:
        if ctx.prev_sigterm_handler is not None and hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, ctx.prev_sigterm_handler)
    except (OSError, ValueError):
        pass
    logger.info("\n🧹 清理资源...")

    _terminate_stats_plot_proc()

    if ctx.mujoco_viewer is not None:
        try:
            ctx.mujoco_viewer.close()
        except Exception as e:
            logger.debug(f"MuJoCo viewer close: %s", e)
        ctx.mujoco_viewer = None

    owns = _fluid_atexit_state.get("owns_shared_services")

    try:
        if ctx.traj_rec is not None:
            ctx.traj_rec.close()
    except Exception as e:
        logger.warning("trajectory recorder close: %s", e)
    try:
        if ctx.mujoco_qpos_sidecar is not None:
            ctx.mujoco_qpos_sidecar.close()
    except Exception as e:
        logger.warning("mujoco qpos sidecar close: %s", e)
    try:
        if ctx.traj_player is not None:
            ctx.traj_player.close()
    except Exception as e:
        logger.warning("trajectory player close: %s", e)
    try:
        if ctx.traj_stats_log_f is not None:
            ctx.traj_stats_log_f.close()
    except Exception as e:
        logger.warning("trajectory stats log close: %s", e)

    if ctx.sph_wrapper:
        ctx.sph_wrapper.close()

    if owns:
        _fluid_send_end_simulation_from_config(config)

    ctx.process_manager.cleanup_all()

    if owns:
        time.sleep(0.2)

    pr_run = config.get("particle_render_run") or {}
    if pr_run.get("mode") == "record" and pr_run.get("record_output_path"):
        try:
            merge_particle_mujoco_sidecar_into_particle_h5(
                pr_run["record_output_path"],
                str(mujoco_qpos_sidecar_tmp_path(pr_run["record_output_path"])),
                session_timestamp=ctx.session_timestamp,
            )
        except Exception as e:
            logger.warning("merge mujoco_frames into particle HDF5 failed: %s", e)

    if ctx.env is not None:
        try:
            if owns:
                _fluid_sync_initial_viewport_to_engine(ctx.env)
        except Exception as e:
            logger.warning(f"退出时 reset_simulation / 同步失败（可忽略）: {e}")
        try:
            ctx.env.close()
        except Exception as e:
            logger.warning(f"env.close() 失败: {e}")

    _fluid_atexit_state["viewport_reset_done"] = True
    _fluid_atexit_state["session_active"] = False
    logger.info("✅ 清理完成")


def run_simulation_with_config(
    config: Dict,
    session_timestamp: Optional[str] = None,
    cpu_affinity: Optional[str] = None,
    max_steps: int = 0,
    max_sim_time: float = 0.0,
) -> None:
    """
    使用配置文件运行仿真

    启动顺序（重要）：
        1. 创建 MuJoCo 环境
        2. 生成 scene.json（依赖环境）
        3. 启动 orcalink（等待 5 秒）
        4. 启动 orcasph --scene <scene.json>（依赖 scene.json）
        5. 连接并开始仿真

    收到 SIGTERM / SIGHUP（如 OrcaLab 停止外部程序）时仅置停止标志，主循环协作退出后
    在 finally 中清理 OrcaLink / OrcaSPH 子进程。

    Args:
        config: 配置字典
        session_timestamp: 会话时间戳（用于统一日志文件名），如果为None则自动生成
        cpu_affinity: CPU 亲和性核心列表（传递给 taskset -c），例如 "0-7" 或 "0,2,4,6"，None 表示不限制
        max_steps: 主循环最大步数；>0 时达到后正常退出（用于无头自动跑与检查）；0 表示不限制
    """
    session_timestamp, orcagym_tmp_dir = _preflight_session(config, session_timestamp)

    _init_atexit_state_for_session(config)

    process_manager = ProcessManager()
    ctx = FluidSimulationContext(
        config=config,
        session_timestamp=session_timestamp,
        cpu_affinity=cpu_affinity,
        orcagym_tmp_dir=orcagym_tmp_dir,
        process_manager=process_manager,
        max_steps=max(0, int(max_steps or 0)),
        max_sim_time=max(0.0, float(max_sim_time or 0)),
    )

    # -----------------------------------------------------------------------
    # 在函数入口（而非主循环入口）注册 SIGTERM，保证启动阶段也能响应：
    # OrcaLab stop_sim() 发 SIGTERM 后等 5 秒再 kill；我们在 handler 里同步
    # 执行全量清理（粒子/视口重置 + 子进程），然后 _exit(0) 退出，不依赖
    # finally 或主循环协作，确保在 5 秒 kill 之前完成。
    # -----------------------------------------------------------------------
    ctx.prev_sigterm_handler = signal.signal(
        signal.SIGTERM,
        _make_sigterm_cleanup_handler(ctx, ctx.shutdown_event),
    )

    try:
        logger.info("=" * 80)
        logger.info("Fluid-MuJoCo 耦合仿真启动")
        logger.info("=" * 80)

        ctx.env = _create_and_reset_gym_env(config)
        _maybe_launch_mujoco_viewer(ctx)
        _maybe_generate_sph_scene(ctx)
        _start_orcalink_if_configured(ctx)
        _start_orcasph_if_configured(ctx)

        _try_start_record_stats_plot_viewer(
            config, session_timestamp, orcagym_tmp_dir
        )

        logger.debug("[DEBUG] About to enter main loop...")
        sys.stdout.flush()
        sys.stderr.flush()

        _connect_sph_bridge_if_enabled(ctx)
        setup_water_jug_trajectory(ctx)

        logger.debug("[DEBUG] About to enter main loop...")
        sys.stdout.flush()
        sys.stderr.flush()
        logger.info("=" * 80)
        logger.info("🎬 仿真主循环开始")
        logger.info("=" * 80)
        sys.stdout.flush()
        sys.stderr.flush()
        print(
            "[PRINT-DEBUG] run_simulation.py - About to enter main loop...",
            file=sys.stderr,
            flush=True,
        )
        print(
            "[PRINT-DEBUG] run_simulation.py - Main loop started...",
            file=sys.stderr,
            flush=True,
        )

        _setup_main_loop_recorders(ctx)
        _run_cooperative_main_loop(ctx)

    except KeyboardInterrupt:
        logger.info("\n⏹️  用户中断仿真")
    except Exception as e:
        logger.error(f"\n❌ 仿真错误: {e}", exc_info=True)
    finally:
        _finalize_simulation_session(ctx)
