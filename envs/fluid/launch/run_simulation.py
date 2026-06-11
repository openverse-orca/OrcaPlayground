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
from .process_utils import ProcessManager, is_tcp_port_accepting_connections
from .mujoco_viewer_style import (
    apply_mujoco_passive_viewer_styles,
    configure_passive_viewer_options,
    draw_coupling_site_markers,
)
from .sph_config import (
    _resolve_coupling_mode,
    generate_orcasph_config,
    prepare_orcasph_config_from_fixed_path,
    setup_python_logging,
)

logger = logging.getLogger(__name__)

# 主循环控制周期（秒）；与 mujoco_trajectory 录制 meta 等一致。
REALTIME_STEP = 0.02


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

    env: Any = None
    sph_wrapper: Any = None
    traj_rec: Any = None
    traj_player: Any = None
    traj_stats_log_f: Any = None
    mujoco_qpos_sidecar: Any = None
    scene_output_path: Optional[Path] = None
    particle_render_override: Any = None
    sphscale: float = 1.0
    prev_sigterm_handler: Any = None
    # MuJoCo 原生被动查看器（短链无 Studio 时用于本地可视化）
    mujoco_passive_viewer: Any = None
    fp_debug_trace: Any = None
    water_jug_ctrl: Any = None
    bench_output_path: Optional[str] = None
    bench_steps: list = field(default_factory=list)


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
            if _owns and not _orcagym_skip_studio(ctx.config):
                try:
                    _fluid_send_end_simulation_from_config(ctx.config)
                except Exception:
                    pass

            # 3. 终止 OrcaSPH / OrcaLink 子进程
            ctx.process_manager.cleanup_all()

            # 4. 等待在途粒子帧被丢弃
            if _owns:
                time.sleep(0.2)

            _close_mujoco_passive_viewer(ctx)

            # 5. 重置刚体位姿并推给 OrcaSim
            if _env is not None and _owns and not _orcagym_skip_studio(ctx.config):
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


def _orcagym_skip_studio(config: Dict) -> bool:
    return bool(config.get("orcagym", {}).get("skip_grpc_load"))


def _default_force_position_debug_output_base(config: Dict) -> Path:
    """
    短链：``scene_3chain/debug_data``；全链路：``~/.orcagym/tmp/debug_data``。

    可由 ``debug.force_position_trace.output_base_dir`` 显式覆盖。
    """
    if _orcagym_skip_studio(config):
        scene_dir = config.get("scene_3chain_dir") or "/home/hjadmin/OrcaApr24/SPH_bug/scene_3chain"
        return Path(scene_dir).expanduser() / "debug_data"
    return Path.home() / ".orcagym" / "tmp" / "debug_data"


def _resolve_debug_mjcf_path(ctx: FluidSimulationContext) -> Optional[Path]:
    """
    解析 CP 监测用 MJCF 路径：显式配置 > 短链 local_xml > OrcaGym 缓存 XML。
    """
    dbg = ctx.config.get("debug", {}).get("force_position_trace", {})
    explicit = dbg.get("mujoco_scene_path")
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    local = ctx.config.get("orcagym", {}).get("local_xml_path")
    if local:
        p = Path(local).expanduser()
        return p if p.is_file() else None
    if ctx.env is not None:
        gym = getattr(ctx.env.unwrapped, "gym", None)
        xml_path = getattr(gym, "_xml_path", None) if gym is not None else None
        if xml_path:
            p = Path(xml_path).expanduser()
            return p if p.is_file() else None
    return None


def _mujoco_gui_enabled(config: Dict) -> bool:
    return bool((config.get("mujoco_gui") or {}).get("enabled"))


def _mujoco_gui_shutdown_on_close(config: Dict) -> bool:
    """
    是否在 MuJoCo 被动查看器窗口关闭时结束整场仿真。

    为 false 时仅停止 viewer.sync()，主循环与 OrcaSPH 继续运行（双界面长跑 / 误关窗场景）。
    未配置时默认为 true，保持单 MuJoCo 查看器时的历史行为。
    """
    return bool((config.get("mujoco_gui") or {}).get("shutdown_on_close", True))


def _start_mujoco_passive_viewer(ctx: FluidSimulationContext) -> None:
    """启动 MuJoCo 原生被动查看器（与 OrcaStudio 视口无关）。"""
    import mujoco
    import mujoco.viewer

    gui_cfg = ctx.config.get("mujoco_gui") or {}
    unwrapped = ctx.env.unwrapped
    mj_model = unwrapped.gym._mjModel
    mj_data = unwrapped.gym._mjData
    apply_mujoco_passive_viewer_styles(mj_model, gui_cfg)
    # 须先 forward + 默认自由相机；否则 launch_passive 默认 lookat=原点、distance=2，场景在远处会全黑
    mujoco.mj_forward(mj_model, mj_data)
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    with viewer.lock():
        configure_passive_viewer_options(viewer, mj_model, gui_cfg)
        if gui_cfg.get("site_markers", True):
            draw_coupling_site_markers(mj_model, mj_data, viewer)
    viewer.sync()
    ctx.mujoco_passive_viewer = viewer
    logger.info(
        "🖥️  MuJoCo 被动查看器已启动（着色=%s SITE=%s lookat=%s distance=%.2f shutdown_on_close=%s）",
        gui_cfg.get("colorize", True),
        gui_cfg.get("show_coupling_sites", True),
        viewer.cam.lookat,
        viewer.cam.distance,
        _mujoco_gui_shutdown_on_close(ctx.config),
    )


def _sync_mujoco_passive_viewer(ctx: FluidSimulationContext) -> None:
    import mujoco

    viewer = ctx.mujoco_passive_viewer
    if viewer is None:
        return
    if not viewer.is_running():
        if _mujoco_gui_shutdown_on_close(ctx.config):
            logger.info("MuJoCo 查看器已关闭，结束主循环")
            ctx.shutdown_event.set()
        else:
            logger.info(
                "MuJoCo 查看器已关闭，仿真继续在后台运行（mujoco_gui.shutdown_on_close=false）；"
                "按 Ctrl+C 结束整场仿真"
            )
            ctx.mujoco_passive_viewer = None
        return
    gui_cfg = ctx.config.get("mujoco_gui") or {}
    unwrapped = ctx.env.unwrapped
    mj_model = unwrapped.gym._mjModel
    mj_data = unwrapped.gym._mjData
    mujoco.mj_forward(mj_model, mj_data)
    if gui_cfg.get("site_markers", True):
        with viewer.lock():
            draw_coupling_site_markers(mj_model, mj_data, viewer)
    viewer.sync()


def _close_mujoco_passive_viewer(ctx: FluidSimulationContext) -> None:
    viewer = ctx.mujoco_passive_viewer
    if viewer is None:
        return
    try:
        viewer.close()
    except Exception as e:
        logger.warning("MuJoCo passive viewer close: %s", e)
    ctx.mujoco_passive_viewer = None


def _resolve_prebuilt_sph_scene(ctx: FluidSimulationContext) -> None:
    """短链模式：使用配置中的固定 sph_scene.json，跳过 SceneGenerator。"""
    oc = ctx.config.get("orcasph", {})
    if oc.get("scene_auto_generate", True):
        return
    scene_path = oc.get("scene_path")
    if not scene_path:
        raise ValueError(
            "orcasph.scene_auto_generate=false 时必须设置 orcasph.scene_path"
        )
    resolved = Path(scene_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"SPH scene 文件不存在: {resolved}")
    ctx.scene_output_path = resolved
    ctx.sphscale = float(oc.get("sphscale", 1.0))
    logger.info("✅ 使用固定 SPH scene: %s\n", resolved)


def _create_and_reset_gym_env(config: Dict) -> Any:
    logger.info("\n📦 步骤 1: 创建 MuJoCo 环境...")
    orcagym_cfg = config["orcagym"]
    skip_studio = _orcagym_skip_studio(config)
    addr_tag = (orcagym_cfg.get("address") or "offline").replace(":", "-")
    env_id = f"{orcagym_cfg['env_name']}-OrcaGym-{addr_tag}-000"

    sim_cfg = config.get("simulation", {})
    mj_timestep = float(sim_cfg.get("timestep", 0.001))
    # 每个主循环宏步（REALTIME_STEP≈OrcaLink 50Hz）内的 MuJoCo 子步数
    frame_skip = max(1, int(round(REALTIME_STEP / mj_timestep)))
    env_kwargs: Dict[str, Any] = {
        "frame_skip": frame_skip,
        "orcagym_addr": orcagym_cfg.get("address") or "localhost:50051",
        "agent_names": [orcagym_cfg["agent_name"]],
        "time_step": mj_timestep,
    }
    logger.info(
        "MuJoCo 子步: dt=%.4fs, frame_skip=%d (宏步物理时间 %.4fs)",
        mj_timestep,
        frame_skip,
        mj_timestep * frame_skip,
    )
    if skip_studio:
        local_xml = orcagym_cfg.get("local_xml_path")
        if not local_xml:
            raise ValueError("orcagym.skip_grpc_load=true 时必须设置 orcagym.local_xml_path")
        env_kwargs["skip_grpc_load"] = True
        env_kwargs["local_xml_path"] = local_xml
        if orcagym_cfg.get("assets_dir"):
            env_kwargs["xml_assets_dir"] = orcagym_cfg["assets_dir"]
        logger.info(
            "🔌 短链 MuJoCo：跳过 OrcaStudio gRPC (50051)，本地 XML=%s",
            local_xml,
        )

    print(
        "[PRINT-DEBUG] run_simulation.py - About to register gymnasium env",
        file=sys.stderr,
        flush=True,
    )
    gym.register(
        id=env_id,
        entry_point="envs.fluid.sim_env:SimEnv",
        kwargs=env_kwargs,
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

    coupling_mode = _resolve_coupling_mode(config)
    scene_generator = SceneGenerator(
        ctx.env.unwrapped,
        config_path=str(scene_config_path),
        runtime_config=sph_config,
        coupling_mode=coupling_mode,
        fluid_config=config,
    )
    scene_data = scene_generator.generate_complete_scene(
        output_path=str(ctx.scene_output_path),
        include_fluid_blocks=config["sph"]["include_fluid_blocks"],
        include_wall=config["sph"]["include_wall"],
    )
    logger.info(f"✅ scene.json 已生成: {ctx.scene_output_path}")
    logger.info(f"   - RigidBodies: {len(scene_data.get('RigidBodies', []))} 个\n")

    ctx.particle_render_override = scene_generator.generate_particle_render_config(
        sph_config
    )
    ctx.sphscale = scene_generator.sphscale


def _init_force_position_debug_trace(ctx: FluidSimulationContext) -> None:
    """若 build_mode=debug 且配置启用，创建 debug_data/<session_timestamp>/ 并设置 ORCA_FP_DEBUG_DIR。"""
    from ..debug.force_position_trace_config import (
        clear_force_position_debug_env,
        is_force_position_trace_enabled,
    )

    if not is_force_position_trace_enabled(ctx.config):
        clear_force_position_debug_env()
        build_mode = str(ctx.config.get("build_mode", "debug")).strip().lower()
        if build_mode == "release":
            logging.getLogger(__name__).info(
                "build_mode=release，已关闭 force_position 调试采集（不写入 debug_data）"
            )
        return

    dbg = ctx.config.get("debug", {}).get("force_position_trace", {})

    base = Path(
        dbg.get("output_base_dir", str(_default_force_position_debug_output_base(ctx.config)))
    ).expanduser()
    session_dir = base / ctx.session_timestamp
    os.environ["ORCA_FP_DEBUG_DIR"] = str(session_dir.resolve())
    os.environ["ORCA_FP_DEBUG_SUBSTEPS"] = str(int(dbg.get("substeps_per_macro", 20)))

    substep_trace = dbg.get("substep_trace", {})
    if substep_trace.get("enabled", False):
        os.environ["ORCA_FP_SUBSTEP_TRACE"] = "1"
        os.environ["ORCA_FP_SUBSTEP_MAX_MACROS"] = str(
            int(substep_trace.get("max_macros", 10))
        )
    else:
        os.environ.pop("ORCA_FP_SUBSTEP_TRACE", None)
        os.environ.pop("ORCA_FP_SUBSTEP_MAX_MACROS", None)

    body_aliases: Dict[str, str] = {}
    alias_file = dbg.get("body_aliases_file")
    if alias_file:
        alias_path = Path(alias_file).expanduser()
        if alias_path.is_file():
            with open(alias_path, encoding="utf-8") as f:
                body_aliases = json.load(f)

    if body_aliases:
        alias_parts = [f"{k}={v}" for k, v in body_aliases.items()]
        os.environ["ORCA_FP_DEBUG_ALIASES"] = ";".join(alias_parts)
    else:
        os.environ.pop("ORCA_FP_DEBUG_ALIASES", None)

    session_dir.mkdir(parents=True, exist_ok=True)
    if alias_file and body_aliases:
        import shutil

        shutil.copy2(alias_path, session_dir / "body_debug_aliases.json")

    surface_sites_by_oid: Dict[str, Dict] = {}
    mjcf_path = _resolve_debug_mjcf_path(ctx)
    if mjcf_path is not None:
        from ..debug.surface_site_registry import (
            pick_highest_z_sph_sites,
            write_surface_sites_csv,
        )

        registry = pick_highest_z_sph_sites(mjcf_path)
        write_surface_sites_csv(registry, session_dir / "surface_sites.csv")
        for body_name, spec in registry.items():
            surface_sites_by_oid[body_name] = spec
    else:
        logging.getLogger(__name__).warning(
            "force_position 调试：未找到 MJCF（跳过 surface_sites.csv）；"
            "全链路可设 debug.force_position_trace.mujoco_scene_path"
        )

    from ..debug.force_position_debug_trace import init_session

    ctx.fp_debug_trace = init_session(session_dir, body_aliases, surface_sites_by_oid)
    logging.getLogger(__name__).info(
        "force_position 调试采集已开启: %s", session_dir.resolve()
    )


def _run_substep_trace_analysis_if_enabled(ctx: FluidSimulationContext) -> None:
    """若启用子步监测，对 cp_substeps_trace.csv 做反推与启动报告。"""
    from ..debug.force_position_trace_config import is_force_position_trace_enabled

    dbg = ctx.config.get("debug", {}).get("force_position_trace", {})
    if not is_force_position_trace_enabled(ctx.config) or not dbg.get(
        "substep_trace", {}
    ).get("enabled", False):
        return

    base = Path(
        dbg.get("output_base_dir", str(_default_force_position_debug_output_base(ctx.config)))
    ).expanduser()
    session_dir = base / ctx.session_timestamp
    analyze_script = Path(
        dbg.get(
            "substep_trace",
            {},
        ).get(
            "analyze_script",
            "/home/hjadmin/OrcaApr24/SPH_bug/scene_3chain/cp_substeps_analyze.py",
        )
    ).expanduser()

    trace_csv = session_dir / "cp_substeps_trace.csv"
    if not trace_csv.is_file():
        logger.info("无 cp_substeps_trace.csv，跳过子步分析")
        return
    if not analyze_script.is_file():
        logger.warning("子步分析脚本不存在: %s", analyze_script)
        return

    try:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, str(analyze_script), str(session_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "子步分析失败 (code=%s): %s",
                result.returncode,
                result.stderr or result.stdout,
            )
        else:
            logger.info("子步分析完成: %s", session_dir)
    except Exception as exc:
        logger.warning("子步分析异常: %s", exc)


def _run_force_position_debug_diff_if_enabled(ctx: FluidSimulationContext) -> None:
    """仿真结束后对当前会话目录运行 cp1_cp4_diff.py。"""
    from ..debug.force_position_trace_config import is_force_position_trace_enabled

    if not is_force_position_trace_enabled(ctx.config):
        return

    dbg = ctx.config.get("debug", {}).get("force_position_trace", {})

    base = Path(
        dbg.get("output_base_dir", str(_default_force_position_debug_output_base(ctx.config)))
    ).expanduser()
    session_dir = base / ctx.session_timestamp
    diff_script = Path(
        dbg.get(
            "diff_script",
            "/home/hjadmin/OrcaApr24/SPH_bug/scene_3chain/cp1_cp4_diff.py",
        )
    ).expanduser()

    if not diff_script.is_file():
        logger.warning("cp1_cp4_diff 脚本不存在: %s", diff_script)
        return
    if not (session_dir / "cp1_pre_publish.csv").is_file():
        logger.info("无 CP1 数据，跳过差值分析")
        return
    if not (session_dir / "cp4_after_substeps.csv").is_file():
        logger.info("无 CP4 数据，跳过差值分析")
        return

    try:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, str(diff_script), str(session_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            logger.info(result.stdout.strip())
        if result.returncode != 0:
            logger.warning(
                "cp1_cp4_diff 退出码 %s: %s",
                result.returncode,
                (result.stderr or "").strip(),
            )
    except Exception as e:
        logger.warning("cp1_cp4_diff 运行失败: %s", e)


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
    fixed_cfg = config["orcasph"].get("orcasph_config_path")
    if fixed_cfg:
        orcasph_config_path, verbose_logging = prepare_orcasph_config_from_fixed_path(
            config,
            Path(fixed_cfg).expanduser().resolve(),
            orcasph_config_path,
        )
    else:
        orcasph_config_path, verbose_logging = generate_orcasph_config(
            config,
            orcasph_config_path,
            particle_render_override=ctx.particle_render_override,
            sphscale=ctx.sphscale,
        )

    orcasph_args = config["orcasph"]["args"].copy()
    orcasph_args.extend(["--config", str(orcasph_config_path)])
    orcasph_args.extend(["--scene", str(ctx.scene_output_path)])

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
    logger.info("⏳ 等待 OrcaSPH 初始化（2 秒）...")
    time.sleep(2)
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


def _log_cp6_after_mj_step_if_enabled(
    ctx: FluidSimulationContext, macro_step: int
) -> None:
    """CP6：mj_step 完成后、下一轮 publish（CP1）前，记录即将发出的 POSITION 包。"""
    from ..debug.force_position_trace_config import is_force_position_trace_enabled

    if not is_force_position_trace_enabled(ctx.config) or ctx.fp_debug_trace is None:
        return
    if ctx.sph_wrapper is None:
        return

    mode = getattr(ctx.sph_wrapper, "current_mode", None)
    ppm = getattr(mode, "position_publish_module", None) if mode is not None else None
    if ppm is None:
        return

    trace = ctx.fp_debug_trace
    if not trace.should_log_cp6(macro_step):
        return

    client = getattr(ctx.sph_wrapper, "orcalink_client", None)
    next_publish_seq = (
        int(client.publish_sequence) if client is not None else macro_step + 1
    )

    positions = ppm._collect_body_positions()
    if not positions:
        return

    trace.log_cp6_pre_next_publish(macro_step, next_publish_seq, positions)


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


def _apply_sph_scene_initial_offset(ctx: FluidSimulationContext) -> None:
    """
    OrcaSPH 启动前：按配置生成带竖直偏移的 sph_scene 副本，避免与 MuJoCo 初值错位。
    """
    from ..utils.sph_scene_initial_offset import apply_sph_scene_initial_offset_to_context

    if ctx.scene_output_path is None:
        return
    patched = apply_sph_scene_initial_offset_to_context(
        ctx.config,
        scene_path=Path(ctx.scene_output_path),
        output_dir=ctx.orcagym_tmp_dir,
        session_timestamp=ctx.session_timestamp,
    )
    if patched is not None:
        ctx.scene_output_path = patched
        logger.info("✅ SPH scene 初值偏移已应用: %s", patched.resolve())


def _init_water_jug_trajectory(ctx: FluidSimulationContext) -> None:
    """短链运动学轨迹：MuJoCo qpos 初值（须在 OrcaSPH 启动前调用一次）。"""
    from ..trajectory.water_jug_trajectory_controller import maybe_create_water_jug_controller

    if ctx.water_jug_ctrl is not None or ctx.env is None:
        return
    ctx.water_jug_ctrl = maybe_create_water_jug_controller(ctx.config, ctx.env.unwrapped)


def _apply_water_jug_trajectory(ctx: FluidSimulationContext, phase: str) -> None:
    """在 mj_step 前/后写入 waterjug（及 fluidblock）轨迹位姿。"""
    ctrl = ctx.water_jug_ctrl
    if ctrl is None or ctx.env is None:
        return
    if phase == "post" and not ctrl.reapply_after_step:
        return
    ctrl.apply(ctx.env.unwrapped, phase=phase)


def _save_fluid_bench_data(ctx: FluidSimulationContext) -> None:
    """将主循环 bench 计时写入 JSON（与 DataCollectionManager bench 字段对齐）。"""
    if not ctx.bench_output_path or not ctx.bench_steps:
        return
    steps = ctx.bench_steps
    n = len(steps)
    if n == 0:
        return
    avg_fluid = sum(s["fluid_ms"] for s in steps) / n
    avg_step = sum(s["step_ms"] for s in steps) / n
    avg_render = sum(s["render_ms"] for s in steps) / n
    avg_total = sum(s["total_ms"] for s in steps) / n
    avg_sleep = sum(s["sleep_ms"] for s in steps) / n
    total_phy = steps[-1].get("phy_time", 0) - steps[0].get("phy_time", 0)
    total_sim = steps[-1].get("sim_time", 0) - steps[0].get("sim_time", 0)
    effective_count = sum(1 for s in steps if s.get("should_step", True))
    pause_count = n - effective_count
    pause_rate = pause_count / n * 100 if n > 0 else 0
    env_step_count = sum(1 for s in steps if s.get("env_stepped", False))
    report = {
        "num_steps": n,
        "loop_count": n,
        "effective_step_count": effective_count,
        "env_step_count": env_step_count,
        "pause_count": pause_count,
        "pause_rate_pct": round(pause_rate, 2),
        "total_sim_time_s": round(total_sim, 4),
        "total_phy_time_s": round(total_phy, 4),
        "sim_over_real_ratio": round(total_sim / total_phy, 4) if total_phy > 0 else 0,
        "avg_step_ms": round(avg_total, 2),
        "avg_fps": round(1000.0 / avg_total, 2) if avg_total > 0 else 0,
        "avg_fluid_ms": round(avg_fluid, 2),
        "avg_step_compute_ms": round(avg_step, 2),
        "avg_render_ms": round(avg_render, 2),
        "avg_sleep_ms": round(avg_sleep, 2),
        "pct_fluid": round(avg_fluid / avg_total * 100, 1) if avg_total > 0 else 0,
        "pct_step": round(avg_step / avg_total * 100, 1) if avg_total > 0 else 0,
        "pct_render": round(avg_render / avg_total * 100, 1) if avg_total > 0 else 0,
        "pct_sleep": round(avg_sleep / avg_total * 100, 1) if avg_total > 0 else 0,
        "macro_dt_s": REALTIME_STEP,
    }
    os.makedirs(os.path.dirname(ctx.bench_output_path) or ".", exist_ok=True)
    with open(ctx.bench_output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": report, "steps": steps}, f, indent=2)
    logger.info(
        "Bench saved: loops=%s env_steps=%s pause_rate=%.1f%% sim/real=%.3f -> %s",
        report["loop_count"],
        report["env_step_count"],
        report["pause_rate_pct"],
        report["sim_over_real_ratio"],
        ctx.bench_output_path,
    )


def _run_cooperative_main_loop(ctx: FluidSimulationContext) -> None:
    """协作式主循环（SIGTERM 由 handler 同步退出；此处响应 SIGHUP / 轨迹耗尽）。"""
    config = ctx.config
    env = ctx.env
    shutdown_event = ctx.shutdown_event

    logger.debug("[DEBUG] Entering main loop (cooperative shutdown on SIGTERM/SIGHUP)...")
    step_count = 0

    while not shutdown_event.is_set():
        t0 = time.perf_counter()
        start_time = datetime.now()

        if ctx.traj_player is not None and ctx.traj_player.exhausted:
            logger.info(
                "MuJoCo trajectory playback finished (%s frames).",
                ctx.traj_player.num_frames,
            )
            break

        if step_count == 0:
            logger.debug("[DEBUG] First iteration - before SPH sync")

        _apply_water_jug_trajectory(ctx, phase="pre")

        t1 = time.perf_counter()
        should_step = True
        if config["orcasph"]["enabled"] and ctx.sph_wrapper is not None:
            try:
                ctx.sph_wrapper.macro_step = step_count
                if step_count == 0:
                    logger.debug("[DEBUG] Calling sph_wrapper.step()...")
                should_step = ctx.sph_wrapper.step()
                if step_count == 0:
                    logger.debug(f"[DEBUG] sph_wrapper.step() returned: {should_step}")
            except Exception as e:
                logger.error(f"SPH 同步失败: {e}")
                config["orcasph"]["enabled"] = False
        t2 = time.perf_counter()

        if step_count == 0:
            logger.debug(f"[DEBUG] Before MuJoCo step, should_step={should_step}")

        # MuJoCo step（轨迹回放：bridge.step 已更新 SPH mocap，此处仅叠加人类操作）
        if should_step:
            if ctx.traj_player is not None:
                ctx.traj_player.push_pending_to_env()
                env.step(None)
                _apply_water_jug_trajectory(ctx, phase="post")
                _log_cp6_after_mj_step_if_enabled(ctx, step_count)
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
            _apply_water_jug_trajectory(ctx, phase="post")
            _log_cp6_after_mj_step_if_enabled(ctx, step_count)
            # §3.3：仅在执行 env.step 之后追加行（与 traj_rec 同控制帧）
            if ctx.mujoco_qpos_sidecar is not None:
                ctx.mujoco_qpos_sidecar.append_row(env, step_count)
            if ctx.traj_rec is not None:
                ctx.traj_rec.append_frame()
            if ctx.mujoco_passive_viewer is not None:
                _sync_mujoco_passive_viewer(ctx)
            elif not _orcagym_skip_studio(config):
                env.render()
            t3 = time.perf_counter()
        else:
            if ctx.mujoco_passive_viewer is not None:
                _sync_mujoco_passive_viewer(ctx)
            elif not _orcagym_skip_studio(config):
                env.render()
            t3 = time.perf_counter()

        if step_count == 0:
            logger.debug("[DEBUG] After render / mujoco viewer sync")

        elapsed = (datetime.now() - start_time).total_seconds()
        sleep_sec = 0.0
        if elapsed < REALTIME_STEP:
            remaining = REALTIME_STEP - elapsed
            sleep_sec = remaining
            if shutdown_event.wait(timeout=remaining):
                break

        if ctx.bench_output_path:
            sim_t = float(env.data.time) if hasattr(env, "data") and hasattr(env.data, "time") else 0.0
            t_end = time.perf_counter()
            ctx.bench_steps.append({
                "loop": len(ctx.bench_steps),
                "sim_time": round(sim_t, 6),
                "phy_time": round(time.time(), 6),
                "fluid_ms": round((t2 - t1) * 1000, 3),
                "step_ms": round((t3 - t2) * 1000, 3),
                "render_ms": 0.0,
                "total_ms": round((t_end - t0) * 1000, 3),
                "sleep_ms": round(sleep_sec * 1000, 3),
                "should_step": should_step,
                "env_stepped": bool(should_step),
            })

        step_count += 1
        if step_count == 1:
            logger.debug("[DEBUG] Completed first iteration successfully")
        if step_count % 100 == 0:
            logger.info(f"仿真步数: {step_count}")

        if ctx.max_steps and step_count >= ctx.max_steps:
            logger.info("⏹️  已达 max_steps=%s，正常结束主循环", ctx.max_steps)
            break

    if shutdown_event.is_set():
        logger.info("\n⏹️  收到停止信号（SIGTERM/SIGHUP），协作退出主循环")


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

    _save_fluid_bench_data(ctx)

    _terminate_stats_plot_proc()

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

    _close_mujoco_passive_viewer(ctx)

    if owns and not _orcagym_skip_studio(config):
        _fluid_send_end_simulation_from_config(config)

    ctx.process_manager.cleanup_all()

    _run_force_position_debug_diff_if_enabled(ctx)
    _run_substep_trace_analysis_if_enabled(ctx)

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
            if owns and not _orcagym_skip_studio(config):
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
    bench_output_path: Optional[str] = None,
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
        bench_output_path: 若指定，主循环逐圈计时并写入 JSON
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
        bench_output_path=bench_output_path,
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
        _resolve_prebuilt_sph_scene(ctx)
        _maybe_generate_sph_scene(ctx)
        _apply_sph_scene_initial_offset(ctx)
        _init_water_jug_trajectory(ctx)
        _init_force_position_debug_trace(ctx)
        _start_orcalink_if_configured(ctx)
        _start_orcasph_if_configured(ctx)

        _try_start_record_stats_plot_viewer(
            config, session_timestamp, orcagym_tmp_dir
        )

        logger.debug("[DEBUG] About to enter main loop...")
        sys.stdout.flush()
        sys.stderr.flush()

        _connect_sph_bridge_if_enabled(ctx)

        if _mujoco_gui_enabled(config):
            _start_mujoco_passive_viewer(ctx)

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
