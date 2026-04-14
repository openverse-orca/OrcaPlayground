"""流体仿真启动编排：Gym、scene、OrcaLink/OrcaSPH、主循环与清理。"""
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

import gymnasium as gym

from ..orcalink_bridge import OrcaLinkBridge
from ..trajectory import TrajectoryRecorder, TrajectoryPlayer
from ..paths import FLUID_PACKAGE_DIR, ORCA_PLAYGROUND_ROOT
from ..utils.scene_generator import SceneGenerator
from .fluid_session import (
    _fluid_atexit_state,
    _fluid_send_end_simulation_from_config,
    _fluid_sync_initial_viewport_to_engine,
    _run_particle_playback_if_requested,
    _terminate_stats_plot_proc,
    _try_start_record_stats_plot_viewer,
)
from .process_utils import ProcessManager, is_tcp_port_accepting_connections
from .sph_config import generate_orcasph_config, setup_python_logging

logger = logging.getLogger(__name__)


def run_simulation_with_config(
    config: Dict,
    session_timestamp: Optional[str] = None,
    cpu_affinity: Optional[str] = None,
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
    """
    # 生成或使用统一时间戳
    if session_timestamp is None:
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 根据配置设置 Python 日志级别（必须在导入其他模块之前）
    setup_python_logging(config)

    if _run_particle_playback_if_requested(config):
        return

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

    process_manager = ProcessManager()
    env = None
    sph_wrapper = None
    traj_rec = None
    traj_player = None
    scene_output_path = None
    shutdown_event = threading.Event()

    _fluid_atexit_state["session_active"] = True
    _fluid_atexit_state["viewport_reset_done"] = False
    _fluid_atexit_state["env_ref"] = None
    _fluid_atexit_state["config_ref"] = config
    _fluid_atexit_state["owns_shared_services"] = False

    # -----------------------------------------------------------------------
    # 在函数入口（而非主循环入口）注册 SIGTERM，保证启动阶段也能响应：
    # OrcaLab stop_sim() 发 SIGTERM 后等 5 秒再 kill；我们在 handler 里同步
    # 执行全量清理（粒子/视口重置 + 子进程），然后 _exit(0) 退出，不依赖
    # finally 或主循环协作，确保在 5 秒 kill 之前完成。
    # -----------------------------------------------------------------------
    def _sigterm_cleanup_handler(_signum, _frame):
        """SIGTERM handler：同步完成全量清理后退出，保证在 OrcaLab kill 前完成。"""
        logger.info("\n⏹️  收到 SIGTERM，开始同步清理（OrcaLab 停止）...")
        shutdown_event.set()
        if not _fluid_atexit_state.get("viewport_reset_done"):
            _owns = _fluid_atexit_state.get("owns_shared_services")
            _env = _fluid_atexit_state.get("env_ref")

            _terminate_stats_plot_proc()

            # 1. 断开 OrcaLink Bridge（停止推位置给 OrcaSPH）
            if sph_wrapper is not None:
                try:
                    sph_wrapper.close()
                except Exception:
                    pass

            # 2. 告知 ParticleRender 结束仿真（停止接收粒子帧）
            if _owns:
                try:
                    _fluid_send_end_simulation_from_config(config)
                except Exception:
                    pass

            # 3. 终止 OrcaSPH / OrcaLink 子进程
            process_manager.cleanup_all()

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

    prev_sigterm_handler = signal.signal(signal.SIGTERM, _sigterm_cleanup_handler)

    try:
        logger.info("=" * 80)
        logger.info("Fluid-MuJoCo 耦合仿真启动")
        logger.info("=" * 80)

        # ============ 步骤 1: 创建 MuJoCo 环境 ============
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
        obs = env.reset()
        print(
            "[PRINT-DEBUG] run_simulation.py - env.reset() completed",
            file=sys.stderr,
            flush=True,
        )
        logger.info("✅ MuJoCo 环境创建成功\n")

        # ============ 步骤 2: 生成 scene.json ============
        particle_render_override = None  # set below when bound site is found
        if config["orcasph"]["enabled"] and config["orcasph"]["scene_auto_generate"]:
            logger.info("📝 步骤 2: 生成 SPH scene.json...")
            scene_uuid = str(uuid4()).replace("-", "_")
            scene_output_path = orcagym_tmp_dir / f"sph_scene_{scene_uuid}.json"

            # 获取 scene_config.json 的路径
            # 优先从 examples/fluid/ 目录查找，如果不存在则尝试 envs/fluid/
            scene_config_path = (
                ORCA_PLAYGROUND_ROOT / "examples" / "fluid" / config["sph"]["scene_config"]
            )
            if not scene_config_path.exists():
                # 如果不存在，尝试 envs/fluid/ 目录
                scene_config_path = FLUID_PACKAGE_DIR / config["sph"]["scene_config"]

            # 加载 sph_sim_config.json（SPH 侧配置模板）
            # 这个配置包含 orcalink_bridge.shared_modules.spring_force 等 SPH 侧参数
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
                env.unwrapped,
                config_path=str(scene_config_path),
                runtime_config=sph_config,  # 传递 SPH 配置（包含 orcalink_bridge.shared_modules.spring_force）
            )
            scene_data = scene_generator.generate_complete_scene(
                output_path=str(scene_output_path),
                include_fluid_blocks=config["sph"]["include_fluid_blocks"],
                include_wall=config["sph"]["include_wall"],
            )
            logger.info(f"✅ scene.json 已生成: {scene_output_path}")
            logger.info(f"   - RigidBodies: {len(scene_data.get('RigidBodies', []))} 个\n")

            # 计算 particle_render 的 grid_resolution / origin 覆盖值。
            # generate_complete_scene 内已调用 _init_particle_radius()，self.particle_radius 已就绪。
            particle_render_override = scene_generator.generate_particle_render_config(sph_config)

        # ============ 步骤 3: 启动 OrcaLink（延时 5 秒）============
        if config["orcalink"]["enabled"] and config["orcalink"]["auto_start"]:
            logger.info("🚀 步骤 3: 启动 OrcaLink Server...")

            # 查找 orcalink 可执行文件（与当前 Python 解释器在同一环境）
            python_bin_dir = Path(sys.executable).parent
            orcalink_bin = python_bin_dir / "orcalink"

            if not orcalink_bin.exists():
                # 尝试通过 shutil.which 查找
                orcalink_path = shutil.which("orcalink")
                if orcalink_path:
                    orcalink_bin = Path(orcalink_path)
                else:
                    raise FileNotFoundError(
                        f"orcalink command not found. "
                        f"Searched: {orcalink_bin}, PATH. "
                        f"Please ensure orca-link is installed: pip install -e /path/to/OrcaLink"
                    )

            # 构建启动参数：从配置中读取 port
            orcalink_port = config["orcalink"].get("port", 50351)
            orcalink_args = ["--port", str(orcalink_port)]

            # 添加其他自定义参数（如果配置中有 args 且不包含 --port）
            if "args" in config["orcalink"]:
                for arg in config["orcalink"]["args"]:
                    if arg not in ["--port", str(orcalink_port)]:
                        orcalink_args.append(arg)

            logger.info(f"启动 OrcaLink，端口: {orcalink_port}")
            log_file = orcagym_tmp_dir / f"orcalink_{session_timestamp}.log"
            process_manager.start_process(
                "OrcaLink",
                str(orcalink_bin),
                orcalink_args,
                log_file,
            )

            # 【关键】等待 OrcaLink 启动完成
            startup_delay = config["orcalink"].get("startup_delay", 5)
            logger.info(f"⏳ 等待 OrcaLink 启动完成（{startup_delay} 秒）...")
            time.sleep(startup_delay)
            logger.info(f"✅ OrcaLink Server 已就绪\n")

        # ============ 步骤 4: 启动 OrcaSPH（依赖 scene.json）============
        if config["orcasph"]["enabled"] and config["orcasph"]["auto_start"]:
            if scene_output_path is None:
                logger.error("❌ 无法启动 OrcaSPH：scene.json 未生成")
                config["orcasph"]["enabled"] = False
            else:
                logger.info("🚀 步骤 4: 启动 OrcaSPH...")

                # 查找 orcasph 可执行文件（与当前 Python 解释器在同一环境）
                python_bin_dir = Path(sys.executable).parent
                orcasph_bin = python_bin_dir / "orcasph"

                if not orcasph_bin.exists():
                    # 尝试通过 shutil.which 查找
                    orcasph_path = shutil.which("orcasph")
                    if orcasph_path:
                        orcasph_bin = Path(orcasph_path)
                    else:
                        raise FileNotFoundError(
                            f"orcasph command not found. "
                            f"Searched: {orcasph_bin}, PATH. "
                            f"Please ensure orca-sph is installed: pip install -e /path/to/SPlisHSPlasH"
                        )

                # 动态生成 orcasph 配置文件
                orcasph_config_path = orcagym_tmp_dir / f"orcasph_config_{session_timestamp}.json"
                orcasph_config_path, verbose_logging = generate_orcasph_config(
                    config,
                    orcasph_config_path,
                    particle_render_override=particle_render_override,
                )

                # 构建启动参数
                orcasph_args = config["orcasph"]["args"].copy()
                orcasph_args.extend(["--config", str(orcasph_config_path)])
                orcasph_args.extend(["--scene", str(scene_output_path)])

                # 根据配置文件自动设置日志级别
                if verbose_logging:
                    orcasph_args.extend(["--log-level", "DEBUG"])
                    logger.info("🔍 启用 DEBUG 日志级别 (verbose_logging=true)")
                else:
                    logger.info("ℹ️  使用默认 INFO 日志级别 (verbose_logging=false)")

                # 如果设置了 CPU 亲和性，使用 taskset 包装启动命令
                if cpu_affinity:
                    logger.info(f"📌 OrcaSPH CPU 亲和性: 核心 {cpu_affinity}")
                    orcasph_cmd = "taskset"
                    orcasph_args = ["-c", cpu_affinity, str(orcasph_bin)] + orcasph_args
                else:
                    orcasph_cmd = str(orcasph_bin)

                log_file = orcagym_tmp_dir / f"orcasph_{session_timestamp}.log"
                process_manager.start_process(
                    "OrcaSPH",
                    orcasph_cmd,
                    orcasph_args,
                    log_file,
                )
                logger.info("⏳ 等待 OrcaSPH 初始化（2 秒）...")
                time.sleep(2)
                logger.info("✅ OrcaSPH 已启动\n")

        _try_start_record_stats_plot_viewer(config, session_timestamp, orcagym_tmp_dir)

        logger.debug("[DEBUG] About to enter main loop...")
        sys.stdout.flush()
        sys.stderr.flush()

        # ============ 步骤 5: 连接并开始仿真 ============
        if config["orcasph"]["enabled"]:
            logger.info("🔗 步骤 5: 初始化 OrcaLinkBridge...")
            # 直接传入配置字典，不再需要 sph_mujoco_config_template.json
            logger.debug("[DEBUG] Creating OrcaLinkBridge instance...")
            print(
                "[PRINT-DEBUG] run_simulation.py - Creating OrcaLinkBridge instance...",
                file=sys.stderr,
                flush=True,
            )
            sph_wrapper = OrcaLinkBridge(env.unwrapped, config=config)
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
            connect_result = sph_wrapper.connect()
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
        else:
            logger.warning("⚠️  OrcaLink 未启用，SPH 集成已禁用")

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

        # SIGHUP（关终端/父 shell 退出）：协作退出主循环，由 finally 做清理
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, lambda *_: shutdown_event.set())

        # ============ 主循环参数 ============
        REALTIME_STEP = 0.02

        # ============ MuJoCo 人类操作轨迹（HDF5）============
        traj_cfg = config.get("mujoco_trajectory") or {}
        sph_mocap_names = frozenset()
        if sph_wrapper is not None:
            sph_mocap_names = sph_wrapper.sph_coupling_mocap_names
        if traj_cfg.get("enabled") and session_timestamp:
            out_p = Path(traj_cfg["output_path"])
            traj_rec = TrajectoryRecorder(
                out_p,
                env.unwrapped,
                session_timestamp,
                REALTIME_STEP,
                sph_mocap_names,
            )
            logger.info("MuJoCo trajectory recording enabled: %s", out_p.resolve())
        pb = traj_cfg.get("playback_path")
        if pb:
            traj_player = TrajectoryPlayer(Path(pb), env.unwrapped)
            logger.info(
                "MuJoCo trajectory playback: %s (%s frames)",
                Path(pb).resolve(),
                traj_player.num_frames,
            )

        # ============ 主循环 ============
        step_count = 0

        logger.debug("[DEBUG] Entering main loop (cooperative shutdown on SIGTERM/SIGHUP)...")
        while not shutdown_event.is_set():
            start_time = datetime.now()

            if traj_player is not None and traj_player.exhausted:
                logger.info(
                    "MuJoCo trajectory playback finished (%s frames).",
                    traj_player.num_frames,
                )
                break

            if step_count == 0:
                logger.debug("[DEBUG] First iteration - before SPH sync")

            # SPH 同步
            should_step = True
            if config["orcasph"]["enabled"] and sph_wrapper is not None:
                try:
                    if step_count == 0:
                        logger.debug("[DEBUG] Calling sph_wrapper.step()...")
                    should_step = sph_wrapper.step()
                    if step_count == 0:
                        logger.debug(f"[DEBUG] sph_wrapper.step() returned: {should_step}")
                except Exception as e:
                    logger.error(f"SPH 同步失败: {e}")
                    config["orcasph"]["enabled"] = False

            if step_count == 0:
                logger.debug(f"[DEBUG] Before MuJoCo step, should_step={should_step}")

            # MuJoCo step（轨迹回放：bridge.step 已更新 SPH mocap，此处仅叠加人类操作）
            if should_step:
                if traj_player is not None:
                    traj_player.push_pending_to_env()
                    obs, reward, terminated, truncated, info = env.step(None)
                    traj_player.advance_cursor()
                else:
                    # 无外围策略：None → SimEnv 内默认搅拌棒占位
                    obs, reward, terminated, truncated, info = env.step(None)
                if traj_rec is not None:
                    traj_rec.append_frame()
                env.render()
            else:
                env.render()

            if step_count == 0:
                logger.debug("[DEBUG] After render")

            # 实时同步（wait 可在睡眠期间立即响应停止标志）
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

        if shutdown_event.is_set():
            logger.info("\n⏹️  收到停止信号（SIGTERM/SIGHUP），协作退出主循环")

    except KeyboardInterrupt:
        logger.info("\n⏹️  用户中断仿真")
    except Exception as e:
        logger.error(f"\n❌ 仿真错误: {e}", exc_info=True)
    finally:
        try:
            if prev_sigterm_handler is not None and hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, prev_sigterm_handler)
        except (OSError, ValueError):
            pass
        logger.info("\n🧹 清理资源...")

        _terminate_stats_plot_proc()

        owns = _fluid_atexit_state.get("owns_shared_services")

        # ── 轨迹 HDF5 关闭 ──────────────────────────────────────────────────────
        try:
            if traj_rec is not None:
                traj_rec.close()
        except Exception as e:
            logger.warning("trajectory recorder close: %s", e)
        try:
            if traj_player is not None:
                traj_player.close()
        except Exception as e:
            logger.warning("trajectory player close: %s", e)

        # ── 第 1 步：先断开 OrcaLink Bridge（停止向 OrcaLink/SPH 推位置）──────────
        if sph_wrapper:
            sph_wrapper.close()

        # ── 第 2 步：告知 ParticleRender 仿真结束（停止接收粒子帧）────────────────
        if owns:
            _fluid_send_end_simulation_from_config(config)

        # ── 第 3 步：终止 OrcaSPH / OrcaLink 子进程，确保不再有粒子帧被推送 ──────
        process_manager.cleanup_all()

        # ── 第 4 步：短暂等待，让 ParticleRender 处理完最后可能在途的粒子帧 ───────
        # OrcaSPH 在收到 SIGTERM 后还可能有 1-2 帧已提交给 ParticleRender，
        # 等待 200 ms 使这些帧全部被 EndSimulation 之后的逻辑丢弃。
        if owns:
            time.sleep(0.2)

        # ── 第 5 步：重置刚体位姿到初始状态并推给 OrcaSim ──────────────────────
        # 此时 OrcaSPH 已终止、ParticleRender 已收到 EndSimulation，
        # gym.render() 推的初始 qpos 不会再被粒子帧覆盖。
        if env is not None:
            try:
                if owns:
                    _fluid_sync_initial_viewport_to_engine(env)
            except Exception as e:
                logger.warning(f"退出时 reset_simulation / 同步失败（可忽略）: {e}")
            try:
                env.close()
            except Exception as e:
                logger.warning(f"env.close() 失败: {e}")

        _fluid_atexit_state["viewport_reset_done"] = True
        _fluid_atexit_state["session_active"] = False
        logger.info("✅ 清理完成")
