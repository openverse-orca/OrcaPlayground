"""
Fluid 模块工具函数 - 封装启动流程
"""
import subprocess
import signal
import atexit
import threading
import time
import os
import json
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def _fluid_subprocess_preexec() -> None:
    """
    仅应在 subprocess.Popen(..., preexec_fn=...) 的子进程中调用。

    - setsid：与原先一致，便于按进程组向子树发信号。
    - PR_SET_PDEATHSIG(SIGTERM)：父进程（流体主控 Python）任意原因退出时，由内核向本子进程发
      SIGTERM。解决「关掉外部程序终端 / 强杀父进程」时未执行 finally，orcasph/orcalink 仍残留、
      引擎里停仿真也无法结束独立 SPH 进程的问题。
    """
    if hasattr(os, "setsid"):
        os.setsid()
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        # linux/prctl.h
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        pass


def is_tcp_port_accepting_connections(host: str, port: int, timeout: float = 0.3) -> bool:
    """若 host:port 上已有服务接受 TCP 连接则返回 True。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def send_end_simulation(server_address: str, reason: str = "simulation_finished") -> bool:
    """
    向 ParticleRender gRPC 服务发送 EndSimulation 请求，使流体渲染重置到初始不可见状态。

    实现委托给 ``orcasph_client.particle_render_rpc``（不在此处直接使用 grpc / protos）。

    Args:
        server_address: ParticleRender gRPC 服务地址（如 "localhost:50251"）
        reason: 可选的结束原因字符串，用于日志

    Returns:
        bool: 调用成功返回 True，否则返回 False
    """
    from orcasph_client.particle_render_rpc import send_particle_render_end_simulation

    return send_particle_render_end_simulation(
        server_address, reason=reason, timeout_sec=5.0, log=logger
    )


def _resolve_particle_render_server(config: Dict) -> Optional[str]:
    """从 fluid 配置 / sph 模板解析 ParticleRender gRPC 地址。"""
    particle_render_cfg = config.get("orcasph", {})
    pr_server = None
    try:
        template_filename = particle_render_cfg.get("config_template", "")
        if template_filename:
            template_path = (
                Path(__file__).parent.parent.parent
                / "examples"
                / "fluid"
                / template_filename
            )
            if not template_path.exists():
                template_path = Path(__file__).parent / template_filename
            if template_path.exists():
                with open(template_path, "r", encoding="utf-8") as tf:
                    tpl = json.load(tf)
                pr_server = tpl.get("particle_render", {}).get("grpc", {}).get(
                    "server_address"
                )
    except Exception:
        pass
    return (
        config.get("particle_render", {}).get("grpc", {}).get("server_address")
        or pr_server
    )


def _fluid_send_end_simulation_from_config(config: Dict) -> None:
    pr_server = _resolve_particle_render_server(config)
    if pr_server and config.get("orcasph", {}).get("enabled", False):
        logger.info(f"📤 发送 EndSimulation 到 ParticleRender ({pr_server})...")
        send_end_simulation(pr_server, reason="simulation_finished")


def _fluid_sync_initial_viewport_to_engine(env) -> None:
    """reset_simulation + 强制 gym.render，把初始 qpos 推到 OrcaSim。

    Ctrl+C 可能在 sph_wrapper.step() 的 loop.run_until_complete() 内抛出
    KeyboardInterrupt，使 asyncio 事件循环处于"已停止但未清空"状态。
    直接再次调用 loop.run_until_complete() 会触发 RuntimeError: This event loop
    is already running。此处改为：
      1. 若循环当前没有在运行（正常情况），直接 run_until_complete。
      2. 若循环被标记为 running（极少发生，通常是嵌套调用），在独立线程里跑以绕过限制。
      3. 若循环已关闭，新建一个临时循环完成同步。
    """
    import asyncio
    import concurrent.futures

    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "reset_simulation"):
        return
    logger.info("🔄 重置 OrcaSim 仿真到初始状态（恢复刚体位姿）...")
    unwrapped.reset_simulation()
    if hasattr(unwrapped, "mj_forward"):
        unwrapped.mj_forward()
    gym_core = getattr(unwrapped, "gym", None)
    loop = getattr(unwrapped, "loop", None)
    if gym_core is None or not hasattr(gym_core, "render"):
        return

    async def _do_render():
        await gym_core.render()

    def _run_in_thread():
        """在新线程里跑一个全新的事件循环执行 render（绕过已运行/已关闭的原循环）。"""
        new_loop = asyncio.new_event_loop()
        try:
            new_loop.run_until_complete(_do_render())
        finally:
            new_loop.close()

    try:
        if loop is None or loop.is_closed():
            # 原循环已关闭，新建临时循环
            _run_in_thread()
        elif loop.is_running():
            # 循环正在运行（极少情况），转线程执行
            logger.debug("事件循环仍在运行，使用线程执行 gym.render()")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                ex.submit(_run_in_thread).result(timeout=5)
        else:
            loop.run_until_complete(_do_render())
        logger.info("✅ 已将初始 qpos 同步到 OrcaSim（UpdateLocalEnv）")
    except Exception as e:
        logger.warning(f"gym.render() 同步失败: {e}")


# 关终端等导致未走 finally 时，解释器退出阶段仍尽力恢复粒子与视口（atexit 不保证在 SIGKILL 下执行）
_fluid_atexit_state: Dict[str, Any] = {
    "session_active": False,
    "viewport_reset_done": False,
    "env_ref": None,
    "config_ref": None,
    # 仅在 gym.make 成功之后置 True；避免第二实例因端口占用等提前退出时仍向共享 ParticleRender 发 EndSimulation，干扰第一实例
    "owns_shared_services": False,
}


def _atexit_fluid_visual_reset() -> None:
    st = _fluid_atexit_state
    if not st["session_active"] or st["viewport_reset_done"]:
        return
    if not st.get("owns_shared_services"):
        return
    cfg = st["config_ref"]
    if cfg is None:
        return
    try:
        logger.info(
            "🧹 atexit: 尽力恢复流体/刚体渲染（例如直接关闭终端未执行 finally）..."
        )
        _fluid_send_end_simulation_from_config(cfg)
        env = st["env_ref"]
        if env is not None:
            try:
                _fluid_sync_initial_viewport_to_engine(env)
            except Exception as e:
                logger.warning(f"atexit 同步 OrcaSim 视口失败: {e}")
            try:
                env.close()
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"atexit 流体视口重置失败: {e}")
    finally:
        st["viewport_reset_done"] = True
        st["session_active"] = False


atexit.register(_atexit_fluid_visual_reset)


class ProcessManager:
    """进程管理器"""
    
    def __init__(self):
        self.processes = {}
        atexit.register(self.cleanup_all)
    
    def start_process(self, name: str, command: str, args: list, 
                     log_file: Optional[Path] = None) -> subprocess.Popen:
        """启动进程"""
        cmd = [command] + args
        logger.info(f"🚀 启动 {name}: {' '.join(cmd)}")
        
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(log_file, 'w', buffering=1)
            process = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                preexec_fn=_fluid_subprocess_preexec,
            )
            process.log_file = log_handle
        else:
            process = subprocess.Popen(cmd, preexec_fn=_fluid_subprocess_preexec)
        
        self.processes[name] = process
        logger.info(f"✅ {name} 已启动 (PID: {process.pid})")
        return process
    
    def terminate_process(self, name: str, timeout: int = 5):
        """终止进程"""
        if name not in self.processes:
            return
        
        process = self.processes[name]
        if process.poll() is None:
            logger.info(f"⏹️  终止 {name} (PID: {process.pid})...")
            try:
                if hasattr(os, 'setsid'):
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=timeout)
                logger.info(f"✅ {name} 已终止")
            except Exception as e:
                logger.error(f"❌ 终止 {name} 失败: {e}")
        
        del self.processes[name]
    
    def cleanup_all(self):
        """清理所有进程"""
        for name in list(self.processes.keys()):
            self.terminate_process(name)


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place (override wins on conflicts)."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def _apply_particle_render_run_mode(orcasph_config: dict, fluid_config: dict) -> None:
    """
    Apply config['particle_render_run'] to particle_render after template + MJCF overrides.

    - live: force recording.enabled false (grpc unchanged from template).
    - record: recording on, grpc outbound off, output_path and optional record_fps from run dict.
    """
    pr_run = fluid_config.get("particle_render_run") or {}
    mode = pr_run.get("mode", "live")
    if "particle_render" not in orcasph_config:
        return
    pr = orcasph_config["particle_render"]
    if mode == "live":
        _deep_merge(pr, {"recording": {"enabled": False}})
        logger.info("[ParticleRender] run mode live: recording.enabled=false")
        return
    if mode == "record":
        rec_path = pr_run.get("record_output_path") or ""
        override: Dict[str, Any] = {
            "grpc": {"enabled": False},
            "recording": {
                "enabled": True,
                "output_path": rec_path,
            },
        }
        _deep_merge(pr, override)
        rf = pr_run.get("record_fps")
        if rf is not None:
            rf_f = float(rf)
            if "recording" not in pr:
                pr["recording"] = {}
            pr["recording"]["record_fps"] = rf_f
            if "grpc" not in pr:
                pr["grpc"] = {}
            pr["grpc"]["update_rate_hz"] = rf_f
        logger.info(
            f"[ParticleRender] run mode record: gRPC disabled, HDF5 output_path={rec_path!r}"
        )
        return


def generate_orcasph_config(
    fluid_config: Dict,
    output_path: Path,
    particle_render_override: Optional[Dict] = None,
) -> tuple[Path, bool]:
    """
    动态生成 orcasph 配置文件
    
    Args:
        fluid_config: 完整的 fluid_config.json 内容
        output_path: 输出配置文件路径
        
    Returns:
        (生成的配置文件路径, verbose_logging配置值)
    """
    orcasph_cfg = fluid_config.get('orcasph', {})
    orcalink_cfg = fluid_config.get('orcalink', {})
    
    # 支持两种方式：外部模板文件（新）或内嵌配置（旧，向后兼容）
    orcasph_config_template = {}
    
    if 'config_template' in orcasph_cfg:
        # 新方式：从外部文件加载模板
        template_filename = orcasph_cfg['config_template']
        # 尝试多个位置查找模板文件
        template_paths = [
            Path(__file__).parent.parent.parent / "examples" / "fluid" / template_filename,
            Path(__file__).parent / template_filename,
            Path(template_filename)  # 相对于当前工作目录
        ]
        
        template_path = None
        for path in template_paths:
            if path.exists():
                template_path = path
                break
        
        if template_path:
            with open(template_path, 'r', encoding='utf-8') as f:
                orcasph_config_template = json.load(f)
            logger.info(f"✅ 从模板加载 SPH 配置: {template_path}")
        else:
            logger.warning(f"⚠️  配置模板文件未找到: {template_filename}，尝试的路径：{template_paths}")
            orcasph_config_template = {}
    elif 'config' in orcasph_cfg:
        # 旧方式：内嵌配置（向后兼容）
        orcasph_config_template = orcasph_cfg['config']
        logger.info("✅ 使用内嵌 SPH 配置（旧格式）")
    else:
        logger.warning("⚠️  未找到 SPH 配置模板，使用空配置")
    
    # 构建完整的 orcasph 配置（合并模板和动态参数）
    orcasph_config = {
        "orcalink_client": {
            "enabled": orcalink_cfg.get('enabled', True),
            "server_address": f"{orcalink_cfg.get('host', 'localhost')}:{orcalink_cfg.get('port', 50351)}",
            **orcasph_config_template.get('orcalink_client', {})
        },
        "orcalink_bridge": orcasph_config_template.get('orcalink_bridge', {}),
        "physics": orcasph_config_template.get('physics', {}),
        "debug": orcasph_config_template.get('debug', {})
    }
    
    # 添加 particle_render 配置（如果模板中存在）
    if 'particle_render' in orcasph_config_template:
        orcasph_config['particle_render'] = orcasph_config_template['particle_render']

    # 用从 MJCF bound site 计算出的值覆盖 particle_render 中的 grid_resolution / origin。
    # 仅当调用方传入了计算结果时才合并（无 bound site 时 particle_render_override=None，不覆盖）。
    if particle_render_override and 'particle_render' in orcasph_config:
        _deep_merge(orcasph_config['particle_render'], particle_render_override)
        logger.info(f"particle_render config overridden from MJCF bound site: {particle_render_override}")

    _apply_particle_render_run_mode(orcasph_config, fluid_config)
    
    # 覆盖关键参数（确保动态值生效）
    orcasph_config['orcalink_client']['server_address'] = f"{orcalink_cfg.get('host', 'localhost')}:{orcalink_cfg.get('port', 50351)}"
    orcasph_config['orcalink_client']['enabled'] = orcalink_cfg.get('enabled', True)
    
    # 写入文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(orcasph_config, f, indent=2, ensure_ascii=False)
    
    # 提取 verbose_logging 配置值
    verbose_logging = orcasph_config.get('debug', {}).get('verbose_logging', False)
    
    logger.info(f"✅ 已生成 orcasph 配置文件: {output_path}")
    return output_path, verbose_logging


def setup_python_logging(config: Dict) -> None:
    """根据配置设置 Python 日志级别"""
    verbose_logging = config.get('debug', {}).get('verbose_logging', False)
    
    # 设置根 logger 的级别
    root_logger = logging.getLogger()
    
    # 清除现有的 handlers（避免重复）
    root_logger.handlers.clear()
    
    # 创建统一的 formatter，包含模块名称
    # 格式: [模块名] 级别: 消息
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    
    # 创建 console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 根据配置设置日志级别
    if verbose_logging:
        root_logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
        logger.info("🔍 Python 日志级别: DEBUG (verbose_logging=true)")
    else:
        root_logger.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
        logger.info("ℹ️  Python 日志级别: INFO (verbose_logging=false)")
    
    # 添加 handler 到根 logger
    root_logger.addHandler(console_handler)
    
    # 配置 OrcaLinkClient 的日志
    try:
        from orcalink_client import setup_logging as setup_orcalink_logging
        setup_orcalink_logging(verbose=verbose_logging, use_root_handler=True)
    except ImportError:
        # 如果 orcalink_client 未安装，跳过
        pass


def _run_particle_playback_if_requested(config: Dict) -> bool:
    """
    If config['particle_render_run']['mode'] == 'playback', stream the HDF5 to OrcaStudio
    via orcasph_client.particle_replay and return True so the caller exits without starting
    MuJoCo / OrcaLink / OrcaSPH.
    """
    pr_run = config.get("particle_render_run") or {}
    if pr_run.get("mode") != "playback":
        return False

    h5 = pr_run.get("playback_h5")
    if not h5:
        logger.error("playback 模式需要 --h5")
        sys.exit(1)
    h5p = Path(h5)
    if not h5p.is_file():
        logger.error(f"HDF5 文件不存在: {h5p}")
        sys.exit(1)

    target = pr_run.get("playback_target")
    if not target:
        target = _resolve_particle_render_server(config)
    if not target:
        logger.error(
            "playback 模式需要 --playback-target，或在 config_template 指向的 "
            "sph_sim_config.json 中配置 particle_render.grpc.server_address"
        )
        sys.exit(1)

    try:
        from orcasph_client.particle_replay import run_playback
    except ImportError:
        logger.error(
            "playback 需要已安装的 orca-sph 包（提供 orcasph_client.particle_replay）。"
            "请执行: pip install orca-sph"
        )
        sys.exit(1)

    fps = float(pr_run.get("playback_fps") or 0.0)
    logger.info(
        "▶️  粒子 HDF5 回放: h5=%s target=%s fps=%s",
        h5p.resolve(),
        target,
        fps if fps > 0 else "(record_fps from file)",
    )
    try:
        run_playback(
            str(h5p.resolve()),
            target,
            playback_fps=fps,
            start_frame=0,
            max_frames=0,
        )
    except KeyboardInterrupt:
        logger.info("回放已中断")
    except (ValueError, OSError) as e:
        logger.error("%s", e)
        sys.exit(1)
    return True


def run_simulation_with_config(config: Dict, session_timestamp: Optional[str] = None, cpu_affinity: Optional[str] = None) -> None:
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
    import gymnasium as gym
    import sys
    import uuid
    from datetime import datetime
    from .orcalink_bridge import OrcaLinkBridge
    from .scene_generator import SceneGenerator
    
    # 生成或使用统一时间戳
    if session_timestamp is None:
        session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
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
        orcagym_cfg = config['orcagym']
        env_id = f"{orcagym_cfg['env_name']}-OrcaGym-{orcagym_cfg['address'].replace(':', '-')}-000"
        
        print("[PRINT-DEBUG] utils.py - About to register gymnasium env", file=sys.stderr, flush=True)
        gym.register(
            id=env_id,
            entry_point="envs.fluid.sim_env:SimEnv",
            kwargs={
                'frame_skip': 20,
                'orcagym_addr': orcagym_cfg['address'],
                'agent_names': [orcagym_cfg['agent_name']],
                'time_step': 0.001
            },
            max_episode_steps=sys.maxsize
        )
        print("[PRINT-DEBUG] utils.py - Gymnasium env registered", file=sys.stderr, flush=True)
        
        print("[PRINT-DEBUG] utils.py - About to call gym.make()", file=sys.stderr, flush=True)
        env = gym.make(env_id)
        _fluid_atexit_state["env_ref"] = env
        _fluid_atexit_state["owns_shared_services"] = True
        print("[PRINT-DEBUG] utils.py - gym.make() completed", file=sys.stderr, flush=True)
        
        print("[PRINT-DEBUG] utils.py - About to call env.reset()", file=sys.stderr, flush=True)
        obs = env.reset()
        print("[PRINT-DEBUG] utils.py - env.reset() completed", file=sys.stderr, flush=True)
        logger.info("✅ MuJoCo 环境创建成功\n")
        
        # ============ 步骤 2: 生成 scene.json ============
        particle_render_override = None  # set below when bound site is found
        if config['orcasph']['enabled'] and config['orcasph']['scene_auto_generate']:
            logger.info("📝 步骤 2: 生成 SPH scene.json...")
            scene_uuid = str(uuid.uuid4()).replace('-', '_')
            scene_output_path = orcagym_tmp_dir / f"sph_scene_{scene_uuid}.json"
            
            # 获取 scene_config.json 的路径
            # 优先从 examples/fluid/ 目录查找，如果不存在则尝试 envs/fluid/
            scene_config_path = Path(__file__).parent.parent.parent / "examples" / "fluid" / config['sph']['scene_config']
            if not scene_config_path.exists():
                # 如果不存在，尝试 envs/fluid/ 目录
                scene_config_path = Path(__file__).parent / config['sph']['scene_config']
            
            # 加载 sph_sim_config.json（SPH 侧配置模板）
            # 这个配置包含 orcalink_bridge.shared_modules.spring_force 等 SPH 侧参数
            sph_config_template_path = Path(__file__).parent.parent.parent / "examples" / "fluid" / config['orcasph']['config_template']
            if not sph_config_template_path.exists():
                sph_config_template_path = Path(__file__).parent / config['orcasph']['config_template']
            
            if sph_config_template_path.exists():
                with open(sph_config_template_path, 'r', encoding='utf-8') as f:
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
                runtime_config=sph_config  # 传递 SPH 配置（包含 orcalink_bridge.shared_modules.spring_force）
            )
            scene_data = scene_generator.generate_complete_scene(
                output_path=str(scene_output_path),
                include_fluid_blocks=config['sph']['include_fluid_blocks'],
                include_wall=config['sph']['include_wall']
            )
            logger.info(f"✅ scene.json 已生成: {scene_output_path}")
            logger.info(f"   - RigidBodies: {len(scene_data.get('RigidBodies', []))} 个\n")

            # 计算 particle_render 的 grid_resolution / origin 覆盖值。
            # generate_complete_scene 内已调用 _init_particle_radius()，self.particle_radius 已就绪。
            particle_render_override = scene_generator.generate_particle_render_config(sph_config)
        
        # ============ 步骤 3: 启动 OrcaLink（延时 5 秒）============
        if config['orcalink']['enabled'] and config['orcalink']['auto_start']:
            logger.info("🚀 步骤 3: 启动 OrcaLink Server...")
            
            # 查找 orcalink 可执行文件（与当前 Python 解释器在同一环境）
            import sys
            import shutil
            
            python_bin_dir = Path(sys.executable).parent
            orcalink_bin = python_bin_dir / 'orcalink'
            
            if not orcalink_bin.exists():
                # 尝试通过 shutil.which 查找
                orcalink_path = shutil.which('orcalink')
                if orcalink_path:
                    orcalink_bin = Path(orcalink_path)
                else:
                    raise FileNotFoundError(
                        f"orcalink command not found. "
                        f"Searched: {orcalink_bin}, PATH. "
                        f"Please ensure orca-link is installed: pip install -e /path/to/OrcaLink"
                    )
            
            # 构建启动参数：从配置中读取 port
            orcalink_port = config['orcalink'].get('port', 50351)
            orcalink_args = ['--port', str(orcalink_port)]
            
            # 添加其他自定义参数（如果配置中有 args 且不包含 --port）
            if 'args' in config['orcalink']:
                for arg in config['orcalink']['args']:
                    if arg not in ['--port', str(orcalink_port)]:
                        orcalink_args.append(arg)
            
            logger.info(f"启动 OrcaLink，端口: {orcalink_port}")
            log_file = orcagym_tmp_dir / f"orcalink_{session_timestamp}.log"
            process_manager.start_process(
                "OrcaLink",
                str(orcalink_bin),
                orcalink_args,
                log_file
            )
            
            # 【关键】等待 OrcaLink 启动完成
            startup_delay = config['orcalink'].get('startup_delay', 5)
            logger.info(f"⏳ 等待 OrcaLink 启动完成（{startup_delay} 秒）...")
            time.sleep(startup_delay)
            logger.info(f"✅ OrcaLink Server 已就绪\n")
        
        # ============ 步骤 4: 启动 OrcaSPH（依赖 scene.json）============
        if config['orcasph']['enabled'] and config['orcasph']['auto_start']:
            if scene_output_path is None:
                logger.error("❌ 无法启动 OrcaSPH：scene.json 未生成")
                config['orcasph']['enabled'] = False
            else:
                logger.info("🚀 步骤 4: 启动 OrcaSPH...")
                
                # 查找 orcasph 可执行文件（与当前 Python 解释器在同一环境）
                python_bin_dir = Path(sys.executable).parent
                orcasph_bin = python_bin_dir / 'orcasph'
                
                if not orcasph_bin.exists():
                    # 尝试通过 shutil.which 查找
                    orcasph_path = shutil.which('orcasph')
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
                    config, orcasph_config_path,
                    particle_render_override=particle_render_override
                )
                
                # 构建启动参数
                orcasph_args = config['orcasph']['args'].copy()
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
                    log_file
                )
                logger.info("⏳ 等待 OrcaSPH 初始化（2 秒）...")
                time.sleep(2)
                logger.info("✅ OrcaSPH 已启动\n")
        
        logger.debug("[DEBUG] About to enter main loop...")
        sys.stdout.flush()
        sys.stderr.flush()

        # ============ 步骤 5: 连接并开始仿真 ============
        if config['orcasph']['enabled']:
            logger.info("🔗 步骤 5: 初始化 OrcaLinkBridge...")
            # 直接传入配置字典，不再需要 sph_mujoco_config_template.json
            logger.debug("[DEBUG] Creating OrcaLinkBridge instance...")
            print("[PRINT-DEBUG] utils.py - Creating OrcaLinkBridge instance...", file=sys.stderr, flush=True)
            sph_wrapper = OrcaLinkBridge(env.unwrapped, config=config)
            logger.debug("[DEBUG] OrcaLinkBridge instance created")
            print("[PRINT-DEBUG] utils.py - OrcaLinkBridge instance created...", file=sys.stderr, flush=True)
            
            logger.info("🔗 连接到 OrcaLink...")
            logger.debug("[DEBUG] Calling sph_wrapper.connect()...")
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            print("[PRINT-DEBUG] utils.py - Calling sph_wrapper.connect()...", file=sys.stderr, flush=True)
            connect_result = sph_wrapper.connect()
            print(f"[PRINT-DEBUG] utils.py - sph_wrapper.connect() returned: {connect_result}", file=sys.stderr, flush=True)
            logger.debug(f"[DEBUG] sph_wrapper.connect() RETURNED: {connect_result}")
            sys.stdout.flush()
            sys.stderr.flush()
            
            if not connect_result:
                logger.warning("⚠️  无法连接到 OrcaLink，SPH 集成已禁用")
                config['orcasph']['enabled'] = False
            else:
                logger.info("✅ OrcaLink 连接成功\n")
                logger.debug("[DEBUG] After OrcaLink connection success message")
        else:
            logger.warning("⚠️  OrcaLink 未启用，SPH 集成已禁用")
        
        import sys
        logger.debug("[DEBUG] About to enter main loop...")
        sys.stdout.flush()
        sys.stderr.flush()
        logger.info("=" * 80)
        logger.info("🎬 仿真主循环开始")
        logger.info("=" * 80)
        sys.stdout.flush()
        sys.stderr.flush()
        print("[PRINT-DEBUG] utils.py - About to enter main loop...", file=sys.stderr, flush=True)
        print("[PRINT-DEBUG] utils.py - Main loop started...", file=sys.stderr, flush=True)

        # SIGHUP（关终端/父 shell 退出）：协作退出主循环，由 finally 做清理
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, lambda *_: shutdown_event.set())

        # ============ 主循环 ============
        step_count = 0
        REALTIME_STEP = 0.02
        
        logger.debug("[DEBUG] Entering main loop (cooperative shutdown on SIGTERM/SIGHUP)...")
        while not shutdown_event.is_set():
            start_time = datetime.now()
            
            if step_count == 0:
                logger.debug("[DEBUG] First iteration - before SPH sync")
            
            # SPH 同步
            should_step = True
            if config['orcasph']['enabled'] and sph_wrapper is not None:
                try:
                    if step_count == 0:
                        logger.debug("[DEBUG] Calling sph_wrapper.step()...")
                    should_step = sph_wrapper.step()
                    if step_count == 0:
                        logger.debug(f"[DEBUG] sph_wrapper.step() returned: {should_step}")
                except Exception as e:
                    logger.error(f"SPH 同步失败: {e}")
                    config['orcasph']['enabled'] = False
            
            if step_count == 0:
                logger.debug(f"[DEBUG] Before MuJoCo step, should_step={should_step}")
            
            # MuJoCo step
            if should_step:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
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

        owns = _fluid_atexit_state.get("owns_shared_services")

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

