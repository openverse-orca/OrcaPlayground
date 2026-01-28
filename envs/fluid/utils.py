"""
Fluid 模块工具函数 - 封装启动流程
"""
import subprocess
import signal
import atexit
import time
import os
import json
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


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
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
            process.log_file = log_handle
        else:
            process = subprocess.Popen(cmd, preexec_fn=os.setsid if hasattr(os, 'setsid') else None)
        
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


def generate_orcasph_config(fluid_config: Dict, output_path: Path) -> tuple[Path, bool]:
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


def run_simulation_with_config(config: Dict, session_timestamp: Optional[str] = None) -> None:
    """
    使用配置文件运行仿真
    
    启动顺序（重要）：
        1. 创建 MuJoCo 环境
        2. 生成 scene.json（依赖环境）
        3. 启动 orcalink（等待 5 秒）
        4. 启动 orcasph --scene <scene.json>（依赖 scene.json）
        5. 连接并开始仿真
    
    Args:
        config: 配置字典
        session_timestamp: 会话时间戳（用于统一日志文件名），如果为None则自动生成
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
    
    orcagym_tmp_dir = Path.home() / ".orcagym" / "tmp"
    orcagym_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    process_manager = ProcessManager()
    env = None
    sph_wrapper = None
    scene_output_path = None
    
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
        print("[PRINT-DEBUG] utils.py - gym.make() completed", file=sys.stderr, flush=True)
        
        print("[PRINT-DEBUG] utils.py - About to call env.reset()", file=sys.stderr, flush=True)
        obs = env.reset()
        print("[PRINT-DEBUG] utils.py - env.reset() completed", file=sys.stderr, flush=True)
        logger.info("✅ MuJoCo 环境创建成功\n")
        
        # ============ 步骤 2: 生成 scene.json ============
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
            
            scene_generator = SceneGenerator(env.unwrapped, config_path=str(scene_config_path))
            scene_data = scene_generator.generate_complete_scene(
                output_path=str(scene_output_path),
                include_fluid_blocks=config['sph']['include_fluid_blocks'],
                include_wall=config['sph']['include_wall']
            )
            logger.info(f"✅ scene.json 已生成: {scene_output_path}")
            logger.info(f"   - RigidBodies: {len(scene_data.get('RigidBodies', []))} 个\n")
        
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
            orcalink_port = config['orcalink'].get('port', 50051)
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
                orcasph_config_path, verbose_logging = generate_orcasph_config(config, orcasph_config_path)
                
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
                
                log_file = orcagym_tmp_dir / f"orcasph_{session_timestamp}.log"
                process_manager.start_process(
                    "OrcaSPH",
                    str(orcasph_bin),
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
        
        # ============ 主循环 ============
        step_count = 0
        REALTIME_STEP = 0.02
        
        logger.debug("[DEBUG] Entering while True loop...")
        while True:
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
            
            # 实时同步
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed)
            
            step_count += 1
            if step_count == 1:
                logger.debug("[DEBUG] Completed first iteration successfully")
            if step_count % 100 == 0:
                logger.info(f"仿真步数: {step_count}")
    
    except KeyboardInterrupt:
        logger.info("\n⏹️  用户中断仿真")
    except Exception as e:
        logger.error(f"\n❌ 仿真错误: {e}", exc_info=True)
    finally:
        logger.info("\n🧹 清理资源...")
        if sph_wrapper:
            sph_wrapper.close()
        if env:
            env.close()
        process_manager.cleanup_all()
        logger.info("✅ 清理完成")

