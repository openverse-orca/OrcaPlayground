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


def generate_orcasph_config(fluid_config: Dict, output_path: Path) -> Path:
    """
    动态生成 orcasph 配置文件
    
    Args:
        fluid_config: 完整的 fluid_config.json 内容
        output_path: 输出配置文件路径
        
    Returns:
        生成的配置文件路径
    """
    orcasph_cfg = fluid_config.get('orcasph', {})
    orcalink_cfg = fluid_config.get('orcalink', {})
    
    # 从 fluid_config 获取 orcasph 配置模板
    orcasph_config_template = orcasph_cfg.get('config', {})
    
    # 构建完整的 orcasph 配置
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
    
    # 确保 server_address 正确（覆盖模板中的值）
    orcasph_config['orcalink_client']['server_address'] = f"{orcalink_cfg.get('host', 'localhost')}:{orcalink_cfg.get('port', 50351)}"
    orcasph_config['orcalink_client']['enabled'] = orcalink_cfg.get('enabled', True)
    
    # 写入文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(orcasph_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ 已生成 orcasph 配置文件: {output_path}")
    return output_path


def run_simulation_with_config(config: Dict) -> None:
    """
    使用配置文件运行仿真
    
    启动顺序（重要）：
        1. 创建 MuJoCo 环境
        2. 生成 scene.json（依赖环境）
        3. 启动 orcalink（等待 5 秒）
        4. 启动 orcasph --scene <scene.json>（依赖 scene.json）
        5. 连接并开始仿真
    """
    import gymnasium as gym
    import sys
    import uuid
    from datetime import datetime
    from .orcalink_bridge import OrcaLinkBridge
    from .scene_generator import SceneGenerator
    
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
        
        env = gym.make(env_id)
        obs = env.reset()
        logger.info("✅ MuJoCo 环境创建成功\n")
        
        # ============ 步骤 2: 生成 scene.json ============
        if config['orcasph']['enabled'] and config['orcasph']['scene_auto_generate']:
            logger.info("📝 步骤 2: 生成 SPH scene.json...")
            orcagym_tmp_dir = Path.home() / ".orcagym" / "tmp"
            orcagym_tmp_dir.mkdir(parents=True, exist_ok=True)
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
            log_file = Path.home() / ".orcagym" / "tmp" / f"orcalink_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
                orcagym_tmp_dir = Path.home() / ".orcagym" / "tmp"
                orcasph_config_path = orcagym_tmp_dir / f"orcasph_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                generate_orcasph_config(config, orcasph_config_path)
                
                # 构建启动参数
                orcasph_args = config['orcasph']['args'].copy()
                orcasph_args.extend(["--config", str(orcasph_config_path)])
                orcasph_args.extend(["--scene", str(scene_output_path)])
                
                log_file = orcagym_tmp_dir / f"orcasph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                process_manager.start_process(
                    "OrcaSPH",
                    str(orcasph_bin),
                    orcasph_args,
                    log_file
                )
                logger.info("⏳ 等待 OrcaSPH 初始化（2 秒）...")
                time.sleep(2)
                logger.info("✅ OrcaSPH 已启动\n")
        
        # ============ 步骤 5: 连接并开始仿真 ============
        if config['orcasph']['enabled']:
            logger.info("🔗 步骤 5: 初始化 OrcaLinkBridge...")
            # 直接传入配置字典，不再需要 sph_mujoco_config_template.json
            sph_wrapper = OrcaLinkBridge(env.unwrapped, config=config)
            
            logger.info("🔗 连接到 OrcaLink...")
            if not sph_wrapper.connect():
                logger.warning("⚠️  无法连接到 OrcaLink，SPH 集成已禁用")
                config['orcasph']['enabled'] = False
            else:
                logger.info("✅ OrcaLink 连接成功\n")
        
        logger.info("=" * 80)
        logger.info("🎬 仿真主循环开始")
        logger.info("=" * 80)
        
        # ============ 主循环 ============
        step_count = 0
        REALTIME_STEP = 0.02
        
        while True:
            start_time = datetime.now()
            
            # SPH 同步
            should_step = True
            if config['orcasph']['enabled'] and sph_wrapper is not None:
                try:
                    should_step = sph_wrapper.step()
                except Exception as e:
                    logger.error(f"SPH 同步失败: {e}")
                    config['orcasph']['enabled'] = False
            
            # MuJoCo step
            if should_step:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
            else:
                env.render()
            
            # 实时同步
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed)
            
            step_count += 1
            if step_count % 100 == 0:
                logger.debug(f"仿真步数: {step_count}")
    
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

