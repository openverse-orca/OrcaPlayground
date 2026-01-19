from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor, LightInfo, CameraSensorInfo, MaterialInfo
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
import numpy as np
import orca_gym.utils.rotations as rotations
import time
import random
import gymnasium as gym
import sys
import uuid
from datetime import datetime
import os
from typing import Optional
from pathlib import Path
import subprocess
import signal
import atexit

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()

# 导入 OrcaLinkBridge
try:
    from envs.fluid import OrcaLinkBridge
    _logger.info("OrcaLinkBridge imported successfully")
except ImportError as e:
    _logger.warning(f"Failed to import OrcaLinkBridge: {e}. SPH integration will be disabled.")


ENV_ENTRY_POINT = {
    "SimulationLoop": "orca_gym.scripts.sim_env:SimEnv",
}

TIME_STEP = 0.001
FRAME_SKIP = 20
REALTIME_STEP = TIME_STEP * FRAME_SKIP
CONTROL_FREQ = 1 / REALTIME_STEP

def register_env(orcagym_addr : str, 
                 env_name : str, 
                 env_index : int, 
                 agent_name : str, 
                 max_episode_steps : int) -> tuple[ str, dict ]:
    orcagym_addr_str = orcagym_addr.replace(":", "-")
    env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
    agent_names = [f"{agent_name}"]
    kwargs = {'frame_skip': FRAME_SKIP,   
                'orcagym_addr': orcagym_addr, 
                'agent_names': agent_names, 
                'time_step': TIME_STEP}           
    gym.register(
        id=env_id,
        entry_point=ENV_ENTRY_POINT[env_name],
        kwargs=kwargs,
        max_episode_steps= max_episode_steps,
        reward_threshold=0.0,
    )
    return env_id, kwargs



def run_simulation(orcagym_addr : str, 
                agent_name : str,
                env_name : str,
                scene_runtime: Optional[OrcaGymSceneRuntime] = None,
                enable_sph: bool = True,
                sph_config_path: Optional[str] = None,
                auto_start_sph: bool = True) -> None:
    """
    运行仿真主循环
    
    Args:
        orcagym_addr: OrcaGym 服务器地址
        agent_name: 代理名称
        env_name: 环境名称
        scene_runtime: 场景运行时（可选）
        enable_sph: 是否启用 SPH 集成
        sph_config_path: SPH 配置文件路径（如果为 None，使用默认位置）
        auto_start_sph: 是否自动启动 SPlisHSPlasH 程序（默认 True）
    """
    env = None
    sph_wrapper = None
    sph_process = None
    try:
        _logger.info(f"simulation running... , orcagym_addr:  {orcagym_addr}")

        env_index = 0
        env_id, kwargs = register_env(orcagym_addr, 
                                      env_name, 
                                      env_index, 
                                      agent_name, 
                                      sys.maxsize)
        _logger.info(f"Registered environment:  {env_id}")

        env = gym.make(env_id)        
        _logger.info("Starting simulation...")

        if scene_runtime is not None:
            if hasattr(env, "set_scene_runtime"):
                _logger.performance("Setting scene runtime...")
                env.set_scene_runtime(scene_runtime)
            elif hasattr(env.unwrapped, "set_scene_runtime"):
                _logger.performance("Setting scene runtime...")
                env.unwrapped.set_scene_runtime(scene_runtime)

        obs = env.reset()
        
        # 初始化 OrcaLinkBridge 和生成配置（如果启用 SPH）
        # 注意：必须在 env.reset() 之后，因为需要从 MuJoCo 模型读取数据
        if enable_sph:
            try:
                # 确定模板配置文件路径
                if sph_config_path is None:
                    # 默认使用模板文件
                    sph_config_path = Path(__file__).parent / "sph_mujoco_config_template.json"
                
                # 确定 scene.json 输出路径（使用 UUID 避免重名）
                orcagym_tmp_dir = Path.home() / ".orcagym" / "tmp"
                orcagym_tmp_dir.mkdir(parents=True, exist_ok=True)
                
                scene_uuid = str(uuid.uuid4()).replace('-', '_')
                scene_output_path = orcagym_tmp_dir / f"sph_scene_{scene_uuid}.json"
                scene_config_path = Path(__file__).parent / "scene_config.json"
                
                # 初始化 OrcaLinkBridge（内部会生成完整配置）
                if Path(sph_config_path).exists():
                    sph_wrapper = OrcaLinkBridge(env.unwrapped, str(sph_config_path))
                    _logger.info("OrcaLinkBridge initialized successfully")
                else:
                    _logger.warning(f"SPH config template not found: {sph_config_path}. SPH integration disabled.")
                    enable_sph = False
            except Exception as e:
                _logger.error(f"Failed to initialize OrcaLinkBridge: {e}. SPH integration disabled.")
                enable_sph = False
        
        # 自动生成 scene.json（如果启用 SPH）
        if enable_sph and sph_wrapper is not None:
            try:
                # 生成 scene.json
                _logger.info("Generating SPH scene.json from current MuJoCo model...")
                from scene_generator import SceneGenerator
                
                scene_generator = SceneGenerator(env.unwrapped, config_path=str(scene_config_path))
                scene_data = scene_generator.generate_complete_scene(
                    output_path=str(scene_output_path),
                    include_fluid_blocks=True,
                    include_wall=True
                )
                _logger.info(f"SPH scene.json generated: {scene_output_path}")
                _logger.info(f"  - RigidBodies: {len(scene_data.get('RigidBodies', []))} bodies")
                
                # 自动启动 SPlisHSPlasH 程序（如果启用）
                if auto_start_sph:
                    try:
                        # 计算 run.sh 的路径（相对于当前脚本）
                        script_dir = Path(__file__).parent
                        run_sh_path = script_dir / ".." / ".." / "run.sh"
                        run_sh_path = run_sh_path.resolve()
                        
                        if not run_sh_path.exists():
                            _logger.warning(f"run.sh not found at {run_sh_path}. Skipping auto-start of SPlisHSPlasH.")
                        else:
                            # 确保 run.sh 有执行权限
                            os.chmod(run_sh_path, 0o755)
                            
                            # 构建启动命令
                            cmd = [
                                str(run_sh_path),
                                "--scene", str(scene_output_path),
                                "--gui"
                            ]
                            
                            # 启动 SPlisHSPlasH 作为独立进程
                            _logger.info(f"Starting SPlisHSPlasH with scene: {scene_output_path}")
                            _logger.info(f"Command: {' '.join(cmd)}")
                            
                            # 创建日志文件路径（带时间戳）
                            log_dir = Path.home() / ".orcagym" / "tmp"
                            log_dir.mkdir(parents=True, exist_ok=True)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            log_file_path = log_dir / f"sph_output_{timestamp}.log"
                            
                            # 打开日志文件用于写入（追加模式，确保实时写入）
                            log_file = open(log_file_path, 'w', buffering=1)  # 行缓冲模式
                            
                            # 使用 subprocess.Popen 启动非阻塞进程，重定向输出到文件
                            sph_process = subprocess.Popen(
                                cmd,
                                stdout=log_file,
                                stderr=subprocess.STDOUT,  # 将 stderr 也重定向到同一个文件
                                cwd=str(run_sh_path.parent),
                                preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # 创建新的进程组
                            )
                            
                            # 保存文件对象到进程对象，防止被垃圾回收导致文件过早关闭
                            sph_process.log_file = log_file
                            
                            _logger.info(f"SPlisHSPlasH started with PID: {sph_process.pid}")
                            _logger.info(f"SPlisHSPlasH output log: {log_file_path}")
                            print(f"\n[SPlisHSPlasH] Output log file: {log_file_path}\n")
                            
                            # 注册清理函数，确保程序退出时终止子进程
                            # 使用闭包捕获 sph_process 变量
                            def make_cleanup_func(proc):
                                def cleanup_sph_process():
                                    if proc and proc.poll() is None:
                                        _logger.info(f"Terminating SPlisHSPlasH process (PID: {proc.pid})...")
                                        try:
                                            if hasattr(os, 'setsid'):
                                                # 终止整个进程组
                                                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                                            else:
                                                proc.terminate()
                                            # 等待进程结束（最多 5 秒）
                                            proc.wait(timeout=5)
                                        except subprocess.TimeoutExpired:
                                            _logger.warning("SPlisHSPlasH process did not terminate gracefully, forcing kill...")
                                            try:
                                                if hasattr(os, 'setsid'):
                                                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                                                else:
                                                    proc.kill()
                                            except Exception as e:
                                                _logger.error(f"Error killing SPlisHSPlasH process: {e}")
                                        except Exception as e:
                                            _logger.error(f"Error terminating SPlisHSPlasH process: {e}")
                                return cleanup_sph_process
                            
                            atexit.register(make_cleanup_func(sph_process))
                            
                    except Exception as e:
                        _logger.error(f"Failed to start SPlisHSPlasH: {e}. Continuing without auto-start.")
                        _logger.error(f"Error details: {e}", exc_info=True)
                        sph_process = None
                
            except Exception as e:
                _logger.error(f"Failed to generate scene.json: {e}. Continuing without scene generation.")
                enable_sph = False
        
        # 在主循环开始前尝试连接 OrcaLink
        if enable_sph and sph_wrapper is not None:
            _logger.info("Attempting to connect to OrcaLink before starting simulation loop...")
            if not sph_wrapper.connect():
                _logger.warning("Failed to connect to OrcaLink, SPH integration disabled")
                enable_sph = False
        
        step_count = 0
        
        while True:
            start_time = datetime.now()

            # SPH 同步步骤（在 env.step 之前）
            should_step_mujoco = True  # 默认允许执行
            if enable_sph and sph_wrapper is not None:
                try:
                    should_step_mujoco = sph_wrapper.step()  # 检查返回值！
                except Exception as e:
                    _logger.error(f"SPH wrapper step failed: {e}. Continuing without SPH integration.")
                    enable_sph = False
                    should_step_mujoco = True  # SPH 失败时不阻塞 MuJoCo

            # 根据流控状态决定是否执行 MuJoCo step
            if should_step_mujoco:
                # MuJoCo 仿真步进
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                # 渲染
                env.render()
            else:
                # 流控暂停，跳过 MuJoCo step
                _logger.debug("[MuJoCo] Step paused by flow control")
                # 保持渲染以更新显示
                env.render()

            # 实时同步
            elapsed_time = datetime.now() - start_time
            if elapsed_time.total_seconds() < REALTIME_STEP:
                time.sleep(REALTIME_STEP - elapsed_time.total_seconds())
            
            step_count += 1
            if step_count % 100 == 0:
                _logger.debug(f"Simulation step: {step_count}")


    except KeyboardInterrupt:
        _logger.info("Simulation stopped by user")        
    except Exception as e:
        _logger.error(f"Simulation error: {e}", exc_info=True)
    finally:
        # 清理资源
        # 首先终止 SPlisHSPlasH 进程
        if sph_process is not None:
            try:
                if sph_process.poll() is None:  # 进程仍在运行
                    _logger.info(f"Terminating SPlisHSPlasH process (PID: {sph_process.pid})...")
                    try:
                        if hasattr(os, 'setsid') and hasattr(os, 'getpgid'):
                            try:
                                # 终止整个进程组
                                pgid = os.getpgid(sph_process.pid)
                                os.killpg(pgid, signal.SIGTERM)
                            except (OSError, ProcessLookupError):
                                # 进程可能已经退出，尝试直接终止
                                sph_process.terminate()
                        else:
                            sph_process.terminate()
                        # 等待进程结束（最多 5 秒）
                        sph_process.wait(timeout=5)
                        _logger.info("SPlisHSPlasH process terminated")
                    except subprocess.TimeoutExpired:
                        _logger.warning("SPlisHSPlasH process did not terminate gracefully, forcing kill...")
                        try:
                            if hasattr(os, 'setsid') and hasattr(os, 'getpgid'):
                                try:
                                    pgid = os.getpgid(sph_process.pid)
                                    os.killpg(pgid, signal.SIGKILL)
                                except (OSError, ProcessLookupError):
                                    # 进程可能已经退出，尝试直接 kill
                                    sph_process.kill()
                            else:
                                sph_process.kill()
                            sph_process.wait()
                        except Exception as e:
                            _logger.error(f"Error killing SPlisHSPlasH process: {e}")
                    except Exception as e:
                        _logger.error(f"Error terminating SPlisHSPlasH process: {e}")
                else:
                    _logger.debug(f"SPlisHSPlasH process already terminated (exit code: {sph_process.returncode})")
            except Exception as e:
                _logger.error(f"Error cleaning up SPlisHSPlasH process: {e}")
        
        if sph_wrapper is not None:
            try:
                sph_wrapper.close()
                _logger.info("OrcaLinkBridge closed")
            except KeyboardInterrupt:
                _logger.warning("OrcaLinkBridge close interrupted by user")
            except Exception as e:
                _logger.error(f"Error closing OrcaLinkBridge: {e}")
        
        if env is not None:
            try:
                env.close()
                _logger.info("Environment closed")
            except KeyboardInterrupt:
                _logger.warning("Environment close interrupted by user")
            except Exception as e:
                _logger.error(f"Error closing environment: {e}")


def str_to_bool(v):
    """将字符串转换为布尔值"""
    import argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    """命令行入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MuJoCo simulation with optional SPH integration')
    parser.add_argument('--orcagym-addr', default='localhost:50051',
                       help='OrcaGym server address (default: localhost:50051)')
    parser.add_argument('--agent-name', default='NoRobot',
                       help='Agent name (default: NoRobot)')
    parser.add_argument('--env-name', default='SimulationLoop',
                       help='Environment name (default: SimulationLoop)')
    parser.add_argument('--disable-sph', action='store_true',
                       help='Disable SPH integration')
    parser.add_argument('--sph-config', default=None,
                       help='Path to SPH configuration file (default: sph_mujoco_config.json in script directory)')
    parser.add_argument('--auto-start-sph', type=str_to_bool, default=True, nargs='?', const=True,
                       help='Enable automatic startup of SPlisHSPlasH program (default: True). Use --auto-start-sph false to disable.')
    
    args = parser.parse_args()
    
    run_simulation(
        orcagym_addr=args.orcagym_addr,
        agent_name=args.agent_name,
        env_name=args.env_name,
        enable_sph=not args.disable_sph,
        sph_config_path=args.sph_config,
        auto_start_sph=args.auto_start_sph
    )


if __name__ == "__main__":
    main()
