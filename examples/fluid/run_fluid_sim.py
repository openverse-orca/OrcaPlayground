#!/usr/bin/env python3
"""
Fluid-MuJoCo 耦合仿真示例

【运行前提】
1. 已启动 OrcaStudio 或 OrcaLab
2. 已加载包含 SPH 标记的流体仿真场景

【启动模式】
- 自动模式（推荐）：脚本自动启动 OrcaLink 和 OrcaSPH
- 手动模式：用户预先启动服务，脚本仅连接

【使用方法】
    python run_fluid_sim.py
    python run_fluid_sim.py --config my_config.json
    python run_fluid_sim.py --manual-mode
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# 获取脚本文件所在目录，然后计算项目根目录
# 从 examples/fluid/run_fluid_sim.py 到项目根目录需要向上两级
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))

# 将项目根目录添加到 PYTHONPATH
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.fluid import run_simulation_with_config


def load_config(config_path: str) -> dict:
    """加载 JSON 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # 生成统一时间戳（在所有其他操作之前）
    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    orcagym_tmp_dir = Path.home() / ".orcagym" / "tmp"
    orcagym_tmp_dir.mkdir(parents=True, exist_ok=True)
    log_file = orcagym_tmp_dir / f"run_fluid_sim_{session_timestamp}.log"
    
    # 设置日志重定向（同时输出到控制台和文件）
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    log_handle = open(log_file, 'w', encoding='utf-8', buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeOutput(sys.stdout, log_handle)
    sys.stderr = TeeOutput(sys.stderr, log_handle)
    
    try:
        print(f"📝 日志文件: {log_file}")
        print("=" * 60)
        
        parser = argparse.ArgumentParser(
            description='Fluid-MuJoCo 耦合仿真',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
【运行前提】
  1. 已启动 OrcaStudio/OrcaLab 并加载场景
  2. 场景中包含带 SPH 标记的刚体

【启动模式】
  自动模式: 脚本自动启动 OrcaLink 和 OrcaSPH（推荐）
  手动模式: 用户预先启动，脚本仅连接（使用 --manual-mode）

【示例】
  python run_fluid_sim.py                    # 默认配置，无 GUI
  python run_fluid_sim.py --gui              # 启用 GUI
  python run_fluid_sim.py --config my.json   # 自定义配置
  python run_fluid_sim.py --manual-mode      # 手动模式
            """
        )
        
        parser.add_argument(
            '--config',
            default='fluid_sim_config.json',
            help='配置文件路径 (默认: fluid_sim_config.json)'
        )
        parser.add_argument(
            '--manual-mode',
            action='store_true',
            help='手动模式：禁用自动启动，需预先启动 orcalink 和 orcasph'
        )
        parser.add_argument(
            '--gui',
            action='store_true',
            help='启用 OrcaSPH GUI 可视化界面（默认禁用）'
        )
        parser.add_argument(
            '--cpu',
            default=None,
            metavar='CORES',
            help='设置 OrcaSPH 进程的 CPU 亲和性（通过 taskset），例如: 0-7 或 0,2,4,6'
        )
        
        args = parser.parse_args()
        
        # 加载配置
        config_path = Path(__file__).parent / args.config
        if not config_path.exists():
            print(f"❌ 错误: 配置文件不存在: {config_path}")
            return 1
        
        config = load_config(str(config_path))
        
        # 设置 OrcaSPH GUI 参数
        if 'orcasph' in config and config['orcasph'].get('enabled', False):
            # 初始化 args 如果不存在
            if 'args' not in config['orcasph']:
                config['orcasph']['args'] = []
            
            # 移除现有的 --gui 参数（如果有）
            config['orcasph']['args'] = [arg for arg in config['orcasph']['args'] if arg != '--gui']
            
            # 根据命令行参数添加 --gui
            if args.gui:
                config['orcasph']['args'].append('--gui')
                print("🎨 OrcaSPH GUI 已启用")
        
        # 手动模式
        if args.manual_mode:
            print("=" * 60)
            print("⚙️  手动模式已启用")
            print("=" * 60)
            print("请确保已手动启动以下服务：")
            print(f"  1. OrcaLink: orcalink --port {config['orcalink']['port']}")
            gui_flag = "--gui" if args.gui else ""
            print(f"  2. OrcaSPH: orcasph --scene <scene.json> {gui_flag}")
            print("=" * 60)
            config['orcalink']['auto_start'] = False
            config['orcasph']['auto_start'] = False
        
        # 运行仿真，传入时间戳
        try:
            run_simulation_with_config(config, session_timestamp=session_timestamp, cpu_affinity=args.cpu)
        except KeyboardInterrupt:
            print("\n✅ 仿真已停止")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0
    finally:
        # 恢复标准输出
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    exit(main())

