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
  python run_fluid_sim.py
  python run_fluid_sim.py --config my_config.json
  python run_fluid_sim.py --manual-mode
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
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"❌ 错误: 配置文件不存在: {config_path}")
        return 1
    
    config = load_config(str(config_path))
    
    # 手动模式
    if args.manual_mode:
        print("=" * 60)
        print("⚙️  手动模式已启用")
        print("=" * 60)
        print("请确保已手动启动以下服务：")
        print(f"  1. OrcaLink: orcalink --port {config['orcalink']['port']}")
        print(f"  2. OrcaSPH: orcasph --scene <scene.json> --gui")
        print("=" * 60)
        config['orcalink']['auto_start'] = False
        config['orcasph']['auto_start'] = False
    
    # 运行仿真
    try:
        run_simulation_with_config(config)
    except KeyboardInterrupt:
        print("\n✅ 仿真已停止")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

