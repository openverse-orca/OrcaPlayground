#!/usr/bin/env python3
"""
MuJoCo ↔ OrcaLink ↔ SPH 短链仿真（无 OrcaStudio Play / 无 50051 gRPC）。

默认使用 fluid_sim_config_short_chain.json，场景文件固定在：
  /home/hjadmin/OrcaApr24/SPH_bug/scene_3chain/

【双界面长跑（推荐）】
  python run_fluid_sim_auto_shortChain.py --gui --mujoco-gui
  # --gui：OrcaSPH 粒子与刚体窗口；--mujoco-gui：MuJoCo 被动查看器
  # 默认关 MuJoCo 窗口不会结束仿真（mujoco_gui.shutdown_on_close=false）；Ctrl+C 结束
  # 若需「关 MuJoCo 即停」：加 --mujoco-shutdown-on-close

【运行】
  cd OrcaPlayground/examples/fluid
  python run_fluid_sim_auto_shortChain.py
  python run_fluid_sim_auto_shortChain.py --max-sim-time 30

【说明】
  临时改动与恢复方法见 SPH_bug/scene_3chain/Temp_3chain.md
"""
from __future__ import annotations

import sys
from pathlib import Path

_SHORT_CHAIN_CONFIG = "fluid_sim_config_short_chain.json"


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    if not any(a == "--config" or a.startswith("--config=") for a in sys.argv[1:]):
        cfg = script_dir / _SHORT_CHAIN_CONFIG
        if not cfg.is_file():
            print(f"❌ 缺少配置文件: {cfg}", file=sys.stderr)
            return 1
        sys.argv[1:1] = ["--config", _SHORT_CHAIN_CONFIG]

    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from run_fluid_sim import main as _run_main

    return int(_run_main())


if __name__ == "__main__":
    sys.exit(main())
