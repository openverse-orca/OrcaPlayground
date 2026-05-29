#!/usr/bin/env python3
"""
Fluid-MuJoCo 耦合 + WaterJug_02_fluid 自动轨迹。

与 run_fluid_sim.py 相同，但默认使用 fluid_sim_config_auto.json，对水壶执行：
  世界 +Z 上移 0.3 m → 绕物体局部 Y 轴旋转 -90°。

run_fluid_sim.py 行为不变（无预定水壶运动）。

【推荐关卡】FluidTest_Hotel_Bar_Fangfang_AutoMove
  export ORCA_LEVEL_NAME=FluidTest_Hotel_Bar_Fangfang_AutoMove
  # 在 OrcaStudio 中 Play 后：
  python run_fluid_sim_auto.py --gui

【MuJoCo 水壶轨迹修正】（fluid_sim_config_auto.json → water_jug_trajectory）
  skip_sph_forces_on_mujoco  默认 true：轨迹阶段不把 SPH 力写回 MuJoCo（避免闪动/过转）
  reapply_after_step         默认 true：mj_step 后再次对齐 qpos
  clear_external_forces      默认 true：对齐前清除水壶 SITE 的 xfrc

【GUI】（与 run_fluid_sim.py 相同，转发至 OrcaSPH / MuJoCo）
  --gui          启用 SPlisHSPlasH（OrcaSPH）可视化窗口
  --mujoco-gui   启用 MuJoCo 原生查看器
  --sph-gui     同 --gui（run_fluid_sim_auto.sh 别名）

【示例】
  python run_fluid_sim_auto.py --gui
  python run_fluid_sim_auto.py --mujoco-gui
  python run_fluid_sim_auto.py --gui --mujoco-gui
  ./run_fluid_sim_auto.sh --all-gui
  python run_fluid_sim_auto.py --config fluid_sim_config_auto.json --max-sim-time 20
"""
from __future__ import annotations

import sys
from pathlib import Path

_AUTO_CONFIG = "fluid_sim_config_auto.json"


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    if not any(a == "--config" or a.startswith("--config=") for a in sys.argv[1:]):
        cfg = script_dir / _AUTO_CONFIG
        if not cfg.is_file():
            print(f"❌ 缺少配置文件: {cfg}", file=sys.stderr)
            return 1
        sys.argv[1:1] = ["--config", _AUTO_CONFIG]

    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from run_fluid_sim import main as _run_main

    return int(_run_main())


if __name__ == "__main__":
    sys.exit(main())
