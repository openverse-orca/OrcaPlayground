#!/usr/bin/env python3
"""
布料-MuJoCo 耦合仿真示例

【运行前提】
1. 已启动 OrcaStudio 或 OrcaLab 并 Play 含布料组件的场景（全链路 P2.3c）
2. 或本地 MJCF 场景（phase1_slide 等离线验收）

【使用方法】
    python run_cloth_sim.py
    python run_cloth_sim.py --config cloth_sim_config.phase1_slide.json
    python -m examples.embodied.cloth.run_cloth_sim --config cloth_sim_config.json
"""

from __future__ import annotations

import os
import sys

# 从 examples/embodied/cloth/run_cloth_sim.py 到 OrcaPlayground 根目录向上三级
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from examples.embodied.cloth.launch.run_cloth_simulation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
