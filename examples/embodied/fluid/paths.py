"""OrcaPlayground 与 examples.embodied.fluid 包路径（供模板 / scene 等解析）。"""
from pathlib import Path

# paths.py 在 examples/embodied/fluid/：fluid → embodied → examples → 仓库根
ORCA_PLAYGROUND_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
FLUID_PACKAGE_DIR: Path = Path(__file__).resolve().parent
