"""OrcaPlayground 与 envs.fluid 包路径（供模板 / scene 等解析）。"""
from pathlib import Path

# envs/fluid/paths.py -> parent=fluid, parent.parent=envs, parent.parent.parent=repo root
ORCA_PLAYGROUND_ROOT: Path = Path(__file__).resolve().parent.parent.parent
FLUID_PACKAGE_DIR: Path = Path(__file__).resolve().parent
