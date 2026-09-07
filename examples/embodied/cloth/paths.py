"""OrcaPlayground 与 examples.embodied.cloth 包路径（供模板 / scene 等解析）。"""
from pathlib import Path

# paths.py 在 examples/embodied/cloth/：cloth → embodied → examples → 仓库根
ORCA_PLAYGROUND_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
CLOTH_PACKAGE_DIR: Path = Path(__file__).resolve().parent
# 多仓库工作区根（Docker 下通常为 /opt，含 OrcaPlayground / XPBD / OrcaLink 等）
ORCA_REPO_ROOT: Path = ORCA_PLAYGROUND_ROOT.parent
