"""
布料-MuJoCo-XPBD 耦合仿真示例

MuJoCo 刚体经 OrcaLink 与 XPBD 布料耦合；配置与场景模板位于本包根目录。
"""

from .paths import CLOTH_PACKAGE_DIR, ORCA_PLAYGROUND_ROOT, ORCA_REPO_ROOT

__all__ = [
    "CLOTH_PACKAGE_DIR",
    "ORCA_PLAYGROUND_ROOT",
    "ORCA_REPO_ROOT",
]
