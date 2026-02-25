"""
Fluid-MuJoCo 耦合仿真环境

该模块通过 OrcaLink 实现 SPH 流体与 MuJoCo 刚体的双向耦合。
"""

from .orcalink_bridge import OrcaLinkBridge, RigidBodyConfig, ConnectionPoint
from .sim_env import SimEnv as FluidSimEnv
from .scene_generator import SceneGenerator, generate_scene_from_env
from .config_generator import ConfigGenerator
from .utils import run_simulation_with_config

__all__ = [
    # 核心类
    "OrcaLinkBridge",
    "RigidBodyConfig",
    "ConnectionPoint",
    "FluidSimEnv",
    # 生成器
    "SceneGenerator",
    "ConfigGenerator",
    "generate_scene_from_env",
    # 工具函数
    "run_simulation_with_config",
]

