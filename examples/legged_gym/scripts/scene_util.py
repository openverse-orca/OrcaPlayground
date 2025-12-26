import shutil
import uuid
import subprocess
import sys
from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor
from orca_gym.utils.rotations import euler2quat
import os
import time
import numpy as np

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


def generate_height_map_file(
    orcagym_addresses: list[str],
):
    _logger.info("=============> Generate height map file ...")

    # 创建临时输出目录
    height_map_dir = os.path.join(os.path.expanduser("~"), ".orcagym", "height_map")
    os.makedirs(height_map_dir, exist_ok=True)
    
    # 生成临时输出文件名
    temp_output_file = os.path.join(height_map_dir, f"height_map_temp_{uuid.uuid4()}.npy")
    
    # 使用模块方式调用，并指定输出文件路径
    # 使用 sys.executable 确保使用当前 Python 解释器
    cmd = [
        sys.executable,
        "-m", "orca_gym.tools.terrains.height_map_generater",
        "--orcagym_addresses", orcagym_addresses[0],
        "--output_file", temp_output_file
    ]
    _logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd()  # 确保在正确的工作目录运行
        )
        _logger.info(f"Command output: {result.stdout}")
        if result.stderr:
            _logger.warning(f"Command stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        _logger.error(f"Command failed with exit code {e.returncode}")
        _logger.error(f"stdout: {e.stdout}")
        _logger.error(f"stderr: {e.stderr}")
        raise RuntimeError(f"Failed to generate height map file. Command exited with code {e.returncode}")
    
    if not os.path.exists(temp_output_file):
        raise FileNotFoundError(f"Height map file was not generated: {temp_output_file}")

    # 用UUID生成一个唯一的最终文件名
    height_map_file = os.path.join(height_map_dir, f"height_map_{uuid.uuid4()}.npy")
    shutil.move(temp_output_file, height_map_file)

    _logger.info(f"=============> Generate height map file done. Height map file:  {height_map_file}")

    return height_map_file

def clear_scene(
    orcagym_addresses: list[str],
):
    _logger.info("=============> Clear scene ...")

    scene = OrcaGymScene(orcagym_addresses[0])
    scene.publish_scene()
    time.sleep(1)
    scene.close()
    time.sleep(1)

    _logger.info("=============> Clear scene done.")


def publish_terrain(
    orcagym_addresses: list[str],
    terrain_asset_paths: list[str],
):
    _logger.info("=============> Publish terrain ...")
    scene = OrcaGymScene(orcagym_addresses[0])
    for terrain_asset_path in terrain_asset_paths:
        terrain = Actor(
            name=f"{terrain_asset_path}",
            asset_path=terrain_asset_path,
            position=[0, 0, 0],
            rotation=euler2quat([0, 0, 0]),
            scale=1.0,
        )
        scene.add_actor(terrain)
        _logger.info(f"    =============> Add terrain {terrain_asset_path} ...")
        time.sleep(0.01)

    scene.publish_scene()
    time.sleep(3)
    scene.close()
    time.sleep(1)

    _logger.info("=============> Publish terrain done.")

def publish_scene(
    orcagym_addresses: list[str],
    agent_name: str,
    agent_asset_path: str,
    agent_num: int,
    terrain_asset_paths: list[str],
):
    _logger.info("=============> Publish scene ...")
    scene = OrcaGymScene(orcagym_addresses[0])
    # 排列成一个方阵，每个机器人间隔0.5米
    sqrt_width = int(np.ceil(np.sqrt(agent_num)))  # 向上取整
    base_offset_x = -(sqrt_width) / 2
    base_offset_y = -(sqrt_width) / 2
    for i in range(agent_num):
        x_pos = (i % sqrt_width) * 0.5 + base_offset_x
        y_pos = (i // sqrt_width) * 0.5 + base_offset_y
        actor = Actor(
            name=f"{agent_name}_{i:03d}",
            asset_path=agent_asset_path,
            position=[x_pos, y_pos, 0],
            rotation=euler2quat([0, 0, 0]),
            scale=1.0,
        )
        scene.add_actor(actor)
        _logger.info(f"    =============> Add agent {agent_name}_{i:03d} ...")
        time.sleep(0.01)

    _logger.info("=============> Publish terrain ...")
    scene = OrcaGymScene(orcagym_addresses[0])
    for terrain_asset_path in terrain_asset_paths:
        terrain = Actor(
            name=f"{terrain_asset_path}",
            asset_path=terrain_asset_path,
            position=[0, 0, 0],
            rotation=euler2quat([0, 0, 0]),
            scale=1.0,
        )
        scene.add_actor(terrain)
        _logger.info(f"    =============> Add terrain {terrain_asset_path} ...")
        time.sleep(0.01)


    scene.publish_scene()
    time.sleep(3)
    scene.close()
    time.sleep(1)

    _logger.info("=============> Publish scene done.")
