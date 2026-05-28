import shutil
import uuid
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

    height_map_dir = os.path.join(os.path.expanduser("~"), ".orcagym", "height_map")
    height_map_file = os.path.join(height_map_dir, f"height_map_{uuid.uuid4()}.npy")
    os.makedirs(height_map_dir, exist_ok=True)

    output_file = os.path.join(height_map_dir, f"height_map_tmp_{uuid.uuid4()}.npy")

    ret = os.system(
        f"python -m orca_gym.tools.terrains.height_map_generater"
        f" --orcagym_addresses {orcagym_addresses[0]}"
        f" --output_file {output_file}"
    )

    if ret != 0:
        _logger.warning(
            f"height_map_generater exited with code {ret}, "
            f"using empty height map file"
        )
        np.save(height_map_file, np.zeros((1, 1), dtype=float))
        return height_map_file

    if os.path.exists(output_file):
        shutil.move(output_file, height_map_file)
    else:
        _logger.warning("height_map_generater did not produce output file, using empty height map")
        np.save(height_map_file, np.zeros((1, 1), dtype=float))

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

    sqrt_width = int(np.ceil(np.sqrt(agent_num)))
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
