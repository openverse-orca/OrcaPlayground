from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor, LightInfo
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
import numpy as np
import orca_gym.utils.rotations as rotations
import sys
import os
# 添加项目根目录到路径，以支持直接运行脚本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from examples.replicator import run_simulation as sim

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()

LIGHT_COUNT = 10

# 聚光灯在 prefab 局部坐标中的发射方向（常见为 -Z）；若运行后光束明显朝上/横飘，可改为 [0,0,1] 或 [0,-1,0]。
_SPOTLIGHT_LOCAL_AXIS = np.array([0.0, 0.0, -1.0], dtype=np.float64)
_WORLD_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _quat_align_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """返回将 v_from 旋转到 v_to 的单位四元数 (wxyz)。"""
    vf = _unit(np.asarray(v_from, dtype=np.float64))
    vt = _unit(np.asarray(v_to, dtype=np.float64))
    dot = float(np.clip(np.dot(vf, vt), -1.0, 1.0))
    if dot > 1.0 - 1e-7:
        return rotations.quat_identity().astype(np.float32)
    if dot < -1.0 + 1e-7:
        ortho = np.cross(vf, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(ortho)) < 1e-6:
            ortho = np.cross(vf, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        ortho = _unit(ortho)
        return np.array([0.0, ortho[0], ortho[1], ortho[2]], dtype=np.float32)
    cross = np.cross(vf, vt)
    w = 1.0 + dot
    q = np.array([w, cross[0], cross[1], cross[2]], dtype=np.float64)
    return _unit(q).astype(np.float32)


def _random_downward_light_quat(max_tilt_rad: float = 0.55) -> np.ndarray:
    """光轴在世界系中主要朝向 -Z，并带小范围倾斜，光斑更接近圆盘而非细长条。"""
    z = _WORLD_DOWN.copy()
    u = np.random.normal(size=3).astype(np.float64)
    u = u - float(np.dot(u, z)) * z
    if float(np.linalg.norm(u)) < 1e-6:
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = _unit(u)
    t = float(np.random.uniform(0.0, max_tilt_rad))
    target = np.cos(t) * z + np.sin(t) * u
    return _quat_align_vectors(_SPOTLIGHT_LOCAL_AXIS, _unit(target))


def create_scene() -> OrcaGymScene:
    """
    构建并发布含多盏随机灯的 Replicator 场景。

    初始 ``LightInfo`` 仅决定开局状态；运行 ``run_simulation(..., scene_runtime=...)``
    后，``examples.replicator.lights_env.LightsEnv`` 会在仿真中持续更新灯光。
    """
    grpc_addr = "localhost:50051"
    scene = OrcaGymScene(grpc_addr)

    actor = Actor(
        name=f"original_red_cup",
        asset_path="assets/e071469a36d3c8aa/default_project/prefabs/cup_of_coffee_usda",
        #asset_path="assets/prefabs/cup_of_coffee_usda",
        position=np.array([np.random.uniform(0.0, 0.5), 
                           np.random.uniform(0.0, 0.5), 
                           np.random.uniform(1.0, 2.0)]),
        rotation=rotations.euler2quat(np.array([np.random.uniform(-np.pi, np.pi), 
                                                np.random.uniform(-np.pi, np.pi), 
                                                np.random.uniform(-np.pi, np.pi)])),
        scale=1.0,
    )
    scene.add_actor(actor)

    actor = Actor(
        name="office_desk",
        asset_path="assets/e071469a36d3c8aa/default_project/prefabs/office_desk_7_mb_usda",
        #asset_path="assets/prefabs/office_desk_7_mb_usda",
        position=np.array([0, 0, 0.0]),
        rotation=rotations.euler2quat(np.array([0.0, 0.0, 0])),
        scale=1.0,
    )
    scene.add_actor(actor)

    for i in range(LIGHT_COUNT):
        actor = Actor(
            name=f"light_with_random_color_scale_intensity_{i}",
            asset_path="assets/e071469a36d3c8aa/default_project/prefabs/spotlight",
            #asset_path="assets/prefabs/spotlight",
            # 布在桌面上方、覆盖办公桌附近，避免过低/过高导致光斑过小或难落在桌面上。
            position=np.array(
                [
                    np.random.uniform(-1.35, 1.35),
                    np.random.uniform(-1.35, 1.35),
                    np.random.uniform(2.15, 3.55),
                ]
            ),
            rotation=_random_downward_light_quat(max_tilt_rad=0.5),
            scale=float(np.random.uniform(2.2, 3.8)),
        )
        scene.add_actor(actor)

    scene.publish_scene()

    # scene.make_camera_viewport_active("default_camera", "CameraViewport")

    for i in range(LIGHT_COUNT):
        light_info = LightInfo(
            color=np.array([np.random.uniform(0.0, 1.0),
                            np.random.uniform(0.0, 1.0),
                            np.random.uniform(0.0, 1.0)]),
            intensity=float(np.random.uniform(520.0, 980.0)),
        )
        scene.set_light_info(f"light_with_random_color_scale_intensity_{i}", light_info)

    _logger.info("Replicator scene published successfully.")

    return scene


def destroy_scene(scene: OrcaGymScene):
    scene.publish_scene()
    scene.close()
    _logger.info("Replicator scene closed successfully.")


if __name__ == "__main__":
    scene = create_scene()

    orcagym_addr = "localhost:50051"
    agent_name = "NoRobot"
    env_name = "Lights"

    scene_runtime = OrcaGymSceneRuntime(scene)

    sim.run_simulation(orcagym_addr, agent_name, env_name, scene_runtime)


    destroy_scene(scene)














