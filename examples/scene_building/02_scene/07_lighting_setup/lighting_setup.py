"""2.2.2 (7) 光照系统配置（spawn spotlight + set_light_info）。

程序化 spawn 多盏 spotlight 资产，并通过 set_light_info 配置光源颜色/强度，
演示多光源组合照明。env 步进时持续旋转光源 body 并刷新 light info，实现动态光照。

模式：在线（需 OrcaLab + spotlight/cup_of_coffee/office_desk 资产）
资产来源：OrcaLab 资产库 https://simassets.orca3d.cn/
资产包：OrcaPlaygroundAssets + run_light_night（同 examples/replicator）

验证点:
    1. spotlight 资产正确 spawn
    2. publish_scene 触发 MJCF 重建
    3. set_light_info 应用光源颜色/强度
    4. 多光源组合照明（10 盏随机色 spotlight）
    5. env 步进时光源 body 持续旋转 + light info 持续刷新

参见:
    03_示例开发计划.md §2.2.2 (7)
    examples/replicator/run_lights.py（动态光源旋转，本课光照逻辑照搬自此）

模块组织:
    - build_lighting_scene(): spawn 场景（桌子 + 杯子 + N 盏 spotlight）+ set_light_info
    - LightsEnv: 自定义 Euler env，step() 内旋转光源 body + 刷新 light info
"""

from __future__ import annotations

import colorsys
import time
from typing import Optional

import numpy as np
from gymnasium import spaces
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import Actor, LightInfo, OrcaGymScene
from orca_gym.scene.orca_gym_scene_runtime import OrcaGymSceneRuntime
import orca_gym.utils.rotations as rotations

# 资产路径（同 examples/replicator/run_lights.py）
_CUP_PATH = "assets/e071469a36d3c8aa/default_project/prefabs/cup_of_coffee_usda"
_DESK_PATH = "assets/e071469a36d3c8aa/default_project/prefabs/office_desk_7_mb_usda"
_SPOTLIGHT_PATH = "assets/e071469a36d3c8aa/default_project/prefabs/spotlight"

# 光源数量（同 examples/replicator/run_lights.py 的 LIGHT_COUNT）
LIGHT_COUNT: int = 10

# 聚光灯在 prefab 局部坐标中的发射方向（常见为 -Z）
_SPOTLIGHT_LOCAL_AXIS: np.ndarray = np.array([0.0, 0.0, -1.0], dtype=np.float64)
_WORLD_DOWN: np.ndarray = np.array([0.0, 0.0, -1.0], dtype=np.float64)

# spawn 区域范围（同 replicator run_lights.create_scene）
_LIGHT_POS_RANGE: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-1.35, 1.35),  # x
    (-1.35, 1.35),  # y
    (2.15, 3.55),   # z
)
_LIGHT_SCALE_RANGE: tuple[float, float] = (2.2, 3.8)
_LIGHT_INTENSITY_RANGE: tuple[float, float] = (520.0, 980.0)
_LIGHT_MAX_TILT_RAD: float = 0.5  # 光轴相对 -Z 的最大倾斜角（弧度）

# 杯子初始位置范围
_CUP_POS_RANGE: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (0.0, 0.5),
    (0.0, 0.5),
    (1.0, 2.0),
)

_QUAT_IDENTITY: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

_logger = get_orca_logger()


def _log(msg: str) -> None:
    """双路输出（logger + print），确保终端可见。"""
    _logger.info(msg)
    print(msg, flush=True)


def _unit(v: np.ndarray) -> np.ndarray:
    """归一化为单位向量。"""
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


def _random_downward_light_quat(max_tilt_rad: float = _LIGHT_MAX_TILT_RAD) -> np.ndarray:
    """光轴在世界系中主要朝向 -Z，并带小范围倾斜（同 replicator）。"""
    z = _WORLD_DOWN.copy()
    u = np.random.normal(size=3).astype(np.float64)
    u = u - float(np.dot(u, z)) * z
    if float(np.linalg.norm(u)) < 1e-6:
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = _unit(u)
    t = float(np.random.uniform(0.0, max_tilt_rad))
    target = np.cos(t) * z + np.sin(t) * u
    return _quat_align_vectors(_SPOTLIGHT_LOCAL_AXIS, _unit(target))


def _make_actor(
    name: str,
    asset_path: str,
    pos: tuple[float, float, float],
    quat: np.ndarray | tuple[float, float, float, float] = _QUAT_IDENTITY,
    scale: float = 1.0,
) -> Actor:
    """构造 Actor 对象。"""
    return Actor(
        name=name,
        asset_path=asset_path,
        position=np.array(pos, dtype=np.float64),
        rotation=np.array(quat, dtype=np.float64),
        scale=scale,
    )


def _random_light_pos() -> tuple[float, float, float]:
    """随机生成光源位置（在 _LIGHT_POS_RANGE 范围内）。"""
    return (
        float(np.random.uniform(*_LIGHT_POS_RANGE[0])),
        float(np.random.uniform(*_LIGHT_POS_RANGE[1])),
        float(np.random.uniform(*_LIGHT_POS_RANGE[2])),
    )


def _random_cup_pos() -> tuple[float, float, float]:
    """随机生成杯子位置（在 _CUP_POS_RANGE 范围内）。"""
    return (
        float(np.random.uniform(*_CUP_POS_RANGE[0])),
        float(np.random.uniform(*_CUP_POS_RANGE[1])),
        float(np.random.uniform(*_CUP_POS_RANGE[2])),
    )


def _random_light_name(index: int) -> str:
    """生成光源 actor 名（同 replicator 命名规则）。"""
    return f"light_with_random_color_scale_intensity_{index}"


def build_lighting_scene(
    scene: OrcaGymScene,
    light_count: int = LIGHT_COUNT,
) -> list[str]:
    """程序化 spawn 光照场景：桌子 + 杯子 + N 盏 spotlight。

    流程（同 examples/replicator/run_lights.create_scene）：
        1. add_actor: 桌子 + 杯子 + N 盏 spotlight
        2. publish_scene: 触发 Studio MJCF 重建
        3. set_light_info: 为每盏 spotlight 设置随机颜色/强度

    Args:
        scene: OrcaGymScene 实例
        light_count: 光源数量

    Returns:
        光源 actor 名列表（供调用方在 env 中查找 body 和后续 set_light_info）
    """
    # 1. 桌子（居中贴地）
    scene.add_actor(_make_actor("office_desk", _DESK_PATH, (0.0, 0.0, 0.0)))

    # 2. 杯子（随机位置，桌子上方）
    scene.add_actor(_make_actor("original_red_cup", _CUP_PATH, _random_cup_pos()))

    # 3. N 盏 spotlight（随机位置/朝向/缩放）
    light_names: list[str] = []
    for i in range(light_count):
        name = _random_light_name(i)
        light_names.append(name)
        scene.add_actor(
            _make_actor(
                name=name,
                asset_path=_SPOTLIGHT_PATH,
                pos=_random_light_pos(),
                quat=_random_downward_light_quat(),
                scale=float(np.random.uniform(*_LIGHT_SCALE_RANGE)),
            )
        )

    _log(f"构建光照场景：桌子 + 杯子 + {light_count} 盏 spotlight")

    # 4. publish_scene 触发 MJCF 重建（光源才能生效）
    scene.publish_scene()
    _log("  publish_scene 完成，actor 已写入 MJCF")

    # 5. 为每盏 spotlight 设置随机颜色/强度
    for name in light_names:
        light_info = LightInfo(
            color=np.array(
                [
                    float(np.random.uniform(0.0, 1.0)),
                    float(np.random.uniform(0.0, 1.0)),
                    float(np.random.uniform(0.0, 1.0)),
                ]
            ),
            intensity=float(np.random.uniform(*_LIGHT_INTENSITY_RANGE)),
        )
        scene.set_light_info(name, light_info)
    _log(f"  set_light_info 完成，{light_count} 盏光源颜色/强度已应用")

    return light_names


# ── 自定义 Euler env：步进时旋转光源 body + 刷新 light info ──────────────
# 光源动画逻辑照搬自 examples/replicator/lights_env.LightsEnv。
# 每帧 step() 内：
#   1. _rotate_light_bodies_in_batch: set_mocap_pos_and_quat 回写光源 body transform
#      （env.reset() 后 MuJoCo mocap body 从 MJCF 重载，需主动同步到 O3DE）
#   2. _update_light_info_group: set_light_info 分批刷新颜色/强度（避免 RPC 过载）


class LightsEnv(OrcaGymEulerEnv):
    """带光源动画的 Euler 仿真环境。

    step() 内每帧旋转光源 mocap body 并刷新 light info，实现动态光照。
    光源动画逻辑照搬自 examples/replicator/lights_env.LightsEnv。
    """

    metadata = {'render_modes': ['human', 'none'], 'version': '0.0.1', 'render_fps': 60}

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list,
        time_step: float,
        light_count: int = LIGHT_COUNT,
        **kwargs,
    ):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )

        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

        # 光源配置（name + body_name，命名同 replicator）
        # light_count 必须与 build_lighting_scene 实际 spawn 的光源数量一致，
        # 否则 env 会为未 spawn 的光源调用 set_light_info，OrcaLab 报 "Light actor name not found"。
        self._light_config = [
            {
                "name": _random_light_name(i),
                "body_name": f"{_random_light_name(i)}_SpotLight",
            }
            for i in range(light_count)
        ]
        light_count = len(self._light_config)
        self.scene_runtime: Optional[OrcaGymSceneRuntime] = None
        self._light_rotation_delta = np.zeros((light_count, 3), dtype=np.float32)
        self._light_rotation_update_phase = 0
        self._light_hue_phase = np.linspace(0.0, 1.0, light_count, endpoint=False, dtype=np.float32)
        self._light_hue_speed = np.random.uniform(0.12, 0.30, size=light_count).astype(np.float32)
        self._light_value_phase = np.random.uniform(0.0, 2.0 * np.pi, size=light_count).astype(np.float32)
        self._light_intensity_base = np.random.uniform(600.0, 900.0, size=light_count).astype(np.float32)
        self._light_intensity_amplitude = np.random.uniform(200.0, 400.0, size=light_count).astype(np.float32)
        self._missing_light_bodies: set[str] = set()
        self._rotatable_lights: list[dict] = []
        self._rotation_bodies_resolved = False
        # 分批刷新 light info，避免每帧全量 RPC 导致过载
        self._light_info_group_count = max(4, light_count // 2)
        self._light_info_group_phase = 0
        self._animation_start_time = time.perf_counter()
        _logger.info(f"Initialized {light_count} animated lights.")

        self._set_obs_space()
        self._set_action_space()

    def _set_obs_space(self) -> None:
        self.observation_space = self.generate_observation_space(self._get_obs().copy())

    def _set_action_space(self) -> None:
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(max(1, self.nu),),
            dtype=np.float32,
        )

    def render_callback(self, mode: str = 'human') -> None:
        if mode == "human":
            self.render()
        else:
            raise ValueError("Invalid render mode")

    def step(self, action) -> tuple:
        ctrl = np.zeros(self.nu, dtype=np.float32)
        self.do_simulation(ctrl, self.frame_skip)
        obs = self._get_obs().copy()

        info: dict = {}
        terminated = False
        truncated = False
        reward = 0.0

        # 旋转光源 body + 刷新 light info（动态光照核心）
        self._rotate_lights()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict:
        obs = {
            "joint_pos": self.data.qpos[:self.nq].copy(),
            "joint_vel": self.data.qvel[:self.nv].copy(),
            "joint_acc": self.data.qacc[:self.nv].copy(),
        }
        return obs

    def reset_model(self) -> tuple[dict, dict]:
        self.ctrl = np.zeros(self.nu, dtype=np.float32)
        obs = self._get_obs().copy()
        return obs, {}

    def get_observation(self, obs: Optional[dict] = None) -> dict:
        if obs is not None:
            return obs
        else:
            return self._get_obs().copy()

    def set_scene_runtime(self, scene_runtime: OrcaGymSceneRuntime) -> None:
        self.scene_runtime = scene_runtime
        _logger.performance("Scene runtime is set.")

    def _timer_now(self) -> float:
        return time.perf_counter() - self._animation_start_time

    def _rotate_lights(self) -> None:
        """旋转 spotlight body 并刷新 light info（同 replicator.LightsEnv._rotate_lights）。"""
        sim_time = self._timer_now()
        if not self._rotation_bodies_resolved:
            self._resolve_rotatable_light_bodies()

        if self._light_rotation_update_phase == 0:
            self._light_rotation_delta = np.random.uniform(
                low=-np.pi * 0.01,
                high=np.pi * 0.01,
                size=self._light_rotation_delta.shape,
            ).astype(np.float32)

        self._rotate_light_bodies_in_batch()
        self._update_light_info_group(sim_time)

        self._light_rotation_update_phase = (self._light_rotation_update_phase + 1) % 180

    def _resolve_rotatable_light_bodies(self) -> None:
        """查找场景中存在的光源 body，缺失的降级为仅刷新 light info。"""
        self._rotation_bodies_resolved = True
        self._rotatable_lights = []
        for index, light in enumerate(self._light_config):
            body_name = light["body_name"]
            try:
                self.get_body_xpos_xmat_xquat([body_name])
                self._rotatable_lights.append({"index": index, "body_name": body_name})
            except Exception:
                self._missing_light_bodies.add(body_name)
                _logger.warning(
                    f"Light body `{body_name}` not found; falling back to color/intensity animation only."
                )

    def _rotate_light_bodies_in_batch(self) -> None:
        """每帧通过 set_mocap_pos_and_quat 回写光源 body transform 到 Studio。

        env.reset() 后 MuJoCo mocap body 从 MJCF 重载 transform，但 O3DE 光源实体的
        transform 可能未同步，导致光源位置/朝向错误。每帧回写确保同步。
        """
        if not self._rotatable_lights:
            return

        body_names = [light["body_name"] for light in self._rotatable_lights]
        # Euler 体系 get_body_xpos_xmat_xquat 返回 dict[body_name -> {"xpos","xmat","xquat"}]
        body_poses = self.get_body_xpos_xmat_xquat(body_names)
        mocap_updates = {}
        for light in self._rotatable_lights:
            light_index = light["index"]
            body_name = light["body_name"]
            rotation_delta = rotations.euler2quat(self._light_rotation_delta[light_index])
            pose = body_poses[body_name]
            mocap_updates[body_name] = {
                "pos": np.asarray(pose["xpos"], dtype=np.float64),
                "quat": rotations.quat_mul(
                    np.asarray(pose["xquat"], dtype=np.float64), rotation_delta
                ),
            }

        self.set_mocap_pos_and_quat(mocap_updates)

    def _update_light_info_group(self, sim_time: float) -> None:
        """分批刷新 light info（颜色/强度），避免每帧全量 RPC 过载。"""
        if self.scene_runtime is None:
            return

        for i, light in enumerate(self._light_config):
            if i % self._light_info_group_count != self._light_info_group_phase:
                continue
            self.scene_runtime.set_light_info(
                light["name"],
                self._build_dynamic_light_info(i, sim_time),
            )
        self._light_info_group_phase = (self._light_info_group_phase + 1) % self._light_info_group_count

    def _build_dynamic_light_info(self, light_index: int, sim_time: float) -> LightInfo:
        """根据仿真时间生成动态颜色/强度的 LightInfo。"""
        hue = (self._light_hue_phase[light_index] + sim_time * self._light_hue_speed[light_index]) % 1.0
        value = 0.75 + 0.25 * np.sin(sim_time * 3.4 + self._light_value_phase[light_index])
        color = np.array(
            colorsys.hsv_to_rgb(float(hue), 0.85, float(np.clip(value, 0.45, 1.0))),
            dtype=np.float32,
        )
        intensity_wave = 0.5 + 0.5 * np.sin(sim_time * 4.6 + self._light_value_phase[light_index])
        intensity = self._light_intensity_base[light_index] + self._light_intensity_amplitude[light_index] * intensity_wave
        return LightInfo(color=color, intensity=float(intensity))

