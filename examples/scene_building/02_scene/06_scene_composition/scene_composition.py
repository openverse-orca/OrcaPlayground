"""2.2.2 (6) 机器人厨房助手（g1_pick + kitchen_Night_2 多源合并）。

叙事性 spawn：机器人厨师在厨房台面前备餐，演示场景包 + 厨具 + 机器人三源共存。
厨房场景与厨具由用户在 OrcaLab 手动加载 kitchen_Night_2 关卡，本脚本只 spawn
机器人 g1_pick 到指定工位（做菜/清理），通过命名空间前缀 robot_ 体现多源隔离。

模式：在线（需 OrcaLab + kitchen_Night_2 资产 + g1_pick spawnable）
资产来源：OrcaLab 资产库 https://simassets.orca3d.cn/

验证点:
    1. g1_pick 机器人正确 spawn 到厨房场景
    2. 机器人与厨房/厨具共存（三源合并）
    3. 两种状态（做菜/清理）机器人工位不同
    4. env 步进时机器人保持站立（物理稳定）

参见:
    03_示例开发计划.md §2.2.2 (6)
    第 8 课 run_random_variation.py（厨房坐标系来源）
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger
from orca_gym.scene.orca_gym_scene import Actor, OrcaGymScene

# g1_pick 机器人 spawnable 路径（816f95ce16021282 资产库）
_ROBOT_PATH = "assets/816f95ce16021282/default_project/prefabs/g1_pick_usda"

# 机器人 actor 名（命名空间前缀 robot_ 体现多源隔离）
ROBOT_ACTOR_NAME = "robot_chef"

# 两种状态的机器人工位配置
# 坐标系来自第 8 课 out.xml 分析：
#   主台面 x∈[-5.239, -4.499], y∈[-5.261, 0.339], z_top≈0.985
#   灶台火眼（锅）y≈-2.4 和 y≈-1.8
#   机器人贴着台面站立（x≈-4.6，距台面边缘 -4.499 约 0.1m），pos.z=0 脚踩地面
#   朝向：台面在 -x 方向，机器人需绕 z 轴 180° 面向台面
#   quat=(0,0,0,1) = 绕 z 轴 180°（w=cos(π/2), z=sin(π/2)）
#
# 洗菜池说明：MuJoCo 模型中无独立洗菜池碰撞体（见第 8 课注释），
# 但视觉模型存在于台面 y≈0 端，cleaning 状态机器人贴近该区域（y=0.2）。
STATE_CONFIGS: dict[str, dict[str, Any]] = {
    "cooking": {
        "desc": "做菜状态：机器人在灶台前备菜",
        "pos": (-4.4, -2.0, 0.0),
        "quat": (0.0, 0.0, 0.0, 1.0),
    },
    "cleaning": {
        "desc": "清理状态：机器人在洗菜池前清理",
        "pos": (-4.4, 0.2, 0.0),
        "quat": (0.0, 0.0, 0.0, 1.0),
    },
}

DEFAULT_STATE: str = "cooking"

# spawn 间隔（同 02/03 课）
SPAWN_INTERVAL: float = 1.0

_logger = get_orca_logger()


class RobotChefEnv(OrcaGymEulerEnv):
    """机器人厨师 env：仅步进物理 + render，不跑 RL。

    机器人 g1_pick spawn 后由 Studio 转为 MJCF body，env 创建时自动拉取。
    step() 内调用 do_simulation 推进 MuJoCo，render() 推送状态到 O3DE。
    """

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,
        **kwargs,
    ) -> None:
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )
        self.nu = self.model.nu

    def reset_model(self) -> tuple[dict, dict]:
        self.ctrl = np.zeros(self.nu, dtype=np.float32)
        self.mj_forward()
        return {"qpos": self.data.qpos.copy()}, {}

    def step(self, action) -> tuple:
        # 必须调用 do_simulation 推进 MuJoCo，否则 O3DE 渲染端收不到状态更新
        self.do_simulation(action, self.frame_skip)
        return self._get_obs(), 0.0, False, False, {}

    def _get_obs(self) -> dict:
        return {"qpos": self.data.qpos.copy()}


def _make_actor(
    name: str,
    asset_path: str,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
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


def build_robot_chef_scene(
    scene: OrcaGymScene,
    state: str = DEFAULT_STATE,
    interval: float = SPAWN_INTERVAL,
) -> str:
    """spawn 机器人 g1_pick 到指定工位。

    厨房场景与厨具由用户在 OrcaLab 手动加载 kitchen_Night_2 关卡，
    本函数只 spawn 机器人，通过 append_scene 增量发布（不销毁已有 actor）。

    Args:
        scene: OrcaGymScene 实例
        state: 工位状态，"cooking"（灶台前）或 "cleaning"（洗菜池前）
        interval: spawn 间隔（秒，仅日志提示用）

    Returns:
        机器人 actor name（ROBOT_ACTOR_NAME）

    Raises:
        ValueError: state 不在 STATE_CONFIGS 中
    """
    if state not in STATE_CONFIGS:
        raise ValueError(
            f"未知状态 {state!r}，可选: {list(STATE_CONFIGS.keys())}"
        )

    cfg = STATE_CONFIGS[state]
    _logger.info(f"构建机器人厨房助手场景：{state}（{cfg['desc']}）")
    _logger.info(
        f"  机器人 {ROBOT_ACTOR_NAME} @ ({cfg['pos'][0]:.2f}, {cfg['pos'][1]:.2f}, {cfg['pos'][2]:.2f})"
    )

    use_append = hasattr(scene, "append_scene")
    if not use_append:
        _logger.warning(
            "OrcaGymScene.append_scene 不存在，降级为 publish_scene"
            "（前序 actor 会被销毁，厨房场景需重新加载）。"
        )

    actor = _make_actor(
        name=ROBOT_ACTOR_NAME,
        asset_path=_ROBOT_PATH,
        pos=cfg["pos"],
        quat=cfg["quat"],
    )
    scene.add_actor(actor)
    if use_append:
        scene.append_scene()
    else:
        scene.publish_scene()
    _logger.info(f"  已 spawn: {ROBOT_ACTOR_NAME}")

    return ROBOT_ACTOR_NAME


# ── cleaning 状态：碗杯精准摆放 + 放倒 ──

# 洗菜池在台面 y≈0 端
# 台面 x∈[-5.239, -4.499], y∈[-5.261, 0.339], z_top≈0.985
# 机器人 cleaning 工位 (-4.4, 0.2, 0.0)，面向 -x（台面）
#
# 处理策略：
#   1. 粉色咖啡杯（Coffecup）→ 池子外侧台面站立（放倒会穿模）
#   2. 其他碗杯（Glass/Porcelain）→ 池子外侧台面上放倒（绕 x 轴 90°），防止摔落
#   3. 其他物体保持原位
#   4. 池子视觉区域内不放任何东西，防止物理穿帮

# 粉色咖啡杯精准放置（站立，quat 保持默认 (1,0,0,0)，放倒会穿模）
# 池子外侧：x=-4.6（靠机器人侧台面边缘，避开池子视觉区域）
STAND_PLACEMENTS: list[dict[str, Any]] = [
    {"keyword": "Coffecup", "pos": (-4.6, 0.2, 0.985), "label": "粉色咖啡杯→池子外侧(站立)"},
]

# 放倒 quat：绕 x 轴 90° = (w=cos45°, x=sin45°, 0, 0) = (0.7071, 0.7071, 0, 0)
TIP_OVER_QUAT: tuple[float, float, float, float] = (0.7071, 0.7071, 0.0, 0.0)

# 需要放倒的碗杯（Glass/Porcelain）→ 池子外侧台面上放倒
TIP_OVER_PLACEMENTS: list[dict[str, Any]] = [
    {"keyword": "Glass", "pos": (-4.8, 0.2, 0.985), "label": "透明杯→放倒"},
    {"keyword": "Porcelain", "pos": (-4.6, -0.2, 0.985), "label": "瓷杯→放倒"},
]

# 不动的物体（同第 8 课 FIXED_OBJECTS，锅盖+锅保持原位）
FIXED_OBJECTS: list[str] = ["Pot_02_b", "Pot_02_a"]

# 内置 body（spawn 前就存在，操作 freejoint 时排除）
_BUILTIN_BODIES: frozenset[str] = frozenset({
    "world",
    "ActorManipulator_Anchor",
    "ActorManipulator_dummy",
})


def place_utensils_in_sink(env: OrcaGymEulerEnv) -> dict[str, dict[str, Any]]:
    """将碗杯精准摆放到洗菜池外侧台面，粉色杯站立其余放倒。

    处理策略：
      1. STAND_PLACEMENTS（Porcelain）→ 池子外侧台面站立
      2. TIP_OVER_PLACEMENTS（Glass/Coffecup）→ 池子外侧台面放倒（绕 x 轴 90°）
      3. 其他物体保持原位

    Args:
        env: 已初始化的 RobotChefEnv 实例

    Returns:
        操作报告 {joint_name: {body_name, action, new_pos}}
    """
    qpos = env.data.qpos.copy()
    qvel = env.data.qvel.copy()

    joint_dict = env.model.get_joint_dict()
    report: dict[str, dict[str, Any]] = {}

    used_stand: set[str] = set()
    used_tipover: set[str] = set()

    for jname, info in joint_dict.items():
        if int(info["Type"]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue

        body_id = int(info["BodyID"])
        body_name = env.model.body_id2name(body_id)
        if body_name in _BUILTIN_BODIES:
            continue

        adr = env.jnt_qposadr(jname)
        old_pos = qpos[adr : adr + 3].copy()

        # 固定物体保持原位
        if any(kw in jname or kw in body_name for kw in FIXED_OBJECTS):
            report[jname] = {"body_name": body_name, "action": "固定", "new_pos": old_pos}
            continue

        # 1. 优先匹配站立放置（粉色瓷杯，放倒会穿模）
        new_pos = None
        label = ""
        tip_over = False
        for placement in STAND_PLACEMENTS:
            if placement["keyword"] in body_name and placement["keyword"] not in used_stand:
                new_pos = placement["pos"]
                label = placement["label"]
                used_stand.add(placement["keyword"])
                break

        # 2. 匹配放倒放置（透明杯/咖啡杯）
        if new_pos is None:
            for placement in TIP_OVER_PLACEMENTS:
                if placement["keyword"] in body_name and placement["keyword"] not in used_tipover:
                    new_pos = placement["pos"]
                    label = placement["label"]
                    used_tipover.add(placement["keyword"])
                    tip_over = True
                    break

        # 3. 未匹配 → 保持原位
        if new_pos is None:
            report[jname] = {"body_name": body_name, "action": "不动", "new_pos": old_pos}
            continue

        # 写入位置
        new_pos_arr = np.array(new_pos, dtype=np.float64)
        qpos[adr : adr + 3] = new_pos_arr

        # 放倒物体：修改 quat（绕 x 轴 90°）
        if tip_over:
            qpos[adr + 3 : adr + 7] = np.array(TIP_OVER_QUAT, dtype=np.float64)

        # 清零速度（freejoint 可能无 dofadr，记录后跳过）
        try:
            dof_adr = env.jnt_dofadr(jname)
            qvel[dof_adr : dof_adr + 6] = 0.0
        except Exception as exc:
            _logger.debug(f"跳过 {jname} 速度清零（无 dofadr）: {exc}")

        report[jname] = {"body_name": body_name, "action": label, "new_pos": new_pos_arr}

    env.set_joint_qpos(qpos)
    env.set_joint_qvel(qvel)
    env.mj_forward()

    return report
