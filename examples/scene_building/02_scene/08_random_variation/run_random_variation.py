"""2.2.2 (8) 入口脚本：厨房场景区域随机变体（freejoint 位置随机化）。

流程:
    1. 连接已运行中的厨房场景（OrcaLab 运行模式，MuJoCo 已初始化）
    2. 收集场景内所有 freejoint 物体（杯子、盘子、锅等）
    3. 按预定义区域（A-盘架区 / B-灶台右侧）随机放置物体
    4. 固定锅盖 + 对应锅（Pot_02_a/b），隐藏篮子（Basket_Kitchen）
    5. env.step 循环步进物理，env.render 推送视口

用法:
    python examples/scene_building/02_scene/08_random_variation/run_random_variation.py
    python examples/scene_building/02_scene/08_random_variation/run_random_variation.py --seed 42 --steps 300

详见: 08_random_variation.md（前置条件、验证点、资产订阅 kitchen_Night_2）
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import mujoco
import numpy as np

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.log.orca_log import get_orca_logger

_logger = get_orca_logger()

# 需订阅的资产包名（运行前提醒用户检查）
_REQUIRED_ASSET_PACK: str = "kitchen_Night_2"


class KitchenProbeEnv(OrcaGymEulerEnv):
    """最简探针 env：仅用于读取/写入场景 freejoint，不跑 RL。"""

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
        # 必须调用 do_simulation 推进 MuJoCo 物理引擎，
        # 否则 O3DE 渲染端收不到状态更新
        self.do_simulation(action, self.frame_skip)
        return self._get_obs(), 0.0, False, False, {}

    def _get_obs(self) -> dict:
        return {"qpos": self.data.qpos.copy()}


# 两个有效放置区域
# 台面碰撞体（来自 out.xml 静态几何）：
#   主台面板 pos=(-4.869, -2.461, 0.961) size=(0.37, 2.8, 0.024)
#   → x∈[-5.239, -4.499], y∈[-5.261, 0.339], z_top≈0.985
#   灶台火眼 y≈-2.4 和 y≈-1.8（固定锅占据）
# 注：MuJoCo 模型中不存在独立的"中岛台"/"洗菜池"碰撞体，
#     只有主台面一条长台面可用，故沿 y 轴划分两个区域。
ZONES = [
    {"name": "A-盘架区", "x": (-5.2, -4.55), "y": (-3.7, -2.7), "desc": "主台面左侧（盘架、杯具原区）"},
    {"name": "B-灶台右侧", "x": (-5.2, -4.55), "y": (-1.5, 0.0), "desc": "主台面右侧（灶台右方开放区）"},
]

# 台面顶部 z（所有物体统一放置到此高度上方，避免用 old_pos[2] 导致穿模）
COUNTER_TOP_Z = 0.985

# 固定不动的物体
FIXED_OBJECTS = ["Pot_02_b", "Pot_02_a"]  # 锅盖 + 对应锅

# 远处隐藏的物体
HIDDEN_KEYWORDS = ["Basket_Kitchen"]
HIDDEN_POS = np.array([0.0, 0.0, -1000.0])


def _place_objects(
    env: OrcaGymEulerEnv,
    free_joints: dict[str, int],
    rng: np.random.Generator,
) -> dict[str, dict[str, Any]]:
    """将每个 freejoint 放置到预定义区域内。

    策略：
      - FIXED_OBJECTS 保持原位
      - HIDDEN_KEYWORDS 移到远处隐藏
      - 其他物体随机分配到 ZONES 内的某个位置
      - z 坐标统一使用 COUNTER_TOP_Z + 微小抬升（避免穿模）
    """
    qpos = env.data.qpos.copy()
    qvel = env.data.qvel.copy()

    joint_dict = env.model.get_joint_dict()
    report: dict[str, dict[str, Any]] = {}

    for jname, adr in free_joints.items():
        body_id = int(joint_dict[jname]["BodyID"])
        body_name = env.model.body_id2name(body_id)

        old_pos = qpos[adr : adr + 3].copy()

        # 检查是否固定不动
        if any(kw in jname or kw in body_name for kw in FIXED_OBJECTS):
            report[jname] = {
                "body_name": body_name,
                "action": "固定",
                "new_pos": old_pos.copy(),
            }
            continue

        # 检查是否隐藏
        if any(kw in jname or kw in body_name for kw in HIDDEN_KEYWORDS):
            qpos[adr : adr + 3] = HIDDEN_POS
            report[jname] = {
                "body_name": body_name,
                "action": "隐藏",
                "new_pos": HIDDEN_POS.copy(),
            }
            continue

        # 随机选择一个区域
        zone = rng.choice(ZONES)
        x_min, x_max = zone["x"]
        y_min, y_max = zone["y"]

        new_x = float(rng.uniform(x_min, x_max))
        new_y = float(rng.uniform(y_min, y_max))
        # 统一用台面顶部高度 + 微小抬升，避免 old_pos[2]（可能在抽屉上）导致穿模
        new_z = COUNTER_TOP_Z + float(rng.uniform(0.02, 0.05))
        new_pos = np.array([new_x, new_y, new_z])

        # 写回 qpos
        qpos[adr : adr + 3] = new_pos

        # 清零速度
        try:
            dof_adr = env.jnt_dofadr(jname)
            qvel[dof_adr : dof_adr + 6] = 0.0
        except Exception:
            pass

        report[jname] = {
            "body_name": body_name,
            "action": f"→{zone['name']}",
            "new_pos": new_pos,
        }

    # 写入 MuJoCo
    env.set_joint_qpos(qpos)
    env.set_joint_qvel(qvel)
    env.mj_forward()

    return report


def _log(msg: str) -> None:
    """双路输出（logger + print），确保终端可见。"""
    _logger.info(msg)
    print(msg, flush=True)


def _check_asset_pack() -> None:
    """提醒用户检查所需资产包是否已订阅。"""
    _log(f"[资产检查] 本示例需在 OrcaStudio 资产库中订阅: {_REQUIRED_ASSET_PACK}")
    _log(f"  若未订阅，请在 OrcaStudio 资产库搜索 {_REQUIRED_ASSET_PACK} 并点击订阅")


def main() -> None:
    parser = argparse.ArgumentParser(description="厨房场景区域随机变体")
    parser.add_argument("--addr", type=str, default="localhost:50051", help="OrcaStudio/OrcaLab gRPC 地址")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，默认 None 表示每次真随机")
    parser.add_argument("--steps", type=int, default=500, help="扰动后仿真步数")
    args = parser.parse_args()

    _check_asset_pack()

    _log(f"连接厨房场景 @ {args.addr}（seed={args.seed}）")
    env = KitchenProbeEnv(
        frame_skip=1,
        orcagym_addr=args.addr,
        agent_names=["agent"],
        time_step=0.002,
    )

    try:
        obs, info = env.reset()
        _log("  env.reset() 完成\n")

        # 收集 freejoint
        joint_dict = env.model.get_joint_dict()
        free_joints: dict[str, int] = {}
        hidden_count = 0

        for name, info in joint_dict.items():
            if int(info["Type"]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            body_id = int(info["BodyID"])
            body_name = env.model.body_id2name(body_id)
            if "ActorManipulator" in body_name:
                continue

            adr = env.jnt_qposadr(name)

            # 提前隐藏
            if any(kw in name or kw in body_name for kw in HIDDEN_KEYWORDS):
                qpos = env.data.qpos.copy()
                qpos[adr : adr + 3] = HIDDEN_POS
                env.set_joint_qpos(qpos)
                env.mj_forward()
                hidden_count += 1
                _log(f"  已隐藏: {body_name}")
                continue

            free_joints[name] = adr

        _log(f"\n发现 {len(free_joints)} 个可移动物体（已隐藏 {hidden_count} 个）\n")

        _log("原始位置：")
        for jname, adr in sorted(free_joints.items()):
            body_id = int(joint_dict[jname]["BodyID"])
            body_name = env.model.body_id2name(body_id)
            pos = env.data.qpos[adr : adr + 3]
            fixed = " [固定]" if any(kw in jname for kw in FIXED_OBJECTS) else ""
            _log(f"  {body_name:50s} pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}){fixed}")

        # 执行区域放置
        rng = np.random.default_rng(args.seed)
        report = _place_objects(env, free_joints, rng)

        _log(f"\n放置完成（{len(report)} 个物体）：")
        for jname, r in report.items():
            if r["action"] == "固定":
                _log(f"  {r['body_name']:50s} [保持原位]")
            elif r["action"] == "隐藏":
                _log(f"  {r['body_name']:50s} [已隐藏]")
            else:
                np_ = r["new_pos"]
                _log(f"  {r['body_name']:50s} {r['action']:6s} → ({np_[0]:+.3f}, {np_[1]:+.3f}, {np_[2]:+.3f})")

        # 正常仿真循环（不干预，让物理引擎自由推进）
        # 关键：每步必须调用 env.render() 才能把状态推送到 O3DE 渲染端
        if args.steps > 0:
            _log(f"\n开始仿真 {args.steps} 步（Ctrl+C 退出）...")
            for i in range(args.steps):
                obs, reward, terminated, truncated, info = env.step(np.zeros(env.nu))
                env.render()  # 推送状态到 OrcaStudio/O3DE
                if (i + 1) % 100 == 0:
                    _log(f"  步进 {i + 1}/{args.steps}")
                if terminated or truncated:
                    break
            _log("  仿真完成")

    finally:
        env.close()
        _log("\nenv 已关闭")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        _logger.error(f"脚本异常退出: {exc}\n{tb}")
        print(f"[ERROR] 脚本异常退出: {exc}", file=sys.stderr, flush=True)
        print(tb, file=sys.stderr, flush=True)
        sys.exit(1)
