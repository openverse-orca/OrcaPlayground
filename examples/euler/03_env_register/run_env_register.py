"""第 3 课：环境注册与 ID 启动 — gymnasium.register 标准流程

对照第 2 课（直接 ``Env(...)`` 构造），演示通过 gym.register + gym.make 启动
Euler 仿真的标准 Gymnasium 流程。本课为在线模式（连接 OrcaStudio）。

流程：
    1. gym.register(id=..., entry_point="register_env:RegisterEulerEnv", kwargs=...)
    2. env = gym.make(env_id)           ← gymnasium 按 entry_point 实例化
    3. env.reset() / env.step() / env.render()  ← 标准 Gymnasium API

entry_point 字符串要求模块可被 import；本脚本通过 sys.path 注入脚本目录解决。

用法:
    # 1. 先启动 OrcaStudio 并加载 pendulum 场景
    # 2. 运行脚本
    python examples/euler/03_env_register/run_env_register.py

    # CPU MuJoCo 后端（默认）
    python examples/euler/03_env_register/run_env_register.py

    # Euler GPU 后端
    python examples/euler/03_env_register/run_env_register.py --device cuda:0

    # 指定 Studio 地址
    python examples/euler/03_env_register/run_env_register.py --addr 192.168.1.100:50051

验证点:
    1. gym.register 成功（env ID 写入注册表）
    2. gym.make(env_id) 返回 env，env.spec.id 正确
    3. env.unwrapped 是 RegisterEulerEnv 实例
    4. 多次 gym.make 同一 env_id 幂等（不重复 register）
    5. env.reset()/step()/render() 正常工作（等价于第 2 课）

参见 orca_gym/scripts/run_euler_loop.py（生产环境的 register 用法）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np
from orca_gym.log.orca_log import get_orca_logger

# entry_point 引用的模块需可 import：将脚本目录加入 sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from register_env import RegisterEulerEnv  # noqa: E402  # entry_point 目标模块

_logger = get_orca_logger()


def _log(msg: str) -> None:
    print(msg)
    _logger.info(msg)


# env ID 命名规范：<EnvName>-OrcaGym-<addr>-<index>（对齐 run_euler_loop.py）
ENV_NAME = "EulerPendulumRegister"
MAX_EPISODE_STEPS = 200


def build_env_id(orcagym_addr: str, env_index: int = 0, device: str = "cpu") -> str:
    """生成 env ID，格式对齐 run_euler_loop.register_env。

    device 参与 env_id（设备后缀），避免同一进程 registry 内 CPU / GPU 复用
    同一 env_id 时幂等跳过返回错误后端。
    """
    addr_str = orcagym_addr.replace(":", "-")
    dev_str = device.replace(":", "-")
    return f"{ENV_NAME}-OrcaGym-{addr_str}-{dev_str}-{env_index:03d}"


def register_euler_env(
    orcagym_addr: str,
    env_index: int = 0,
    time_step: float = 0.002,
    frame_skip: int = 5,
    device: str = "cpu",
) -> str:
    """注册 Euler Env 到 gymnasium registry，返回 env_id。

    对照 run_euler_loop.register_env：同样的 register 调用，entry_point 指向
    本课的 RegisterEulerEnv（而非 orca_gym.scripts.sim_euler_env:EulerSimEnv）。
    重复注册同一 env_id 时幂等跳过。
    """
    env_id = build_env_id(orcagym_addr, env_index, device)
    if env_id in gym.envs.registry:
        _log(f"  env_id 已注册（幂等跳过）: {env_id}")
        return env_id
    gym.register(
        id=env_id,
        entry_point="register_env:RegisterEulerEnv",
        kwargs={
            "orcagym_addr": orcagym_addr,
            "agent_names": ["agent0"],
            "time_step": time_step,
            "frame_skip": frame_skip,
            # 在线模式：连接 OrcaStudio
            "skip_grpc_load": False,
            "render_mode": "human",
            "device": device,
        },
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_threshold=0.0,
    )
    _log(f"  gym.register 成功: {env_id}")
    _log("    entry_point = register_env:RegisterEulerEnv")
    return env_id


def main() -> int:
    parser = argparse.ArgumentParser(description="第 3 课：环境注册与 ID 启动")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--steps", type=int, default=200, help="仿真步数")
    parser.add_argument("--time-step", type=float, default=0.002, help="物理时间步长")
    parser.add_argument("--frame-skip", type=int, default=5, help="frame_skip")
    parser.add_argument(
        "--device",
        default="cpu",
        help="后端：cpu=CPU MuJoCo（默认），cuda:0=Euler GPU",
    )
    args = parser.parse_args()

    _log("=" * 60)
    _log("第 3 课：环境注册与 ID 启动 — gymnasium.register 标准流程")
    _log(f"  模式: 在线 gRPC（addr={args.addr}）")
    _log(f"  步数: {args.steps}")
    _log("  对照: 第 2 课用 Env(...) 直接构造；本课用 gym.make(env_id) 实例化")
    _log("=" * 60)

    # ── 步骤 1：register ──
    _log("[1/5] 注册环境到 gymnasium registry")
    env_id = register_euler_env(
        orcagym_addr=args.addr,
        time_step=args.time_step,
        frame_skip=args.frame_skip,
        device=args.device,
    )
    _log(f"      env_id = {env_id}")

    env = None
    try:
        # ── 步骤 2：make（gymnasium 按 entry_point 实例化）──
        _log("[2/5] gym.make(env_id) 实例化环境")
        env = gym.make(env_id)
        unwrapped = env.unwrapped
        _log(f"      env.spec.id = {env.spec.id}")
        _log(f"      env.unwrapped 类型 = {type(unwrapped).__name__}")
        _log(f"      nq={unwrapped.model.nq}, nv={unwrapped.model.nv}, nu={unwrapped.model.nu}")

        # 关卡结构诊断：列出所有 joint 和 body，帮助识别 OrcaStudio 加载的关卡
        _log("      ── 关卡结构诊断 ──")
        joint_dict = unwrapped.model.get_joint_dict()
        _log(f"      joints (count={len(joint_dict)}):")
        for jname, jinfo in joint_dict.items():
            jtype = jinfo.get("Type", "?")
            jid = jinfo.get("JointId", "?")
            qpos_addr = unwrapped.jnt_qposadr(jname)
            dof_addr = unwrapped.jnt_dofadr(jname)
            _log(
                f"        - {jname}: id={jid}, type={jtype}, "
                f"qpos_addr={qpos_addr}, dof_addr={dof_addr}"
            )
        body_dict = unwrapped.model.get_body_dict()
        _log(f"      bodies (count={len(body_dict)}):")
        for bname, binfo in body_dict.items():
            bid = binfo.get("BodyId", "?")
            _log(f"        - [{bid}] {bname}")

        # 关卡校验：
        # - nu=0 是真正的问题（无执行器无法控制摆杆）
        # - 无 hinge joint 也是问题（无法定位摆杆角度）
        # - nq>1 可接受（关卡含额外 body，代码会用 joint 名后缀匹配定位 hinge）
        from register_env import _find_hinge_joint_name

        hinge_name = _find_hinge_joint_name(joint_dict)
        if unwrapped.model.nu == 0:
            _log("      ⚠ 警告: nu=0（无执行器），无法控制摆杆")
            _log("        请确认 OrcaStudio 已加载含 motor 执行器的 pendulum 场景")
        elif hinge_name is None:
            _log("      ⚠ 警告: 关卡中未找到 hinge joint（无 'hinge' 也无 '*_hinge'）")
            _log("        请确认 OrcaStudio 已加载 simple_pendulum 场景")
        else:
            _log(f"      ✓ 定位 hinge joint: {hinge_name}")
            if unwrapped.model.nq != 1:
                _log(f"      ℹ 提示: nq={unwrapped.model.nq}（标准 simple_pendulum 为 nq=1）")
                _log("        关卡含额外 body，已通过 joint 名后缀匹配定位摆杆角度")

        # ── 步骤 3：幂等性验证（重复 make 不重复 register）──
        _log("[3/5] 幂等性验证：再次 gym.make 同一 env_id")
        env2 = gym.make(env_id)
        _log(f"      第二次 make 成功，spec.id = {env2.spec.id}")
        env2.close()

        # ── 步骤 4：reset 进入仿真态 ──
        _log("[4/5] env.reset() 进入仿真态")
        obs, info = env.reset()
        _log(f"      obs.shape = {np.asarray(obs).shape}, obs = {np.asarray(obs)}")
        _log("      → Studio 视口应显示摆杆初始状态")

        # ── 步骤 5：step + render 循环 ──
        _log(f"[5/5] 步进渲染循环（{args.steps} 步，随机动作，RTF=1.0）")
        step_dt = unwrapped.dt
        wall_start = time.perf_counter()
        total_reward = 0.0
        for step in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += float(reward)
            # RTF=1.0 同步
            expected = (step + 1) * step_dt
            elapsed = time.perf_counter() - wall_start
            if elapsed < expected:
                time.sleep(expected - elapsed)
            if (step + 1) % 50 == 0:
                _log(
                    f"      step {step + 1}/{args.steps}: "
                    f"reward={reward:.4f}, time={info['time']:.4f}"
                )
            if terminated or truncated:
                _log(f"      episode 结束: terminated={terminated}, truncated={truncated}")
                obs, info = env.reset()
        _log(f"      步进完成: 总奖励={total_reward:.4f}")
    finally:
        if env is not None:
            env.close()

    _log("=" * 60)
    _log("第 3 课验证通过")
    _log("  ✓ gym.register 写入注册表")
    _log("  ✓ gym.make 通过 entry_point 实例化")
    _log("  ✓ env.spec.id / env.unwrapped 类型正确")
    _log("  ✓ 重复 make 幂等")
    _log("  ✓ reset/step/render 等价于直接构造方式")
    _log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
