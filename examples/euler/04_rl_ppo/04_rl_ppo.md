# 第 4 课：SB3 PPO 强化学习 — 训练倒立摆保持直立

> 阶段三（P3B）强化学习课。基于 Stable Baselines3 PPO 算法训练
> `SimpleEulerEnv`（单铰链倒立摆）。本课验证 `OrcaGymEulerEnv` 与
> Gymnasium API 契约的兼容性，使第三方 RL 库可直接消费。

---

## 1. 课程目标

| # | 验证点 | 期望 |
|---|--------|------|
| 1 | 训练（离线 VecEnv 并发） | reward 从 -2500 趋近 0（摆杆学会保持直立） |
| 2 | 评估默认 online | Studio 视口实时显示智能体行为 |
| 3 | Studio 未启动自动退化 | 离线无头，评估指标正常计算 |
| 4 | RTF 同步 | 在线渲染时仿真时间 ≈ 真实时间 |

> 训练默认**离线无头**（SubprocVecEnv 多进程并行仿真，最高效），评估默认 online（单 env 可视化）。
> 训练产物 `ppo_pendulum.zip` 保存在 `04_rl_ppo/models/`（`.gitignore` 忽略）。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用，已安装 `stable-baselines3`
- ✅ `orca_gym` 包已安装
- ✅ **训练**：不需要 OrcaStudio/OrcaLab（离线）
- ✅ **评估 online**：需要 OrcaStudio/OrcaLab + pendulum 场景

---

## 3. 目录结构（自包含）

```
examples/euler/04_rl_ppo/
├── 04_rl_ppo.md        ← 本教程
├── train_ppo.py        ← 训练 / 评估脚本入口
├── simple_env.py       ← SimpleEulerEnv（资产路径指向 examples/euler/assets/scenes/）
└── models/             # 训练产物（.gitignore 忽略）
    └── ppo_pendulum.zip
```

**资产路径**：`simple_env.py` 顶部通过 `__file__` 上溯 2 层定位 `examples/euler/`，
指向 `examples/euler/assets/scenes/simple_pendulum.xml`。

**依赖**：`orca_gym.environment.euler.*` + `orca_gym.log` + `stable_baselines3`，
零 Local 体系引用。

---

## 4. 运行步骤

### 4.1 训练

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认训练（16 并发 VecEnv，buffer=16×128=2048）
python examples/euler/04_rl_ppo/train_ppo.py

# 快速验证（单 env，20k 步，约 30 秒）
python examples/euler/04_rl_ppo/train_ppo.py --n-envs 1 --total-timesteps 20000
```

> **VecEnv 原理**：`SubprocVecEnv` 为每个 env fork 一个子进程并行仿真，
> PPO 每次更新从 `n_envs × n_steps` 的 buffer 采样。
> - `buffer = n_envs × n_steps` 应适中（默认 16×128=2048），过大会导致每次更新
>   batch 数过多（`buffer / batch_size`），模型吸收不了，样本效率低
> - `n_envs` 越大单次 rollout 越快，但受 CPU 核数和内存限制（默认 16 通用）
> - 训练强制离线无头（本地 MuJoCo），评估默认 online

### 4.2 评估

```bash
# 评估（默认 online human，Studio 未启动自动退化离线）
python examples/euler/04_rl_ppo/train_ppo.py --eval --eval-episodes 5

# 评估时强制离线无头
python examples/euler/04_rl_ppo/train_ppo.py --eval --render-mode none

# 评估时慢动作观察（RTF=0.5，仿真比真实时间慢一半）
python examples/euler/04_rl_ppo/train_ppo.py --eval --rtf 0.5

# 评估时快进（不 sleep，快速跑完评估回合）
python examples/euler/04_rl_ppo/train_ppo.py --eval --rtf 0
```

---

## 5. 预期输出

### 5.1 训练

```
============================================================
第 4 课：SB3 PPO 训练 — 倒立摆
  模式: 离线
  并发环境数: 16（VecEnv）
  总步数: 100000
  render_mode: none
  学习率: 0.0003
  n_steps: 128（per env，buffer = n_steps × n_envs = 2048）
  seed: 0
============================================================
[1/4] VecEnv 创建成功: n_envs=16, obs_space=(3,), action_space=(1,)
      device: cuda（GPU 训练，CPU 训练 MLP 较慢）
[2/4] PPO 模型创建成功
[3/4] 开始训练...
  [train] step=2048, episodes=10, mean_reward=-2596.93 ± 914.49
  [train] step=28672, episodes=10, mean_reward=-96.55 ± 83.42
  [train] step=40960, episodes=10, mean_reward=-3.25 ± 0.70
  [train] step=100352, episodes=10, mean_reward=-0.43 ± 0.13
[4/4] 训练完成，模型已保存: .../04_rl_ppo/models/ppo_pendulum.zip
```

### 5.2 评估

```
============================================================
第 4 课：SB3 PPO 评估 — 倒立摆
  模型: models/ppo_pendulum.zip
  模式: 在线 gRPC
  render_mode: human
  评估回合数: 5
  RTF: 1.0（按真实时间同步）
============================================================
[1/3] 模型加载成功
[2/3] 评估结果: mean_reward=-0.43 ± 0.13
[3/3] 可视化运行...
  [eval] ep=1/3, ep_reward=-0.42, time=0.4000
  ...
```

**通过条件**：
- ✅ 训练 100k 步后 `mean_reward` 趋近 0（典型 -0.4 ~ -1.0）
- ✅ 评估 `mean_reward` 为小负值（< -5），摆杆基本直立
- ✅ Studio 视口显示摆杆保持平衡

---

## 6. 本课概念

| 概念 | 说明 |
|------|------|
| Gymnasium API 契约 | `reset()`/`step()` 返回值格式，使 SB3 等 RL 库可直接消费 |
| Box 观测空间 | `[cos(theta), sin(theta), theta_dot]`，避免 2π 周期性问题 |
| Pendulum-v1 cost 函数 | `reward = -(theta² + 0.1·theta_dot² + 0.001·action²)`，直立时为 0 |
| episode 截断 | `MAX_EPISODE_STEPS=200`，使 SB3 Monitor 能收集 episode 奖励 |
| Monitor 包装器 | SB3 工具，收集 episode 奖励供回调使用（单 env）；VecEnv 用 VecMonitor |
| SubprocVecEnv | SB3 向量化环境，每个 env 在独立子进程并行仿真，多核加速 rollout |
| VecMonitor | VecEnv 版 Monitor，收集各子进程 episode 奖励 |
| make_env 工厂 | 返回 thunk（callable），架构 §6.8.1 E2：SubprocVecEnv 要求可序列化的工厂闭包 |
| MlpPolicy | PPO 的 MLP 策略网络，推荐用 GPU（详见 SB3 issue #1245） |

### 奖励函数

```
reward = -(theta² + 0.1 * theta_dot² + 0.001 * action²)
```

- `theta=0`（直立）时 reward=0（最优）
- `theta=π`（倒挂）时 reward≈-9.87（最差）
- 训练目标：reward 从大负值趋近 0

| 训练阶段 | mean_reward | 说明 |
|---------|-------------|------|
| 初期（~2k 步） | -2500 ~ -3500 | 随机策略，摆杆倒下 |
| 中期（~28k 步） | -50 ~ -200 | 开始学习平衡 |
| 后期（~40k 步） | -3 ~ -10 | 接近平衡 |
| 收敛（~100k 步） | -0.4 ~ -1.0 | 稳定平衡 |

---

## 7. 命令行参数

### 环境参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio/OrcaLab gRPC 地址 |
| `--no-skip-grpc` | `False` | 启用 gRPC（默认离线模式） |
| `--time-step` | `0.002` | 物理时间步长（秒） |
| `--frame-skip` | `5` | 每个动作执行的物理步数 |
| `--render-mode` | `none` | 渲染模式（训练时默认 `none`） |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total-timesteps` | `100000` | 训练总步数 |
| `--n-envs` | `16` | 并发环境数（SubprocVecEnv，1=单 env Monitor） |
| `--seed` | `0` | 随机种子（每个子 env 用 `seed + rank` 保证可复现） |
| `--learning-rate` | `3e-4` | 学习率 |
| `--n-steps` | `128` | PPO 每次更新 per env 的步数（buffer = n_steps × n_envs，默认 2048） |
| `--batch-size` | `64` | minibatch 大小（每次更新 batch 数 = buffer / batch_size = 32） |
| `--device` | `cuda` | PyTorch 设备（默认 GPU 训练，CPU 训练 MLP 较慢） |

### 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--eval` | `False` | 评估模式（加载模型而非训练） |
| `--model-path` | `models/ppo_pendulum.zip` | 已训练模型路径 |
| `--eval-episodes` | `10` | 评估回合数 |
| `--rtf` | `1.0` | 实时因子（仅 online 评估生效） |

---

## 8. 模型文件

训练产物保存在 `04_rl_ppo/models/`（已被 `.gitignore` 忽略）：

```python
from stable_baselines3 import PPO
model = PPO.load("examples/euler/04_rl_ppo/models/ppo_pendulum.zip")
```

---

## 9. 故障排查

### Q1：`UserWarning: You are trying to run PPO on the GPU`

**原因**：SB3 检测到 GPU 可用，默认用 GPU，但对 MLP 策略推荐 CPU
（[issue #1245](https://github.com/DLR-RM/stable-baselines3/issues/1245)）。

**说明**：项目统一采用 GPU 训练（`--device cuda`，默认值），实测 CPU 训练 MLP
较慢。此警告可忽略，不影响训练正确性。

### Q2：`Error 304: OS call failed`

**原因**：在 TRAE sandbox 内运行，CUDA 驱动因 capabilities 限制无法初始化。

**解决**：GPU 训练需 TRAE 命令白名单旁路 sandbox（详见 AGENTS.md 规则 3），
或使用 `--device cpu` 退化到 CPU 训练。

### Q3：`ModuleNotFoundError: No module named 'simple_env'`

**原因**：未激活 `orca` 环境，或脚本目录不在 `sys.path`。

**解决**：

```bash
cd /path/to/OrcaPlayground
conda activate orca
python examples/euler/04_rl_ppo/train_ppo.py --total-timesteps 100000
```

> 本目录自包含，运行时 Python 自动将脚本所在目录加入 `sys.path[0]`，
> 同目录 `from simple_env import ...` 可直接生效，无需额外 `PYTHONPATH`。

### Q4：评估时 gRPC 连接超时

**原因**：Studio 未启动。

**说明**：脚本内置 `_probe_studio()` 探测，Studio 不可达时自动退化到离线无头
评估，指标照常计算。如需可视化，请先启动 Studio 并加载 pendulum 场景。

### Q5：`BrokenPipeError` / 子进程异常退出（n_envs 过大）

**原因**：`n_envs` 超出机器资源（CPU 核数不足、内存不足、文件描述符上限）。

**解决**：
- 降低 `--n-envs`（如 8/16），匹配机器 CPU 核数
- 提高 ulimit：`ulimit -n 65535`（文件描述符）
- 监控内存：每个子进程加载一份 MuJoCo 模型，`n_envs × 模型大小` 需 < 可用内存

### Q6：样本效率低 / 训练只更新 1 次就结束

**原因**：`buffer = n_envs × n_steps` 过大，导致每次更新 batch 数过多
（`buffer / batch_size`），模型吸收不了；或 buffer > total_timesteps 只做 1 次更新。

**解决**：调小 `--n-steps`，使 `buffer = n_envs × n_steps` 适中
（建议 1024~4096，batch 数 = buffer / batch_size 在 16~64 之间）。

例：`--n-envs 16 --n-steps 128` → buffer=2048，batch 数=32（合理）。
