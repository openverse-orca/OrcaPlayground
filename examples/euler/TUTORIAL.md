# OrcaGym Euler 示例课程

本课程通过一系列由浅入深的示例，演示如何使用 `OrcaGymEulerEnv` 开发自己的仿真。
每个示例既是 OrcaGymEuler 开发阶段的验证载体，也是面向用户的教学用例。

> 课程设计参考 [OrcaFlow examples/TUTORIAL.md](../../../OrcaFlow/examples/TUTORIAL.md)，
> 每个 example 对应一个开发阶段（P3 → P5），引入 1-2 个新概念。

## 快速开始

```bash
# 进入 OrcaPlayground 根目录
cd /path/to/OrcaPlayground

# 安装依赖（第 3 课需要 SB3，第 1/2 课仅需 orca_gym）
pip install -r examples/euler/requirements.txt

# 第 1 课：Hello Euler（离线，不需要 OrcaStudio）
python examples/euler/01_hello_euler/hello_euler.py

# 第 2 课：在线渲染（需要先启动 OrcaStudio）
python examples/euler/02_online_render/online_render.py

# 第 3 课：SB3 PPO 训练（离线，约 2-3 分钟）
python examples/euler/03_rl_ppo/train_ppo.py --total-timesteps 100000
```

---

## 目录结构

```
examples/euler/
├── TUTORIAL.md                     ← 你正在阅读的教程
├── requirements.txt                # 课程依赖
├── .gitignore                      # 忽略训练产物
├── 01_hello_euler/
│   └── hello_euler.py              # 第 1 课：第一个 OrcaGymEulerEnv（P3）
├── 02_online_render/
│   └── online_render.py            # 第 2 课：在线渲染与交互（P3A）
├── 03_rl_ppo/
│   ├── train_ppo.py                # 第 3 课：SB3 PPO 强化学习（P3B）
│   └── models/                     # 训练产物（.gitignore）
├── 04_query_api/                   # 第 4 课：状态查询 API（P4，待开发）
├── 05_force_apply/                 # 第 5 课：外力应用与约束（P4，待开发）
└── 06_solver_coupling/             # 第 6 课：Euler 求解器耦合（后续 phase，预留）

envs/euler/                         # 课程使用的 Env 子类
├── simple_env.py                   # SimpleEulerEnv（单铰链倒立摆，Gymnasium API）
└── scenes/
    └── simple_pendulum.xml         # MuJoCo 场景（摆杆 + motor）
```

---

## 课程总览

| 课 | 阶段 | 主题 | 新概念 | 状态 | 需要 OrcaStudio |
|----|------|------|--------|------|----------------|
| 1 | P3 | Hello Euler | `OrcaGymEulerEnv` 基本用法、离线模式、模型加载、步进、状态访问 | ✅ 已实现 | 否 |
| 2 | P3A | 在线渲染与交互 | gRPC 在线模式、render 循环、sync/异步渲染、override_ctrls、Studio UI 交互 | ✅ 已实现 | 是 |
| 3 | P3B | SB3 PPO 强化学习 | Gymnasium API 契约、SB3 PPO、奖励函数设计、Box 观测、episode 截断 | ✅ 已实现 | 否（离线训练） |
| 4 | P4 | 状态查询 API | `query_joint_*`、`query_site_*`、`query_sensor_data`、`query_contact_*` | ⏳ 待开发 | 否 |
| 5 | P4 | 外力应用与约束 | `apply_body_force`、`clear_*_force`、`update_equality_constraints`、`set_*` | ⏳ 待开发 | 否 |
| 6 | 后续 phase | Euler 求解器耦合 | 与 Euler 求解器的深度耦合（具体内容待设计） | 🔒 预留 | — |

---

## 环境准备

### 1. 安装依赖

```bash
cd /path/to/OrcaPlayground
pip install -r examples/euler/requirements.txt
```

### 2. 前置条件

| 依赖 | 说明 |
|------|------|
| OrcaGym | 已安装（`orca_gym` 包可在 Python 中 import） |
| MuJoCo | OrcaGym 自带的 `mujoco` 包 |
| OrcaStudio | **仅第 2 课、第 3 课在线模式需要**，需启动 gRPC 服务（默认 `localhost:50051`） |
| stable-baselines3 | **仅第 3 课需要**，`requirements.txt` 已包含 |

### 3. 验证安装

```bash
python -c "from envs.euler.simple_env import SimpleEulerEnv; print('OK')"
python -c "import stable_baselines3; print('sb3:', stable_baselines3.__version__)"
```

---

## 第 1 课：Hello Euler — 第一个 OrcaGymEulerEnv 程序

**文件**：`01_hello_euler/hello_euler.py`

**运行**：

```bash
python examples/euler/01_hello_euler/hello_euler.py
```

**预期输出**：

```
[1/5] 环境创建成功: nq=1, nv=1, nu=1
[2/5] 状态访问: qpos.shape=(1,), time=0.0000
[3/5] 求解器配置: timestep=0.002, integrator=1
[4/5] reset 成功: obs.shape=(3,), obs=[...]
[5/5] step 50/200: obs=[...], reward=-0.2442, time=0.1000
[5/5] 步进完成: 总奖励=-2922.8292（随机动作，无学习意义）
第 1 课验证通过
```

### 本课概念

| 概念 | 说明 |
|------|------|
| `OrcaGymEulerEnv` | OrcaGym Euler 体系的环境基类，符合 Gymnasium API |
| 离线模式 | `skip_grpc_load=True`，本地 MuJoCo 仿真，不需要 OrcaStudio |
| `env.data` | `OrcaGymDataView`，访问 `qpos`/`qvel`/`time` 等物理状态 |
| `env.sim_config` | `SimConfig`，求解器配置（timestep、integrator 等） |
| `do_simulation` | 基类步进方法：`set_ctrl` → `mj_step` × frame_skip → `sync_to_view` |

### 代码解析

```python
env = SimpleEulerEnv(
    orcagym_addr="localhost:50051",
    time_step=0.002,
    frame_skip=5,
    skip_grpc_load=True,   # 离线模式
)
```

- `skip_grpc_load=True`：跳过 gRPC，直接用本地 MuJoCo 仿真。
- `time_step=0.002`：物理步长 2ms。
- `frame_skip=5`：每个 `env.step()` 执行 5 个物理步（控制频率 100Hz）。

```python
obs, info = env.reset()
for step in range(args.steps):
    action = env.action_space.sample()   # 随机动作（不做 RL）
    obs, reward, terminated, truncated, info = env.step(action)
```

- `env.reset()` 返回 `(obs, info)`，符合 Gymnasium API。
- `env.step(action)` 返回 `(obs, reward, terminated, truncated, info)` 五元组。

> **注意**：本课用随机动作验证链路畅通，**不做强化学习**。reward 是 cost 函数（负值），不应呈现学习趋势。

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--steps` | `200` | 仿真步数 |
| `--time-step` | `0.002` | 物理时间步长（秒） |
| `--frame-skip` | `5` | 每个动作执行的物理步数 |

---

## 第 2 课：在线渲染与交互 — 连接 OrcaStudio

**文件**：`02_online_render/online_render.py`

**运行**：

```bash
# 1. 先启动 OrcaStudio 并加载 pendulum 场景
# 2. 运行脚本
python examples/euler/02_online_render/online_render.py
```

**预期输出**：

```
[1/4] gRPC 连接成功: nq=1, nv=1, nu=1
[2/4] reset 成功: obs.shape=(3,)
      → 此时 Studio 视口应显示摆杆初始状态（竖直向上）
[3/4] 开始步进渲染循环（可在 Studio UI 手动控制执行器 / 拖拽物体）
  step 100/500: reward=-0.2442, time=1.0000
  step 200/500: reward=-0.1833, time=2.0000
[3/4] 步进完成: 总奖励=-1234.5678
[4/4] 环境关闭，gRPC 连接断开
第 2 课验证通过
```

### 本课概念

| 概念 | 说明 |
|------|------|
| 在线模式 | `skip_grpc_load=False`，通过 gRPC 连接 OrcaStudio，同步渲染 |
| `render_mode="human"` | 渲染到 Studio 视口（`"none"` 不渲染） |
| `sync_render` | `True`：每个物理步都渲染（帧率最高）；`False`：按 fps 节流（默认，CPU 占用低） |
| `override_ctrls` | Studio UI 返回的手动控制值，在 `set_ctrl` 中覆盖程序动作 |
| `do_body_manipulation` | Studio UI 拖拽物体的占位接口 |

### 代码解析

```python
env = SimpleEulerEnv(
    orcagym_addr=args.addr,
    skip_grpc_load=False,       # 在线模式：连接 OrcaStudio
    render_mode="human",        # 渲染到 Studio 视口
    sync_render=args.sync_render,
)
```

```python
for step in range(args.steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()   # 将物理状态同步到 Studio 视口
```

- `env.render()` 在在线模式下通过 gRPC 把当前物理状态推送到 Studio。
- `sync_render=True` 时 `step()` 内部已渲染，`render()` 立即返回。
- `sync_render=False` 时 `render()` 按 fps 节流，可能跳过部分帧。

> **交互提示**：运行时可在 OrcaStudio UI 手动拖动执行器滑块（`override_ctrls` 会覆盖程序动作），或拖拽物体（`do_body_manipulation`）。

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio gRPC 地址 |
| `--steps` | `500` | 仿真步数 |
| `--sync-render` | `False` | 同步渲染（每个物理步都渲染） |

---

## 第 3 课：SB3 PPO 强化学习 — 训练倒立摆保持直立

**文件**：`03_rl_ppo/train_ppo.py`

**运行**：

```bash
# 离线训练（100k 步，约 2-3 分钟，CPU）
python examples/euler/03_rl_ppo/train_ppo.py --total-timesteps 100000

# 快速验证（20k 步，约 30 秒）
python examples/euler/03_rl_ppo/train_ppo.py --total-timesteps 20000

# 评估已训练模型
python examples/euler/03_rl_ppo/train_ppo.py --eval --eval-episodes 5
```

**预期输出**：

```
[1/4] 环境创建成功: obs_space=(3,), action_space=(1,)
[2/4] PPO 模型创建成功
[3/4] 开始训练...
  [train] step=2048, episodes=10, mean_reward=-2596.93 ± 914.49
  [train] step=28672, episodes=10, mean_reward=-96.55 ± 83.42
  [train] step=40960, episodes=10, mean_reward=-3.25 ± 0.70
  [train] step=100352, episodes=10, mean_reward=-0.43 ± 0.13
[4/4] 训练完成，模型已保存: .../03_rl_ppo/models/ppo_pendulum.zip
```

### 本课概念

| 概念 | 说明 |
|------|------|
| Gymnasium API 契约 | `reset()`/`step()` 返回值格式，使 SB3 等 RL 库可直接消费 |
| Box 观测空间 | `[cos(theta), sin(theta), theta_dot]`，避免 2π 周期性问题 |
| Pendulum-v1 cost 函数 | `reward = -(theta² + 0.1·theta_dot² + 0.001·action²)`，直立时为 0 |
| episode 截断 | `MAX_EPISODE_STEPS=200`，使 SB3 Monitor 能收集 episode 奖励 |
| Monitor 包装器 | SB3 工具，收集 episode 奖励供回调使用 |
| MlpPolicy | PPO 的 MLP 策略网络，推荐用 CPU（详见 SB3 issue #1245） |

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

### 命令行参数

**环境参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio gRPC 地址 |
| `--no-skip-grpc` | `False` | 启用 gRPC（默认离线模式） |
| `--time-step` | `0.002` | 物理时间步长（秒） |
| `--frame-skip` | `5` | 每个动作执行的物理步数 |
| `--render-mode` | `none` | 渲染模式（训练时默认 `none`） |

**训练参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total-timesteps` | `20000` | 训练总步数 |
| `--learning-rate` | `3e-4` | 学习率 |
| `--n-steps` | `2048` | PPO 每次更新的步数 |
| `--batch-size` | `64` | minibatch 大小 |
| `--device` | `cuda` | PyTorch 设备（默认 GPU 训练，CPU 训练 MLP 较慢） |

**评估参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--eval` | `False` | 评估模式（加载模型而非训练） |
| `--model-path` | `models/ppo_pendulum.zip` | 已训练模型路径 |
| `--eval-episodes` | `10` | 评估回合数 |

### 模型文件

训练产物保存在 `03_rl_ppo/models/`（已被 `.gitignore` 忽略）：

```python
from stable_baselines3 import PPO
model = PPO.load("examples/euler/03_rl_ppo/models/ppo_pendulum.zip")
```

---

## 第 4 课：状态查询 API（待开发）

**阶段**：P4 — API 完备化

**预期文件**：`04_query_api/query_api.py`

**将验证的概念**：

| 概念 | 方法 |
|------|------|
| 关节状态查询 | `query_joint_qpos` / `query_joint_qvel` / `query_joint_qacc` |
| Site 查询 | `query_site_pos_and_quat` / `query_site_pos_and_mat` / `query_site_size` |
| 传感器查询 | `query_sensor_data` |
| 执行器力矩 | `query_actuator_torques` |
| 接触查询 | `query_contact_simple` / `query_contact_force` |
| Body 位姿 | `get_body_xpos_xmat_xquat` |

> 本课将在 P4 阶段开发完成后补充完整内容。

---

## 第 5 课：外力应用与约束（待开发）

**阶段**：P4 — API 完备化

**预期文件**：`05_force_apply/force_apply.py`

**将验证的概念**：

| 概念 | 方法 |
|------|------|
| 施加外力 | `apply_body_force(body_name, force, torque)` |
| 清除外力 | `clear_body_force` / `clear_all_forces` |
| 约束更新 | `update_equality_constraints` |
| 状态设置 | `set_joint_qpos` / `set_joint_qvel` / `set_mocap_pos_and_quat` |
| 通过 Site 施力 | `mj_apply_force_at_site` |

> 本课将在 P4 阶段开发完成后补充完整内容。

---

## 第 6 课：Euler 求解器耦合（预留，后续 phase）

**阶段**：后续 phase（不在 Phase 1 范围内）

**预期文件**：`06_solver_coupling/`（待设计）

**预留说明**：

本课将演示 OrcaGymEulerEnv 与 Euler 求解器的深度耦合能力。具体内容（API、场景、验证点）
将在后续 phase 的设计文档中确定，届时补充完整教程。

> 本课为预留章节，当前不提供示例代码。

---

## 常见问题

### Q1：第 1 课随机采样，为何奖励看似"上升"？

**原因**：这不是学习效果，是奖励函数目标定义的巧合。

- **修复前**：奖励为 `cos(theta)`，目标是摆杆水平。随机扰动后摆杆恰好经过水平位置，导致 reward 接近 1，呈现"上升"假象。
- **修复后**：奖励改为 Pendulum-v1 标准 cost `-(theta² + ...)`，目标是直立。随机动作下 reward 为大负值，**不再呈现上升趋势**。

### Q2：第 3 课训练时出现 `UserWarning: You are trying to run PPO on the GPU`

**原因**：SB3 检测到 GPU 可用，默认用 GPU，但对 MLP 策略推荐 CPU（[issue #1245](https://github.com/DLR-RM/stable-baselines3/issues/1245)）。

**说明**：项目统一采用 GPU 训练（`--device cuda`，默认值），实测 CPU 训练 MLP 较慢。此警告可忽略，不影响训练正确性。

### Q3：训练时出现 `Error 304: OS call failed`

**原因**：在 TRAE sandbox 内运行，CUDA 驱动因 capabilities 限制无法初始化。

**解决**：GPU 训练需 TRAE 命令白名单旁路 sandbox（详见 AGENTS.md 规则 3），或使用 `--device cpu` 退化到 CPU 训练。

### Q4：第 2 课在线模式连接 OrcaStudio 失败

**排查**：
1. 确认 OrcaStudio 已启动并监听 `localhost:50051`
2. 确认场景中已加载 pendulum 模型
3. 确认 `--addr` 参数正确
4. 第 2 课默认在线模式（`skip_grpc_load=False`），无需额外参数
