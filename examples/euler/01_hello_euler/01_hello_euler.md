# 第 1 课：Hello Euler — 第一个 OrcaGymEulerEnv 程序

> 阶段三（P3）端到端联调第一课。本课用随机动作驱动单铰链倒立摆，验证
> `OrcaGymEulerEnv` 体系的端到端 API 契约，聚焦**离线模式**（不需要 OrcaStudio/OrcaLab）。

---

## 1. 课程目标

验证 `OrcaGymEulerEnv` 的 5 项基础能力，离线模式下跑通端到端链路：

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 模型加载 | `SimpleEulerEnv(...)` | `nq=1, nv=1, nu=1` |
| 2 | 状态访问 | `env.data.qpos` / `env.data.qvel` | shape 正确，time=0.0000 |
| 3 | 求解器配置 | `env.sim_config.timestep` / `integrator` | 0.002 / 1 |
| 4 | reset | `env.reset()` → `(obs, info)` | obs.shape=(3,) |
| 5 | 步进 | `env.step(action)` → 五元组 | 200 步无异常，总奖励为负 |

> **本课不做强化学习**，仅用 `env.action_space.sample()` 验证链路畅通。
> reward 是 cost 函数（负值），不应呈现学习趋势。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ `orca_gym` 包已安装
- ✅ **不需要** OrcaStudio/OrcaLab（离线模式）

---

## 3. 目录结构（自包含）

```
examples/euler/01_hello_euler/
├── 01_hello_euler.md   ← 本教程
├── hello_euler.py      ← 脚本入口
└── simple_env.py       ← SimpleEulerEnv（资产路径指向 assets/scenes/）
```

**资产路径**：`simple_env.py` 顶部通过 `__file__` 上溯 4 层定位项目根，
指向 `assets/scenes/simple_pendulum.xml`。

**依赖**：仅 `orca_gym.environment.euler.*` + `orca_gym.log`，零 Local 体系引用。

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认 200 步
python examples/euler/01_hello_euler/hello_euler.py

# 指定步数
python examples/euler/01_hello_euler/hello_euler.py --steps 500
```

---

## 5. 预期输出

```
============================================================
第 1 课：Hello Euler — 第一个 OrcaGymEulerEnv 程序
  模式: 离线（不需要 OrcaStudio/OrcaLab）
  步数: 200
============================================================
[1/5] 环境创建成功: nq=1, nv=1, nu=1
[2/5] 状态访问: qpos.shape=(1,), time=0.0000
[3/5] 求解器配置: timestep=0.002, integrator=1
[4/5] reset 成功: obs.shape=(3,), obs=[...]
[5/5] step 50/200: obs=[...], reward=-0.2442, time=0.1000
[5/5] step 100/200: ...
[5/5] step 150/200: ...
[5/5] step 200/200: ...
[5/5] 步进完成: 总奖励=-2922.8292（随机动作，无学习意义）
============================================================
第 1 课验证通过
============================================================
```

**通过条件**：
- ✅ 5 项 `[X/5]` 全部打印，无异常退出
- ✅ 退出码 0

---

## 6. 本课概念

| 概念 | 说明 |
|------|------|
| `OrcaGymEulerEnv` | OrcaGym Euler 体系的环境基类，符合 Gymnasium API |
| 离线模式 | `skip_grpc_load=True`，本地 MuJoCo 仿真，不需要 OrcaStudio/OrcaLab |
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

---

## 7. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--steps` | `200` | 仿真步数 |
| `--time-step` | `0.002` | 物理时间步长（秒） |
| `--frame-skip` | `5` | 每个动作执行的物理步数 |

---

## 8. 故障排查

### Q1：`ModuleNotFoundError: No module named 'simple_env'`

**原因**：未激活 `orca` 环境，或脚本目录不在 `sys.path`。

**解决**：

```bash
cd /path/to/OrcaPlayground
conda activate orca
python examples/euler/01_hello_euler/hello_euler.py
```

> 本目录自包含，运行时 Python 自动将脚本所在目录加入 `sys.path[0]`，
> 同目录 `from simple_env import ...` 可直接生效，无需额外 `PYTHONPATH`。

### Q2：`FileNotFoundError: simple_pendulum.xml`

**原因**：`assets/scenes/simple_pendulum.xml` 缺失。

**解决**：确认 `assets/scenes/simple_pendulum.xml` 存在（迁移后资产统一在 `assets/`）。
