# 第 3 课：环境注册与 ID 启动 — gymnasium.register 标准流程

> 阶段三（P3B）环境注册课。对照第 2 课（直接 ``Env(...)`` 构造），演示通过
> `gym.register` + `gym.make` 启动 Euler 仿真的标准 Gymnasium 流程。本课为
> 在线模式（连接 OrcaStudio/OrcaLab），聚焦生产环境必备能力：env ID 注册、
> entry_point 动态导入、幂等性管理。

---

## 1. 课程目标

验证 5 项能力：

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 环境注册 | `gym.register(id=..., entry_point=..., kwargs=...)` | env ID 写入 gymnasium registry |
| 2 | ID 启动 | `gym.make(env_id)` | gymnasium 按 entry_point 动态 import 并构造 |
| 3 | spec / unwrapped | `env.spec.id` / `env.unwrapped` | id 正确，unwrapped 是 `RegisterEulerEnv` |
| 4 | 幂等性 | 重复 `gym.make` 同一 env_id | 不重复 register，不报错 |
| 5 | 标准 API | `reset()` / `step()` / `render()` | 行为等价于第 2 课直接构造方式 |

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaStudio/OrcaLab 已启动，并加载 pendulum 场景（`simple_pendulum.xml`）
- ✅ Studio 中已点击「运行」启动仿真，gRPC 监听 `localhost:50051`

---

## 3. 目录结构（自包含）

```
examples/euler/03_env_register/
├── 03_env_register.md   ← 本教程
├── register_env.py      ← RegisterEulerEnv（entry_point 目标模块）
└── run_env_register.py  ← 脚本入口（register + make + 仿真循环）
```

**资产路径**：`register_env.py` 顶部通过 `__file__` 上溯 2 层定位 `examples/euler/`，
指向 `examples/euler/assets/scenes/simple_pendulum.xml`。

**依赖**：仅 `orca_gym.environment.euler.*` + `orca_gym.log` + `gymnasium`，
零 Local 体系引用。

**entry_point 可导入性**：`entry_point="register_env:RegisterEulerEnv"` 要求
`register_env` 模块可被 import。入口脚本通过 `sys.path.insert(0, _SCRIPT_DIR)`
注入脚本目录解决（见 `run_env_register.py` 顶部）。

---

## 4. 运行步骤

### 步骤 1（人工）：启动 OrcaStudio/OrcaLab 并加载 pendulum 场景

1. 打开 OrcaStudio/OrcaLab
2. 资产搜索 `pendulum` 或导入 `examples/euler/assets/scenes/simple_pendulum.xml`
3. 点击「运行」启动仿真

### 步骤 2（人工）：运行课程脚本

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认在线模式，RTF=1.0 实时同步，200 步
python examples/euler/03_env_register/run_env_register.py

# 指定 Studio 地址
python examples/euler/03_env_register/run_env_register.py --addr 192.168.1.100:50051

# 增加仿真步数
python examples/euler/03_env_register/run_env_register.py --steps 1000
```

### 步骤 3（人工）：观察 Studio 视口

- `[2/5]` 阶段控制台输出关卡结构诊断（joints / bodies 列表）
- `[4/5]` reset 后视口应显示摆杆初始状态
- `[5/5]` 循环中视口应显示摆杆实时运动（随机动作）

---

## 5. 预期输出

```
============================================================
第 3 课：环境注册与 ID 启动 — gymnasium.register 标准流程
  模式: 在线 gRPC（addr=localhost:50051）
  步数: 200
  对照: 第 2 课用 Env(...) 直接构造；本课用 gym.make(env_id) 实例化
============================================================
[1/5] 注册环境到 gymnasium registry
  gym.register 成功: EulerPendulumRegister-OrcaGym-localhost-50051-000
    entry_point = register_env:RegisterEulerEnv
      env_id = EulerPendulumRegister-OrcaGym-localhost-50051-000
[2/5] gym.make(env_id) 实例化环境
      env.spec.id = EulerPendulumRegister-OrcaGym-localhost-50051-000
      env.unwrapped 类型 = RegisterEulerEnv
      nq=1, nv=1, nu=1
      ── 关卡结构诊断 ──
      joints (count=1):
        - hinge: id=0, type=hinge, qpos_addr=0, dof_addr=0
      bodies (count=2):
        - [0] world
        - [1] pendulum
      ✓ 定位 hinge joint: hinge
[3/5] 幂等性验证：再次 gym.make 同一 env_id
      第二次 make 成功，spec.id = EulerPendulumRegister-OrcaGym-localhost-50051-000
[4/5] env.reset() 进入仿真态
      obs.shape = (3,), obs = [ 0.995 -0.1    0.  ]
      → Studio 视口应显示摆杆初始状态
[5/5] 步进渲染循环（200 步，随机动作，RTF=1.0）
  step 50/200: reward=-0.1234, time=0.5000
  step 100/200: reward=-0.2345, time=1.0000
  ...
      步进完成: 总奖励=-12.3456
============================================================
第 3 课验证通过
  ✓ gym.register 写入注册表
  ✓ gym.make 通过 entry_point 实例化
  ✓ env.spec.id / env.unwrapped 类型正确
  ✓ 重复 make 幂等
  ✓ reset/step/render 等价于直接构造方式
============================================================
```

**通过条件**：
- ✅ `[1/5]` gym.register 成功，env_id 格式为 `<EnvName>-OrcaGym-<addr>-<index>`
- ✅ `[2/5]` `env.unwrapped` 类型为 `RegisterEulerEnv`，关卡诊断显示 hinge joint
- ✅ `[3/5]` 第二次 make 不报 `env id already registered` 错误
- ✅ 视口显示摆杆实时运动
- ✅ 退出码 0

---

## 6. 本课概念

| 概念 | 说明 |
|------|------|
| `gym.register` | 将 env ID + entry_point + kwargs 写入 gymnasium 全局注册表 |
| `gym.make` | 按 env ID 从注册表查找 entry_point，动态 import 模块并构造 env |
| `entry_point` | `"模块:类名"` 字符串，gymnasium 据此定位类；模块必须可 import |
| env ID 命名规范 | `<EnvName>-OrcaGym-<addr>-<index>`（addr 中 `:` 替换为 `-`），对齐 `run_euler_loop.py` |
| 幂等注册 | register 前检查 `env_id in gym.envs.registry`，已注册则跳过 |
| `max_episode_steps` | register 时声明截断阈值，gymnasium 自动包 `TimeLimit` |
| `env.spec` | gymnasium 注册信息（id、kwargs 等），来自 registry 而非 env 本身 |
| `env.unwrapped` | 剥离所有包装器（如 TimeLimit），拿到原生 `RegisterEulerEnv` |
| joint 名前缀 | OrcaStudio 在线模式可能在 joint 名前加 USD 路径前缀（如 `simple_pendulum_usda_1_hinge`），需后缀匹配 |

### 代码解析

```python
# 注册：entry_point 指向本课的 RegisterEulerEnv
gym.register(
    id=env_id,                                  # EulerPendulumRegister-OrcaGym-localhost-50051-000
    entry_point="register_env:RegisterEulerEnv",  # gymnasium 动态 import
    kwargs={
        "orcagym_addr": orcagym_addr,
        "skip_grpc_load": False,                # 在线模式：连接 OrcaStudio
        "render_mode": "human",
    },
    max_episode_steps=MAX_EPISODE_STEPS,
)

# 幂等性：register 前检查（重复注册会抛异常）
if env_id in gym.envs.registry:
    return env_id  # 已注册，跳过

# 启动：gymnasium 按 entry_point 实例化
env = gym.make(env_id)
```

- 第 2 课 `SimpleEulerEnv(...)` 是**直接构造**；本课 `gym.make(env_id)` 是
  **注册表驱动**，env ID 可跨模块/跨进程共享（生产环境用法）。
- 生产用法参见 `orca_gym/scripts/run_euler_loop.py`：多 env 按地址 + 索引
  生成不同 env_id，分别注册并 `gym.make`。

---

## 7. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio/OrcaLab gRPC 地址 |
| `--steps` | `200` | 仿真步数 |
| `--time-step` | `0.002` | 物理时间步长（秒） |
| `--frame-skip` | `5` | 每个动作执行的物理步数 |

---

## 8. 故障排查

### Q1：gRPC 连接失败 / 超时

**原因**：OrcaStudio/OrcaLab 未启动或地址不对。

**解决**：
1. 确认 OrcaStudio/OrcaLab 已启动并监听 `localhost:50051`
2. 确认场景中已加载 pendulum 模型
3. 确认 `--addr` 参数正确

### Q2：`ModuleNotFoundError: No module named 'register_env'`

**原因**：未激活 `orca` 环境，或 entry_point 目标模块不在 `sys.path`。

**解决**：

```bash
cd /path/to/OrcaPlayground
conda activate orca
python examples/euler/03_env_register/run_env_register.py
```

> 入口脚本已通过 `sys.path.insert(0, _SCRIPT_DIR)` 注入脚本目录，
> 使 entry_point 字符串 `register_env:RegisterEulerEnv` 可被 gymnasium
> 动态 import。若在其他脚本中引用本模块，需自行注入路径或用包导入。

### Q3：`ValueError: Environment id already registered`（其他脚本中）

**原因**：未做幂等检查直接重复 register 同一 env_id。

**解决**：register 前检查 `env_id in gym.envs.registry`（本课
`register_euler_env()` 已内置此逻辑）。

### Q4：诊断输出 `⚠ 警告: nu=0（无执行器）` 或未找到 hinge joint

**原因**：OrcaStudio 加载的关卡不是标准 `simple_pendulum`（无 motor
执行器，或 joint 名既非 `hinge` 也无 `_hinge` 后缀）。

**解决**：
1. 确认 Studio 已加载 `simple_pendulum.xml`（含 motor 执行器）
2. 查看 `[2/5]` 阶段输出的 joints 列表，确认实际 joint 名
3. 关卡含额外 body（nq>1）是可接受的——代码会按后缀 `_hinge`
   自动定位摆杆 joint
