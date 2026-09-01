# 第 11 课：求解器与场景配置深度 — SimConfig 四属性对比实验

> 阶段四（P4）补充课程。本课系统演示 `env.sim_config` 全部可配置项对动力学行为的影响，
> 通过四组对比实验（timestep / integrator / gravity / iterations）观察能量保持、
> 积分器特性、重力对周期的影响、接触求解迭代次数的作用。
>
> **场景**：复用 `simple_pendulum.xml`（单铰链倒立摆），离线模式纯物理对比，不需要 OrcaStudio/OrcaLab。

---

## 1. 课程目标

验证 `SimConfig` 四项属性的配置生效性与动力学影响，覆盖数值判定 + 物理趋势观察：

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | timestep 精度 | `time_step={0.002..0.2}`（构造参数） | 小步长能量保持良好，大步长（≥0.1）漂移明显 |
| 2 | 积分器能量保持 | `sim_config_overrides={"integrator": Euler/RK4/implicitfast}` | 保守系统下 symplectic Euler 长期保持优于 RK4 |
| 3 | gravity 影响周期 | `sim_config_overrides={"gravity": (0,0,g)}` | 地球周期短、月球周期长（√(g) 关系）、失重不摆动 |
| 4 | iterations 构造期下发 | `sim_config_overrides={"iterations": {10..500}}` | 构造期下发生效（构造完成后 getter 即返回设定值） |

> **本课为对比实验课**，不做强化学习，不依赖 OrcaStudio/OrcaLab。
> `action=0`（零控）自由摆动，通过 `env.energy()` 读取总能量观察漂移。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ `orca_gym` 包已安装
- ✅ **不需要** OrcaStudio/OrcaLab（离线模式，`skip_grpc_load=True`）
- ✅ `examples/euler/assets/scenes/simple_pendulum.xml` 存在

---

## 3. 目录结构（自包含）

```
examples/euler/11_scene_config/
├── 11_scene_config.md     ← 本教程
├── run_scene_config.py    ← 脚本入口（四组实验调度 + 结果打印）
└── scene_config_env.py    ← SceneConfigEulerEnv（暴露 energy() 方法）
```

**资产路径**：`scene_config_env.py` 顶部通过 `__file__` 上溯 2 层定位 `examples/euler/`，
指向 `examples/euler/assets/scenes/simple_pendulum.xml`。

**依赖**：仅 `orca_gym.environment.euler.*` + `orca_gym.log`，零 Local 体系引用。

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 运行全部 4 个实验
python examples/euler/11_scene_config/run_scene_config.py

# 只跑某个实验
python examples/euler/11_scene_config/run_scene_config.py --exp timestep
python examples/euler/11_scene_config/run_scene_config.py --exp integrator
python examples/euler/11_scene_config/run_scene_config.py --exp gravity
python examples/euler/11_scene_config/run_scene_config.py --exp iterations

# GPU 后端（Euler.SolverMujoco，需 CUDA 可用）
python examples/euler/11_scene_config/run_scene_config.py --device cuda:0
```

### GPU 后端说明

四个实验均支持 GPU 后端（`--device cuda:0`）。Euler 后端下 `mj_model.opt` 在
solver 构造时随 `put_model` 上传 GPU 后固化，**init 后 opt 字段只读**（构造后
`env.sim_config.<field> = ...` 赋值会触发 setter 守卫抛 `RuntimeError`，见
OrcaGym P1 设计）。因此 SimConfig 覆盖统一经构造参数在后端固化前下发：

- `timestep` → `time_step` 构造参数（既有通道）；
- `integrator` / `gravity` / `iterations` → `sim_config_overrides` 构造参数
  （如 `SceneConfigEulerEnv(sim_config_overrides={"integrator": 0})`）。

构造完成后 `env.sim_config.<field>` 的 getter 即返回覆盖值（双后端一致，getter
委托 host `mj_model.opt`）。GPU float32 与 CPU float64 存在数值容差，趋势判据
一致即可。

---

## 5. 预期输出

### 实验 1：timestep 对比（自由摆动 20 秒）

```
timestep |    E_init  |   E_final  |       漂移
----------------------------------------------------
     0.002 |     0.0000 |     0.0000 |    +0.0000
      0.01 |     0.0000 |    -0.0001 |    -0.0001
      0.05 |     0.0000 |    -0.0033 |    -0.0033
       0.1 |     0.0000 |    -0.0132 |    -0.0132
       0.2 |     0.0000 |    -0.0529 |    -0.0529
```

**趋势**：timestep 越大，半隐式 Euler 的离散化误差累积越明显，能量漂移单调增大。

### 实验 2：integrator 对比（timestep=0.05 放大差异，20 秒）

```
    integrator |    E_init  |   E_final  |       漂移
------------------------------------------------------------
         Euler |     0.0000 |    -0.0033 |    -0.0033
           RK4 |     0.0000 |    -0.4891 |    -0.4891
  implicitfast |     0.0000 |    -0.0033 |    -0.0033
```

**趋势**：无阻尼保守系统下，symplectic Euler（半隐式）能量在真实值附近振荡，长期保持好；
RK4 短期精度高但非保结构，能量单调漂移。

### 实验 3：gravity 对比（自由摆动，记录 theta 在 t=1..5 秒）

```
gravity |   t=1.0s   t=2.0s   t=3.0s   t=4.0s   t=5.0s
------------------------------------------------------
    地球 |  +0.563  -0.268  +0.658  -0.354  +0.601
    月球 |  +0.721  -0.105  +0.753  -0.141  +0.748
    失重 |  +0.785  +0.785  +0.785  +0.785  +0.785
```

**趋势**：月球重力（g=1.62）摆动周期变长（√(g) 关系），失重下摆杆保持初始角度不摆动。

### 实验 4：iterations 对比（构造期下发生效验证）

```
   iterations(before) |    iterations(after) |   1秒后能量
------------------------------------------------------------
                  100 |                   10 |     0.0000
                  100 |                  100 |     0.0000
                  100 |                  500 |     0.0000
```

**结论**：构造期下发生效（after 等于设定值，before 为 XML 默认 100）；
`simple_pendulum` 无接触，iterations 主要影响接触求解，本场景动力学差异极小，
需接触场景验证。双后端行为一致即为本实验的验证目标。

### 通过条件

- ✅ 4 个实验全部打印，无异常退出
- ✅ timestep 漂移随步长单调增大
- ✅ RK4 漂移显著大于 Euler
- ✅ 失重场景 theta 保持不变
- ✅ iterations 构造期下发生效（after 等于设定值）

---

## 6. 本课概念

| 概念 | 说明 |
|------|------|
| `SimConfig` | `OrcaGymEulerEnv` 暴露的求解器配置对象，通过 `env.sim_config.*` 读写 |
| `timestep` | 物理积分步长（秒），影响精度与性能 |
| `integrator` | 积分器类型（`Euler=0`/`RK4=1`/`implicitfast=3`） |
| `gravity` | 三维重力向量，修改可模拟不同星球环境 |
| `iterations` | 求解器迭代次数，主要影响接触约束求解精度 |
| symplectic 积分器 | 保结构积分器，对保守系统长期能量保持好 |
| 能量保持 | 保守系统总能量（KE+PE）应保持不变，漂移反映积分器误差 |

### 代码解析

#### 6.1 创建环境（离线模式）

```python
env = SceneConfigEulerEnv(
    time_step=0.002,
    frame_skip=5,
    skip_grpc_load=True,   # 离线模式，不需要 OrcaStudio
)
```

- `skip_grpc_load=True`：跳过 gRPC，直接用本地 MuJoCo 仿真。
- `time_step` / `frame_skip`：物理步长与控制频率。

#### 6.2 配置求解器（构造期下发）

```python
# timestep 走 time_step 构造参数（既有通道）
env = SceneConfigEulerEnv(
    time_step=0.05,
    frame_skip=5,
    skip_grpc_load=True,   # 离线模式，不需要 OrcaStudio
    # integrator/gravity/iterations 走 sim_config_overrides 构造期下发
    sim_config_overrides={
        "integrator": 0,                        # Euler（半隐式 symplectic）
        # "gravity": np.array([0, 0, -1.62]),   # 月球重力
        # "iterations": 100,                    # 求解器迭代次数
    },
)
```

- `skip_grpc_load=True`：跳过 gRPC，直接用本地 MuJoCo 仿真。
- `time_step` / `frame_skip`：物理步长与控制频率。
- `sim_config_overrides`：在后端固化前下发（Euler 后端 init 后 opt 只读），
  合法键为 `integrator`/`gravity`/`iterations`（`timestep` 键会被拒绝，
  与 `time_step` 构造参数双通道冲突）。构造完成后
  `env.sim_config.integrator` 等 getter 即返回覆盖值。

> **API 隔离**：通过构造参数 + `env.sim_config.*` 公共 API 配置，不访问
> `_mjModel.opt.*` 内部属性。CPU 后端下运行期 setter（如
> `env.sim_config.iterations = 100`）仍可用；Euler 后端下 init 后赋值会触发
> 只读守卫，请统一使用构造期下发。

#### 6.3 能量计算（`scene_config_env.py`）

```python
def energy(self) -> float:
    theta = float(self.data.qpos[0])
    theta_dot = float(self.data.qvel[0])
    mass = float(self.body_subtree_mass("pendulum"))
    L = 0.5  # 质心到铰链距离
    g_z = float(self.sim_config.gravity[2])
    I = mass * L * L / 3.0  # 细杆近似转动惯量
    ke = 0.5 * I * theta_dot**2
    pe = mass * g_z * L * float(np.cos(theta))
    return float(ke + pe)
```

- **动能 KE**：`0.5 * I * θ̇²`，I 取细杆近似 `m*L²/3`。
- **势能 PE**：`m * g_z * L * cos(θ)`，θ=0 直立时最高。
- **参考点**：铰链位置；能量对比看相对漂移，I 的绝对精度不影响趋势判断。

#### 6.4 固定初始角度（配置对比起点一致）

```python
def reset_model(self) -> tuple[np.ndarray, dict]:
    qpos = self.init_qpos.copy()
    qpos[0] = INITIAL_THETA  # π/4，无随机扰动
    qvel = self.init_qvel.copy()
    self.set_joint_qpos(qpos)
    self.set_joint_qvel(qvel)
    ...
```

不同配置下起点一致，能量/轨迹差异完全来自求解器配置。

---

## 7. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--exp` | `all` | 运行哪个实验（`timestep`/`integrator`/`gravity`/`iterations`/`all`） |
| `--device` | `cpu` | 后端选择：`cpu`=CPU MuJoCo（默认），`cuda:0`=Euler.SolverMujoco GPU |

---

## 8. 故障排查

### Q1：`ModuleNotFoundError: No module named 'scene_config_env'`

**原因**：未激活 `orca` 环境，或脚本目录不在 `sys.path`。

**解决**：

```bash
cd /path/to/OrcaPlayground
conda activate orca
python examples/euler/11_scene_config/run_scene_config.py
```

> 本目录自包含，运行时 Python 自动将脚本所在目录加入 `sys.path[0]`，
> 同目录 `from scene_config_env import ...` 可直接生效。

### Q2：`FileNotFoundError: simple_pendulum.xml`

**原因**：`examples/euler/assets/scenes/simple_pendulum.xml` 缺失。

**解决**：确认 `examples/euler/assets/scenes/simple_pendulum.xml` 存在（资产统一在 `examples/euler/assets/`）。

### Q3：能量漂移方向与预期相反

**原因**：`g_z` 为重力 z 分量（负值），势能参考点取铰链位置。θ=0 直立时势能最负（最高点），
θ=π 倒立时势能最正（最低点）。能量漂移看绝对值趋势，不看符号。

**解决**：关注 `E_final - E_init` 的绝对值大小，而非符号。

### Q4：iterations 实验无动力学差异

**原因**：`simple_pendulum` 无接触，iterations 主要影响接触约束求解。

**解决**：本实验仅验证构造期下发生效。接触场景差异需扩展 XML（添加地面/接触对）后验证，
待 OrcaGym 扩展 `sim_config` 接触配置 API 后补充。

### Q5：Euler 后端构造后修改 `env.sim_config.*` 抛 `RuntimeError`

**原因**：Euler 后端下 `mj_model.opt` 在 solver 构造时上传 GPU 后固化，init 后
opt 字段只读（P1 设计的 setter 守卫）。

**解决**：改用构造期下发——`timestep` 走 `time_step` 构造参数，
`integrator`/`gravity`/`iterations` 走 `sim_config_overrides` 构造参数
（见 §4 "GPU 后端说明"）。

---

## 9. API 隔离合规

本课严格遵循 AGENTS.md 规则 4（API 隔离）：

| 操作 | 正确（本课使用） | 禁止 |
|------|------------------|------|
| 读取状态 | `env.data.qpos` / `env.data.qvel` | `env._gym._sim._mjData.qpos` |
| 求解器配置 | `env.sim_config.timestep/integrator/gravity/iterations` | `env._gym._sim._mjModel.opt.timestep` |
| 步进 | `env.do_simulation(action, frame_skip)` | `env._gym._sim._mjData.step()` |
| 质量查询 | `env.body_subtree_mass("pendulum")` | `env._gym._sim._mjModel.body_subtreemass` |

---

## 10. 扩展方向

- **接触场景**：扩展 XML 添加地面与接触对，验证 iterations 对接触求解精度的影响
- **更多积分器**：待 OrcaGym 暴露 `implicit`/`implicitfast` 的完整配置后补充对比
- **能量精确计算**：从 MJCF inertia 读取精确转动惯量，替代细杆近似
- **contact/flags 配置**：待 OrcaGym 扩展 `SimConfig` 后补充深度配置演示
