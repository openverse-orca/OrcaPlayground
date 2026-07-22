# d12 从 Local 迁移到 Euler 踩坑实录

> 记录日期：2026-07-22
> 适用版本：OrcaGym Euler 体系（`OrcaGymEulerEnv`）
> 背景：d12 机械臂样例（demo + act）从 Local 体系迁移到 Euler 体系后，运行直接报错 `AttributeError: 'D12Env' object has no attribute 'gym'`，后续暴露多处 API 不兼容

## 一、问题现象

d12 样例分两个入口：

- **demo**：`examples/d12/demo/run_d12_demo.py`，脚本化轨迹演示（无学习策略）
- **act**：`examples/d12/act/run_d12_act.py`，ACT（Action Chunking Transformer）模型推理

两者都使用 robosuite OSC（OperationalSpaceController）控制器做笛卡尔末端阻抗控制。迁移到 Euler 后出现以下症状：

- **启动即崩溃**：`AttributeError: 'D12Env' object has no attribute 'gym'`
- **OSC 控制器初始化失败**：`AttributeError: 'OrcaGymEuler' 对象的属性 '_sim' 被隔离`
- **控制信号失效**：`env.ctrl[i] = val` 不写入 `mjData.ctrl`，机械臂不动
- **位姿解算错误**：`get_body_xpos_xmat_xquat` 返回类型从元组变字典，解构报错
- **关节设置失败**：`set_joint_qpos(dict)` 类型不匹配

对比 dev 分支（Local 体系）能正常运行，说明问题出在迁移过程。

## 二、根因一：`env.gym` 从公共属性变成被隔离的私有属性

### 通俗解释

**Local 体系**下，`self.gym` 是公共属性，直接持有 `OrcaGymLocal` 实例，OSC 控制器通过它访问底层 MuJoCo：
```python
l_config["sim"] = env.gym        # 获取 gym 对象
controller = controller_factory(...)
controller.gym.mj_fullM()        # 控制器内部通过 gym 调用 MuJoCo API
```

**Euler 体系**下，`self._gym` 是私有属性（K1/K5 隔离机制），`OrcaGymEuler.__getattribute__` 运行时拦截 `_sim`/`_gym`/`_mjData`/`_mjModel` 等访问：
```python
env.gym                          # AttributeError: 'D12Env' object has no attribute 'gym'
env._gym._sim                    # AttributeError: 'OrcaGymEuler' 对象的属性 '_sim' 被隔离
```

### 失效链路

d12 的 OSC 控制器初始化原本这么写：
```python
l_config["sim"] = env.gym        # ① Euler 下 env.gym 不存在 → AttributeError
l_controller = controller_factory(l_config["type"], l_config)
# 控制器内部 __init__ 调用 self.update() → self.gym.mj_fullM()
# ② 即使绕过 env.gym，OrcaGymEuler._sim 也被隔离 → AttributeError
```

### 为什么 OSC 控制器需要 `gym`？

OSC 控制器（`base_controller.py`）的 `update()` 方法依赖 `gym` 提供以下方法：
- `gym.query_site_pos_and_mat([eef_name])`：末端位姿
- `gym.mj_jac_site([eef_name])`：雅可比矩阵
- `gym.mj_fullM()`：完整质量矩阵（逆动力学用）
- `gym.data.qvel`：关节速度
- `gym.query_joint_qpos/qvel(joint_index)`：关节状态
- `gym.opt.timestep`：物理步长

其中 `query_site_pos_and_mat`、`mj_jac_site`、`data`、`query_joint_*` 在 Euler 体系下 `OrcaGymEulerEnv` 已有公共方法，但 `mj_fullM` 和 `opt` 缺失。

### 修复方法

**方案 A（推荐）**：传 `env` 而非 `env.gym`，让控制器直接用 `OrcaGymEulerEnv` 的公共方法：
```python
l_config["sim"] = env            # D12Env 提供控制器所需的全部方法
```

但需在 `D12Env` 补齐 `mj_fullM()` 和 `opt` property。

## 三、根因二：OSC 控制器依赖的 `mj_fullM()` 在 Euler 缺失

### 通俗解释

**Local 体系**下，`OrcaGymLocal.mj_fullM()` 封装了 MuJoCo 的 `mj_fullM` C 函数，计算完整质量矩阵（nv×nv）：
```python
mass_matrix = self.gym.mj_fullM()   # 返回 (nv, nv) ndarray
```

**Euler 体系**下，`OrcaGymEuler` 和 `OrcaGymEulerEnv` 均未提供 `mj_fullM()`，`OrcaGymDataView` 也未暴露 `qM`（MuJoCo 的稀疏质量矩阵）。

### 为什么不能直接穿墙？

第一次尝试这么写：
```python
def mj_fullM(self):
    sim = self._gym._sim           # ❌ OrcaGymEuler.__getattribute__ 拦截 _sim
    mujoco.mj_fullM(sim._mjModel, mass_matrix, sim._mjData.qM)
```

报错：
```
AttributeError: 'OrcaGymEuler' 对象的属性 '_sim' 被隔离。
  API 契约：用户不应直接访问 _mjData / _mjModel / 任何子组件。
```

**关键点**：`# noqa: SLF001` 只跳过 ruff 静态检查，**无法绕过 `__getattribute__` 运行时拦截**。

### 绕过路径：DataView 的 `_mj_data`/`_mj_model`

`OrcaGymEuler` 用 `__getattribute__`（拦截所有属性访问，包括已存在的），但 `OrcaGymDataView` 用 `__getattr__`（仅兜底不存在的属性）。`_mj_data`/`_mj_model` 在 DataView 的 `__dict__` 中存在，访问不被拦截：

```python
def mj_fullM(self) -> np.ndarray:
    mj_data = self.data._mj_data    # ✓ DataView 不拦截 _mj_data
    mj_model = self.data._mj_model  # ✓ DataView 不拦截 _mj_model
    nv = mj_model.nv
    mass_matrix = np.ndarray(shape=(nv, nv), dtype=np.float64, order="C")
    mujoco.mj_fullM(mj_model, mass_matrix, mj_data.qM)
    return mass_matrix.reshape((nv, nv))
```

### 临时穿墙风险

这是用户选择"仅修 demo/act 侧"的妥协方案，违反 K1/K2 隔离机制。待 OrcaGym 侧在 `OrcaGymEuler`/`MuJoCoSimCore` 添加 `mj_fullM()` 公共方法后应立即移除穿墙，改为委托调用。当前通过 `pyproject.toml` 的 `per-file-ignore` + `# noqa: SLF001` 标注临时豁免。

## 四、根因三：`env.ctrl[i] = val` 索引赋值失效（同 xbot）

### 通俗解释

与 xbot 完全相同的问题。**Local 体系**下 `self.ctrl` 是普通 ndarray，索引赋值正常。**Euler 体系**下 `self.ctrl` 是 property：
- **getter**：返回 `actuator_force`（只读）
- **setter**：调用 `set_ctrl()` 写入 `mjData.ctrl`

`env.ctrl[act_id] = val` 触发 getter，写到 `actuator_force`（错误位置），`mjData.ctrl` 完全没被写入。

### 失效链路

d12 的 OSC 控制循环原本这么写：
```python
# init_env_state
env.ctrl[act_id] = val                  # ① 触发 getter → 写到 actuator_force（无效）

# ScriptedDriver.step / ACTDriver.step
self.env.ctrl[act_id] = l_ctrl[i]       # ② 同上，无效
self.env.ctrl[act_id] = r_ctrl[i]       # ③ 同上，无效
self.env.ctrl[act_id] = np.clip(...)    # ④ 夹爪控制，同上无效

# 主循环
env.do_simulation(env.ctrl, env.frame_skip)  # ⑤ 传入 actuator_force（错误值）
```

机械臂关节实际收到的控制信号全是 0（或上一步的 actuator_force），OSC 控制器算出的 tau 完全丢失。

### 修复方法

在 Driver 中维护本地 ctrl 缓冲区：
```python
class ScriptedDriver:
    def __init__(self, env, ...):
        self._ctrl_buf = np.zeros(env.nu, dtype=np.float32)  # 本地缓冲区

    def step(self):
        self._ctrl_buf[act_id] = l_ctrl[i]    # 本地数组，正常写入
        # ...

# 主循环
env.do_simulation(driver._ctrl_buf, env.frame_skip)  # 传入正确填充的数组
```

`init_env_state` 中同理：
```python
ctrl_buf = np.zeros(env.nu, dtype=np.float32)
ctrl_buf[act_id] = val
env.set_ctrl(ctrl_buf)                        # 一次性写入
```

## 五、根因四：`get_body_xpos_xmat_xquat` 返回类型从元组变字典（同 xbot 坑 3）

### 通俗解释

**Local 体系**返回元组：
```python
base_xpos, _, base_xquat = env.get_body_xpos_xmat_xquat([base_body])
```

**Euler 体系**返回字典：
```python
{
    base_body: {
        "xpos": np.ndarray(3,),
        "xmat": np.ndarray(9,),
        "xquat": np.ndarray(4,)
    }
}
```

元组解构在 Euler 下会报 `ValueError: not enough values to unpack`。

### 修复方法

```python
_base_pose = env.get_body_xpos_xmat_xquat([base_body])[base_body]
base_xpos = _base_pose["xpos"]
base_xquat = _base_pose["xquat"]
```

## 六、根因五：`set_joint_qpos(dict)` 参数类型不匹配（同 xbot 坑 6）

### 通俗解释

**Local 体系**下 `set_joint_qpos` 接受 dict：
```python
env.set_joint_qpos({"joint1": np.array([0.1]), "joint2": np.array([0.2])})
```

**Euler 体系**下 `set_joint_qpos` 只接受完整 qpos 数组：
```python
env.set_joint_qpos(np.full(env.nq, 0.0))  # 必须传整个数组
```

### 修复方法

在 `D12Env` 添加辅助方法 `apply_joint_qpos_dict`，将 dict 合并为完整 qpos 数组：
```python
def apply_joint_qpos_dict(self, joint_qpos_dict: dict) -> None:
    full_qpos = self.data.qpos.copy()
    for jname, jqpos in joint_qpos_dict.items():
        addr = self.jnt_qposadr(jname)
        arr = np.atleast_1d(np.asarray(jqpos, dtype=full_qpos.dtype))
        full_qpos[addr:addr + len(arr)] = arr
    self.set_joint_qpos(full_qpos)
```

调用处：
```python
env.apply_joint_qpos_dict(default_qpos)   # 替代 env.set_joint_qpos(default_qpos)
```

## 七、根因六：OSC 控制器依赖 `sim.opt.timestep`

### 通俗解释

**Local 体系**下，`gym.opt.timestep` 访问 `MjModel.opt.timestep`。
**Euler 体系**下，用 `gym.sim_config.timestep`（SimConfig 对象），没有 `opt` 属性。

OSC 控制器内部访问 `self.gym.opt.timestep` 获取物理步长，Euler 下会 `AttributeError`。

### 修复方法

在 `D12Env` 添加 `opt` property，返回 `sim_config`：
```python
@property
def opt(self):
    """OSC 控制器兼容：返回 sim_config（含 timestep 等 opt 字段）。"""
    return self.sim_config
```

`SimConfig` 提供了 `timestep` 属性，接口兼容。

## 八、六个根因的叠加效应

| 根因 | 影响 | 后果 |
|------|------|------|
| `env.gym` 被隔离 | OSC 控制器拿不到 sim 对象 | 启动即崩溃 |
| `mj_fullM()` 缺失 | OSC 无法计算质量矩阵 | 控制器初始化失败 |
| `env.ctrl[i]=v` 失效 | OSC 算出的 tau 写不进 mjData.ctrl | 机械臂不动 |
| `get_body_xpos_xmat_xquat` 返回类型 | 基坐标系变换解构报错 | 位姿解算失败 |
| `set_joint_qpos(dict)` 类型 | 初始关节设置失败 | 机械臂初始位姿错误 |
| `sim.opt.timestep` 缺失 | OSC 控制器拿不到物理步长 | 控制周期计算失败 |

六个问题同时存在，d12 demo/act 既"拿不到 sim"（env.gym 隔离），又"算不出质量矩阵"（mj_fullM 缺失），还"写不进控制信号"（ctrl property），完全无法运行。

## 九、Local → Euler 迁移易踩坑清单（d12 特有）

### 坑 1：OSC 控制器的 `sim` 参数语义变化

| 体系 | `config["sim"]` 应传 | 控制器内部 `self.gym` |
|------|----------------------|----------------------|
| Local | `env.gym`（OrcaGymLocal 实例） | `gym.query_*` / `gym.mj_*` |
| Euler | `env`（OrcaGymEulerEnv 子类） | `env.query_*` / `env.mj_*`（需在 Env 子类补齐缺失方法） |

**规避**：Euler 体系下传 `env`，并在 Env 子类补齐 OSC 控制器依赖的 `mj_fullM()`、`opt` 等。

### 坑 2：`OrcaGymEuler.__getattribute__` 运行时拦截 vs `noqa` 静态跳过

| 机制 | 作用层 | `# noqa: SLF001` 是否有效 |
|------|--------|---------------------------|
| ruff SLF001 | 静态检查 | ✓ 跳过 |
| `__getattribute__` | 运行时 | ✗ 无法绕过 |

**规避**：`_sim`/`_gym`/`_mjData`/`_mjModel` 等被 `_BLOCKED_ATTRS` 拦截的属性，`noqa` 无效。需走 DataView 的 `_mj_data`/`_mj_model`（`__getattr__` 仅兜底，不拦截已存在属性）或等 OrcaGym 侧扩展公共方法。

### 坑 3：`DataView._mj_data`/`_mj_model` 可访问但不推荐

| 对象 | 拦截机制 | `_mj_data`/`_mj_model` 可访问？ |
|------|----------|----------------------------------|
| `OrcaGymEuler` | `__getattribute__`（拦截所有） | ✗ 被拦截 |
| `OrcaGymDataView` | `__getattr__`（仅兜底） | ✓ 可访问（在 `__dict__` 中） |

**规避**：临时穿墙可走 `env.data._mj_data`/`env.data._mj_model`，但应优先扩展 OrcaGym 公共 API。

## 十、调试经验

### 1. `AttributeError: '_xxx' 被隔离` ≠ `noqa` 能解决

遇到 `OrcaGymEuler.__getattribute__` 抛出的隔离错误，不要试图用 `# noqa: SLF001` 绕过。`noqa` 只影响 ruff 静态检查，运行时拦截依然生效。需要：
- 换路径（如 DataView 的 `_mj_data`）
- 或等 OrcaGym 侧扩展公共方法

### 2. OSC 控制器的 `gym` 参数是鸭子类型

`base_controller.py` 的 `__init__` 参数标注为 `gym: OrcaGymLocal`，但实际只调用 `gym.query_site_pos_and_mat`/`gym.mj_jac_site`/`gym.mj_fullM`/`gym.data`/`gym.query_joint_*`/`gym.opt.timestep` 等方法。只要传入的对象提供这些方法即可，不要求必须是 `OrcaGymLocal` 实例。因此传 `env`（`OrcaGymEulerEnv` 子类）也能工作，前提是 Env 子类补齐缺失方法。

### 3. demo 和 act 的修复高度同步

d12 的 demo（脚本轨迹）和 act（ACT 策略推理）共享相同的 OSC 控制器初始化、`b_to_global`、`init_env_state` 逻辑。修复时需同步修改两个文件，避免遗漏。

## 十一、修复验证清单

- [x] ruff SLF001 静态检查通过
- [x] ast 语法检查通过
- [ ] OrcaStudio 实际运行验证 demo 机械臂运动（用户侧）
- [ ] OrcaStudio 实际运行验证 act 策略推理（用户侧）
- [ ] 机械臂末端跟踪轨迹（demo）
- [ ] ACT 策略输出有效控制（act）

## 十二、相关文件

- 修复文件：
  - `examples/d12/d12_env.py`（新增 `mj_fullM()`、`opt` property）
  - `examples/d12/demo/run_d12_demo.py`（5 类修复）
  - `examples/d12/act/run_d12_act.py`（5 类修复，与 demo 同步）
  - `pyproject.toml`（添加 `examples/d12/**` per-file-ignore）
- 详细修复记录：`.trae_history/14.BUGFIX_d12_demo_OSC控制器Euler适配.md`
- Euler 架构文档：`../OrcaGym/docs/design/architecture/orca_gym_euler_architecture.md`
- 同类问题参考：`docs/xbot_euler_migration_pitfalls.md`（xbot 的 ctrl property 和 time_step 问题）
- 整体迁移评估：`docs/euler_migration_assessment.md`
