# xbot 从 Local 迁移到 Euler 踩坑实录

> 记录日期：2026-07-21
> 适用版本：OrcaGym Euler 体系（`OrcaGymEulerEnv`）
> 背景：xbot 样例从 Local 体系（`OrcaGymLocalEnv`）迁移到 Euler 体系后，机器人无法站立，持续前倾摔倒

## 一、问题现象

机器人迁移到 Euler 体系后出现以下症状：

- **持续前倾下落**：base_z 从 0.880m 持续降到 0.861m，pitch 从 0° 持续增到 -1.7°
- **力矩饱和**：tau_max 从 20.9 飙到 71.1（tau_limit=200），说明 PD 控制器在拼命纠正但无效
- **几乎不移动**：10 步只移动 3mm
- **轨迹完全复现**：多个 episode 的数据一字不差，说明观测和动作都固定不变
- **10 步就 reset**：episode 提前结束

对比 dev 分支（Local 体系）能正常站立行走，说明问题出在迁移过程。

## 二、根因一：`self.ctrl` 从普通属性变成了 property

### 通俗解释

**Local 体系**下，`self.ctrl` 就是一个普通的 numpy 数组，你可以随便往里面写值：
```python
self.ctrl = np.zeros(12)      # 创建一个数组
self.ctrl[0] = 1.0            # 往数组第 0 位写 1.0  ✓ 正常写入
```

**Euler 体系**下，`self.ctrl` 变成了一个 property（带 getter 和 setter 的"假属性"）：
- **getter**（读取时触发）：返回 `actuator_force`（MuJoCo 计算出的实际力矩输出，**只读**）
- **setter**（赋值时触发）：调用 `set_ctrl()` 写入 `mjData.ctrl`

问题出在"索引赋值"这种写法：
```python
self.ctrl[0] = 1.0   # 实际等价于 (self.ctrl.__getitem__)[0] = 1.0
                     # 触发 getter 返回 actuator_force，再往 actuator_force 写 1.0
                     # mjData.ctrl 完全没被写入！
```

### 失效链路

xbot 的 PD 控制循环原本这么写：
```python
self.ctrl.fill(0.0)                          # ① 触发 getter → 返回 actuator_force → fill 无效
self.ctrl[actuator_idx] = tau[i]             # ② 触发 getter → 写到 actuator_force（错误位置）
self.do_simulation(self.ctrl, 1)             # ③ 触发 getter → 传入 actuator_force（错误值）
```

`do_simulation` 收到的是 `actuator_force`（上一步的力矩输出或 0），而不是 PD 控制器刚算出来的 tau。机器人的关节实际收到的控制信号和策略意图完全脱节，自然站不稳。

### 为什么 `self.ctrl = np.zeros(...)` 看起来正常？

因为这是**整体赋值**，触发 setter，调用 `set_ctrl()` 正确写入 `mjData.ctrl`。问题只出现在 `self.ctrl[i] = ...` 这种**索引赋值**上（触发 getter）。

### 修复方法

用本地变量做缓冲区，避免触发 property getter：
```python
ctrl_buf = np.zeros(self.nu, dtype=np.float32)
ctrl_buf[actuator_idx] = tau[i]              # 本地数组，正常写入
self.do_simulation(ctrl_buf, 1)              # 传入正确填充的数组
```

## 三、根因二：Euler 没有 `self.time_step` 属性

### 通俗解释

**Local 体系**在 `__init__` 里设置了 `self.time_step = time_step`（保存物理步长）。
**Euler 体系**的 `set_time_step` 方法只设置了 `self.realtime_step`，**忘了设置 `self.time_step`**。

xbot 在计算角速度（omega）时用到了这个属性：
```python
dt = self.time_step * self.frame_skip   # Euler 下抛 AttributeError！
omega = delta_euler / dt
```

### 为什么没报错？

因为这段代码被一个"大杂烩" try/except 包住了：
```python
try:
    # ... 一堆代码，包括上面的 dt 计算
    omega = delta_euler / dt
except:
    pass   # 吞掉所有异常，omega 保持初始值 0
```

**结果**：omega 始终为 0，策略收到的角速度观测全是 0，**完全无法感知自己正在前倾**，自然无法做出纠正动作。

### 为什么策略会"瞎"了还能力矩饱和？

因为 PD 控制器是直接基于关节位置（q）算 tau 的，不依赖 omega。策略输出动作 → 转成 target_q → PD 算 tau → 试图让关节到位。但策略本身拿不到正确的姿态信息（omega=0），输出的动作方向是错的，PD 控制器忠实地执行错误命令，力矩越来越大（饱和），机器人在重力作用下持续前倾。

### 修复方法

用 `self.dt`（Euler 提供的 property）替代 `self.time_step * self.frame_skip`：
```python
# self.dt = sim_config.timestep * frame_skip = 0.001 * 10 = 0.01s
dt = self.dt
```

## 四、两个根因的叠加效应

| 根因 | 影响 | 后果 |
|------|------|------|
| ctrl property | PD 控制器的 tau 写不进 mjData.ctrl | 关节实际收到的控制信号错误 |
| omega=0 | 策略观测里的角速度永远是 0 | 策略无法感知姿态变化，输出错误动作 |

两个问题同时存在，机器人既"听不到指令"（PD tau 失效），又"看不到姿态"（omega=0），自然完全无法保持平衡。

## 五、Local → Euler 迁移易踩坑清单

### 坑 1：`self.ctrl` 索引赋值失效

| 体系 | `self.ctrl` 本质 | `self.ctrl[i] = x` |
|------|------------------|---------------------|
| Local | 普通 ndarray 实例属性 | 正常写入 ✓ |
| Euler | property（getter 返回 actuator_force） | 写到错误位置 ✗ |

**规避**：Euler 体系下用本地缓冲区 + `do_simulation(ctrl_buf)`，不要用 `self.ctrl[i] = x`。

### 坑 2：`self.time_step` 属性缺失

| 体系 | `self.time_step` | `self.realtime_step` | `self.dt` |
|------|-------------------|----------------------|-----------|
| Local | ✓ 存在（= time_step） | ✓ 存在 | ✗ 不存在 |
| Euler | ✗ 不存在 | ✓ 存在 | ✓ 存在（property） |

**规避**：Euler 体系下用 `self.dt` 获取控制周期，不要用 `self.time_step`。

### 坑 3：`get_body_xpos_xmat_xquat` 返回类型不同

| 体系 | 返回类型 | 取值方式 |
|------|----------|----------|
| Local | 元组 `(xpos, xmat, xquat)` | `xpos, xmat, xquat = self.get_body_xpos_xmat_xquat([name])` |
| Euler | 字典 `{name: {"xpos", "xmat", "xquat"}}` | `pose = self.get_body_xpos_xmat_xquat([name])[name]; xpos = pose["xpos"]` |

**规避**：迁移时改成字典取值方式。

### 坑 4：`self.gym` 从公共属性变成私有属性

| 体系 | 访问方式 |
|------|----------|
| Local | `self.gym.xxx()`（公共属性） |
| Euler | `self.xxx()`（公共方法委托），禁止 `self._gym.xxx()` 穿墙 |

**规避**：所有 `self.gym.xxx()` 改为 `self.xxx()`（如果 Euler 提供了对应公共方法）。

### 坑 5：`update_data` 在 Euler 无等价方法

| 体系 | 仿真步进后的状态同步 |
|------|---------------------|
| Local | `self.gym.mj_step(n)` + `self.gym.update_data()` 手动同步 |
| Euler | `do_simulation()` 内部自动调用 `sync_to_view()`，无需手动同步 |

**规避**：Euler 体系下移除所有 `self.gym.update_data()` 调用。

### 坑 6：`set_joint_qpos` 参数类型不同

| 体系 | 参数类型 |
|------|----------|
| Local | `dict[joint_name -> qpos_array]` |
| Euler | 完整 qpos 数组 `np.ndarray` |

**规避**：Euler 体系下传整个 qpos 数组，不要传字典。

## 六、调试经验

### 1. 多 episode 轨迹完全相同 = 观测固定

如果多个 episode 的数据一字不差，通常意味着：
- reset 没有随机化（正常）
- 策略是确定性的（正常）
- **观测计算有 bug，某些维度永远是 0 或固定值**（异常，需排查）

本次问题中 omega 永远是 0 就是这种情况。

### 2. try/except: pass 是隐形杀手

xbot 的观测构建函数里有多个 `try/except: pass`，会吞掉所有异常。这种写法在迁移时极其危险，因为：
- 属性缺失（AttributeError）被吞掉
- 类型错误（TypeError）被吞掉
- 索引错误（IndexError）被吞掉
- 看起来"运行正常"，实际上观测是错的

**建议**：迁移完成后把所有 `except: pass` 改成 `except Exception as e: _logger.warning(f"... {e}")`，至少记录下来。

### 3. 力矩饱和 ≠ 控制生效

看到 tau_max 很大（饱和）不能说明"控制是有效的"，只能说明"PD 控制器在拼命算 tau"。要验证 tau 是否真的作用到了关节，可以：
- 检查 `mjData.ctrl` 是否等于 PD 算出的 tau
- 检查 `actuator_force` 是否合理
- 观察关节实际运动是否符合预期

## 七、修复验证清单

- [x] ruff SLF001 静态检查通过
- [ ] OrcaStudio 实际运行验证步态恢复（用户侧）
- [ ] 跑 100+ 步不摔倒
- [ ] base_z 保持稳定（不再持续下降）
- [ ] pitch 保持小角度（不再持续前倾）
- [ ] tau_max 不饱和（< 100）

## 八、相关文件

- 修复文件：`examples/xbot/xbot_simple_env.py`
- 详细修复记录：`.trae_history/9.BUGFIX_Euler_ctrl_property导致PD控制失效.md`
- Euler 架构文档：`../OrcaGym/docs/design/architecture/orca_gym_euler_architecture.md`
- 其他迁移问题汇总：`docs/orcagym_issues.md`
- 整体迁移评估：`docs/euler_migration_assessment.md`
