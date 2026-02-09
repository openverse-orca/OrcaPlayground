本示例源项目地址：git@github.com:LeCAR-Lab/ASAP.git
只移植了由 isaac 训练的模型策略的运行代码。

## ⚠️ 资产准备

- **资产**：位于 **OrcaPlaygroundAssets**，导入后预制体为 `g1_29dof_old_usda`（源文件在 `robots/g1/`）。策略 ONNX 与运行配置仍从本示例目录加载。
- **是否需要手动拖动到布局中**：**否**（脚本会通过 spawn/replicator 自动创建场景）。

## 🔧 手动拖入资产进行调试

手动拖动资产的操作方式与命名建议见**项目根目录 [README - 手动拖动资产（调试时）](../../README.md#-手动拖动资产调试时)**。

**本示例修改前样例代码（手动拖入时，不调用 spawn）**：

```python
# 不调用 publish_g1_scene(...)，依赖场景中已存在对应名称的 actor
agent_name = "g1"
env_id, kwargs = register_env(orcagym_addr, env_name, 0, agent_name, sys.maxsize)
env = gym.make(env_id)
```

---

相关操作说明如下：

## 运行

```bash
python examples/g1/run_g1_sim.py
```

## 键盘控制

### 策略控制
| 按键 | 功能 |
|------|------|
| `]` | 启用策略控制（站立） |
| `o` | 停止策略控制 |
| `i` | 重置到初始状态 |
| `=` | 切换站立/行走模式 |

### Mimic 动作
| 按键 | 功能 |
|------|------|
| `[` | 执行/取消 Mimic 动作 |
| `;` | 切换到下一个 Mimic 动作 |
| `'` | 切换到上一个 Mimic 动作 |

### 移动控制（行走模式下）
| 按键 | 功能 |
|------|------|
| `w/s` | 前进/后退 |
| `a/d` | 左移/右移 |
| `q/e` | 左转/右转 |
| `z` | 停止移动 |

### 其他
| 按键 | 功能 |
|------|------|
| `1/2` | 增加/减少站立高度 |
| `4/5/6/7` | 调整 Kp 增益 |
| `0` | 重置 Kp 增益 |
