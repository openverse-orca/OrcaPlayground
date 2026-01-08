# XBot在OrcaGym中运行
参考自http://github.com/roboterax/humanoid-gym.git

## ⚠️ 重要：资产准备

> **📦 相关资产**：https://simassets.orca3d.cn/ **Xbot资产包**
> 
> **🔧 是否需要手动拖动到布局中**：**是**
> 
> **📝 对应模型**：`Xbot_usda`

## ✅ 当前状态
**`run_xbot_orca.py`已经可以在OrcaGym中稳定运行！**

机器人现在可以：
- ✅ 稳定站立
- ✅ 平稳行走
- ✅ 使用humanoid-gym预训练模型
- ✅ 策略文件集成在项目内（config目录）

## 🚀 使用方法

### 方式 1：使用 OrcaLab 启动（推荐）

在 OrcaLab 中配置了 XBot 仿真启动项，可以直接使用：

配置位置：`.orcalab/config.toml`

```toml
[[external_programs.programs]]
name = "xbot_orca"
display_name = "run_xbot_orca"
command = "python"
args = [ "-m", "examples.xbot.run_xbot_orca",]
description = "启动XBot仿真"
```

在 OrcaLab 中选择 `run_xbot_orca` 即可启动 XBot 仿真。

**注意**：使用 OrcaLab 启动时，默认使用 GPU（CUDA）进行推理。如需使用 CPU，请使用命令行方式并添加 `--device cpu` 参数。

### 方式 2：命令行启动

#### 方法 1: 自动运行（固定速度）

```bash
# 使用 GPU（默认）
python examples/xbot/run_xbot_orca.py

# 或使用模块方式
python -m examples.xbot.run_xbot_orca

# 使用 CPU
python examples/xbot/run_xbot_orca.py --device cpu
```

**速度调整**: 编辑`run_xbot_orca.py`第154-156行：
```python
CMD_VX = 0.4   # 前向速度 (m/s)
CMD_VY = 0.0   # 侧向速度 (m/s)
CMD_DYAW = 0.0 # 转向速度 (rad/s)
```

#### 方法 2: 键盘控制（WASD）⭐

```bash
# 使用 GPU（默认）
python examples/xbot/run_xbot_keyboard.py

# 或使用模块方式
python -m examples.xbot.run_xbot_keyboard

# 使用 CPU
python examples/xbot/run_xbot_keyboard.py --device cpu
```

**按键说明**:
- `W` - 前进
- `S` - 后退
- `A` - 左转
- `D` - 右转
- `Q` - 左平移
- `E` - 右平移
- `LShift` - 加速（Turbo模式，2倍速度）
- `Space` - 停止
- `R` - 手动重置环境 ⭐
- `Esc` - 退出程序

**特性**:
- ✅ **不会自动重置**: 即使检测到摔倒或超时，机器人也会继续运行
- ✅ **手动控制**: 只有按R键才会重置环境
- ✅ **实时速度调整**: 按住按键即时响应

## 📊 性能指标

使用humanoid-gym预训练模型：
- Episode长度: 262步 (26.2秒)
- 行走距离: 1.05m
- 平均速度: 0.4 m/s
- 姿态稳定: Roll/Pitch < 5°

## 🔧 核心组件

### 环境
- **`envs/xbot_gym/xbot_simple_env.py`** - XBot环境实现
  - 基于`OrcaGymLocalEnv`
  - 实现PD控制、观察空间、decimation

### 配置和策略
- **`config/policy_example.pt`** - 预训练策略模型 ⭐
  - 来自humanoid-gym项目
  - 已集成在项目内
  - 无需外部依赖

- **`config/xbot_train_config.yaml`** - 训练配置文件

### 运行脚本
- **`run_xbot_orca.py`** - 自动运行脚本（固定速度）
  - 加载预训练策略
  - 设置固定命令速度
  - 实时监控和诊断

- **`run_xbot_keyboard.py`** - 键盘控制脚本 ⭐
  - WASD控制移动方向
  - 实时调整速度
  - 支持Turbo加速模式


## 📝 关键配置

```python
# PD控制参数
kps = [200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15]
kds = [10.0] * 12
tau_limit = 200.0
action_scale = 0.25

# 仿真参数
timestep = 0.001s  # 1ms
decimation = 10    # 策略100Hz
frame_stack = 15   # 观察堆叠
```

## ⚠️ 注意事项

1. **OrcaStudio 必须先启动**：默认地址为 `localhost:50051`
2. **场景中需要添加机器人**：确保场景中存在名为 `XBot-L` 的机器人预制体
3. **初始高度约 0.88m**：OrcaStudio 默认 spawn 高度
4. **设备选择**：
   - 默认使用 GPU（CUDA）进行推理，性能更好
   - 如果没有 GPU 或遇到 CUDA 问题，使用 `--device cpu` 参数
5. **策略文件**：预训练策略文件位于 `examples/xbot/config/policy_example.pt`，已集成在项目内

## 🎯 下一步

- ✅ 核心功能已完成
- ✅ 可以稳定运行
- 📈 如需训练自定义模型，可参考humanoid-gym项目

---

**项目状态**: ✅ 可用
