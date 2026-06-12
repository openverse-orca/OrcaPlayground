# Franka RL - 多机械臂强化学习训练

Franka Panda 多机械臂并行强化学习训练示例，基于 **SB3 + HER** 实现。支持同一 MuJoCo 场景中多台机械臂独立训练，通过局部坐标系隔离实现策略共享。

## ⚠️ 重要：场景与机器人准备

> **📦 相关资产**：https://simassets.orca3d.cn/ **panda 资产包**
>
> **🔧 是否需要手动拖动到布局中**：**是**
>
> **📝 对应模型模板**：`desktop_collaborative_robot_arm`（Franka Panda）
>
> **🧭 UI 资产详情**：请在资产面板中搜索 `desktop_collaborative_robot_arm`，将对应 actor 拖入布局后点击"资产详情"确认路径
>
> **运行方式**：脚本会在启动前扫描场景中的 joint / actuator 后缀，自动识别机器人实例名
>
> **失败行为**：如果关节或驱动器没有完全匹配，会直接报错退出
>
> **📐 多臂布局提示**：机械臂数量较多（≥8）时，部分实例可能被放置在远离原点的位置（x/y 超出 ±8）。此时需在 `franka_config.py` 中扩大 `mocap_pos_range`，否则 mocap 目标被裁剪会导致 weld constraint 失效、机械臂动作异常。当前默认值已设为 `[[-20,20],[-20,20],[0,2]]`，如仍不够可继续扩大。

## 🔧 手动拖入资产进行调试

1. 在资产面板中搜索 `desktop_collaborative_robot_arm`
2. 将 actor 拖入布局，调整到期望的初始位置和朝向
3. 可拖入多台机械臂（最多支持 16 台并行），每台会自动分配独立 agent
4. 选中 actor，打开"资产详情"，确认关节后缀与模板匹配（`joint1`~`joint7`、`finger_joint1`/`finger_joint2` 等）
5. 启动脚本，脚本会扫描并绑定场景中的完整匹配实例

## 🚀 快速开始

### 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# Franka RL 额外依赖（SB3 + sb3-contrib）
pip install stable-baselines3 sb3-contrib
```

### 训练

```bash
# Reach 任务（TQC + HER）
python examples/franka_rl/run_franka_rl.py \
    --config examples/franka_rl/configs/reach_tqc_config.yaml \
    --train

# Pick & Place 任务（SAC + HER）
python examples/franka_rl/run_franka_rl.py \
    --config examples/franka_rl/configs/pick_place_sac_config.yaml \
    --train

```

### 测试

```bash
python examples/franka_rl/run_franka_rl.py \
    --config examples/franka_rl/configs/reach_tqc_config.yaml \
    --test \
    --ckpt /path/to/model.zip
```

## 📁 目录结构

```
examples/franka_rl/
├── README.md                          # 本文件
├── run_franka_rl.py                   # 统一入口（训练/测试）
├── configs/
│   ├── reach_tqc_config.yaml          # Reach 任务 TQC 配置
│   └── pick_place_sac_config.yaml     # Pick&Place 任务 SAC 配置
└── scripts/
    └── sb3_her_rl.py                  # SB3 + HER 训练/测试核心逻辑

envs/franka_rl/
├── franka_gym_env.py                  # Gym 环境（OrcaGymAsyncEnv 子类）
├── franka_agent.py                    # 机器人 Agent（观测/奖励/mocap 控制）
├── franka_config.py                   # 机器人参数与任务配置
└── franka_robot_locator.py            # 场景扫描与机器人发现
```

## 🧭 架构概览

### 训练流水线

```
run_franka_rl.py                         # 统一入口：解析配置、扫描场景
  ├── locate_scene_franka()              # 扫描场景中的 Franka 实例
  └── sb3_her_rl.py
        ├── register_env()               # 注册 Gym 环境
        ├── make_env()                   # 创建 env 工厂函数
        ├── OrcaGymAsyncSubprocVecEnv    # 多进程向量化环境
        │     └── FrankaGymEnv           # Gym 环境
        │           └── FrankaAgent      # 机器人 Agent（每个 agent 独立）
        └── TQC/SAC + HerReplayBuffer    # RL 算法
```

### 多机械臂隔离机制

本示例的核心设计是**多机械臂环境隔离**，使同一场景中的多台机械臂可以独立训练并共享策略：

| 隔离维度 | 实现方式 |
|----------|----------|
| **观测隔离** | 每个 agent 的 obs 转换为**相对于基座的局部坐标**（`ee_position - base_xpos`），不同位置机械臂的 obs 分布相同 |
| **目标隔离** | 每个 agent 独立采样 `desired_goal`，goal 噪声基于自身 EE 位置 |
| **奖励隔离** | 每个 agent 独立计算 `reward = -distance(achieved_goal, desired_goal)` |
| **终止隔离** | 每个 agent 独立判断 `terminated` / `truncated` |
| **重置隔离** | 只有 terminated/truncated 的 agent 才会 reset，不影响其他 agent |

**`OrcaGymAsyncSubprocVecEnv`** 将 `num_envs = n_subenvs × n_agents`，SB3 将每个 agent 视为独立 env，实现完全隔离训练。

### 局部坐标系设计

多台机械臂分布在场景不同位置，世界坐标系下的 obs 值完全不同，无法共享策略。本示例将 obs 转换为**相对于基座（link0）的局部坐标**：

```
世界坐标:  arm_1 EE = [0.614, 0.481, 0.247]   base = [-0.023, 0.481, 0.050]
局部坐标:  arm_1 EE = [0.638, 0.000, 0.197]   ← 所有机械臂的局部 EE 位置一致

世界坐标:  arm_4 EE = [-2.051, -3.967, 0.247]  base = [-2.690, -3.967, 0.050]
局部坐标:  arm_4 EE = [0.638, 0.000, 0.197]   ← 与 arm_1 完全相同
```

转换后的 obs 维度不变，但所有机械臂的 obs 分布相同，**一个策略即可控制所有位置的机械臂**。

### 控制方式：Mocap + PD Position Control

本示例使用 **MuJoCo mocap body + weld constraint** 实现末端执行器的位置控制：

1. **动作空间**：3D 位置增量（reach）或 3D 位置增量 + 1D 夹爪（pick_and_place）
2. **Mocap 驱动**：动作映射为 mocap body 位置偏移，通过 weld constraint 拉动末端执行器
3. **关节控制**：7 个关节使用 `general` 类型 actuator（PD position control），2 个夹爪关节直接控制
4. **重力补偿**：`qfrc_bias` 补偿重力，保持机械臂在零动作时稳定

## 📋 配置文件说明

### Reach 任务（TQC + HER）

```yaml
framework: "sb3"
orcagym_addresses: ["localhost:50051"]
model_name: "panda"
task: "reach"                    # 末端到达目标位置
reward_type: "dense"             # dense: -distance, sparse: 0/1
model_type: "tqc"                # TQC 算法（支持 HER）
training_episode: 100
start_her_episode: 1.0           # 从第 1 个 episode 开始使用 HER

training:
  subenv_num: 1                  # 并行子环境数
  render_mode: "human"           # "human" 可视化, "none" 无渲染
```

### Pick & Place 任务（SAC + HER）

```yaml
task: "pick_and_place"           # 抓取物体放到目标位置
model_type: "sac"                # SAC 算法
training_episode: 200
```

### 关键配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `model_name` | 机器人型号（对应 `FrankaRobotConfig` 键名） | `"panda"` |
| `task` | 任务类型：`"reach"` 或 `"pick_and_place"` | `"reach"` |
| `model_type` | RL 算法：`"tqc"` / `"sac"` / `"ddpg"` / `"ppo"` | `"tqc"` |
| `reward_type` | 奖励类型：`"dense"`（-距离）或 `"sparse"`（0/1） | `"dense"` |
| `training_episode` | 训练 episode 数 | `100` |
| `subenv_num` | 并行子环境数（每个子环境包含场景中所有机械臂） | `1` |
| `render_mode` | 渲染模式：`"human"` 可视化，`"none"` 无渲染 | `"human"` |

## 🦾 支持的任务

### Reach（末端到达）

- **目标**：控制末端执行器到达随机目标位置
- **动作空间**：3D（x/y/z 位置增量），夹爪锁定
- **观测空间**：18 维（EE 位置 3 + EE 速度 3 + 物体位置 3 + 物体旋转 3 + 物体线速度 3 + 物体角速度 3）
- **成功条件**：EE 与目标距离 < 0.05m
- **最大步数**：250 步

### Pick & Place（抓取放置）

- **目标**：抓取物体并放到目标位置
- **动作空间**：4D（x/y/z 位置增量 + 夹爪开合）
- **观测空间**：25 维（EE 位置 3 + EE 速度 3 + 夹爪状态 2 + 物体位置 3 + 物体旋转 3 + 物体线速度 3 + 物体角速度 3 + ...）
- **成功条件**：物体与目标距离 < 0.05m
- **最大步数**：500 步

## 📊 观测空间详解

### Reach 任务（block_gripper=True）

| 维度 | 内容 | 说明 |
|------|------|------|
| 0-2 | `ee_position_local` | 末端执行器位置（相对于基座） |
| 3-5 | `ee_velocity` | 末端执行器速度 |
| 6-8 | `object_position_local` | 物体位置（相对于基座，reach 任务为占位） |
| 9-11 | `object_rotation` | 物体旋转（欧拉角） |
| 12-14 | `object_velp` | 物体线速度 |
| 15-17 | `object_velr` | 物体角速度 |

### HER 目标空间

| 键 | 维度 | 说明 |
|----|------|------|
| `achieved_goal` | 3 | 当前 EE 位置（局部坐标） |
| `desired_goal` | 3 | 目标位置（局部坐标） |

## 🔧 机器人配置

`FrankaRobotConfig["panda"]` 中的关键参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `neutral_joint_values` | `[0, 0.41, 0, -1.85, 0, 2.26, 0.79, 0, 0]` | 初始关节角度（下探抓取姿态） |
| `mocap_pos_range` | `[[-20,20],[-20,20],[0,2]]` | Mocap 位置范围（多臂布局需扩大） |
| `actuator_type` | `"position"` | PD 位置控制 |
| `kps` | `[20, 20, ...]` | 位置增益 |
| `kds` | `[0.5, 0.5, ...]` | 速度增益 |
| `distance_threshold` | `0.05` | 成功判定距离阈值（m） |
| `goal_xy_range` | `0.5` | 目标采样 xy 范围（m） |
| `goal_z_range` | `0.3` | 目标采样 z 范围（m） |

## 🧪 场景扫描与机器人发现

`franka_robot_locator.py` 实现了自动场景扫描：

1. **扫描场景**：通过 `probe_scene_model()` 获取场景中所有关节、驱动器、site、body 名称
2. **模板匹配**：根据 `FrankaRobotConfig` 中定义的关节后缀模板（`joint1`~`joint7`、`finger_joint1` 等）匹配机器人实例
3. **前缀提取**：匹配成功的实例提取前缀作为 agent 名称（如 `desktop_collaborative_robot_arm_1`）
4. **完整性校验**：要求所有关节、驱动器、site、body 都完整匹配，否则报错退出

## 🆘 常见问题

### Q: 机械臂"躺下"或关节脱落

**原因**：MuJoCo XML 中使用了 `motor` 类型 actuator（力矩控制），零控制力无法抵抗重力。

**解决**：确保 XML 使用 `general` 类型 actuator（PD 位置控制），并添加了 mocap weld constraint。当前 `panda_mocap_moto.xml` 已修复。

### Q: 多台机械臂"向外探头"

**原因**：`mocap_pos_range` 太小，导致 mocap 位置被裁剪到错误范围。

**解决**：`mocap_pos_range` 已扩大到 `[[-20,20],[-20,20],[0,2]]`，适配多臂布局。若机械臂超过 8 台或布局范围更大，需继续扩大该范围。

### Q: 所有机械臂爪夹在顶端

**原因**：`neutral_joint_values` 设为全零，导致机械臂竖直向上。

**解决**：已恢复为 `[0, 0.41, 0, -1.85, 0, 2.26, 0.79, 0, 0]`（下探抓取姿态）。

### Q: 场景帧率为 0

**原因**：`render_mode` 设为 `"none"` 时不会同步状态到 OrcaStudio。

**解决**：训练配置中设置 `render_mode: "human"`。

### Q: 物体被甩飞后无法恢复

**原因**：物体初始 `qpos` 未正确保存和恢复。

**解决**：`FrankaGymEnv.__init__()` 中显式查询并保存物体初始 `qpos`，reset 时恢复。

### Q: 训练时如何增加并行环境数？

修改配置文件中的 `subenv_num`。每个子环境会创建独立的 MuJoCo 仿真进程，包含场景中的所有机械臂。总并行数 = `subenv_num × agent_num`。

### Q: 如何添加新的任务？

1. 在 `franka_config.py` 的 `TaskConfig` 中添加新任务配置
2. 在 `franka_agent.py` 中实现对应的 `sample_goal()` 和 `compute_reward()` 逻辑
3. 创建对应的 YAML 配置文件

## 📞 获取帮助

- 查看环境代码：`envs/franka_rl/`
- 查看训练脚本：`examples/franka_rl/scripts/sb3_her_rl.py`
- SB3 HER 文档：https://stable-baselines3.readthedocs.io/en/master/modules/her.html
- MuJoCo Mocap 文档：https://mujoco.readthedocs.io/en/stable/modeling.html#kinematic-trees
