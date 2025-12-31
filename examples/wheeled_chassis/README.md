# Wheeled Chassis 使用指南

轮式底盘仿真示例，支持差速驱动和阿克曼转向两种底盘类型。

## 🚀 基本使用

### 方式 1：使用 OrcaLab 启动（推荐）

在 OrcaLab 中配置了轮式底盘仿真启动项，可以直接使用：

配置位置：`.orcalab/config.toml`

```toml
[[external_programs.programs]]
name = "wheeled_chassis"
display_name = "run_wheeled_chassis"
command = "python"
args = [ "-m", "examples.wheeled_chassis.run_wheeled_chassis",]
description = "启动轮式底盘仿真"
```

在 OrcaLab 中选择 `run_wheeled_chassis` 即可启动轮式底盘仿真。

### 方式 2：命令行启动

从项目根目录运行：

```bash
# 使用默认参数（localhost:50051, agent_name=openloong_gripper_2f85_mobile_base_usda_1）
python examples/wheeled_chassis/run_wheeled_chassis.py

# 或使用模块方式
python -m examples.wheeled_chassis.run_wheeled_chassis

# 带参数运行
python examples/wheeled_chassis/run_wheeled_chassis.py \
    --orcagym_addr localhost:50051 \
    --agent_name openloong_gripper_2f85_mobile_base_usda_1 \
    --env_name WheeledChassis
```

### 命令行参数说明

`run_wheeled_chassis.py` 参数：

- `--orcagym_addr`：OrcaStudio 远程地址（可选，默认：`localhost:50051`）
- `--agent_name`：机器人名称（可选，默认：`openloong_gripper_2f85_mobile_base_usda_1`）
- `--env_name`：环境名称（可选，默认：`WheeledChassis`）

## 📋 支持的底盘类型

### 1. WheeledChassis（差速驱动底盘）

使用 `run_wheeled_chassis.py` 运行差速驱动底盘仿真。

**特点**：
- 左右轮独立控制
- 通过左右轮速度差实现转向
- 适用于移动机器人、AGV 等

**默认机器人**：`openloong_gripper_2f85_mobile_base_usda_1`

### 2. Ackerman（阿克曼转向底盘）

使用 `run_ackerman.py` 运行阿克曼转向底盘仿真。

**特点**：
- 前轮转向，后轮驱动
- 符合汽车转向原理
- 适用于车辆仿真

**默认机器人**：`hummer_h2_usda_1`

**注意**：`run_ackerman.py` 目前使用硬编码参数，如需修改请直接编辑文件中的参数：

```python
orcagym_addr = "localhost:50051"
agent_name = "hummer_h2_usda_1"
env_name = "Ackerman"
```

## 🔧 环境说明

### WheeledChassisEnv

差速驱动底盘环境，主要特性：

- **动作空间**：归一化到 `[-1, 1]` 区间，控制左右轮速度
- **观察空间**：包含关节位置、速度、加速度
- **控制频率**：50 Hz（`TIME_STEP * FRAME_SKIP = 0.001 * 20`）

### AckermanEnv

阿克曼转向底盘环境，主要特性：

- **动作空间**：控制前轮转向和驱动轮速度
- **观察空间**：包含关节位置、速度、加速度
- **控制频率**：50 Hz（`TIME_STEP * FRAME_SKIP = 0.001 * 20`）

## 💡 使用提示

1. **确保 OrcaStudio 正在运行**：默认地址为 `localhost:50051`
2. **在场景中添加机器人**：确保场景中存在对应的机器人预制体
3. **修改机器人名称**：如果场景中的机器人名称与默认值不同，请使用 `--agent_name` 参数指定
4. **键盘控制**：环境支持键盘输入控制（通过 OrcaStudio）

## 🔍 代码结构

```
examples/wheeled_chassis/
├── run_wheeled_chassis.py    # 差速驱动底盘仿真脚本
├── run_ackerman.py           # 阿克曼转向底盘仿真脚本
└── README.md                 # 本文件

envs/wheeled_chassis/
├── wheeled_chassis_env.py    # 差速驱动底盘环境定义
└── ackerman_env.py           # 阿克曼转向底盘环境定义
```

