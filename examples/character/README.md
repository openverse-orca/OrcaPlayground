# Character 使用指南

角色动画仿真示例，支持键盘控制和路径点控制。

## ⚠️ 重要：资产准备

> **📦 相关资产**：https://simassets.orca3d.cn/ **Remy资产包**
> 
> ** 您可以订阅Remy在已经打开的布局拖动Remy进去进行使用 **
> 
> **🔧 是否需要手动拖动到布局中**：**是**
> 
> **📝 run_character.py 对应模型模板**：`remy`
>
> **运行方式**：脚本会在启动前扫描场景中的 `Animation` body 和控制关节后缀，自动识别角色实例名

## 🚀 基本使用

### 方式 1：使用 OrcaLab 启动（推荐）

在 OrcaLab 中配置了角色仿真启动项，可以直接使用：

配置位置：`.orcalab/config.toml`

```toml
[[external_programs.programs]]
name = "character"
display_name = "run_character"
command = "python"
args = [ "-m", "examples.character.run_character",]
description = "启动动画仿真"
```

在 OrcaLab 中选择 `run_character` 即可启动角色仿真。

### 方式 2：命令行启动

从项目根目录运行：

```bash
# 使用默认参数（脚本会自动扫描场景中的角色实例名）
python examples/character/run_character.py

# 或使用模块方式
python -m examples.character.run_character

# 带参数运行
python examples/character/run_character.py \
    --orcagym_addr localhost:50051 \
    --env_name Character
```

### 命令行参数说明

- `--orcagym_addr`：OrcaStudio 远程地址（可选，默认：`localhost:50051`）
- `--env_name`：环境名称（可选，默认：`Character`）

### 前置准备

#### 在关卡中添加 Remy.prefab

1. 引擎已经内置，在实例化预制体菜单项，直接联想就可以找到
2. 拖动 Remy 到需要的位置
3. 不需要保持默认名称，脚本会在启动时自动扫描角色实例并绑定

## 📋 配置说明

配置文件位置：`envs/character/character_config/remy.yaml`

### 控制方式

支持两种控制方式，可以通过按键切换：

1. **keyboard（键盘控制）**

   - `W/A/S/D` 控制角色移动
   - 可以在配置文件中修改按键映射
2. **waypoint（路径点控制）**（默认）

   - 角色按照预设路径点移动
   - 可以修改路径点坐标（相对于角色初始位置）
   - 可以配置路径点之间的距离阈值、角度阈值、站立等待时间

### 配置文件结构

```yaml
asset_path: "assets/prefabs/Remy"  # 角色资产路径
body_name: "Animation"              # 身体名称
joint_names: {...}                  # 关节名称映射
speed:                              # 移动速度配置
  Forward: 1.7
  Backward: -1.2
  TurnLeft: 1.5707963
  TurnRight: -1.5707963
  Acceleration: 0.1

control_type:                       # 控制方式配置
  active_type: "keyboard"           # 默认激活的控制方式
  switch_key:                       # 切换按键
    waypoint: "1"
    keyboard: "2"

keyboard_control:                   # 键盘控制按键映射
  move_forward: "W"
  move_backward: "S"
  turn_left: "A"
  turn_right: "D"

waypoint_control: [                 # 路径点列表
  {"Coordinates" : [0, 5], "Idle Action" : "Standing", "Duration" : 5.0},
  {"Coordinates" : [5, 5], "Idle Action" : "Standing", "Duration" : 5.0},
  ...
]

waypoint_distance_threshold: 0.2    # 路径点距离阈值
waypoint_angle_threshold: 5         # 路径点角度阈值（度）
```

### 修改控制方式

1. **修改默认控制方式**：在配置文件中修改 `control_type.active_type` 的值（`"keyboard"` 或 `"waypoint"`）
2. **修改切换按键**：在配置文件中修改 `control_type.switch_key` 的按键映射
3. **修改键盘按键**：在 `keyboard_control` 部分修改按键映射
4. **修改路径点**：在 `waypoint_control` 列表中修改路径点坐标和行为

## 🔧 在自己的 Python 程序中添加 Remy

Character 需要用到 `SenenRuntime` 类，用来向 Orca 发送动画指令，因此需要在 env 中添加 `SenenRuntime` 类的回调入口。

### 步骤

1. **添加 `set_scene_runtime` 方法**

   - 参考 `envs/character/character_env.py` 中定义的 `set_scene_runtime` 方法
   - 在你的 env 中添加一个 `set_scene_runtime` 方法
2. **在程序中调用 `set_scene_runtime`**

   - 参考 `run_character.py` 中定义的 `run_simulation` 方法
   - 在你的程序中调用 `set_scene_runtime` 方法，将 `SenenRuntime` 类的实例传入
   - **注意**：这一步需要在 `env.reset()` 之前完成
3. **在 env 中集成 Character**

   - 参考 `envs/character/character_env.py`
   - 在你的 env 中添加 Remy 的实例
   - 在 `step()` 函数调用 Character 的 `on_step()` 方法
   - 在 `reset()` 函数调用 Character 的 `on_reset()` 方法（在 `character_env.py` 中是 `reset_model`）
