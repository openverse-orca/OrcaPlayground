# Replicator 使用指南

场景复制示例，演示如何使用 OrcaGym 的场景管理功能创建和配置 Actor、Camera 和 Light。

## ⚠️ 重要：资产准备

> **📦 相关资产**：https://simassets.orca3d.cn/ **OrcaPlaygroundAssets资产包**
> 
> **🔧 是否需要手动拖动到布局中**：**否**（脚本会自动创建场景）

## 🚀 基本使用

### 方式 1：使用 OrcaLab 启动（推荐）

在 OrcaLab 中可以配置场景复制示例启动项，可以直接使用：

配置位置：`.orcalab/config.toml`

```toml
[[external_programs.programs]]
name = "replicator_actors"
display_name = "run_actors"
command = "python"
args = ["-m", "examples.replicator.run_actors"]
description = "启动 Actor 复制示例"
```

在 OrcaLab 中选择对应的外部程序即可启动。

### 方式 2：命令行启动

从项目根目录运行：

```bash
# Actor 复制示例（生成多个杯子、桌子、购物车等）
python -m examples.replicator.run_actors

# 或直接运行
python examples/replicator/run_actors.py

# Camera 复制示例
python -m examples.replicator.run_cameras

# Light 复制示例
python -m examples.replicator.run_lights
```

## 📋 示例说明

### 1. run_actors.py - Actor 复制示例

演示如何创建和配置多个 Actor（物体）：

**功能**：
- 自动清空场景，避免名称冲突
- 创建一个原始红色杯子（固定位置）
- 创建 10 个随机颜色和尺寸的杯子（随机位置和旋转）
- 创建一个购物车（cart_basket）
- 创建一个办公桌（office_desk）
- 为随机杯子设置随机材质颜色

**使用的资产**：
- `cup_of_coffee_usda` - 咖啡杯
- `cart_basket_usda` - 购物车
- `office_desk_7_mb_usda` - 办公桌

**运行效果**：
- 场景中会出现 1 个原始杯子、10 个随机杯子、1 个购物车和 1 个办公桌
- 随机杯子具有不同的颜色和尺寸

### 2. run_cameras.py - Camera 复制示例

演示如何创建和配置相机：

**功能**：
- 创建一个默认相机（default_camera）
- 设置相机为活动视口
- 可以扩展添加多个相机（代码中已注释示例）

**使用的资产**：
- `cameraviewport_mujoco` - MuJoCo 相机视口

**运行效果**：
- 场景中会出现一个相机，并自动设置为活动视口

### 3. run_lights.py - Light 复制示例

演示如何创建和配置灯光：

**功能**：
- 创建一个原始红色杯子
- 创建一个办公桌
- 创建一个默认相机
- 创建 10 个随机颜色、尺寸和强度的聚光灯（spotlight）
- 为随机灯光设置随机颜色和强度

**使用的资产**：
- `cup_of_coffee_usda` - 咖啡杯
- `office_desk_7_mb_usda` - 办公桌
- `cameraviewport` - 相机视口
- `spotlight` - 聚光灯

**运行效果**：
- 场景中会出现杯子、桌子、相机和 10 个随机灯光
- 灯光具有不同的颜色、位置和强度

## 🔧 代码结构

```
examples/replicator/
├── run_actors.py          # Actor 复制示例
├── run_cameras.py         # Camera 复制示例
├── run_lights.py          # Light 复制示例
├── run_simulation.py      # 通用仿真运行函数
├── actors_env.py          # Actors 环境定义
├── cameras_env.py         # Cameras 环境定义
├── lights_env.py          # Lights 环境定义
└── README.md              # 本文件
```

### 核心函数

所有示例都使用 `run_simulation.py` 中的通用函数：

```python
def run_simulation(
    orcagym_addr: str,
    agent_name: str,
    env_name: str,
    scene_runtime: Optional[OrcaGymSceneRuntime] = None
) -> None
```

**参数说明**：
- `orcagym_addr`：OrcaStudio 地址（默认：`localhost:50051`）
- `agent_name`：代理名称（通常为 `"NoRobot"`）
- `env_name`：环境名称（`"Actors"`、`"Cameras"` 或 `"Lights"`）
- `scene_runtime`：场景运行时对象（可选，用于相机和灯光示例）

## 💡 使用提示

1. **确保 OrcaStudio 正在运行**：默认地址为 `localhost:50051`
2. **资产路径**：确保资产路径正确，资产应位于 `assets/` 目录下
3. **场景清空**：`run_actors.py` 会自动清空场景，避免名称冲突
4. **随机性**：每次运行 `run_actors.py` 和 `run_lights.py` 会产生不同的随机效果
5. **自定义场景**：可以修改脚本中的 `create_scene()` 函数来自定义场景内容

## 🔍 自定义示例

### 添加新的 Actor

在 `create_scene()` 函数中添加：

```python
actor = Actor(
    name="my_actor",
    asset_path="assets/path/to/your/asset",
    position=np.array([x, y, z]),
    rotation=rotations.euler2quat(np.array([roll, pitch, yaw])),
    scale=1.0,
)
scene.add_actor(actor)
```

### 设置材质属性

```python
material_info = MaterialInfo(
    base_color=np.array([r, g, b, a]),  # RGBA 颜色值 (0-1)
)
scene.set_material_info("actor_name", material_info)
```

### 设置灯光属性

```python
light_info = LightInfo(
    color=np.array([r, g, b]),         # RGB 颜色值 (0-1)
    intensity=100.0,                    # 强度值
)
scene.set_light_info("light_name", light_info)
```

### 设置相机为活动视口

```python
scene.make_camera_viewport_active("camera_name", "CameraViewport")
```

## 📝 注意事项

1. **资产路径格式**：资产路径应使用相对路径，格式为 `assets/...`
2. **Actor 名称唯一性**：确保每个 Actor 的名称唯一，否则会出现冲突
3. **场景发布**：修改场景后需要调用 `scene.publish_scene()` 才能生效
4. **场景清理**：使用完毕后调用 `scene.close()` 关闭场景连接
