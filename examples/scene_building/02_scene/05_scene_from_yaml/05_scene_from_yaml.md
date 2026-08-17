# 第 5 课：YAML 配置驱动 spawn 场景

> 场景构建子系统第 5 课（02_scene 子系统）。本课从 YAML 声明式描述场景，
> 演示配置驱动开发（SceneSpec → ActorSpec → add_actor + 物理仿真）。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | YAML 解析正确 | `parse_yaml(path)` | SceneSpec 含 world/actors |
| 2 | SceneSpec → add_actor 链路 | `build_scene_from_yaml` | Actor 按配置 spawn |
| 3 | publish_scene 后 Studio 可见 | `scene.publish_scene()` | 视口渲染场景 |
| 4 | 材质应用 | `scene.set_material_info` | YAML 中带 material 的 actor 变色 |
| 5 | 物理仿真 | `env.sim_config.gravity` + `env.step` | 方块自由落体 |

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaLab 已启动
- ✅ 已在 OrcaLab 资产库中订阅 **OrcaPlaygroundAssets** 资产包
- ✅ 在 OrcaLab 中加载 **orcalab_day** 关卡

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/05_scene_from_yaml/
├── 05_scene_from_yaml.md  ← 本教程
├── scene_from_yaml.py     ← 核心逻辑（parse_yaml + build_scene_from_yaml）
├── run_scene_from_yaml.py ← 脚本入口（spawn + 仿真）
└── scene_demo.yaml        ← 默认 YAML 配置样例
```

---

## 4. YAML Schema

```yaml
world:
  gravity: [0, 0, -9.81]
  timestep: 0.002
  integrator: RK4
actors:
  - name: box
    spawnable_path: <path>
    asset_type: xml
    pos: [0, 0, 1]
    quat: [1, 0, 0, 0]        # 可选，默认 identity
    scale: 1.0                # 可选，默认 1.0
    material:                 # 可选
      base_color: [0.8, 0.2, 0.2, 1]
      metallic: 0.0
      roughness: 0.5
  - name: chair
    spawnable_path: <path>
    asset_type: usdz
    pos: [1, 0, 0]
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `world.gravity` | `[x, y, z]` | 否 | 重力加速度，默认 `[0, 0, -9.81]` |
| `world.timestep` | float | 否 | 仿真步长，默认 `0.002` |
| `world.integrator` | str | 否 | 积分器，默认 `RK4` |
| `actors[].name` | str | 是 | Actor 唯一名 |
| `actors[].spawnable_path` | str | 是 | spawnable 资产路径 |
| `actors[].asset_type` | str | 否 | `xml`/`usdz`/`asset_pack`，默认 `xml` |
| `actors[].pos` | `[x, y, z]` | 是 | 初始位置 |
| `actors[].quat` | `[w, x, y, z]` | 否 | 四元数旋转，默认 `[1, 0, 0, 0]` |
| `actors[].scale` | float | 否 | 缩放，默认 `1.0` |
| `actors[].material` | dict | 否 | PBR 材质（base_color/metallic/roughness） |

> **注意**：
> - `material.metallic` 和 `material.roughness` 在 ActorCollector 中定义，
>   但 OrcaGymScene.set_material_info 当前只应用 `base_color`。

---

## 5. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 使用默认 scene_demo.yaml
python examples/scene_building/02_scene/05_scene_from_yaml/run_scene_from_yaml.py

# 使用自定义 YAML
python examples/scene_building/02_scene/05_scene_from_yaml/run_scene_from_yaml.py --config my_scene.yaml

# 覆盖 YAML 中的重力
python examples/scene_building/02_scene/05_scene_from_yaml/run_scene_from_yaml.py --gravity 0 0 -2
```

**流程**：
1. 解析 YAML → SceneSpec → ActorCollector
2. `spawn_all` + `publish_scene` —— spawn 到 Studio（编辑模式）
3. 对带 material 的 actor 调用 `set_material_info`
4. 在 OrcaLab 中点击「运行」按钮进入运行模式（MuJoCo 初始化）
5. 脚本自动重试创建 Euler env，拉取已 spawn 的场景 MJCF
6. `env.sim_config.gravity` 应用重力
7. `env.step(action)` 循环步进物理，`env.render()` 推送视口
8. 方块自由落体，Ctrl+C 退出

---

## 6. 默认配置（scene_demo.yaml）

```yaml
world:
  gravity: [0, 0, -9.81]
  timestep: 0.002
  integrator: RK4
actors:
  - name: falling_cup         # 红色咖啡杯，自由落体
    spawnable_path: assets/e071469a36d3c8aa/default_project/prefabs/cup_of_coffee_usda
    asset_type: usdz
    pos: [0.0, 0.0, 1.5]
  - name: reference_desk      # 办公桌，静态参考物
    spawnable_path: assets/e071469a36d3c8aa/default_project/prefabs/office_desk_7_mb_usda
    asset_type: usdz
    pos: [0.0, 0.0, 0.0]
  - name: reference_basket    # 购物篮，静态参考物
    spawnable_path: assets/e071469a36d3c8aa/default_project/prefabs/cart_basket_usda
    asset_type: usdz
    pos: [0.5, 0.5, 0.0]
```

---

## 7. 本课概念

### 7.1 配置驱动的优势

| 方式 | 特点 | 适用场景 |
|------|------|----------|
| **代码驱动**（前 4 课） | Python 函数调用 add_actor | 快速原型、复杂逻辑 |
| **YAML 驱动**（本课） | 声明式描述，代码与配置分离 | 场景复用、版本管理、非程序员编辑 |

### 7.2 数据流

```
YAML 文件
  ↓ parse_yaml()
SceneSpec (world + actors)
  ↓ build_scene_from_yaml()
ActorCollector
  ↓ spawn_all(scene)
scene.add_actor(...) × N
  ↓ scene.publish_scene()
Studio MJCF 重建
  ↓ env.reset()
MuJoCo 物理仿真
```
## 8. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `scene_demo.yaml` | YAML 配置文件路径（相对路径基于脚本目录） |
| `--addr` | `localhost:50051` | OrcaLab gRPC 地址 |
| `--gravity` | None | 覆盖 YAML 中的重力（X Y Z）。不指定则用 YAML 值 |