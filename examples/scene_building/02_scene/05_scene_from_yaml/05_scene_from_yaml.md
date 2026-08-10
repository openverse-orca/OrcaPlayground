# 第 5 课：YAML 配置驱动 spawn 场景

> 场景构建子系统第 5 课（02_scene 子系统）。本课从 YAML 声明式描述场景，
> 演示配置驱动开发（SceneSpec → ActorSpec → add_actor）。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | YAML 解析正确 | `parse_yaml(path)` | SceneSpec 含 world/actors/lights |
| 2 | SceneSpec → add_actor 链路 | `build_scene_from_yaml` | Actor 按配置 spawn |
| 3 | publish_scene 后可见 | `spawn_all(scene)` | Studio 视口渲染场景 |

> **当前状态**：骨架（TODO 未实现）。YAML schema 已定义，待 OrcaGymScene 接入。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaStudio/OrcaLab 已启动
- ✅ 已准备 YAML 配置文件（参考下方 schema）

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/05_scene_from_yaml/
├── 05_scene_from_yaml.md ← 本教程
├── scene_from_yaml.py     ← 核心逻辑（parse_yaml + build_scene_from_yaml）
└── run_scene_from_yaml.py ← 脚本入口（骨架，argparse + TODO）
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
  - name: chair
    spawnable_path: <path>
    asset_type: usdz
    pos: [1, 0, 0]
    material:
      base_color: [0.8, 0.2, 0.2, 1]
      metallic: 0.0
      roughness: 0.5
lights:
  - name: point_light
    light_type: point
    pos: [2, 2, 3]
    intensity: 100
```

---

## 5. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

python examples/scene_building/02_scene/05_scene_from_yaml/run_scene_from_yaml.py --config scene.yaml
```

---

## 6. 本课概念

| 概念 | 说明 |
|------|------|
| `SceneSpec` | 场景规格数据类（world + actors + lights） |
| `ActorSpec` | Actor 规格（name + spawnable_path + pos + quat + material + scale） |
| `parse_yaml` | YAML → SceneSpec 解析器 |
| 配置驱动 | 声明式描述场景，便于复用与版本管理 |

---

## 7. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio gRPC 地址 |
| `--config` | `scene.yaml` | YAML 配置文件路径 |

---

## 8. 参见

- 设计文档：`03_示例开发计划.md §2.2.2 (5)`
- 架构设计：`01_架构设计.md §5.1`（SceneSpec/ActorSpec 数据结构）
