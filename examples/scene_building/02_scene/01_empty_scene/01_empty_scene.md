# 第 1 课：最简 spawnable 场景（自由落体方块）

> 场景构建子系统第 1 课（02_scene 子系统）。本课搭建最简场景：地面 + 重力 + 一个自由落体方块，
> 演示世界坐标系与重力配置。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | add_actor spawn 成功 | `ActorCollector.add_actor` | 视口出现方块 |
| 2 | publish_scene 后可见 | `OrcaGymScene.publish_scene` | Studio 视口渲染方块 |
| 3 | 方块自由落体 | `ActorCollector.set_world(gravity=)` | 方块按重力下落 |
| 4 | 修改 gravity 观察加速度 | `set_world(gravity=(0,0,-2))` | 弱重力下方块下落变慢 |
| 5 | 坐标系方向 | z 轴向上 | 方块从 z=1 落到 z=0 |

> **当前状态**：骨架（TODO 未实现）。资产路径 `_BOX_SPAWNABLE_PATH` 待确认。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaStudio/OrcaLab 已启动并监听 `--addr`
- ✅ 已订阅方块 spawnable 资产包
- ✅ 加载一个空关卡并点击运行

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/01_empty_scene/
├── 01_empty_scene.md      ← 本教程
├── empty_scene.py         ← 核心逻辑（build_empty_scene）
└── run_empty_scene.py     ← 脚本入口（骨架，argparse + TODO）
```

**依赖**：`_common/actor_collector.py`（ActorCollector + set_world + add_actor + spawn_all）

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

python examples/scene_building/02_scene/01_empty_scene/run_empty_scene.py
python examples/scene_building/02_scene/01_empty_scene/run_empty_scene.py --box-pos 0 0 2
python examples/scene_building/02_scene/01_empty_scene/run_empty_scene.py --gravity 0 0 -2
```

> **注意**：当前为骨架，运行仅打印 warning。需在 `run_empty_scene.py` 中实现
> `OrcaGymScene` 接入与 `publish_scene` 调用。

---

## 5. 本课概念

| 概念 | 说明 |
|------|------|
| `ActorCollector` | 场景构建工具类，封装 actor 收集与批量 spawn |
| `set_world(gravity=)` | 设置世界参数（重力、timestep、integrator） |
| `add_actor` | 添加 actor 规格（spawnable_path + pos + asset_type） |
| `spawn_all(scene)` | 批量调用 `scene.add_actor` 并 `publish_scene` |

---

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio gRPC 地址 |
| `--box-pos` | `0 0 1` | 方块初始位置 |
| `--gravity` | `0 0 -9.81` | 重力加速度 |

---

## 7. 参见

- 设计文档：`03_示例开发计划.md §2.2.2 (1)`
- API 缺口：`00_设计方案_总体架构.md §3.2.7`（ScenePhysicsBridge，自由落体需物理步进接口）
