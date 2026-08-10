# 第 6 课：多源 spawnable 合并

> 场景构建子系统第 6 课（02_scene 子系统）。本课合并多源资产（机器人 XML + 物体 USDZ + 场景资产包），
> 演示命名空间隔离与多类型共存。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 多源 spawnable 合并 | `add_robot` + `add_actor` × 2 | 三类资产共存 |
| 2 | actor name 无冲突 | 手动前缀 | robot_a / obstacle / base_scene |
| 3 | 机器人 + 物体 + 场景共存 | `spawn_all` | Studio 视口显示完整场景 |

> **当前状态**：骨架（TODO 未实现）。三个资产路径待确认。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaStudio/OrcaLab 已启动
- ✅ 已订阅机器人/物体/场景资产包

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/06_scene_composition/
├── 06_scene_composition.md ← 本教程
├── scene_composition.py     ← 核心逻辑（build_scene_composition）
└── run_scene_composition.py ← 脚本入口（骨架，argparse + TODO）
```

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

python examples/scene_building/02_scene/06_scene_composition/run_scene_composition.py
```

---

## 5. 本课概念

| 概念 | 说明 |
|------|------|
| 多源合并 | XML（机器人）+ USDZ（物体）+ 资产包（场景）三种格式 |
| 命名空间隔离 | actor name 唯一性，避免冲突 |
| `add_robot` | ActorCollector 的机器人专用添加方法 |

### 代码解析

```python
# 三类资产合并
collector.add_robot(name="robot_a", spawnable_path=_ROBOT_PATH, pos=(0, 0, 1.05))
collector.add_actor(name="obstacle", spawnable_path=_OBSTACLE_PATH, pos=(2, 0, 0.5), asset_type="usdz")
collector.add_actor(name="base_scene", spawnable_path=_SCENE_PATH, pos=(0, 0, 0), asset_type="asset_pack")
```

---

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio gRPC 地址 |

---

## 7. API 缺口

需确认 `add_actor` 对 XML/USDZ/资产包三种格式的命名空间隔离机制；
若不支持，降级为手动加前缀（如 `robot_a_` / `obstacle_` / `scene_`）。

---

## 8. 参见

- 设计文档：`03_示例开发计划.md §2.2.2 (6)`
