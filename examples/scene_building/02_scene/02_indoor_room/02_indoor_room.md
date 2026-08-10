# 第 2 课：程序化室内房间（墙体/家具 spawnable）

> 场景构建子系统第 2 课（02_scene 子系统）。本课程序化 spawn 房间：4 面墙 + 地板 + 家具，
> 演示批量布局与几何参数化。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 房间几何正确 | `add_actor` × 4 墙 + 1 地板 | 墙体围成矩形房间 |
| 2 | 家具位于房间内 | `add_actor`（table + chair） | 家具在墙体内侧 |
| 3 | 碰撞检测 | 物理引擎 | 家具不穿墙 |

> **当前状态**：骨架（TODO 未实现）。资产路径 `_WALL_SPAWNABLE_PATH` / `_FLOOR_SPAWNABLE_PATH` /
> `_FURNITURE_SPAWNABLE_PATHS` 待确认。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaStudio/OrcaLab 已启动并监听 `--addr`
- ✅ 已订阅墙体/地板/家具 spawnable 资产包
- ✅ 加载一个空关卡并点击运行

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/02_indoor_room/
├── 02_indoor_room.md      ← 本教程
├── indoor_room.py         ← 核心逻辑（build_indoor_room）
└── run_indoor_room.py     ← 脚本入口（骨架，argparse + TODO）
```

**依赖**：`_common/actor_collector.py`

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

python examples/scene_building/02_scene/02_indoor_room/run_indoor_room.py --width 5 --height 3
```

> **注意**：当前为骨架，运行仅打印 warning。

---

## 5. 本课概念

| 概念 | 说明 |
|------|------|
| 程序化布局 | 通过循环 / 坐标计算批量生成 actor |
| 几何参数化 | `width` / `height` 参数控制房间尺寸 |
| 半尺寸定位 | `half_w = width / 2`，墙体位于 ±half_w 处 |

### 代码解析

```python
# 4 面墙围绕原点布局
half_w = width / 2
half_h = height / 2
collector.add_actor("wall_north", _WALL_PATH, pos=(0, half_w, half_h))
collector.add_actor("wall_south", _WALL_PATH, pos=(0, -half_w, half_h))
collector.add_actor("wall_east",  _WALL_PATH, pos=(half_w, 0, half_h))
collector.add_actor("wall_west",  _WALL_PATH, pos=(-half_w, 0, half_h))
```

---

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio gRPC 地址 |
| `--width` | `5.0` | 房间宽度（米） |
| `--height` | `3.0` | 房间高度（米） |

---

## 7. 参见

- 设计文档：`03_示例开发计划.md §2.2.2 (2)`
