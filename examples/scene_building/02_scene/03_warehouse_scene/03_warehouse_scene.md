# 第 3 课：仓库货架场景（批量 spawn）

> 场景构建子系统第 3 课（02_scene 子系统）。本课 spawn 仓库货架阵列，演示批量生成与过道布局。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 货架阵列生成 | 嵌套循环 `add_actor` | rows × cols 个货架 |
| 2 | 过道宽度合理 | `_SHELF_SPACING = (2.0, 2.0)` | 货架间过道 2m |
| 3 | 命名规范 | `f"shelf_{r}_{c}"` | 每个 actor 唯一命名 |

> **当前状态**：骨架（TODO 未实现）。资产路径 `_SHELF_SPAWNABLE_PATH` 待确认。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaStudio/OrcaLab 已启动
- ✅ 已订阅货架 spawnable 资产包

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/03_warehouse_scene/
├── 03_warehouse_scene.md  ← 本教程
├── warehouse_scene.py     ← 核心逻辑（build_warehouse_scene）
└── run_warehouse_scene.py ← 脚本入口（骨架，argparse + TODO）
```

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

python examples/scene_building/02_scene/03_warehouse_scene/run_warehouse_scene.py --rows 3 --cols 5
```

---

## 5. 本课概念

### 代码解析

```python
# 批量 spawn 货架阵列
for r in range(rows):
    for c in range(cols):
        pos = (r * _SHELF_SPACING[0], c * _SHELF_SPACING[1], 0)
        collector.add_actor(
            name=f"shelf_{r}_{c}",          # 唯一命名
            spawnable_path=_SHELF_PATH,
            pos=pos,
            asset_type="usdz",
        )
```

---

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio gRPC 地址 |
| `--rows` | `3` | 货架行数 |
| `--cols` | `5` | 货架列数 |

---

## 7. 参见

- 设计文档：`03_示例开发计划.md §2.2.2 (3)`
