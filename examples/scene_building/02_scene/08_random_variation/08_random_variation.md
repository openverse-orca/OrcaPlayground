# 第 8 课：场景随机变体（批量 spawn + 域随机化）

> 场景构建子系统第 8 课（02_scene 子系统）。本课通过域随机化批量 spawn 物体，
> 演示布局扰动与可复现性。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 不同 seed 生成不同布局 | `np.random.default_rng(seed)` | seed 变化 → 布局变化 |
| 2 | 同 seed 可复现 | 固定 seed | 两次运行结果一致 |
| 3 | 物体参数在合理范围 | `_POS_RANGE` / `_COLOR_RANGE` | 位置/颜色在预设范围 |

> **当前状态**：骨架（TODO 未实现）。物体路径 `_OBJECT_SPAWNABLE_PATH` 待确认。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaStudio/OrcaLab 已启动
- ✅ 已订阅物体 spawnable 资产包

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/08_random_variation/
├── 08_random_variation.md ← 本教程
├── random_variation.py     ← 核心逻辑（build_random_variation）
└── run_random_variation.py ← 脚本入口（骨架，argparse + TODO）
```

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

python examples/scene_building/02_scene/08_random_variation/run_random_variation.py --seed 42 --num-objects 20
```

---

## 5. 本课概念

| 概念 | 说明 |
|------|------|
| 域随机化 | 随机扰动物体参数，增强 sim-to-real 鲁棒性 |
| `np.random.default_rng(seed)` | 可复现的随机数生成器 |
| `_POS_RANGE` | 位置随机范围：x/y ∈ [-5, 5]，z ∈ [0.5, 2] |
| `_COLOR_RANGE` | 颜色随机范围：RGB ∈ [0, 1] |

### 代码解析

```python
rng = np.random.default_rng(seed)
for i in range(num_objects):
    pos = (
        float(rng.uniform(*(_POS_RANGE[0]))),  # x: [-5, 5]
        float(rng.uniform(*(_POS_RANGE[1]))),  # y: [-5, 5]
        float(rng.uniform(*(_POS_RANGE[2]))),  # z: [0.5, 2]
    )
    rgba = (
        float(rng.uniform(*_COLOR_RANGE)),
        float(rng.uniform(*_COLOR_RANGE)),
        float(rng.uniform(*_COLOR_RANGE)),
        1.0,
    )
    collector.add_actor(name=f"obj_{i}", spawnable_path=_OBJECT_PATH, pos=pos, material=MaterialInfo(base_color=rgba))
```

---

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio gRPC 地址 |
| `--seed` | `42` | 随机种子（可复现） |
| `--num-objects` | `20` | 物体数量 |

---

## 7. 参见

- 设计文档：`03_示例开发计划.md §2.2.2 (8)`
- 复用：第 9 课 `generate_obstacle_layout` 随机布局逻辑（改为 spawnable 路径）
