# 第 8 课：场景随机变体（批量 spawn + 域随机化）

> 场景构建子系统第 8 课（02_scene 子系统）。本课通过域随机化批量 spawn 物体，
> 演示布局扰动与可复现性。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 不同 seed 生成不同布局 | `np.random.default_rng(seed)` | seed 变化 → 布局变化 |
| 2 | 同 seed 可复现 | 固定 seed | 两次运行结果一致 |
| 3 | 物体落在台面有效区域 | `ZONES` / `COUNTER_TOP_Z` | 位置在台面范围内，不穿模/不掉地 |
| 4 | 固定物体保持原位 | `FIXED_OBJECTS` | 锅盖 + 对应锅不动 |
| 5 | 隐藏物体不可见 | `HIDDEN_KEYWORDS` | 篮子移到远处 |

> **实现说明**：本课采用 freejoint 位置随机化范式（操作已有场景物体），
> 区别于 spawn 范式（`random_variation.py` 骨架，待实现）。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaStudio/OrcaLab 已启动
- ✅ 已在 OrcaStudio 资产库中订阅 **kitchen_Night_2** 资产包
- ✅ 在 OrcaLab 中加载 kitchen_Night_2 关卡并点击「运行」按钮进入运行模式

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/08_random_variation/
├── 08_random_variation.md ← 本教程
├── random_variation.py     ← spawn 范式骨架（待实现，本课未使用）
└── run_random_variation.py ← 入口脚本（freejoint 位置随机化，已实现）
```

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认真随机
python examples/scene_building/02_scene/08_random_variation/run_random_variation.py

# 指定 seed 可复现
python examples/scene_building/02_scene/08_random_variation/run_random_variation.py --seed 42 --steps 300
```

---

## 5. 本课概念

| 概念 | 说明 |
|------|------|
| freejoint 位置随机化 | 操作已有场景 freejoint 的 qpos，重置物体位置 |
| `np.random.default_rng(seed)` | 可复现的随机数生成器（seed=None 时真随机） |
| `ZONES` | 有效放置区域：A-盘架区 / B-灶台右侧（沿主台面 y 轴划分） |
| `COUNTER_TOP_Z` | 台面顶部高度 0.985，物体统一放置到此高度上方 |
| `FIXED_OBJECTS` | 固定不动的物体（锅盖 Pot_02_b + 对应锅 Pot_02_a） |
| `HIDDEN_KEYWORDS` | 隐藏物体关键词（Basket_Kitchen 移到 z=-1000） |

### 代码解析

```python
rng = np.random.default_rng(seed)
for jname, adr in free_joints.items():
    # 固定/隐藏检查略
    zone = rng.choice(ZONES)
    new_x = float(rng.uniform(*zone["x"]))
    new_y = float(rng.uniform(*zone["y"]))
    new_z = COUNTER_TOP_Z + float(rng.uniform(0.02, 0.05))
    qpos[adr : adr + 3] = [new_x, new_y, new_z]
env.set_joint_qpos(qpos)
env.mj_forward()
```

---

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio/OrcaLab gRPC 地址 |
| `--seed` | `None` | 随机种子（None=真随机，指定值可复现） |
| `--steps` | `500` | 扰动后仿真步数 |

---

## 7. 参见

- 设计文档：`03_示例开发计划.md §2.2.2 (8)`
- spawn 范式骨架：`random_variation.py`（待实现，本课未使用）
