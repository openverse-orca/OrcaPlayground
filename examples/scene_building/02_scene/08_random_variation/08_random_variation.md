# 第 8 课：场景随机变体（厨房家具域随机化）

> 场景构建子系统第 8 课（02_scene 子系统）。本课随机 spawn 厨房家具
> （数量为清单的 1~3 倍，位置/朝向由 seed 控制），演示布局扰动与可复现性。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 厨房家具正确 spawn | `_KITCHEN_ASSETS` 清单 | 家具出现在场景中 |
| 2 | 数量随机（1~3 倍） | `_MULTIPLY_RANGE` | 每次运行家具数量不同（6/12/18 个） |
| 3 | 不同 seed 生成不同布局 | `np.random.default_rng(seed)` | seed 变化 → 家具数量/位置/朝向变化 |
| 4 | 同 seed 可复现 | 固定 seed | 两次运行结果一致 |
| 5 | 资产可复制多份 | name 加索引区分 | 同一资产 spawn 多个实例，name 不冲突 |

> **实现说明**：本课采用 spawn 范式，所有 actor 由 `ActorCollector` 收集后统一
> `spawn_all(scene)` + `publish_scene()` 发布到 OrcaLab。不创建 env、不运行物理仿真，
> 聚焦 spawn 流程与域随机化。家具数量为清单长度的 1~3 倍，由 seed 随机确定。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaLab 已启动
- ✅ 已在 OrcaLab 资产库中订阅 **kitchen_night** 资产包
- ✅ 在 OrcaLab 中加载 **orcalab_day** 关卡

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/08_random_variation/
├── 08_random_variation.md ← 本教程
├── random_variation.py     ← 核心逻辑（build_random_variation）
└── run_random_variation.py ← 入口脚本（argparse + spawn）
```

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认
python examples/scene_building/02_scene/08_random_variation/run_random_variation.py

# 指定 seed 生成不同布局
python examples/scene_building/02_scene/08_random_variation/run_random_variation.py --seed 7

# 指定 OrcaLab 地址
python examples/scene_building/02_scene/08_random_variation/run_random_variation.py --addr localhost:50051
```

---

## 5. 本课概念

| 概念 | 说明 |
|------|------|
| `_KITCHEN_ASSETS` | 厨房资产清单：turner_wood_01 / desk_dinningroom / pot_01 / pot_02_a / coffecup_01 / chair_001 |
| `_MULTIPLY_RANGE` | 家具数量倍数范围：(1, 3)，即清单长度的 1~3 倍（6/12/18 个） |
| `_POS_RANGE` | 家具位置范围：x/y ∈ [-3, 3]，z = 0（地面高度） |
| `_random_z_rotation` | 绕 z 轴随机旋转 θ ∈ [0, 2π)，保持物体直立只改朝向 |
| `np.random.default_rng(seed)` | 可复现的随机数生成器（seed 固定时两次运行结果一致） |
| `ActorCollector` | 批量收集 Actor 规格，统一 `spawn_all(scene)` 提交 |
| 资产复制 | 同一资产可 spawn 多个实例，name 加索引区分（如 `pot_01_01`、`pot_01_02`） |

### 代码解析

```python
rng = np.random.default_rng(seed)
collector = ActorCollector()
collector.set_world()

# 随机确定家具数量（清单长度的 1~3 倍）
multiply = rng.integers(1, 4)  # 1, 2, 或 3
num_objects = len(_KITCHEN_ASSETS) * multiply

# name 计数器：同一资产复制多份时加索引区分
name_counter = {}
for _ in range(num_objects):
    pos = (rng.uniform(-3, 3), rng.uniform(-3, 3), 0.0)
    quat = _random_z_rotation(rng)

    idx = rng.integers(0, len(_KITCHEN_ASSETS))
    asset_name, spawnable_path = _KITCHEN_ASSETS[idx]

    # name 加索引区分（同一资产可复制多份）
    name_counter[asset_name] = name_counter.get(asset_name, 0) + 1
    actor_name = f"{asset_name}_{name_counter[asset_name]:02d}"

    collector.add_actor(
        name=actor_name,
        spawnable_path=spawnable_path,
        pos=pos, quat=quat,
        asset_type="asset_pack",
    )

collector.spawn_all(scene)
scene.publish_scene()
```

---

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaLab gRPC 地址 |
| `--seed` | `None` | 随机种子（省略则全随机，指定值可复现） |
