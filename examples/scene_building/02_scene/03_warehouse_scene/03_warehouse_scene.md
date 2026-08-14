# 第 3 课：仓库货架场景（批量 spawn）

> 场景构建子系统第 3 课（02_scene 子系统）。本课 spawn 仓库货架阵列，演示批量生成与过道布局。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 货架阵列批量生成 | 嵌套循环 `add_actor` | rows × cols 个货架 |
| 2 | 过道宽度合理 | `_SHELF_SPACING = (货架尺寸 + 2.0, ...)` | 货架间过道 2m |
| 3 | 阵列中心对齐 | `offset = -(n-1)/2 * spacing` | 阵列居中于地面 |
| 4 | 命名规范 | `f"shelf_{r}_{c}"` | 每个 actor 唯一命名 |
| 5 | 增量发布 | `append_scene()` | 前序 actor 不被销毁 |

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaLab 已启动
- ✅ 已在 OrcaLab 资产库中订阅 **PlayGroundScene** 资产包
- ✅ 在 OrcaLab 中加载 **orcalab_day** 关卡

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/03_warehouse_scene/
├── 03_warehouse_scene.md  ← 本教程
├── warehouse_scene.py     ← 核心逻辑（build_warehouse_scene）
└── run_warehouse_scene.py ← 脚本入口
```

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认 B 密集（3×4）
python examples/scene_building/02_scene/03_warehouse_scene/run_warehouse_scene.py

# A 稀疏（2×3）
python examples/scene_building/02_scene/03_warehouse_scene/run_warehouse_scene.py --rows 2 --cols 3
```

---

## 5. 本课概念

### 5.1 布局计算

货架尺寸 1.2m × 0.4m（来自 `metal_shelf.xml`），过道宽度 2.0m。

```python
# 货架间距（中心到中心 = 货架尺寸 + 过道）
_SHELF_SPACING = (1.2 + 2.0, 0.4 + 2.0)  # (3.2, 2.4)

# 阵列中心化偏移
offset_x = -(rows - 1) / 2.0 * _SHELF_SPACING[0]
offset_y = -(cols - 1) / 2.0 * _SHELF_SPACING[1]
```

占地公式：`(n-1) * spacing + shelf_size`
- X 方向：`(rows-1) * 3.2 + 1.2`
- Y 方向：`(cols-1) * 2.4 + 0.4`

### 5.2 推荐配置（10×10 地面内）

为避免超出 10×10 地面边界，且兼顾不同视觉效果，推荐以下两组配置：

| 配置 | rows×cols | 占地 X×Y（m） | 视觉特点 | 命令 |
|------|-----------|---------------|----------|------|
| A 稀疏 | 2×3 | 4.4 × 5.2 | 过道宽敞，货架四周留白，凸显空间感 | `--rows 2 --cols 3` |
| B 密集 | 3×4 | 7.6 × 7.6 | 货架排满中央区域，过道与货架比例均衡 | `--rows 3 --cols 4` |

**占地校验**（10×10 地面边界）：
- A 稀疏 2×3: X=4.4m ✓, Y=5.2m ✓（四周各留 2.4m+ 缓冲）
- B 密集 3×4: X=7.6m ✓, Y=7.6m ✓（四周各留 1.2m 缓冲）
- 3×5（满铺）: X=7.6m ✓, Y=10.0m ⚠️（Y 方向贴边，无缓冲，不推荐）

**默认配置**：`DEFAULT_ROWS=3, DEFAULT_COLS=4`（B 密集），兼顾视觉密度与边界缓冲。

### 5.3 批量 spawn

```python
for r in range(rows):
    for c in range(cols):
        pos_x = offset_x + r * _SHELF_SPACING[0]
        pos_y = offset_y + c * _SHELF_SPACING[1]
        scene.add_actor(_make_actor(f"shelf_{r}_{c}", _SHELF_PATH, (pos_x, pos_y, 0.0)))
        scene.append_scene()  # 增量发布，前序 actor 不被销毁
```

### 5.4 资产路径

| 资产 | spawnable 路径 |
|------|----------------|
| 地面 | `assets/816f95ce16021282/default_project/prefabs/floor_10x10_usda` |
| 货架 | `assets/816f95ce16021282/default_project/prefabs/metal_shelf_usda` |

---

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaLab gRPC 地址 |
| `--rows` | `3` | 货架行数（X 方向，推荐 2 或 3） |
| `--cols` | `4` | 货架列数（Y 方向，推荐 3 或 4） |
| `--interval` | `0.5` | spawn 间隔（秒） |

---

## 7. 参见

- 设计文档：`03_示例开发计划.md §2.2.2 (3)`
- 资产源文件：`OrcaGym_Assets/room/floor_10x10.xml`、`OrcaGym_Assets/room/metal_shelf.xml`
