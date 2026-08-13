# 第 4 课：户外地形（程序化 spawn + 球体滚落仿真）

> 场景构建子系统第 4 课（02_scene 子系统）。本课 spawn 程序化生成的地形，再启动物理仿真观察球体滚落。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 地形 spawnable 加载 | `add_actor` + `publish_scene` | 视口出现所选地形 |
| 2 | 三种地形视觉差异 | slope/stairs/gentle | 杂乱 / 阶梯 / 平整 |
| 3 | 球体在不同地形上滚落 | Euler env + `do_simulation` | 斜坡滚动 vs 微动 |

---

## 2. 前置条件

- conda `orca` 环境可用
- OrcaStudio/OrcaLab 已启动
- 已订阅仓库资产包 room（含 terrain_slope + terrain_stairs + terrain_gentle + sphere）

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/04_outdoor_terrain/
├── 04_outdoor_terrain.md      ← 本教程
├── outdoor_terrain.py         ← 核心逻辑（build_outdoor_terrain，三选一）
└── run_outdoor_terrain.py     ← 脚本入口（--terrain 参数）
```

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

# spawn 全部三种地形（沿 X 轴排列）+ 球体 + 启动仿真
python examples/scene_building/02_scene/04_outdoor_terrain/run_outdoor_terrain.py

# 只 spawn 指定地形
python examples/scene_building/02_scene/04_outdoor_terrain/run_outdoor_terrain.py --terrain stairs
python examples/scene_building/02_scene/04_outdoor_terrain/run_outdoor_terrain.py --terrain gentle
python examples/scene_building/02_scene/04_outdoor_terrain/run_outdoor_terrain.py --terrain slope

# 自定义球体位置（放在指定地形上方）
python examples/scene_building/02_scene/04_outdoor_terrain/run_outdoor_terrain.py --terrain stairs --sphere-pos -3 0 3
```

**流程**：
1. spawn 地形 + 球体到 Studio（编辑模式）
2. 在 OrcaLab 中点击「运行」按钮进入运行模式（MuJoCo 初始化）
3. 脚本自动重试创建 Euler env，拉取已 spawn 的场景
4. `env.step(action)` 循环步进物理，`env.render()` 推送到视口
5. 球体自由落体并沿地形滚动，Ctrl+C 退出

---

## 5. 本课概念

### 5.1 三种地形对比

| 地形 | `--terrain` | 高度范围 | 最大倾斜 | 颜色 | 球体行为 |
|------|-------------|----------|----------|------|----------|
| **杂乱起伏坡地** | `slope` | -0.4~2.1m | 42° | 沙土色 | 沿复杂曲面滚动/弹跳 |
| **完整阶梯斜坡** | `stairs` | 0~2.27m | 18°（统一） | 土褐色 | 稳定沿单一方向滚下 |
| **几乎平整** | `gentle` | ±9cm | ≤5° | 灰蓝色 | 轻微晃动或静止 |

### 5.2 地形生成方式

#### terrain_slope（杂乱起伏）
- Perlin-like 多频正弦叠加驱动高度场
- 每个 body 沿高度场梯度方向倾斜（body euler），最大 ±40°
- 64 个独立 body，每个带 euler 旋转

#### terrain_stairs（完整阶梯斜坡）
- 统一倾角 18°（舒适滑行角度）
- 沿 X 方向线性下降：左高右低
- 每个台阶高度差 ≈ 0.32m，总落差 2.27m

#### terrain_gentle（几乎平整）
- 极低振幅正弦波（±9cm）
- 最大倾斜仅 5°
- 用于对比测试：球在几乎平整地面上无明显运动

### 5.3 spawn 流程

```python
# 不指定 --terrain：三种地形沿 X 轴排列（slope@-10 / stairs@0 / gentle@+10）
# 指定 --terrain：只 spawn 该地形（居中）
specs = [
    ("terrain_slope",  SLOPE_PATH,  (-10, 0, 0)),   # 仅全部模式
    ("terrain_stairs", STAIRS_PATH, (0, 0, 0)),     # 两种模式都有
    ("terrain_gentle", GENTLE_PATH, (10, 0, 0)),    # 仅全部模式
    ("test_sphere",    SPHERE_PATH, sphere_pos),     # 球体
]
for name, path, pos in specs:
    scene.add_actor(_make_actor(name, path, pos))
# 统一 publish_scene 触发 Studio 重建 MJCF
scene.publish_scene()
```

### 5.4 资产路径

| 资产 | spawnable 路径 |
|------|----------------|
| 杂乱起伏 | `assets/816f95ce16021282/default_project/prefabs/terrain_slope_usda` |
| 阶梯斜坡 | `assets/816f95ce16021282/default_project/prefabs/terrain_stairs_usda` |
| 几乎平整 | `assets/816f95ce16021282/default_project/prefabs/terrain_gentle_usda` |
| 测试球体 | `assets/816f95ce16021282/default_project/prefabs/sphere_usda` |

### 5.5 球体资产说明

`sphere.xml`：
- 半径 0.15m，青色（`0.20 0.90 0.80`）
- 密度 100（轻量，滚动明显）
- 自由关节（freejoint），可自由落体和滚动
- 摩擦 0.8（足够滚动但不打滑）

---

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio/OrcaLab gRPC 地址 |
| `--terrain` | None（全部） | 地形类型: `slope`(杂乱) / `stairs`(阶梯) / `gentle`(平整)。不指定则 spawn 全部三种（沿 X 轴排列，间距 10m） |
| `--sphere-pos` | `0 0 2.5` | 测试球体初始位置（X Y Z） |
| `--interval` | `1.0` | spawn 间隔（秒） |

---


