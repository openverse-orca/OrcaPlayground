# 第 1 课：最简 spawnable 场景（自由落体方块）

> 场景构建子系统第 1 课（02_scene 子系统）。本课搭建最简场景：地面 + 重力 + 一个自由落体方块，
> 演示世界坐标系与重力配置。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | add_actor spawn 成功 | `ActorCollector.add_actor` | 视口出现方块 |
| 2 | publish_scene 后可见 | `OrcaGymScene.publish_scene` | OrcaLab 视口渲染方块 |
| 3 | 方块自由落体 | `env.sim_config.gravity` | 方块按重力下落 |
| 4 | 修改 gravity 观察加速度 | `--gravity 0 0 -2` | 弱重力下方块下落变慢 |
| 5 | 坐标系方向 | z 轴向上 | 方块从 z=1 落到 z=0 |

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaLab 已启动
- ✅ 已在 OrcaLab 资产库中订阅 **PlayGroundScene** 资产包
- ✅ 在 OrcaLab 中加载 **orcalab_day** 关卡

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/01_empty_scene/
├── 01_empty_scene.md      ← 本教程
├── empty_scene.py         ← 核心逻辑（build_empty_scene）
└── run_empty_scene.py     ← 脚本入口
```

**依赖**：`_common/actor_collector.py`（ActorCollector + set_world + add_actor + spawn_all）

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认重力（地球）
python examples/scene_building/02_scene/01_empty_scene/run_empty_scene.py

# 弱重力对比
python examples/scene_building/02_scene/01_empty_scene/run_empty_scene.py --gravity 0 0 -2

# 失重对比
python examples/scene_building/02_scene/01_empty_scene/run_empty_scene.py --gravity 0 0 0
```

运行流程：
1. 脚本 spawn 方块到 OrcaLab
2. 在 OrcaLab 中点击「运行」按钮进入运行模式
3. 脚本自动创建 Euler env 并步进物理，方块开始自由落体

---

## 5. 本课概念

| 概念 | 说明 |
|------|------|
| `ActorCollector` | 场景构建工具类，封装 actor 收集与批量 spawn |
| `set_world(gravity=)` | 在 collector 中收集重力意图（spawn 接口 proto 不携带 gravity 字段） |
| `add_actor` | 添加 actor 规格（spawnable_path + pos + asset_type） |
| `spawn_all(scene)` | 批量调用 `scene.add_actor` 并 `publish_scene` |
| `env.sim_config.gravity` | 通过 Euler env 的 SimConfig 应用重力到本地 mjModel |
| `env.do_simulation` | 步进本地物理仿真 |
| `env.render()` | 将本地 mjData 状态推送到 OrcaLab 视口 |

---

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaLab gRPC 地址 |
| `--box-pos` | `0 0 1` | 方块初始位置 |
| `--gravity` | `0 0 -9.81` | 重力加速度 |
| `--sim-steps` | `500` | 仿真步数 |
