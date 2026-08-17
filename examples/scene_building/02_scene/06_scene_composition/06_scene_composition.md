# 第 6 课：机器人厨房助手（多源合并）

> 场景构建子系统第 6 课（02_scene 子系统）。本课演示场景包 + 厨具 + 机器人三源共存，
> 机器人厨师 g1_pick 在厨房台面前备餐，通过命名空间前缀（kitchen_/utensil_/robot_）
> 体现多源隔离，并支持做菜/清理两种工位状态。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | g1_pick 机器人正确 spawn | `add_actor` + `append_scene` | Studio 视口显示机器人 |
| 2 | 机器人与厨房/厨具共存 | `RobotChefEnv` | 三源合并，无冲突 |
| 3 | 两种工位状态切换 | `--state cooking\|cleaning` | 机器人工位不同 |
| 4 | env 步进时机器人保持站立 | `do_simulation` + `render` | 物理稳定，不倒地 |

> **实现说明**：厨房场景与厨具由用户在 OrcaLab 手动加载 kitchen_Night_2 关卡，
> 本脚本只 spawn 机器人 g1_pick。"三源合并"体现在场景中同时存在
> kitchen_（厨房台面）/utensil_（厨具）/robot_（机器人）三类命名空间的 actor。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaLab 已启动
- ✅ 已在 OrcaLab 资产库中订阅 **PlayGroundScene** 和 **Kitchen_Night_2** 资产包
- ✅ 在 OrcaLab 中加载 **Kitchen_Night_2** 关卡


---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/06_scene_composition/
├── 06_scene_composition.md ← 本教程
├── scene_composition.py     ← 核心逻辑（build_robot_chef_scene + RobotChefEnv）
└── run_scene_composition.py ← 入口脚本（argparse + spawn + 仿真）
```

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 做菜状态（默认，机器人在灶台前备菜）
python examples/scene_building/02_scene/06_scene_composition/run_scene_composition.py

# 清理状态（机器人在洗菜池前清理）
python examples/scene_building/02_scene/06_scene_composition/run_scene_composition.py --state cleaning

# 指定步数（默认无限循环直至 Ctrl+C）
python examples/scene_building/02_scene/06_scene_composition/run_scene_composition.py --sim-steps 500
```

---

## 5. 本课概念

| 概念 | 说明 |
|------|------|
| 多源合并 | 厨房场景包 + 厨具 + 机器人三类 actor 共存于同一 MuJoCo 模型 |
| 命名空间隔离 | `kitchen_` / `utensil_` / `robot_` 前缀自然区分资产来源 |
| 叙事性 spawn | 机器人按"厨房工作站"逻辑布局（灶台前/洗菜池前） |
| `STATE_CONFIGS` | 两种工位状态的位置配置（cooking / cleaning） |
| `append_scene` | 增量发布，不销毁用户已加载的厨房场景 |

### 代码解析

```python
# 两种工位状态（坐标系来自第 8 课 out.xml 分析）
STATE_CONFIGS = {
    "cooking":  {"pos": (-3.5, -2.0, 0.0), ...},  # 灶台前
    "cleaning": {"pos": (-3.5,  0.5, 0.0), ...},  # 洗菜池前
}

# spawn 机器人到指定工位
scene.add_actor(Actor(name="robot_chef", asset_path=_ROBOT_PATH, pos=cfg["pos"]))
scene.append_scene()

# env 步进物理
env = RobotChefEnv(agent_names=["g1_pick"], ...)
env.step(ctrl)   # do_simulation 推进 MuJoCo
env.render()     # 推送状态到 O3DE
```

---

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaLab gRPC 地址 |
| `--state` | `cooking` | 机器人工位状态（`cooking` / `cleaning`） |
| `--sim-steps` | `0` | 仿真步数（<=0 表示无限循环直至 Ctrl+C） |

---

## 7. 坐标系参考（来自第 8 课 out.xml 分析）

| 区域 | x 范围 | y 范围 | z_top | 说明 |
|------|--------|--------|-------|------|
| 主台面 | [-5.239, -4.499] | [-5.261, 0.339] | 0.985 | 厨房长台面 |
| 灶台火眼 | - | y≈-2.4, y≈-1.8 | - | 锅 Pot_02_a/b 位置 |
| 机器人 cooking | -4.4 | -2.0 | 0.0 | 灶台前，面向 -x（绕 z 轴 180°） |
| 机器人 cleaning | -4.4 | 0.2 | 0.0 | 洗菜池前，面向 -x（绕 z 轴 180°） |
| 粉色咖啡杯（Coffecup） | -4.6 | 0.2 | 0.985 | 池子外侧前，站立（放倒会穿模），和机器人对齐 |
| 透明杯（Glass） | -4.8 | 0.2 | 0.985 | 池子外侧前，放倒（绕 x 轴 90°） |
| 瓷杯（Porcelain） | -4.6 | -0.2 | 0.985 | 池子外侧后，放倒，一前一后 |
