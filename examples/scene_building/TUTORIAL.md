# OrcaPlayground 场景构建课程

本课程通过一系列由浅入深的示例，演示如何以 **全 spawn** 形式构建 OrcaGym 仿真场景：
通过 `OrcaGymScene.add_actor` 订阅资产库的 spawnable 资产（XML/USDZ/资产包），统一调用
`publish_scene()` 发布到 OrcaStudio/OrcaLab。

> 课程设计参考 [examples/euler/TUTORIAL.md](../euler/TUTORIAL.md)，每个示例既是一个独立的
> 场景构建用例，也是后续强化学习训练（Lesson 4 PPO、Lesson 8 行走控制）和其他课题的**场景前置准备**。
> 基本逻辑参考 [examples/replicator](../replicator/README.md)。

## 为什么需要"全 spawn"

传统 MuJoCo 工作流需要用户手写 MJCF XML 描述场景几何、关节、接触对，门槛高且难以复用。
OrcaStudio/OrcaLab 提供了资产订阅机制：spawnable 资产（机器人 XML、物体 USDZ、完整场景
资产包）预先在资产库中注册，脚本通过 `asset_path` 字符串引用即可加载，无需关心底层 MJCF
生成。本课程所有示例统一采用 spawn 范式，让用户聚焦**场景组织与配置**，而非 XML 细节。

学完本课程后，你将能够：

- 加载三类 spawnable 资产（机器人 / 物体 / 完整场景）到 OrcaStudio/OrcaLab
- 程序化批量 spawn 多个 actor 构建仓库、室内、随机化等场景
- 用 YAML 声明式描述场景，实现配置驱动开发
- 为后续 RL 训练（如 PPO 训练倒立摆、G1 行走）准备可复用的标准化场景

## 快速开始

```bash
# 进入 OrcaPlayground 根目录
cd /path/to/OrcaPlayground
conda activate orca

# 第 1 课：加载机器人 spawnable（需 OrcaStudio/OrcaLab + 机器人资产）
python examples/scene_building/01_assets/run_load_mjcf_robot.py

# 第 2 课：加载 USD 物体（需 OrcaStudio/OrcaLab + 物体资产）
python examples/scene_building/01_assets/run_load_usd_object.py --object chair

# 第 3 课：最简场景（需 OrcaStudio/OrcaLab + 方块资产）
python examples/scene_building/02_scene/run_empty_scene.py

# 第 4 课：仓库货架批量 spawn
python examples/scene_building/02_scene/run_warehouse_scene.py --rows 3 --cols 5

# 第 5 课：YAML 配置驱动
python examples/scene_building/02_scene/run_scene_from_yaml.py --config scene.yaml
```

> 注：目录名以数字开头（如 `01_assets`）不能作为 Python 包路径，因此使用脚本直接运行
> 方式（`python examples/...`），而非 `python -m` 模块方式。
> 详细教程见各示例目录下的源码注释。本 TUTORIAL 为总纲，具体实现待补充。

---

## 目录结构

```
examples/scene_building/
├── README.md                      ← 你正在阅读的总纲
├── 01_assets/                     # 第一阶段：资产管理（全 spawn）
│   ├── load_mjcf_robot.py         #   机器人 XML spawnable 加载
│   ├── load_usd_object.py         #   USDZ 物体 spawnable 加载
│   ├── load_usd_scene.py          #   资产包场景 spawnable 加载
│   └── run_*.py                   #   入口脚本
├── 02_scene/                      # 第二阶段：场景构建（全 spawn）
│   ├── empty_scene.py             #   最简 spawnable 场景
│   ├── indoor_room.py             #   程序化室内
│   ├── warehouse_scene.py         #   仓库货架（批量 spawn）
│   ├── outdoor_terrain.py         #   户外地形
│   ├── scene_from_yaml.py         #   YAML 配置驱动
│   ├── scene_composition.py       #   多源 spawnable 合并
│   ├── lighting_setup.py          #   光照系统
│   ├── random_variation.py        #   域随机化
│   └── run_*.py                   #   入口脚本
└── _common/                       # 共享工具
    └── actor_collector.py         #   ActorCollector + spawnable 资产规格收集
```

> **自包含原则**：每个示例目录包含全部所需 `.py` 文件，import 使用同目录引用，
> 资产路径通过模块常量集中定义。零 `envs/` 共享目录依赖。

---

## 课程总览

### 第一阶段：资产管理（Lesson 1-3）

学习如何加载三类 spawnable 资产。本阶段是后续所有场景构建的基础。

| 课 | 主题 | 新概念 | 资产格式 | OrcaStudio/OrcaLab | 入口 |
|----|------|--------|---------|-----------|------|
| 1 | 加载机器人 spawnable | `add_actor`、`publish_scene`、关节扫描 | XML | 是 | `run_load_mjcf_robot` |
| 2 | 加载 USD 物体 spawnable | PBR 材质（`MaterialInfo`）、碰撞体 | USDZ | 是 | `run_load_usd_object` |
| 3 | 加载完整 USD 场景 | 多 actor 组合、场景层级 | 资产包 | 是 | `run_load_usd_scene` |

### 第二阶段：场景构建（Lesson 4-11）

学习程序化构建场景。本阶段产出的场景可直接用于后续 RL 训练与其他课题。

| 课 | 主题 | 新概念 | 模式 | 为后续课题准备 | 入口 |
|----|------|--------|------|---------------|------|
| 4 | 最简场景 | 世界坐标系、重力、自由落体 | 在线 | Euler Lesson 1（pendulum 场景） | `run_empty_scene` |
| 5 | 程序化室内 | 批量 spawn、几何布局 | 在线 | 体操作（Lesson 10） | `run_indoor_room` |
| 6 | 仓库货架 | 阵列 spawn、碰撞优化 | 在线 | 导航/抓取课题 | `run_warehouse_scene` |
| 7 | 户外地形 | 高度图 spawnable | 在线 | 户外 locomotion | `run_outdoor_terrain` |
| 8 | YAML 配置驱动 | `SceneSpec`/`ActorSpec`、声明式描述 | 在线 | 可复现场景 | `run_scene_from_yaml` |
| 9 | 多源合并 | 命名空间隔离、XML+USDZ+资产包 | 在线 | 复合任务场景 | `run_scene_composition` |
| 10 | 光照系统 | 光源 spawn、阴影、HDR 环境 | 在线 | 视觉课题/数据采集 | `run_lighting_setup` |
| 11 | 域随机化 | 随机布局、可复现 seed | 在线 | RL 泛化训练 | `run_random_variation` |

> 第二阶段所有示例为**在线模式**，需 OrcaStudio/OrcaLab 运行并订阅对应资产包。

---

## 课程简介

### 第 1 课：加载机器人 spawnable — XML 资产订阅入口

通过 `OrcaGymScene.add_actor` 加载机器人 spawnable 资产（G1/Franka XML），扫描关节、
执行器、Body 信息，演示阻尼/摩擦参数修改。本课是所有涉及机器人的后续课题（Lesson 5-10
状态查询、Lesson 8 行走控制）的**资产加载前置**。

- **运行**：`python -m examples.scene_building.01_assets.run_load_mjcf_robot --robot g1`
- **需要 OrcaStudio/OrcaLab**：是
- **资产**：机器人 XML spawnable（资产库订阅）

---

### 第 2 课：加载 USD 物体 spawnable — PBR 材质与碰撞

加载家具/工具 USD spawnable，设置 PBR 材质（`MaterialInfo`），验证碰撞体生效。
本课演示非机器人资产的加载流程，为后续室内场景（Lesson 5）、随机化场景（Lesson 11）
做资产准备。

- **运行**：`python -m examples.scene_building.01_assets.run_load_usd_object --object chair`
- **需要 OrcaStudio/OrcaLab**：是
- **资产**：USDZ 物体 spawnable

---

### 第 3 课：加载完整 USD 场景 — 资产包多 actor 组合

加载完整 USD 场景（多 actor 组合的资产包），演示场景引用与层级。本课是"一键加载
预制场景"的范式，适合快速搭建复杂环境，为 Lesson 9 多源合并做铺垫。

- **运行**：`python -m examples.scene_building.01_assets.run_load_usd_scene --scene living_room`
- **需要 OrcaStudio/OrcaLab**：是
- **资产**：场景资产包 spawnable

---

### 第 4 课：最简场景 — 地面 + 重力 + 自由落体

搭建最简场景：地面 + 重力 + 一个自由落体方块 spawnable，演示世界坐标系与重力配置。
本课对应 Euler Lesson 1 的场景前置：学完本课后可直接接入 `OrcaGymEulerEnv` 进行
pendulum/box 仿真。

- **运行**：`python -m examples.scene_building.02_scene.run_empty_scene`
- **需要 OrcaStudio/OrcaLab**：是
- **为后续准备**：Euler Lesson 1-4（pendulum/box 仿真）

---

### 第 5 课：程序化室内房间 — 墙体/家具批量 spawn

程序化 spawn 房间：4 面墙 + 地板 + 天花板 + 摆放家具 spawnable。本课演示**批量布局**
能力，产出的房间场景可用于后续体操作（Euler Lesson 10）、室内导航等课题。

- **运行**：`python examples/scene_building/02_scene/run_indoor_room.py --width 5 --height 3`
- **需要 OrcaStudio/OrcaLab**：是
- **为后续准备**：Euler Lesson 10（体操作）、室内导航课题

---

### 第 6 课：仓库货架场景 — 阵列 spawn 与碰撞优化

spawn 仓库：货架阵列、过道、碰撞优化（简化碰撞 Mesh）。本课演示**大规模批量 spawn**
与碰撞性能权衡，产出的仓库场景适合导航、抓取、SLAM 等课题。

- **运行**：`python examples/scene_building/02_scene/run_warehouse_scene.py --rows 3 --cols 5`
- **需要 OrcaStudio/OrcaLab**：是
- **为后续准备**：导航/抓取课题

---

### 第 7 课：户外地形 — 高度图 spawnable

spawn 高度图地形 spawnable，演示纹理混合。本课产出户外地形场景，适合户外
locomotion、足式机器人越野等课题。

- **运行**：`python examples/scene_building/02_scene/run_outdoor_terrain.py --size 20`
- **需要 OrcaStudio/OrcaLab**：是
- **为后续准备**：户外 locomotion 课题

---

### 第 8 课：YAML 配置驱动 — 声明式场景描述

从 YAML 声明式描述 spawn 场景，演示配置驱动开发（`SceneSpec`/`ActorSpec` 数据结构）。
本课是**可复现场景**的关键：用 YAML 描述场景，团队协作时只需共享 YAML 文件即可
复现相同布局，为 RL 训练的实验可复现性提供基础。

- **运行**：`python examples/scene_building/02_scene/run_scene_from_yaml.py --config scene.yaml`
- **需要 OrcaStudio/OrcaLab**：是
- **为后续准备**：RL 训练实验可复现性

---

### 第 9 课：多源 spawnable 合并 — XML + USDZ + 资产包

合并多个 spawnable 资产（机器人 XML + 物体 USDZ + 场景资产包），演示命名空间隔离。
本课演示**复合场景**构建，适合需要"机器人 + 物体 + 环境"共存的复杂任务（如抓取、
操作、交互）。

- **运行**：`python examples/scene_building/02_scene/run_scene_composition.py`
- **需要 OrcaStudio/OrcaLab**：是
- **为后续准备**：抓取/操作/交互课题

---

### 第 10 课：光照系统 — 光源 spawn 与 HDR 环境

演示 HDR 环境光、点光源、聚光灯、阴影配置（基于 spawnable 场景）。本课产出的光照
场景适合视觉课题、数据采集、sim-to-real 视觉迁移。

- **运行**：`python examples/scene_building/02_scene/run_lighting_setup.py`
- **需要 OrcaStudio/OrcaLab**：是
- **为后续准备**：视觉课题、数据采集（Euler Lesson 9 视频录制）

---

### 第 11 课：场景随机变体 — 域随机化与可复现

域随机化：随机 spawn 物体位置/尺寸/颜色/摩擦，演示布局扰动。本课是 **RL 泛化训练**
的核心：通过随机化场景提升策略的鲁棒性，可复现的 seed 保证实验可对比。

- **运行**：`python examples/scene_building/02_scene/run_random_variation.py --seed 42 --num-objects 20`
- **需要 OrcaStudio/OrcaLab**：是
- **为后续准备**：RL 泛化训练（Euler Lesson 4 PPO 扩展）

---

## 环境准备

### 1. 安装依赖

```bash
cd /path/to/OrcaPlayground
conda activate orca
# OrcaGym 已随 orca 环境安装
```

### 2. 前置条件

| 依赖 | 说明 |
|------|------|
| OrcaGym | 已安装（`orca_gym` 包可在 Python 中 import） |
| OrcaStudio/OrcaLab | **所有课程需要**，需启动 gRPC 服务（默认 `localhost:50051`） |
| **OrcaPlaygroundAssets 资产包** | **所有课程需要**，需在 OrcaStudio/OrcaLab 中订阅（见下） |

### 3. 订阅 OrcaPlaygroundAssets 资产包

本课程所有示例通过 `OrcaGymScene.add_actor` 订阅 spawnable 资产，所需的资产由
**OrcaPlaygroundAssets 资产包** 提供。

> **📦 相关资产**：https://simassets.orca3d.cn/ **OrcaPlaygroundAssets 资产包**
>
> **🔧 是否需要手动拖动到布局中**：**否**
>
> 脚本通过 `asset_path` 引用 Studio 中已订阅的 spawnable 资产，运行时自动加载。

资产路径格式（参考 [examples/replicator](../replicator/README.md)）：

```
assets/<asset_hash>/default_project/prefabs/<asset_name>
```

**订阅步骤**：

1. 打开 OrcaStudio/OrcaLab
2. 进入资产商店 / 资产管理，搜索并订阅 **OrcaPlaygroundAssets** 资产包
3. 等待资产导入完成（spawnable 出现在上述路径）
4. 加载一个**空关卡**，点击运行
5. 运行对应课程的入口脚本

> 各课程所需的具体 spawnable 资产路径见源码中的 `_*_SPAWNABLE_PATH` 常量，
> 实现时需从资产库确认并替换占位符。

### 4. 验证安装

```bash
conda activate orca
python -c "from orca_gym.scene.orca_gym_scene import OrcaGymScene; print('OK')"
```

---

## 核心模式

所有示例遵循 [examples/replicator](../replicator/run_actors.py) 的三段式流程：

```python
from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor, MaterialInfo
import orca_gym.utils.rotations as rotations
import numpy as np

# 1. 构造场景（gRPC 连接 Studio）
scene = OrcaGymScene("localhost:50051")

# 2. add_actor 订阅 spawnable 资产
actor = Actor(
    name="my_actor",
    asset_path="assets/<hash>/default_project/prefabs/<asset_name>",
    position=np.array([0, 0, 1.0]),
    rotation=rotations.euler2quat(np.array([0, 0, 0])),
    scale=1.0,
)
scene.add_actor(actor)

# 3. publish_scene 发布到 Studio
scene.publish_scene()

# 4. （可选）设置材质
material_info = MaterialInfo(base_color=np.array([0.8, 0.2, 0.2, 1.0]))
scene.set_material_info("my_actor", material_info)

# 5. 接入仿真循环（参考 examples/replicator/run_simulation.py）
#    通过 gym.register + gym.make 启动 OrcaGymEulerEnv
```

### ActorCollector 工具类

`_common/actor_collector.py` 提供 spawnable 资产规格收集器，替代逐个 `scene.add_actor`
调用，便于批量管理与命名空间隔离：

```python
from examples.scene_building._common.actor_collector import ActorCollector

collector = ActorCollector()
collector.set_world(gravity=(0, 0, -9.81))
collector.add_actor("box", spawnable_path="<方块资产>", pos=(0, 0, 1))
collector.add_robot("g1", spawnable_path="<G1 资产>", pos=(0, 0, 1.05))
collector.spawn_all(scene)
scene.publish_scene()
```

---

## 与其他课程的关系

本课程是 OrcaPlayground 的**场景构建基础课**，产出的场景可直接用于：

| 后续课程 | 依赖的本课程示例 | 说明 |
|---------|----------------|------|
| Euler Lesson 1-4（pendulum/box 仿真） | Lesson 4（最简场景） | 提供基础场景前置 |
| Euler Lesson 5-10（G1 在线验证） | Lesson 1（机器人加载） | 提供机器人 spawn 前置 |
| Euler Lesson 4（PPO 训练） | Lesson 8（YAML 配置） | 提供可复现训练场景 |
| Euler Lesson 9（视频录制） | Lesson 10（光照系统） | 提供视觉数据采集场景 |
| RL 泛化训练 | Lesson 11（域随机化） | 提供随机化训练场景 |
| 导航/抓取课题 | Lesson 5-6（室内/仓库） | 提供任务场景 |

---

## 常见问题

### Q1：运行时报错 "spawnable 路径不存在"

**原因**：`asset_path` 引用的 spawnable 资产未在 OrcaStudio/OrcaLab 中订阅或导入。

**解决**：

1. 确认已订阅 **OrcaPlaygroundAssets** 资产包（见"环境准备"）
2. 等待资产导入完成
3. 确认源码中 `_*_SPAWNABLE_PATH` 常量与资产库实际路径一致

### Q2：`scene.add_actor` 报名称冲突

**原因**：Actor 的 `name` 必须唯一，重复名称会导致冲突。

**解决**：参考 replicator，运行前构造临时 scene 调用 `publish_scene()` 清空现有内容：

```python
temp_scene = OrcaGymScene(addr)
temp_scene.publish_scene()  # 清空
temp_scene.close()
```

### Q3：修改场景后 Studio 视口无变化

**原因**：修改场景后未调用 `scene.publish_scene()`。

**解决**：所有 `add_actor` / `set_material_info` / `set_light_info` 操作后必须调用
`scene.publish_scene()` 才能生效。

### Q4：光照示例运行后光源未生效

**原因**：OrcaGym 光源配置 API 可能存在缺口。

**解决**：本课程标注了 API 缺口（见源码 TODO），降级方案为脚本引导用户在 Studio UI
手动配置光照，脚本仅 spawn 演示物体并读取打印当前光照状态。待 OrcaGym 扩展光源 API
后补全。

### Q5：`ModuleNotFoundError` 或语法错误（数字开头目录）

**原因**：目录名以数字开头（如 `01_assets`、`02_scene`）不能作为 Python 包路径，
不能使用 `python -m examples.scene_building.01_assets.xxx` 方式运行。

**解决**：使用脚本直接运行方式，从项目根目录执行：

```bash
cd /path/to/OrcaPlayground
python examples/scene_building/02_scene/run_empty_scene.py
```

---

## 📚 相关文档

- [00_设计方案_总体架构.md](../../.trae_history/Playground_PRD_v2_Examples_Plan/00_设计方案_总体架构.md)
- [01_架构设计.md](../../.trae_history/Playground_PRD_v2_Examples_Plan/01_架构设计.md)
- [03_示例开发计划.md](../../.trae_history/Playground_PRD_v2_Examples_Plan/03_示例开发计划.md)
- [Euler 课程总纲](../euler/TUTORIAL.md)（叙事风格参考）
- [Replicator 示例](../replicator/README.md)（基本逻辑参考）
