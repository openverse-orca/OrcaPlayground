# OrcaPlayground — 机器人仿真新手村

**OrcaPlayground 是一个"可读、可改、可跑"的机器人仿真教学示例库**：
如果你会一点 Python，但从没碰过物理仿真或机器人，这里有 48 节
循序渐进的动手课，带你从"往场景里放一个方块"一路走到"控制机械臂
完成任务并判断成败"——每一课的定量结果都能和物理公式对上号。

> 一句话介绍：**在 OrcaLab 里，用真实的物理引擎，学会让机器人为你动起来。**

当前版本：26.10.1.19 | 版本分支：release/26.10.1.19

📌 **版本兼容性**：请使用与当前 Playground 版本配套的 OrcaLab 版本，详见下方版本对照表。

📌 **版本发布说明**：本项目采用版本分支策略进行版本管理：

| OrcaLab 版本 | Playground 版本 |
|-------------|-----------------|
| OrcaLab 26.4.3 | release/26.4.3 |
| OrcaLab 26.5.1 | release/26.5.1 |
| OrcaLab 26.6.3 | release/26.6.1 |
| OrcaLab 26.7.1 ～ 26.7.3 | release/26.7.1 |
| OrcaLab 26.8.1 | release/26.8.1 |
| OrcaLab 26.8.2 | release/26.8.2 |
| OrcaLab 26.9.1 | release/26.9.1 |
| OrcaLab 26.10.1.19 | release/26.10.1.19（当前） |

正式版本发布后，会将 main 分支的代码合入对应的版本分支（如 release/26.10.1.19） 版本分支命名格式：release/<主版本>.<次版本>.<修订号> 每个正式版本对应一个版本分支，用于版本追踪和热修复 开发阶段请使用 main 或 dev 分支，正式发布后合入版本分支

## 这里和别的教程有什么不同

- **数字说话**：每课都有"实测 vs 理论"的验证点。单摆周期实测 1.667 s
  vs 理论 1.666 s、伺服静差 1.92° 逐位命中 mgL·sinθ/kp、恒力矩下
  α = τ/I 精确到千分位——你写下的每行代码都被物理世界严格验收
- **源码即教材**：样例库以"可读可改的源码包"形式交付（不装进
  site-packages），每课四件套 `run.py` / `README.md` / `example.yaml`，
  配方区集中所有可调参数，改一个常数就能做实验
- **三种玩法自由切换**：每课既可以在 OrcaLab 里自己拖拽摆场景（层 3
  场景无关设计），也可以 `--default-scene` 让脚本自动摆好教学场景
- **挑战驱动**：每课带一个 swap / combine / predict 型小挑战，
  "先预测再运行对照"是贯穿全程的学习方法

## 课程地图（已交付 01–30 课）

| 阶段 | 主题 | 你将学会 |
|---|---|---|
| **1. 场景基础**（01–06） | 资产订阅 / 加载 / 组合 | 把物体、机器人、整张场景放进仿真世界 |
| **2. 场景编辑**（07–12） | 移动 / 旋转 / 缩放 / 改色 / 复制 / 删除 | 像搭积木一样编辑场景 |
| **3. 仿真时间**（13–18） | 步进 / 重置 / 读状态 / 重力 / 时步 | 让时间流动，理解仿真循环的本质 |
| **4. 物理交互**（19–24） | 外力 / 碰撞 / 关节 / 质量 / 摩擦 / 堆叠 | 牛顿定律在仿真里的精确模样 |
| **5. 单关节控制**（25–30） | 读关节 / 写状态 / 三种伺服 / 手写 PD | 从"上帝模式"到真电机，亲手复刻 PD 控制器 |
| 6. 机器人控制（31–36） | 差速小车 / 机械臂 / 夹爪 | ⏳ 规划中 |
| 7–8. 任务实践（37–48） | 相机 / 测距 / 任务判定 / 随机化评测 | ⏳ 规划中 |
| 选修 | GPU 后端等 9 个专题 | ⏳ 暂缓发布 |

完整规划见 [docs/beginner/REQUIREMENTS.md](docs/beginner/REQUIREMENTS.md)，
全部课程使用的配套资产包（教具 / 机器人模型 / 美术与物理资源储备）
见 [docs/beginner/ASSETS.md](docs/beginner/ASSETS.md)。

## 如何开始（从零到跑起第一课，只需这一次）

**第 1 步 · 拿到本仓库源码**
```bash
git clone https://github.com/openverse-orca/OrcaPlayground.git OrcaPlayground
cd OrcaPlayground
```
（示例库是"可读可改的源码包"，不装进 site-packages——你改的每一行都直接生效。）

**第 2 步 · 准备 Python 环境**
需要 `orca` conda 环境（内置本库全部依赖，含 orca_gym 开发版）：
```bash
conda activate orca
```
> 还没有这个环境？先克隆并安装 OrcaGym 开发版
> （https://github.com/openverse-orca/OrcaGym ，安装指引见其 README），再回来继续。

**第 3 步 · 启动 OrcaLab**
打开 OrcaLab 客户端并进入 Studio 视口（课程通过 `localhost:50051` 与它通信）。

**第 4 步 · 订阅教学资产包**
在资产库 [https://simassets.orca3d.cn/](https://simassets.orca3d.cn/) 搜索并订阅
**新手实例库配套资产**（英文名 OrcaPlaygroundAssets）（详细步骤见[第 01 课 README](examples/euler/beginner/stage1_scene_basics/lesson_01_hello_world/README.md)；
包里有哪些教具、机器人和美术资源，见[配套资产包介绍](docs/beginner/ASSETS.md)）。
订阅后要在 OrcaLab 中进行资产同步：顶栏 资产 → 同步资产（也可以重新启动 OrcaLab）。

**第 5 步 · 跑你的第一课**
```bash
python -m examples.euler.beginner.stage1_scene_basics.lesson_01_hello_world.run
```
终端出现"场景已就绪"、视口出现地面和方块——恭喜，链路全通。
之后每一课都只是换一个模块名运行，环境再也不用动。

## 项目约定

- **目标用户**：会运行少量 Python，无机器人 / 物理仿真 / 强化学习经验
- **技术底座**：所有课程基于 `OrcaGymEulerEnv`（Euler 体系），状态
  读写与控制全部走公共 API——学到的调用方式在真实项目里原样可用
- **质量门槛**：阶段 4 / 5 全部课程实跑验证通过（数字命中理论），
  ruff `--select SLF001` 全仓零报警
- 所有 AI 代理与贡献者必须遵守 [AGENTS.md](AGENTS.md)：
  使用 `orca` conda 环境、禁止穿墙访问 `_` 前缀内部属性

## 旧样例去哪了（老用户必读）

本仓库 `dev` 分支已于 2026-10-14 由新手主线（原 `feat/playground-beginner-redesign`）
**整体重建**。重建前 dev 上的旧样例**没有删除**，全部冻结在分支
`code-frozen-dev-20261014`（重建前 dev 的完整快照，末位提交 `71c38f9`），
包括：

- `examples/embodied/` — 布料（cloth）、无人机（drone_driver）、G1、
  数字人（character / Remy）、流体（fluid）、d12、replicator、
  轮式底盘（wheeled_chassis）、xbot、zq_sa01 等高级样例
- `examples/euler/` — 旧编号课程 00–12（含 SLAM 建图）与
  `sensor_provider/`（灵巧手、传感器）
- `examples/scene_building/` — 旧场景搭建教程
- `tests/sensor_provider/` 及 `DEVELOPER_GUIDE.md` 等配套文件

找回某个旧样例（以 cloth 为例）：

```bash
git fetch origin
git checkout code-frozen-dev-20261014 -- examples/embodied/cloth
```

或完整检出旧线：`git checkout code-frozen-dev-20261014`。
