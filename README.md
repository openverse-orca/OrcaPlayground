# OrcaPlayground — 机器人仿真新手村

**OrcaPlayground 是一个"可读、可改、可跑"的机器人仿真教学示例库**：
如果你会一点 Python，但从没碰过物理仿真或机器人，这里有 48 节
循序渐进的动手课，带你从"往场景里放一个方块"一路走到"控制机械臂
完成任务并判断成败"——每一课的定量结果都能和物理公式对上号。

> 一句话介绍：**在 OrcaLab 里，用真实的物理引擎，学会让机器人为你动起来。**

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

完整规划见 [docs/beginner/REQUIREMENTS.md](docs/beginner/REQUIREMENTS.md)。

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
**OrcaPlaygroundAssets**（详细步骤见[第 01 课 README](examples/euler/beginner/stage1_scene_basics/lesson_01_hello_world/README.md)）。
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

## 分支说明

当前分支 `feat/playground-beginner-redesign` 正在从零重建新手主线，
最终将**整体替换 `dev` 分支**：旧 Euler 课程（01–11）与 embodied
高级样例在合入主线时移除，替换前在 `dev` 上打 `legacy/pre-redesign`
存档标签供历史访问。
