# OrcaLocomotion Quick Start

本示例通过 PyPI 包 `orca-locomotion` 在 OrcaLab 中回放 Unitree Go2 / G1 运动控制策略。
包内自带三个可直接运行的 Checkpoint 示例，不需要克隆OrcaLocomotion 仓库。
更多训练、资产和源码说明见 [OrcaLocomotion](https://github.com/openverse-orca/OrcaLocomotion/tree/dev)。

| 仿真 | 实机 |
| --- | --- |
| <img src="output.gif" alt="OrcaLocomotion simulation preview" width="480"> | <img src="physical.gif" alt="OrcaLocomotion physical preview" width="480"> |

## 安装包

建议使用 OrcaLab 的 Python 3.12 环境：

```bash
conda create -n orcalab python=3.12
conda activate orcalab

## 安装 OrcaLab
pip install orca-lab 

## 安装 OrcaLocomotion 包
pip install --extra-index-url https://py.mujoco.org --extra-index-url https://pypi.nvidia.com orca-locomotion==0.1.6
```

安装后使用如下脚本,导出包内示例脚本和 预制模型 Checkpoint：

```bash
python -m orca_rl.examples.play_examples
```

导出后当前目录会出现：

```text
orca_locomotion_example/
├── checkpoints/
├── play_go2_flat.sh
├── play_go2_rough.sh
└── play_g1_flat.sh
```

## OrcaLab 资产订阅

推理前需要在 OrcaLab 中订阅 `unitree_robots` 资产包。Go2 和 G1 机器人预制件均来自该资产包。

运行 `Unitree-Go2-Rough` 还需要订阅 rough terrain 对应的渲染资产(在订阅搜索 框直接搜索)：

```text
OrcaPrimitiveTerrainXml
MjlabRough5x5xml_20260527
```

## 启动 OrcaLab

先启动 OrcaLab，然后选择：

```text
运行 -> 开始模拟 -> 无仿真程序 -> 启动
```

## 三个示例任务

#### 1. Unitree-Go2-Flat

Go2 平地速度跟踪策略，面向四足机器人在平整地面上的基础移动能力。
策略会根据给定速度命令完成前进、后退和横向移动，适合展示 Go2 在平地场景中的稳定步态和速度跟随效果。

```bash
bash orca_locomotion_example/play_go2_flat.sh
```

#### 2. Unitree-G1-Flat

G1 平地速度跟踪策略，面向人形机器人在平整地面上的基础行走能力。
策略会根据给定速度命令完成平地行走，适合展示 G1 在双足运动中的姿态保持、步态切换和速度跟随效果。

```bash
bash orca_locomotion_example/play_g1_flat.sh
```

#### 3. Unitree-Go2-Rough

![注意：运行环境要求](https://img.shields.io/badge/%E6%B3%A8%E6%84%8F-%E8%BF%90%E8%A1%8C%E7%8E%AF%E5%A2%83%E8%A6%81%E6%B1%82-red)

> **推荐系统：Ubuntu 22.04 或 Ubuntu 24.04。**
>
> **Windows 原生环境、WSL 和虚拟机环境当前不兼容，不推荐用于运行该示例。**

Go2 粗糙地形速度跟踪策略，当前示例模型专注于通过碎石崎岖路面，示例中位于左上侧的地形可用于测试。
如果需要其他策略，如带有跨越、可通过性、斜坡、楼梯等能力的策略，请参照 OrcaLocomotion 里的训练部分自行训练并导入。
这个示例带键盘控制：按住方向键或小键盘方向键才会发送速度命令，松开后速度归零。

```bash
bash orca_locomotion_example/play_go2_rough.sh
```

键盘控制：

```text
Up / 8       前进
Down / 2     后退
Left / 4     左移
Right / 6    右移
Z / 7        左转
C / 9        右转
Space / 5    速度归零
Q            退出
```
