# 传感器插件演示：完整灵巧手与四种小型场景

这里既有完整机械臂与五指灵巧手，也有四种传感器示例：TouchGrid、ContactGrid、Rangefinder、SevenPad。
场景、模型资源和预编译插件随示例提供，运行后可以观察实际传感器输出的实时图表。
完整灵巧手提供两种展示方式，复用这些插件，不是另外两种传感器。

## 先选想看的内容

| 想看什么 | `--example` | 运行入口 |
| --- | --- | --- |
| 完整灵巧手：五指各一对法向力/切向力曲线 | `hand` | [完整灵巧手](#完整灵巧手) |
| 完整灵巧手：五指各一幅 4×4 接触力网格及一路测距 | `hand_grid` | [完整灵巧手](#完整灵巧手) |
| 两块触面的 4×4 平面力网格 | `touch_grid` | [四种小型场景](#四种小型场景) |
| 两个测量位置的 4×4 角度力网格 | `contact_grid` | [四种小型场景](#四种小型场景) |
| 两路距离，未命中时显示 NO HIT | `rangefinder` | [四种小型场景](#四种小型场景) |
| 七触面合力、近似电容和距离 | `seven_pad` | [四种小型场景](#四种小型场景) |

## 运行前准备

本示例依赖配套的 OrcaGym `feat/sensor-provider-integration` 分支，当前验证提交为 `8896ad77`。
**该功能尚未正式发布**；现有 PyPI 版本不保证包含本例所需接口。验证本 PR 时，先按
[配套源码安装步骤](developer.md#配套版本与本地安装)安装对应实现。正式发布后，可按
[OrcaGym 官方安装说明](https://github.com/openverse-orca/OrcaGym#安装)安装支持该功能的官方 `orca-gym` 发布版本。

在支持的平台上，Host 随 OrcaGym 提供并自动加载；不需要另外安装 Host、厂商 SDK 或编译插件。
当前演示面向 Linux x86_64（glibc ≥ 2.35），具体系统兼容范围以 OrcaGym 发布说明为准。
这里的平台限制只针对插件演示，不表示 OrcaGym 其他功能不能在 Windows 使用。

以下所有命令都在 **OrcaPlayground 仓库根目录**执行，使用 `orca` conda 环境。若尚未创建环境，先运行下面的命令；已有环境可跳过：

```bash
conda create -n orca python=3.12
```

激活环境：

```bash
conda activate orca
```

请在此环境中按上述配套版本说明完成 OrcaGym 的安装，然后安装示例的绘图依赖：

```bash
python -m pip install -r examples/euler/sensor_provider/requirements.txt
```

本示例的插件和模型资源使用普通 Git 保存，正常克隆即可取得完整文件，无需 Git LFS。

如果使用压缩包，请确认其中包含完整的 `examples/euler/sensor_provider/providers/` 和
`examples/euler/sensor_provider/scenes/` 目录。本示例直接加载自带场景，无需向 OrcaLab 布局拖入资产。

## 运行

### 四种小型场景

每条命令分别运行一个场景，并打开对应的实时图表：

```bash
# TouchGrid：两块触面的平面力网格。
python -m examples.euler.sensor_provider.run --example touch_grid

# ContactGrid：两个测量位置的角度力网格。
python -m examples.euler.sensor_provider.run --example contact_grid

# Rangefinder：两路距离，未命中时显示 NO HIT。
python -m examples.euler.sensor_provider.run --example rangefinder

# SevenPad：七触面合力、近似电容和距离。
python -m examples.euler.sensor_provider.run --example seven_pad
```

### 完整灵巧手

两种模式都加载[完整灵巧手场景](scenes/dexhand/scene.xml)，执行渐进接近和闭手动作。
选择其中一条命令运行，打开的是传感器图表窗口：

```bash
# 完整灵巧手：每指法向力、切向力。
python -m examples.euler.sensor_provider.run --example hand

# 完整灵巧手：五指网格和测距。
python -m examples.euler.sensor_provider.run --example hand_grid
```

`hand` 使用五个 SevenPad 实例，默认关闭噪声；`hand_grid` 在 EulerEnv 中使用五个
ContactGrid 和五个 Rangefinder 实例。模型组成、安装位置与输出含义见[灵巧手模型说明](scenes/dexhand/README.md)。

### 保存图表与结束运行

默认演示 10 s 仿真；结束后保留最终图表，关闭窗口退出。提前关闭会停止运行。
`--fps` 只控制绘图刷新，不改变物理或插件计算频率。

没有图形界面时，可保存 PNG；目标文件必须尚不存在：

```bash
python -m examples.euler.sensor_provider.run --example hand --output build/hand-forces.png
```

同样可以为其他示例添加 `--output`。也可以直接执行
`python examples/euler/sensor_provider/run.py --example touch_grid`。

## 如何理解读数

图表读取实际 DLL 输出，不伪造读数。小型载荷夹具稳定后约为 0.981 N 和 1.962 N。
灵巧手使用已有渐进接近/闭手控制，是开环演示，不承诺抓取成功或每指持续接触。
电容是教学近似的 raw 数值，不是物理电容标定。算法和模型不关联实际传感器厂商。

完整场景统一放在 `scenes/<场景名>/scene.xml`，所需资源随场景放在同一目录。
`hand` 与 `hand_grid` 共用 `scenes/dexhand/scene.xml`。
模型格式兼容 MuJoCo MJCF。几何、碰撞和安装位姿使用标准模型元素，传感器实例声明放在
`custom` 中，由 Orca 解析并调用插件。格式兼容不代表其他读取工具会执行 Orca 的传感器算法。

更多：[进阶用法](advanced.md) · [灵巧手模型说明](scenes/dexhand/README.md) · [插件版本与校验](providers/README.md)
