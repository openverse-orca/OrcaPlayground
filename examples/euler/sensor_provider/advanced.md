# 进阶用法

普通运行只需 [README](README.md) 的统一入口。这里介绍诊断和程序化调用，
使用 README 指定的配套 OrcaGym 和随包插件，无需另装 Host 或 SDK。

## 运行边界

`run.py` 选择对应的 Euler 环境运行器；控制轨迹、物理模型和 DLL 算法保持一致。

- 四个夹具由 `playground.py` 读取独立 `scenes/<型号>/scene.xml`。
- `hand` 使用 `dexhand_seven_pad.py` 与 `ProviderSceneEnv`，精确映射五指各七个触面，逐物理步记录法向/切向矢量合力的模长。
- `hand_grid` 使用 `dexhand_euler.py`，继承 `OrcaGymEulerEnv`，物理和采样都在唯一的环境中执行；输出进入任务定义的 Gym observation。

六个入口均由 `OrcaGymEulerEnv` 推进唯一的物理仿真，并管理传感器生命周期。
四个夹具与 `hand` 共用应用侧 `ProviderSceneEnv`；`hand_grid` 沿用其任务环境。

默认所有演示运行 10 s。`--steps` 在夹具中计 5 子步的逻辑步、`hand` 中计物理步、
`hand_grid` 中计 4 子步的 Gym 步。改变控制采样周期、模型参数或仿真版本可能改变接触曲线。

## 传感器绑定

`scenes/<场景名>/scene.xml` 是完整挂载场景，`providers/<型号>/model.xml` 是单个型号的资产。
灵巧手的两种演示共用 `scenes/dexhand/scene.xml`，其网格和纹理放在同目录的 `meshes/` 中。
ContactGrid / Rangefinder 使用标准 `custom/text + tuple` 精确绑定 site；
TouchGrid / SevenPad 使用 `object/<alias>` tuple 精确绑定已有 body/site，不根据实例名称猜对象，也不加载型号几何资产。
SevenPad 的噪声参数与随机种子写在 XML 的 numeric/text 声明中。DLL 路径不从 XML 读取，只加载应用选定的可信包。

Rangefinder 的 `oscillating_targets` keyframe 提供初始位移；环境在加载和 reset 时恢复它，不为初始化额外步进。
`hand` 在临时 XML 中将十个 site 声明替换为五个 SevenPad 对象声明，只调整绑定和资源查找路径，保留模型几何、原文件及网格。
手动查看模型时，也需要选择相同初始状态，才能观察到相同振荡。
模型格式兼容 MuJoCo MJCF；面向维护者的格式兼容性验证见 [开发者说明](developer.md)。

## 调整灵巧手

详细入口支持 `--xml`、噪声和控制配置，默认仍使用随包插件及 OrcaGym Host：

```bash
python -m examples.euler.sensor_provider.dexhand_seven_pad \
  --control grasp --plot --steps 10000 --noise-scale 0
python -m examples.euler.sensor_provider.dexhand_euler \
  --control grasp --plot --steps 2500 --frame-skip 4
```

`--pregrasp-mode ramp` 和 `--ring-target-scale 1.004` 保持既有默认。
`step` 模式仅供阶跃对照，可能产生很大冲击，不应用滤波或缩放伪装成正常力值。
模型圆柱使用 `<freejoint/>`，不继承手指关节阻尼。

兼容性验证、本地构建调试和插件更新见 [开发者说明](developer.md)，
不属于用户运行步骤。
