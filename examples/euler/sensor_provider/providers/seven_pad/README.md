# SevenPad：七触面的合力、噪声与电容近似响应

本目录提供一个七触面传感器的型号模型、契约和算法源码，采用 MIT 许可证。算法演示接触力矢量求和、各实例独立的随机噪声，以及简化的线性电容响应。七块触面均由基本长方体组成，不对应真实厂商产品，也不是实物标定算法或电气仿真。

**完整五指灵巧手演示**位于 OrcaPlayground 仓库的 `examples/euler/sensor_provider/`，包含完整手部模型、控制和实时图表。`hand` 模式在每根手指上使用一个 SevenPad 实例，五指共 35 块触面。先阅读该目录的 `README.md`，确认 OrcaGym 支持版本和准备方式。准备好环境后，在 **OrcaPlayground 根目录**、`orca` 环境运行：

```bash
python -m examples.euler.sensor_provider.run --example hand
```

图表显示各指的法向力和切向力，来自 SevenPad 的计算输出。

厂商包 ID（`provider_id`）和型号 ID（`type_id`）均为 `com.orca.examples.seven_pad`。SDK 目标和厂商包版本均为 `1.0.0`，C ABI 为 2。`contract.json` 使用 schema 4，生成输入结构 `SevenPadInput`，契约 ID 为 `com.orca.examples.seven_pad.input.v1`。

## 型号模型与输入

`sensor_base` 是模块的安装 site（带位置和姿态的参考点）。七块固定触面对应 `force_f1` 至 `force_f7`，每块尺寸为 `0.012 × 0.012 × 0.004 m`，一块在中心、六块环绕。触面顶端和沿 +Z 测距的参考系 `range_frame` 均位于模块局部 `Z = 0.012 m`。外壳只用于显示；模型不包含原生传感器、插件、网格资产或第三方几何。

Orca 按契约采集以下输入，算法动态库不调用物理引擎：

- `proximity`：使用 `orca.raycast.v1`，从 `range_frame` 沿局部 +Z 测距，最大距离 0.1 m，`exclude_self=true`。距离单位为 m，未命中记为 -1。
- `contacts`：使用 `orca.contact.v1`，按 `force_f1` 至 `force_f7` 的顺序采集接触，统一在 `range_frame` 中表达，`exclude_internal=true`，容量为 1024 条。每条只包含触面编号 `pad_index`、法向力 `normal[3]`、切向力 `tangent[3]`，分别映射能力字段 `surface_index`、`normal_force`、`tangential_force`。

`pad_index` 从 0 开始，因此 `force_f1` 对应 0，`force_f7` 对应 6。生成的 C 结构通过 `contacts_count` 表示本步有效条数，通过 `contacts[i]` 访问记录。请使用生成头文件，不要手工编写内存布局。

输入能力的完整规则见 SDK 根目录下的 `docs/capabilities/raycast.md` 和 `docs/capabilities/contact.md`。

## 算法与输出

对每块触面 i，先分别累加其法向力**矢量**和切向力**矢量**，得到 `N_i` 和 `T_i`；再跨全部触面求和，得到总矢量 `N` 和 `T`。同一触面内以及不同触面上的相反力都可能抵消。算法不按接触对方分组，也不先将各接触力取模后相加。

启用噪声时，分别给 `N` 和 `T` 的三个分量叠加独立高斯噪声：均值为 0，标准差为 `noise_scale` N，每次成功计算使用六个高斯样本。输出为连续的 `float64[11]`，下表的偏移按 double 元素计数：

| 偏移 | 通道 | 含义 |
| --- | --- | --- |
| 0 | `distance` | 原样输出 `proximity`，单位 m，保留未命中标记 -1 |
| 1 | `normal_force` | 加噪声后的法向总矢量 N 的模长，单位 N |
| 2 | `tangential_force` | 加噪声后的切向总矢量 T 的模长，单位 N |
| 3 | `direction` | 加噪声后 `atan2(T_y, T_x)` 的结果，归一化到 `[0, 2π)`，单位 rad；XY 分量同时为零时取 0 |
| 4–10 | `capacitance[7]` | 各触面的 `capacitance_base + capacitance_gain * norm(N_i + T_i)`，依次对应 `force_f1` 至 `force_f7` |

`norm` 表示三维矢量的欧氏长度。电容通道使用**未加噪声**的触面合力，是确定性的线性教学近似，单位为任意 `raw` 值，不是法拉或实测电容。无接触时，每块触面的该通道恰好等于 `capacitance_base`。

## 实例参数与重放

参数按 `provider.json` 中的声明顺序传入，每个参数都须有限且在 `[0, 1000000]` 内：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `noise_scale` | 0.01 | 总力矢量各分量的噪声标准差，单位 N；设为 0 完全关闭噪声 |
| `capacitance_base` | 1.0 | 七路 raw 电容通道的共同基线 |
| `capacitance_gain` | 0.1 | 每块触面的合力模长到 raw 值的线性系数 |

每个实例独立保存参数、随机数引擎和高斯分布状态。`reset` 保留参数，重新设置种子并清除分布缓存；在相同构建和平台上，用相同 seed 重放相同输入序列可得到完全相同的输出。不同平台或 C++ 标准库之间不保证逐位相同。矢量分量的噪声均值为零，并不代表取模后的力大小误差均值也为零。

距离、编号或记录数无效、输入含 NaN/无穷值，以及算术溢出，都会返回错误状态。11 个输出全部检查通过后才一起写入；失败时不会部分更新输出，也不会推进随机序列。

## 构建厂商包

以下命令需要完整 SDK，并在 SDK 根目录执行。若正在阅读已构建厂商包中的 README，源码位于完整 SDK 的 `examples/seven_pad/`。使用 CMake 和 C++17 编译器即可，随包契约工具不需要另装 Python：

```bash
cmake -S examples/seven_pad -B build/seven-pad -DCMAKE_BUILD_TYPE=Release
cmake --build build/seven-pad --config Release
```

产物位于 `build/seven-pad/providers/seven_pad/`，包含动态库（当前 Linux 平台为 `liborca_seven_pad.so`）、`provider.json`、`contract.json`、`model.xml`、展示配置 `presentation.json`、本 README 和 SDK 的 MIT 许可证。

完整 SDK 的三个原生验收测试包含 SevenPad 的合力、电容、实例独立和噪声重放检查；运行方法见 SDK 的 `examples/acceptance/README.md`。
