# TouchGrid：可重复挂载的 4×4 触觉力网格

这是完整的厂商接入样板，不是经过标定的真实传感器模型。型号只定义一次；场景可以挂载多个实例，各实例由 Orca 独立解析名称、采集数据、调用算法动态库并返回结果。

填写契约前先看完整 SDK 中的 `docs/capabilities/index.md`。本例的参数、默认值、`fields` 映射和 `exclude_internal` 说明见 `docs/capabilities/contact.md`，`sample_time` 见 `docs/capabilities/sampling.md`；这些路径均相对于 SDK 根目录，无需先阅读 JSON Schema。

| 文件 | 厂商定义的内容 |
| --- | --- |
| `provider.json` | 稳定的 `type_id`、动态库文件名、型号资产、契约及算法配置 |
| `model.xml` | 整块触面的结构、外观、安装基准和固定内部名称 |
| `contract.json` | 算法需要的采样时间与接触记录，以及 `float64[4,4]` 输出 |
| `touch_grid.cpp` | 类型描述符注册与 `create/reset/compute/destroy` 回调 |

## 型号与实例

XML 中的 `sensor_base` 是安装 site（带位置和姿态的参考点），`force_f1` 是触面 body（刚体节点），其直属 `force_f1_shape` 是唯一的碰撞与显示 geom（几何体）。`surface_frame` 在上表面中心，定义接触位置和力的参考坐标系。XML 不包含原生传感器或插件，也不包含场景载荷。

模型长宽各 0.06 m、厚度 0.02 m，安装原点在底面中心。场景挂载后，Orca 为每个实例加上自己的命名空间；厂商始终只声明 `force_f1` 和 `surface_frame`，不需要知道场景中的左右名称。

4×4 是算法划分的输出网格，模型中仍然只有一块物理触面。

## 数据如何进入算法、如何返回

1. Orca 根据契约找到当前实例的 `force_f1` 和 `surface_frame`。
2. 每个物理子步采集该触面的接触，转换到 `surface_frame`；同时取得同一源状态的采样时间。
3. Host 按生成的 C 布局填充 `TouchGridInput`：`sample_time`、`contacts_count`、`contacts[64]`。每条接触含触面索引、对方刚体 ID、位置、法向力、切向力。超过 64 条报错，不截断。
4. Host 调用该实例的 `compute` 回调。算法按接触点的局部 x/y 选择网格，将法向力大小乘以 `gain` 后累加到对应格。
5. Host 保存 16 个输出值；OrcaGym 读取时得到独立拥有内存的 NumPy `float64[4,4]` 数组。

行索引沿局部 +Y、列索引沿局部 +X 增大；每格单位 N，是力，不是压力 Pa。超过算法采样范围的接触不计入网格。当前算法不使用对方刚体 ID、切向力及时间进行额外建模；保留这些输入是为了展示完整的接触结构传递。

具体读取为 `grid[row, column]`：`row` 对应 y，`column` 对应 x，索引均从负坐标端的 0 增至正坐标端的 3。默认两个轴的范围都是 `[-0.03, 0.03]` m，每格宽 0.015 m；内部分界点归入较大索引，正端点 0.03 m 归入最后一格。例如局部 `(x, y) = (-0.02, 0.02)` m 的接触写入 `grid[3, 0]`。C 数组使用行优先布局，元素偏移为 `row * 4 + column`。

`global_parameters` 在实例创建时传入，默认 `extent_x=0.06`、`extent_y=0.06`、`gain=1.0`。前两者是以 `surface_frame` 为中心的算法采样宽度，不会缩放 XML。配置属于各实例自己的 handle；reset 不改变配置。

## 构建厂商包

以下命令需要完整 SDK，并在 SDK 根目录执行；算法源码位于其中的 `examples/touch_grid/`。

```bash
cmake -S examples/touch_grid -B build/touch-grid -DCMAKE_BUILD_TYPE=Release
cmake --build build/touch-grid --config Release
```

产物在 `build/touch-grid/providers/touch_grid/`，包含 `provider.json`、`model.xml`、`contract.json`、动态库（当前 Linux 平台为 `liborca_touch_grid.so`）和许可证。构建使用 SDK 预编译工具，不需要安装 Python；算法动态库不依赖 Python 或物理引擎运行时。

需要查看生成的结构时，可单独运行：

```bash
bin/linux-x86_64/orca-sensor-tool contract examples/touch_grid/contract.json \
  build/touch_grid_input.h \
  --model examples/touch_grid/model.xml --mount-site sensor_base --sdk-version 1.0.0
```

本包固定以 SDK `1.0.0` 配置生成和构建；厂商包版本同为 `1.0.0`；契约 ID 为 `com.orca.examples.touch_grid.input.v3`；契约声明格式为 schema 4；C ABI 保持 2。协议编号不随首次公开版本重置。手工生成时也必须选择同一 SDK 目标，不能随工具升级自动改标签。契约身份或内容发生变化后应重新生成头文件并重建动态库，注册时会校验指纹是否一致。

同进程动态库只应加载可信代码；捕获 C++ 异常不提供段错误、越界或死锁隔离。
