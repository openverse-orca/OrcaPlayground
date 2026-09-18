# ContactGrid：Orca 汇总接触力，算法计算每格合力大小

这个型号演示一个独立的厂商库：`com.orca.examples.contact_grid`。它不解析物理引擎内部数据，不读取接触点列表，也不自己将接触点划分网格。

1. `contract.json` 请求一个 `4×4` 的接触力网格，水平和垂直视场全角均为 `120°`。
2. Orca 按场景实例绑定的 site（带位置和姿态的测量参考点）采集、旋转和汇总接触力，将结果装入生成的 C 数组 `force_grid[4][4][3]`。
3. 厂商动态库读取每格的 `Fx/Fy/Fz`，计算向量模长并乘以实例参数 `gain`，返回 `float64[4,4]`。

每格单位为 N，不是 Pa。输入先汇总向量，算法再取模；相反方向的力可能抵消，不能解释成逐接触力模长之和。本例不是胶面形变、压力重建或标定后的真实触觉模型。

## 厂商收到什么

生成结构包含 `sample_time` 和固定大小的 `force_grid[4][4][3]`，没有字典查找或可变数组计数。每格三个分量在绑定 site 的局部坐标系表达；计算结果为行优先的 16 个数。没有接触时平台传入全零网格，算法输出全零。

网格是**角度划分**，不是固定毫米宽度的线性 XY 网格：site 的 +Z 为前方，只有局部 `z>0` 的接触点参与；方位角为 `atan2(x,z)`，仰角为 `atan2(y,sqrt(x*x+z*z))`。行随朝 +Y 的仰角增大，列随朝 +X 的方位角增大，视场边界包含在内。完整采集语义见 SDK 中的 `docs/capabilities/contact_grid.md`。

`model.xml` 仅提供示例外形与 `sensor_base` 安装参考，不包含原生 sensor/plugin。其安装 site 在底部，触面位于前方 20 mm；模型只定义一块物理触面，4×4 格子不是 16 个 geom。输入契约不引用 `force_f1` 等型号内部对象；测量 site 由场景实例绑定。

## 参数、版本与构建

- `gain`：默认 1，范围 `[0, 1000000]`，创建实例时传入；各实例独立。
- `reset` 保留 gain。本例无随机噪声，seed 不改变计算结果。
- SDK 目标为 `1.0.0`，C ABI 为 2，厂商包版本为 `1.0.0`。
- 输入、参数及计算结果必须有限，即不含 NaN 或正负无穷；错误通过状态码返回，成功时写满输出。

以下命令需要完整 SDK，并在 SDK 根目录执行。若正在阅读已构建厂商包中的 README，源码位于完整 SDK 的 `examples/contact_grid/`。

```bash
cmake -S examples/contact_grid -B build/contact-grid -DCMAKE_BUILD_TYPE=Release
cmake --build build/contact-grid --config Release
```

产物为 `build/contact-grid/providers/contact_grid/`，包含包入口、型号 XML、输入输出契约、动态库（当前 Linux 平台为 `liborca_contact_grid.so`）、README 和 MIT 许可证。构建使用 SDK 自带的工具生成头文件和清单，不需要安装 Python。OrcaGym 使用方按实例 ID 读取结果，无需直接调用 C 接口。

## 在完整灵巧手中查看效果

OrcaPlayground 仓库的 `examples/euler/sensor_provider/` 提供完整五指灵巧手演示。`hand_grid` 模式在每根手指上各绑定一个 ContactGrid 和一个 Rangefinder，显示五指的接触力网格与测距。先按该目录的 `README.md` 确认 OrcaGym 支持版本并准备环境。准备好后，在 **OrcaPlayground 根目录**、`orca` 环境运行：

```bash
python -m examples.euler.sensor_provider.run --example hand_grid
```
