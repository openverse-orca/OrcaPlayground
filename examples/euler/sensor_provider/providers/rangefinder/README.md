# Rangefinder：读取射线距离并添加偏差

这个型号的 ID 为 `com.orca.examples.rangefinder`，使用与 ContactGrid 独立的厂商包和动态库。它请求一个物理距离，并对有效命中施加可配置偏差，用于演示多个型号各自接收输入和输出结果。

1. `contract.json` 声明 `orca.site_raycast.v1`：从场景实例绑定的 site（带位置和姿态的测量参考点）沿局部 +Z 测距，最大查询距离为 0.1 m。
2. Orca 执行查询，将 `sample_time` 和 `distance` 两个标量组装进生成的 `RangefinderInput`。
3. 厂商算法只读取距离并写入一个输出值，不访问模型或物理引擎；OrcaGym 使用方按实例读取 `float64[1]` 输出。

模型只含示例外壳与 `sensor_base` 安装 site，不含原生 rangefinder、sensor 或 plugin。外壳位于安装原点后方，明确不参与碰撞；测量 site 由场景实例绑定，契约不要求厂商填写模型内部对象名。

## 输入与输出含义

- `distance == -1` 表示没有命中：算法原样输出 -1，不叠加偏差。
- 有效命中输入非负，输出为 `max(0, distance + bias_m)`，单位 m。
- 负偏差可能将命中结果压到零，但不会造成 -1，避免与“未命中”混淆。
- 正偏差可能让输出大于 0.1 m；0.1 m 限制的是平台的几何查询范围，不是加入算法偏差后的显示范围。
- `bias_m` 默认 0，范围 `[-1, 1]`，创建时传入并由实例独立保存；reset 保留参数。本例没有随机噪声，seed 不影响结果。

这是数据链路样板，不是经实物标定的测距传感器。具体哪些场景几何参与射线、如何排除自身，见 SDK 中的 `docs/capabilities/site_raycast.md`；厂商算法不自行筛选目标。

## 版本与构建

SDK 目标为 `1.0.0`，C ABI 为 2，厂商包版本为 `1.0.0`。以下命令需要完整 SDK，并在 SDK 根目录执行。若正在阅读已构建厂商包中的 README，源码位于完整 SDK 的 `examples/rangefinder/`。

```bash
cmake -S examples/rangefinder -B build/rangefinder -DCMAKE_BUILD_TYPE=Release
cmake --build build/rangefinder --config Release
```

产物为 `build/rangefinder/providers/rangefinder/`，包含包入口、型号 XML、输入输出契约、动态库（当前 Linux 平台为 `liborca_rangefinder.so`）、README 和 MIT 许可证。构建使用 SDK 自带的工具生成头文件与清单，不需要安装 Python；算法动态库不依赖 Python 或物理引擎运行时。

动态库导出的 `orca_sensor_get_provider()` 返回厂商描述符，其中登记本型号及 `create/reset/compute/destroy` 四个回调。它与 ContactGrid 使用相同的 C ABI，但厂商包 ID（`provider_id`）、型号 ID（`type_id`）、输入结构和动态库都独立；平台通过描述符调用各自算法。

## 在完整灵巧手中查看效果

OrcaPlayground 仓库的 `examples/euler/sensor_provider/` 提供完整五指灵巧手演示。`hand_grid` 模式在每根手指上各绑定一个 Rangefinder 和一个 ContactGrid，显示五指的测距与接触力网格。先按该目录的 `README.md` 确认 OrcaGym 支持版本并准备环境。准备好后，在 **OrcaPlayground 根目录**、`orca` 环境运行：

```bash
python -m examples.euler.sensor_provider.run --example hand_grid
```
