# 传感器示例开发与维护

本目录维护六个 EulerEnv 演示及其完整场景、预编译插件和图表。Host 与契约工具由配套
OrcaGym 提供；Playground 不复制 Host 或厂商算法源码。

## 配套版本与本地安装

当前配套功能尚未正式发布，验证基线为 OrcaGym `feat/sensor-provider-integration` 的
`8896ad77`。该提交支持已装配模型的 site 和对象绑定；仅支持旧 site 接口的版本不能运行全部示例。
合并前以配套 PR 的最终提交为准，不将当前开发版本号当作已发布的最低 OrcaGym 版本。

在 `orca` conda 环境中，进入已检出配套实现的 **OrcaGym 源码目录**执行：

```bash
python -m pip install -e .
```

然后回到 **OrcaPlayground 仓库根目录**：

```bash
python -m pip install -r examples/euler/sensor_provider/requirements.txt
python -m examples.euler.sensor_provider.run --example touch_grid
```

绘图依赖不会安装或升级 OrcaGym。遇到导入缺失或未知绑定格式时，先确认环境实际加载的
OrcaGym 是否为配套提交，避免可编辑安装仍指向另一个源码工作区：

```bash
python -c "import orca_gym; print(orca_gym.__file__)"
```

正式发布时，须更新依赖说明中的最低 OrcaGym 版本和安装步骤，并在干净环境安装正式
构建产物验收。SDK `1.0.0` 和 ABI `2` 是独立版本，不能代替 OrcaGym 版本。
当前随包 Runtime 面向 Linux x86_64、glibc ≥ 2.35；更新任何二进制后须重新验证目标平台。

## 本地覆盖与回归

详细运行器保留 `--host` 和 `--package` 等显式调试选项。
`--build-dir /path/to/build` 指向同时包含 Host 动态库与 `providers/<型号>/provider.json` 的目录。
正常运行使用安装包里的 Host 和本目录插件，不需要这些覆盖参数。

```bash
python -m examples.euler.sensor_provider.dexhand_euler \
  --build-dir /path/to/build --steps 5
python -m pytest tests/sensor_provider -q
python -m ruff check --select SLF001 envs/ examples/
```

测试直接使用预编译插件，不要求 SDK 源码或 C++ 编译器。六个统一入口都应完成默认 10 s
仿真；无图形界面时使用 `--output` 保存到尚不存在的 PNG 路径。

## SDK 与预编译包

SDK 接口、契约规范、C/C++ 样板和构建说明由 OrcaSensorHost 及其厂商 SDK 维护。
`providers/` 保存版本固定的完整预编译包；更新时接收同一发布产物，并同步
`providers/manifest.json` 的版本、来源和逐文件摘要。各包 README、模型元数据及许可证
属于原始包内容，应完整保留，不能只替换动态库或修改契约版本号。

插件与场景资源使用普通 Git，克隆本示例无需 Git LFS。灵巧手的 35 个网格与 1 个纹理均被
场景引用，更新 XML 时应同步检查资源闭包及 `scenes/dexhand/PROVENANCE.md`。

## 格式兼容性与实现诊断

六个入口通过 `OrcaGymEulerEnv` 推进物理并读取传感器输出。`scene_env.py` 是夹具与五指力
演示的环境，`dexhand_euler.py` 提供网格/测距环境，`provider_paths.py` 只定位随包资源。

场景格式兼容 MuJoCo MJCF。原始查看器可检查几何和状态，但不会执行 Orca 插件算法。
Rangefinder 须选中 `oscillating_targets` keyframe，才能与示例初始状态对齐：

```bash
python -m mujoco.viewer --mjcf examples/euler/sensor_provider/scenes/rangefinder/scene.xml
```

场景载荷、真实插件输出、控制轨迹、reset 与资源释放由测试覆盖。所有实例都绑定已有对象；
示例运行不装配模型，也不需要生成器或外部模型资源目录。
