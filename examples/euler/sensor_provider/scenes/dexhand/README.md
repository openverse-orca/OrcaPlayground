# 完整灵巧手模型

这是供 `hand` 和 `hand_grid` 两种演示共同使用的完整机械臂与五指灵巧手场景。
`hand` 显示五指法向力/切向力，`hand_grid` 显示五指接触力网格与测距；两者均使用自带模型进行物理计算。

[scene.xml](scene.xml) 是已挂载传感器的完整 MJCF，包含机械臂、五指、35 个触面 body、
91 个 geom、26 个执行器和 5 个指尖 site。`meshes/` 包含必需的 36 个网格/纹理文件，
复制模型时应一起复制。不需要生成器、第二份配置 XML 或外部资产目录。

## 运行

先按[演示入口的运行前准备](../../README.md#运行前准备)配置支持传感器插件的 OrcaGym 环境并取得模型资源。
在 **OrcaPlayground 仓库根目录**、`orca` 环境运行，选择其中一种图表：

```bash
# 每指法向力/切向力。
python -m examples.euler.sensor_provider.run --example hand
# 五指网格与测距。
python -m examples.euler.sensor_provider.run --example hand_grid
```

默认运行 10 s 仿真，结束后保留图表窗口。无界面保存图表见[演示入口](../../README.md#保存图表与结束运行)，
控制与自定义包见[进阶说明](../../advanced.md)，模型物理状态诊断见[开发者说明](../../developer.md#格式兼容性与实现诊断)。

## XML 绑定与输出

每个原始指尖 site 同时绑定 ContactGrid 和 Rangefinder，五指共十个独立实例。例如：

```xml
<custom>
  <text name="orca.sensor.v1/touch_f2/plugin" data="com.orca.examples.contact_grid"/>
  <tuple name="orca.sensor.v1/touch_f2/site">
    <element objtype="site" objname="site_l_f_link2_4"/>
  </tuple>
  <numeric name="orca.sensor.v1/touch_f2/config/gain" data="1"/>
  <text name="orca.sensor.v1/range_f2/plugin" data="com.orca.examples.rangefinder"/>
  <tuple name="orca.sensor.v1/range_f2/site">
    <element objtype="site" objname="site_l_f_link2_4"/>
  </tuple>
</custom>
```

模型格式兼容 MuJoCo MJCF。加载模型时检查 site 引用；Orca 读取 `custom` 中的实例声明，
选择已注册的可信厂商包。XML 不包含动态库路径，也不根据实例名称猜测 site。
不加载传感器插件时仍可检查几何、碰撞和物理状态，但不会得到厂商算法输出。

| 插件 | Orca 输入组装 | 插件输出 |
| --- | --- | --- |
| ContactGrid | site 焊接组接触 → site 坐标系 → 120°×120° 角度网格内合力向量 | 每格合力模长，4×4，单位 N |
| Rangefinder | site 局部 +Z 射线，量程 0.1 m，排除同焊接组 | 米制距离，未命中为 -1 |
| SevenPad | 每指明确的七个触面接触，转换到指尖坐标系，附带测距 | 法/切向矢量合力模长、切向角度、七路近似电容、距离 |

`hand_grid` 的环境推进物理并逐子步采样，任务从 `query_provider_sensor_data()` 读取 NumPy 副本，
放入 `provider_sensors` 观测。`reset` 不偷跑物理，返回占位并用 `provider_valid=0` 标记，
首次成功步进后为 1。姿态是积分后状态，读数是最后物理子步的源状态，不是同一时间戳。

`hand` 使用 `ProviderSceneEnv`，在临时 XML 中将十个 site 声明替换为五个 SevenPad 对象声明。
例如第二指内部 `force_f1…force_f7` 精确映射到 `force2_f1…force2_f7`，`range_frame` 映射到
`site_l_f_link2_4`。触面 body 的直接 geom 自动作为本实例几何；匿名 geom 无需补名称。
仅 EulerEnv 推进物理并逐步采样，分发 XML 与资源文件不变。

## 读数和模型限制

法向与切向力分别先作矢量相加，再取模，不是网格响应和，也不是按 site 的 Z/XY 轴拆分。
完整手主入口默认噪声为零；高级 SevenPad 入口允许设置 `noise_scale`。
近似电容为 `capacitance_base + capacitance_gain × norm(触面法向合力 + 切向合力)`，
单位 raw，纯教学近似，无真实产品标定。

控制使用已验证的渐进接近/闭手轨迹，保留关节与执行器限力；圆柱 `<freejoint/>` 避免继承
手指阻尼。这是开环演示，不保证每指接触或抓取成功。不同仿真版本、控制周期和模型配置
可改变接触曲线。neutral 短时运行可能只有零力/NO HIT，不能据此判断算法错误。

模型来源和修正记录见 [PROVENANCE.md](PROVENANCE.md)。算法来自 OrcaSensorHost SDK 教学样板；
本目录只使用 [预编译包](../../providers/README.md)，不维护其 C/C++ 源码。
