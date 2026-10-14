# 第 13 课：开始、暂停、单步

## 本课效果
脚本分三段演示：**暂停**（静置 3 秒倒数——不推进，方块纹丝不动）→ **单步**（每次推进
一个物理步，高度一点点减小）→ **连续**（零控制自由下落，实时节拍，落地后静置 1 秒收尾）
——你第一次控制仿真的时间。

## 准备（二选一，体现你的主导权）
- **你的场景（倡导）**：在 OrcaLab 里拖入一个方块（进入布局模式 → 资产库 → 拖入），
  或自己写 spawn 调用放置（第 07–12 课学过的配方方式）
- **兜底**：什么都不摆，加 `--default-scene`，脚本用默认配方

> 不熟悉拖拽/移动/旋转物体？见官方
> [OrcaLab 基础操作指南](https://docs.orca3d.cn/#/操作指南/OrcaLab基础操作指南_v1.0.md)
> 的「大纲与相机」「编辑」两节。

## 运行
```bash
# 方式 A：用你自己的场景
python -m examples.euler.beginner.stage3_simulation_time.lesson_13_step_simulation.run

# 方式 B：默认配方兜底
python -m examples.euler.beginner.stage3_simulation_time.lesson_13_step_simulation.run --default-scene
```

## 关键代码
本课脚本**不假设方块在哪里**——通过名称发现它（场景无关设计）：
```python
block_name = find_body(env, "cube")   # 关键词对齐资产内部名（cube_small_usda）
```
发现失败会明确提示：拖入方块，或加 `--default-scene`。

时间流动的最小循环（连接、零控制、步进+渲染都收口在 `sim_link`，13–18 课共用）：
```python
env = sim_link.connect_simulation_env(addr)   # 连接 OrcaLab 当前场景并 reset
ctrl = sim_link.zero_ctrl(env)                # 零控制 = 纯被动物理（重力下落）
for _ in range(n_calls):
    sim_link.advance_realtime(env, ctrl, FRAME_SKIP)  # 步进+渲染+实时节拍
    z = sim_link.read_height(env, block_name)
```
`advance_realtime` 让 1 秒仿真 ≈ 1 秒墙钟——不加节拍的话纯 CPU 几毫秒就
跑完 1 秒仿真，视口什么也看不到（仿真时间 vs 墙钟时间的深入对比见第 17 课）。

## 动手改
`--steps` 从 10 改成 30：一次回车下落更多。

## 解释结果
「单步」= 物理引擎推进一个时间步（本课基础步长 0.002 秒）。下落是引擎算出来的，不是动画——
这就是"摆位置"（第 07 课）和"物理运动"（本课起）的本质区别。
暂停的秘密：仿真不会自己走，**你不推进它就停着**——开始/暂停/单步全都由你的循环决定。

## 小挑战
**先猜再验证**：方块从 2 米落到地面大约需要多少步？先写下你的数字，
再改 `--steps` 逐步推进验证你的估算。

> 猜错了最好——那说明物理的直觉正在建立。

## 常见问题
| 现象 | 修复 |
|---|---|
| "未找到方块" | 拖入/自 spawn 一个含 "cube" 名字的方块，或加 `--default-scene` |
| 方块本来就在地上 | "你的场景"模式下拖入的方块贴地摆放没有下落可看——抬高它，或用 `--default-scene`（默认配方把小方块抬到 2 米） |
| 方块不动 | 确认脚本日志走到 `[连续]` 段——单步只有 10 步（0.02 秒），肉眼几乎看不出；视口同步依赖循环内 `render()`，`sim_link.advance_realtime` 已内置 |

## 下一课
[第 14 课：回到初始状态](../lesson_14_reset_simulation/README.md) — reset 的意义。
