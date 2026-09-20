# 第 15 课：读取位置和速度

## 本课效果
下落过程每 0.1 秒打印一行：时间、高度、实测速度、理论速度（v = g·t）。
实测列追着理论列跑——物理引擎里的自由落体和你中学学的公式是同一件事。

## 准备（二选一，体现你的主导权）
- **你的场景（倡导）**：沿用前面课摆好的方块
- **兜底**：加 `--default-scene`，脚本用默认配方（地面 + 抬高一米的方块）

## 运行
```bash
# 方式 A：用你自己的场景
python -m examples.euler.beginner.stage3_simulation_time.lesson_15_read_state.run

# 方式 B：默认配方兜底
python -m examples.euler.beginner.stage3_simulation_time.lesson_15_read_state.run --default-scene
```

## 关键代码
```python
pos = env.data.body_xpos(block_name)    # 位置 (3,) 世界坐标
vel = env.data.body_cvel(block_name)    # [角速度(3), 线速度(3)]
vz = vel[5]                              # 线速度的 z 分量
```
`env.data` 是状态的唯一窗口（OrcaGymDataView）——**按名称查询**，
不碰内部数据结构。这正是 06 课"认出物体"的价值兑现：认出名字，才能读它。

## 动手改
配方区 `PRINT_INTERVAL = 0.1` → `0.05`：采样密一倍，曲线更平滑；
`FALL_TIME` → `2.0`：观察落地后 vz 如何被接触约束"按住"归零。

## 解释结果
- 下落段：vz 为负（向下）且随时间线性增长，和理论列几乎重合
- 落地段：vz 被地面接触约束清零——数据读出的不只是"位置"，还有"物理正在做什么"

传感器、控制器、RL 训练的观测，全都是这样一行行 `env.data.*` 读出来的。

## 小挑战
**换个读数**：把打印内容从 `z / vz` 换成水平分量 `x / y / vx / vy`
（`body_xpos` 的前两个分量、`body_cvel` 的第 3、4 个分量），运行后
解释它们为什么始终是 0（提示：方块只在竖直方向受力）。

> 读状态的真功夫是知道"该读哪个量"。

## 常见问题
| 现象 | 修复 |
|---|---|
| "未找到方块" | 拖入/自 spawn 一个含 "cube" 名字的方块，或加 `--default-scene` |
| vz 理论列符号反了 | 本课地面在 z≈0、方块在上方，向下速度为负是坐标系约定，不是 bug |
| 实测 vz 比理论值小 | 离散积分与读取时机的固有滞后（未建模的空气阻力不会带来偏差），量级一致即可 |

## 下一课
[第 16 课：修改重力](../lesson_16_change_gravity/README.md) — 换个星球做实验。
