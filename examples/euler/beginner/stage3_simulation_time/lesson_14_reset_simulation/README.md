# 第 14 课：回到初始状态

## 本课效果
方块自由落体 → `reset()` 一键把位置、速度、仿真时间全部倒回起点 →
再落一次，分毫不差。重复 3 轮，每次起点完全一致。

## 准备（二选一，体现你的主导权）
- **你的场景（倡导）**：沿用上一课摆好的方块（拖入或 spawn 均可）
- **兜底**：加 `--default-scene`，脚本用默认配方（地面 + 抬高一米的方块）

## 运行
```bash
# 方式 A：用你自己的场景
python -m examples.euler.beginner.stage3_simulation_time.lesson_14_reset_simulation.run

# 方式 B：默认配方兜底
python -m examples.euler.beginner.stage3_simulation_time.lesson_14_reset_simulation.run --default-scene
```

## 关键代码
```python
sim_link.reset_env(env)         # 复位 qpos / qvel / 仿真时间到初始状态（reset_simulation + mj_forward + render）
z_start = sim_link.read_height(env, block_name)
for _ in range(n_calls):
    sim_link.advance_realtime(env, ctrl, sim_link.FRAME_SKIP)   # 自由落体（实时节拍，视口可见）
z_end = sim_link.read_height(env, block_name)
```
reset 是环境级操作：不只方块，**整个场景**的状态都回到 XML 初始值。

## 动手改
配方区 `ROUNDS = 3` → `5`：复位更多轮；`SETTLE_TIME = 1.5` → `0.5`：
方块还在半空中就被"倒带"。

## 解释结果
`reset_simulation()` 不是"把方块放回原处"这么简单——速度清零、时间归零、
内部缓存全部重建。这就是为什么强化学习训练每局开始都要 reset：
**没有可重复的起点，就没有可比较的实验**。

## 小挑战
**改参数验证一致性**：把配方区改成 `ROUNDS = 5`、`SETTLE_TIME = 0.5`，
运行后确认 5 轮的起点数值逐位相同。若把 `SETTLE_TIME` 改成 `0.05`
（只落 1 步）呢？起点是否依然一致？

> 复位的意义不在于"重来"，而在于"每次重来都一样"。

## 常见问题
| 现象 | 修复 |
|---|---|
| "未找到方块" | 拖入/自 spawn 一个含 "cube" 名字的方块，或加 `--default-scene` |
| 每轮起点略有差异 | 数值精度内的微差（1e-6 量级）属正常；量级不同请反馈 |

## 下一课
[第 15 课：读取位置和速度](../lesson_15_read_state/README.md) — 让数据开口说话。
