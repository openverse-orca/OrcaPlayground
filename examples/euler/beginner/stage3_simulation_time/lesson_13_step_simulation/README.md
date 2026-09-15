# 第 13 课：开始、暂停、单步

## 本课效果
暂停时方块停住；每按一次回车，方块下落一点——你第一次控制仿真的时间。

## 准备（二选一，体现你的主导权）
- **你的场景（倡导）**：在 OrcaLab 里拖入一个方块（进入布局模式 → 资产库 → 拖入），
  或自己写 spawn 调用放置（第 07–12 课学过的配方方式）
- **兜底**：什么都不摆，加 `--default-scene`，脚本用默认配方

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
block_name = find_body(env, "block")
```
发现失败会明确提示：拖入方块，或加 `--default-scene`。

## 动手改
`--steps` 从 10 改成 30：一次回车下落更多。

## 解释结果
「单步」= 物理引擎推进一个时间步。下落是引擎算出来的，不是动画——
这就是"摆位置"（第 07 课）和"物理运动"（本课起）的本质区别。

## 小挑战
数一数方块从 1 米落到地面大约需要多少步；改 `--steps` 验证你的估算。

## 常见问题
| 现象 | 修复 |
|---|---|
| "未找到方块" | 拖入/自 spawn 一个含 "block" 名字的方块，或加 `--default-scene` |
| 方块不动 | 骨架阶段已知缺口：env 驱动链路待 P0 实测（见 run.py 顶部说明） |

## 下一课
[第 14 课：回到初始状态](../lesson_14_reset_simulation/README.md) — reset 的意义。
