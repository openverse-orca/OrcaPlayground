# 第 05 课：机器人 + 桌子 + 方块

## 本课效果
机械臂站在桌旁、桌上放着方块——第一个「机器人工作区」，第 45 课抓取任务的最小雏形。

## 准备
同第 01 课。

## 运行
```bash
python -m examples.euler.beginner.lesson_05_compose_scene.run
```

## 关键代码
组合 = 在配方里多写一行，机器人不享受特殊待遇：
```python
ActorSpec(name="arm_1", asset_path=_ARM_PATH, position=(0.0, -0.8, 0.0)),
ActorSpec(name="block_1", ..., position=(0.0, 0.0, _TABLE_TOP_Z + 0.1)),
```

## 动手改
把机械臂的 y 坐标 `-0.8` 改成 `-1.2`，离桌子更远。

## 解释结果
「机器人工作区」没有专门 API——它只是一个多资产配方。你已经在用
组合的方式构造任务环境了。

## 小挑战
再加一个球（`ball_1`），放到机械臂与桌子之间。

## 常见问题
| 现象 | 修复 |
|---|---|
| 机械臂与桌子穿模 | 调大 y 距离（资产尺寸落地后校准默认值） |

## 下一课
[第 06 课：找到指定物体](../lesson_06_identify_object/README.md) — 按名字认出你的目标。
