# 第 05 课：机器人 + 桌子 + 球

## 本课效果
go2 机器狗站在桌旁、桌上放着球——第一个「机器人工作区」，第 45 课抓取任务的最小雏形。

## 准备
同第 01 课（OrcaPlaygroundAssets 订阅）。

## 运行
```bash
python -m examples.euler.beginner.stage1_scene_basics.lesson_05_compose_scene.run
```

## 关键代码
组合 = 在配方里多写一行，机器人不享受特殊待遇：
```python
ActorSpec(name="robot_1", asset_path=GO2, position=(0.0, -1.2, 0.0)),
ActorSpec(name="ball_1", asset_path=BALL, position=(0.0, 0.0, _TABLE_TOP_Z + _BALL_RADIUS)),
```

## 动手改
把机器狗的 y 坐标 `-1.2` 改成 `-2.0`，离桌子更远。

## 解释结果
「机器人工作区」没有专门 API——它只是一个多资产配方。你已经在用
组合的方式构造任务环境了。

## 小挑战
再加一个方块（`block_1`，资产路径参考第 04 课），放到机器狗与桌子之间。

## 常见问题
| 现象 | 修复 |
|---|---|
| 机器狗与桌子穿模 | 调大 y 距离（默认 -1.2 已按 go2 尺寸留出间距） |

## 下一课
[第 06 课：找到指定物体](../lesson_06_identify_object/README.md) — 按名字认出你的目标。
