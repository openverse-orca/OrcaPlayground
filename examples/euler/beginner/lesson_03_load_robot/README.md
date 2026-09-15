# 第 03 课：导入一个机器人

## 本课效果
场景中出现一个稳定摆放的机器人（默认固定基座机械臂）。本课只观察，不控制。

## 准备
同第 01 课。

## 运行
```bash
python -m examples.euler.beginner.lesson_03_load_robot.run             # 机械臂
python -m examples.euler.beginner.lesson_03_load_robot.run --robot car # 小车
```

## 关键代码
机器人与桌椅走完全相同的 spawn 路径，只是资产不同：
```python
ActorSpec(name="robot_1", asset_path=ROBOT_CHOICES[args.robot], position=(0.0, 0.0, 0.0))
```

## 动手改
`--robot car` 换成差速小车，观察它与机械臂形态差异。

## 解释结果
机器人「有关节、能控制」的差别在资产内部（关节/执行器定义），不在
spawn 方式上。从 spawn 视角看，机器人 = 一种资产。

## 小挑战
无（对比观察即可，控制能力从第 27 课建立）。

## 常见问题
| 现象 | 修复 |
|---|---|
| 机器人姿势怪异 | BeginnerAssets 制作时已设稳定初始姿态；若异常请反馈 |

## 下一课
[第 04 课：导入一个场景](../lesson_04_load_scene/README.md) — 一键还原多物体布局。
