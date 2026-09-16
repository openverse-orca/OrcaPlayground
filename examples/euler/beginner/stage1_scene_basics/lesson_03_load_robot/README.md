# 第 03 课：导入一个机器人

## 本课效果
场景中出现一个稳定摆放的机器人（默认 go2 机器狗）。本课只观察，不控制。

## 准备
同第 01 课（OrcaPlaygroundAssets 订阅）。

## 运行
```bash
python -m examples.euler.beginner.stage1_scene_basics.lesson_03_load_robot.run              # go2 机器狗
python -m examples.euler.beginner.stage1_scene_basics.lesson_03_load_robot.run --robot h1  # h1 人形
```

## 关键代码
机器人与桌椅走完全相同的 spawn 路径，只是资产不同：
```python
ActorSpec(name="robot_1", asset_path=ROBOT_CHOICES[args.robot], position=(0.0, 0.0, 0.0))
```

## 动手改
`--robot h1` 换成人形机器人，观察它与机器狗的形态差异。

## 解释结果
机器人「有关节、能控制」的差别在资产内部（关节/执行器定义），不在
spawn 方式上。从 spawn 视角看，机器人 = 一种资产。

## 小挑战
在配方里再加一个机器人 `robot_2`（实例名不能重复），让两个机器人同时在场——
同一份资产，两个实例。

> 验收：视口里两个机器人并存，终端 spawn 日志显示 3 个物体（地面 + 两个机器人）。

## 常见问题
| 现象 | 修复 |
|---|---|
| 机器人姿势怪异 | go2/h1 均为验证过的稳定初始姿态资产；若异常请反馈 |

## 下一课
[第 04 课：导入一个场景](../lesson_04_load_scene/README.md) — 一键还原多物体布局。
