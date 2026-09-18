# 第 04 课：导入一个场景

## 本课效果
视口一键出现完整布局：桌子居中，方块和球分立前后的地面——一个「场景配方」的还原。

## 准备
同第 01 课。

## 运行
```bash
python -m examples.euler.beginner.stage1_scene_basics.lesson_04_load_scene.run
```

## 关键代码
`build_recipe()` 返回的列表就是**场景配方**——场景 = 多个资产的摆放组合：
```python
[
    ActorSpec(name="ground", asset_path=_GROUND_PATH),
    ActorSpec(name="table_1", asset_path=_TABLE_PATH),
    ActorSpec(name="block_1", ..., position=(0.0, -1.6, _REST_Z)),
    ActorSpec(name="ball_1", ..., position=(0.0, 1.6, _REST_Z)),
]
```
运行时终端会逐行打印这份配方。

## 动手改
把 `ball_1` 的 y 坐标从 `1.6` 改成 `2.6`，重跑，球离桌子更远了。

## 解释结果
桌子长边沿 x，方块与球沿 y 分立前后（±1.6 m）避开桌体——**x/y 负责平面上的
间隔，z 负责离地高度**。位置是相对语义的组合，不是孤立的绝对数字。

## 小挑战
在配方里再加一个 `block_2`（名字不能重复），放在球的旁边。

## 常见问题
| 现象 | 修复 |
|---|---|
| 物体重叠 | 调整 position，保证 x/y 至少错开物体尺寸 |
| 物体嵌入桌子 | 桌子长边沿 x（x=±1.2 仍在桌体内）；沿 y 前后摆放，默认 ±1.6 已避开 |
| 报 name 冲突 | 同一场景内实例名必须唯一 |

## 下一课
[第 05 课：机器人 + 桌子 + 球](../lesson_05_compose_scene/README.md) — 组合你的第一个工作区。
