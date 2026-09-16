# 第 02 课：导入一个物体

## 本课效果
视口出现地面与你选定的物体；换 `--object` 重跑，变成另一种物体。

## 准备
同第 01 课（OrcaLab + OrcaPlaygroundAssets 订阅 + 首次重启）。

## 运行
```bash
python -m examples.euler.beginner.stage1_scene_basics.lesson_02_load_object.run              # 桌子
python -m examples.euler.beginner.stage1_scene_basics.lesson_02_load_object.run --object ball
```

## 关键代码
`OBJECT_CHOICES` 把「物体种类」映射到「资产路径」——换物体就是换路径：
```python
OBJECT_CHOICES = {
    "table": _TABLE_PATH,
    "ball": _BALL_PATH,
    "shelf": _SHELF_PATH,
}
```
注意：实例名始终是 `object_1`——资产换了，名字不变。

## 动手改
用 `--object shelf` 重跑，观察金属货架替换桌子。

## 解释结果
`asset_path` 指向资产库里的**资产**（这一类东西），`name` 是场景里的
**实例名**（这一次摆放）。一个资产可以 spawn 多个实例（第 04 课见）。

## 小挑战
把 `_SHELF_PATH` 换成资产库里你感兴趣的其它 OrcaPlaygroundAssets 资产路径。

## 常见问题
| 现象 | 修复 |
|---|---|
| 报"资产包未订阅" | 同第 01 课：确认订阅并重启 OrcaLab |
| 物体悬空/嵌入地面 | 桌子/货架已紧贴地面（`GROUND_Z`）；球默认悬空 0.5 m 演示下落。可改 `GROUND_Z` 中对应值 |

## 下一课
[第 03 课：导入一个机器人](../lesson_03_load_robot/README.md) — 机器人也是资产。
