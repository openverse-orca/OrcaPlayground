# 第 01 课：Hello OrcaGym

## 本课效果
OrcaLab 视口出现地面和一个方块，终端显示连接与 spawn 成功。

## 准备
- OrcaLab 已启动（含 Studio 视口）
- ✅ 已在 OrcaLab 资产库中订阅 **OrcaPlaygroundAssets** 资产包
- 首次订阅后需重启 OrcaLab（引擎侧资产表为一次性填充）

## 运行
```bash
python -m examples.euler.beginner.stage1_scene_basics.lesson_01_hello_world.run
```

## 关键代码
`run.py` 中 `build_default_recipe()` 定义了场景的全部物体——这就是"场景配方"：
```python
ActorSpec(name="ground", asset_path=_GROUND_PATH),
ActorSpec(name="block_1", asset_path=_BLOCK_PATH, position=(0.0, 0.0, 0.5)),
```

## 动手改
本课没有可改参数——先确保链路通。下一课开始你就能改东西了。

## 解释结果
`position` 的三个数字是世界坐标 (x, y, z)，单位是米。0.5 表示方块中心离地半米。

## 小挑战
把 `run.py` 里方块的 x 从 `0.0` 改成 `2.0`，重跑——方块换了个位置出现。
这是你第一次"改了世界"。（本课只摆位置；方块为什么会动，第 13 课再讲。）

> 验收：方块出现在原来的右侧约 2 米处。

## 常见问题
| 现象 | 修复 |
|---|---|
| 报"资产包未订阅" | 确认已订阅 OrcaPlaygroundAssets；重启 OrcaLab 后重试 |
| 视口看不到方块 | 检查视口是否聚焦原点；确认终端出现"场景已就绪" |

## 下一课
[第 02 课：导入一个物体](../lesson_02_load_object/README.md) — 替换另一种物体。
