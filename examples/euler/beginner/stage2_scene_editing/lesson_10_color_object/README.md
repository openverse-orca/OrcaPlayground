# 第 10 课：改变颜色

## 本课效果
方块 spawn 后立刻变成配方区 `BOX_RGBA` 指定的红色。改 RGBA 重跑，变成任意颜色。

## 准备
- 同第 01 课（OrcaLab + 资产包订阅 + 首次重启）
- 本课只用到 `cube_usda` 一个方块资产——颜色不靠换资产，靠运行时改材质
- ⚠️ 本课**重建教学场景**：每次运行会清空场景再摆放（不是热编辑）

## 运行
```bash
python -m examples.euler.beginner.stage2_scene_editing.lesson_10_color_object.run
```

## 关键代码
`run.py` 顶部的**配方区**就是你要改的地方：
```python
# ======================= 配方区（改这里） =======================
BOX_RGBA: tuple[float, float, float, float] = (0.9, 0.1, 0.1, 1.0)
# ================================================================
```
颜色在 spawn 之后由一行接口下发：
```python
scene.set_material_info("box_1", MaterialInfo(base_color=np.array(BOX_RGBA)))
```

## 动手改
把 `BOX_RGBA` 改成绿色 `(0.1, 0.9, 0.2, 1.0)`，重新运行，方块从红变绿。

## 解释结果
- spawn 的 `ActorSpec` 只有 位置/旋转/缩放——形状和姿态在 spawn 时确定；
  **颜色属于材质**，spawn 后由 `set_material_info` 随时改。
- 「一个资产 + 运行时改色」优于「每种颜色做一个变体资产」：资产库更小，
  换色零成本。给目标物打标记、区分同类物体，用的就是这一招。
- RGBA 每个分量取 0~1（不是 0~255）；A 是不透明度，1 = 完全不透明。

## 小挑战
复制 `box_1` 的两行代码（一行 `ActorSpec`、一行 `set_material_info`），
加一颗 `box_2` 并给它**不同颜色**，与 `box_1` 表面间隔 0.8 米。
提示：实例名不能重复；`set_material_info` 的第一个参数是实例名；
方块宽 1 米——表面隔 0.8 米意味着两颗中心要相距 1.8 米。

## 常见问题
| 现象 | 修复 |
|---|---|
| 接口成功但视口不变色 | 确认 `set_material_info` 的名字与 spawn 时的 `name` 一致 |
| 颜色不对/发白 | RGB 每个分量取 0~1；写成 0~255 会趋近纯白 |
| 想要半透明 | 当前不支持：A 分量暂不生效，固定填 1.0 |

## 下一课
[第 11 课：复制与排列](../lesson_11_duplicate_objects/README.md) — 用循环一次摆一排方块。
