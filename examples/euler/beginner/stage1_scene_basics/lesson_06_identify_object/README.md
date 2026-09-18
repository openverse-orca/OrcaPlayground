# 第 06 课：找到指定物体

## 本课效果
终端列出场景全部实例名，并按关键词检索出目标——后续所有课程定位物体的方式。

## 准备
同第 01 课。

## 运行
```bash
python -m examples.euler.beginner.stage1_scene_basics.lesson_06_identify_object.run
```

## 关键代码
「探针」连接后读取场景全部 body 名（只读，不改场景）：
```python
body_names = probe_body_names(args.addr)
matches = [n for n in body_names if SEARCH_KEYWORD.lower() in n.lower()]
```
这一步回答的问题是：**我想操作的东西，在场景里叫什么名字？**

## 动手改
把 `SEARCH_KEYWORD` 从 `"ball"` 改成 `"robot"`，重跑检索机器狗——
一个关键词命中一串 body（每条腿、每个关节都是一个 body）。

## 解释结果
名称发现是层 3「场景无关」的基础：不管场景是你 spawn 的还是手动拖的，
只要按名字能找到，脚本就能对它操作——这就是第 13 课开始的工作方式。

body 名还有个规律：**实例名 + 资产内名字**拼接——`ball_1` + `sphere` →
`ball_1_sphere`、`ground_floor` 同理。所以你在配方里起的 `name` 会成为
前缀，检索时用 `ball`、`robot` 这类短关键词即可命中。

## 小挑战
试试用 `"table"` 检索：命中几个？为什么 `table_1` 和 `ball_1` 都在
同一个 body 名单里？（提示：它们都是场景里的实体。）

## 阶段 1 毕业挑战 🎓
搭一个**你自己的角落**：改 `build_recipe()`，用地面 + 桌子 + 至少 2 个
你选的物体组成一个小场景；再把 `SEARCH_KEYWORD` 换成你的物体名，确认
检索全部命中。

> 验收：终端名单里能看到你的每个实例名。
> 你已经会：放资产（01–03）、写配方（04–05）、按名找人（06）。
> 下一阶段「像搭积木一样编辑场景」将让摆放更自由。

## 常见问题
| 现象 | 修复 |
|---|---|
| 检索无命中 | 检查关键词拼写；先看打印的完整名单里目标叫什么 |
| "MuJoCo has not been initialized" | spawn 后引擎初始化的竞态，脚本已内置退避重试；若重试后仍失败，稍等数秒重跑 |
| 名单很长 | 一个资产可能展开多个子 body（如 `table_1` 有 7 个；`ActorManipulator_*` 是引擎注入），属正常 |

## 下一课
[第 07 课：移动物体](../../stage2_scene_editing/lesson_07_move_object/README.md) — 阶段 2「像搭积木一样编辑场景」开始：你来改配方。
