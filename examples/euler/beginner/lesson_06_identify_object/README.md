# 第 06 课：找到指定物体

## 本课效果
终端列出场景全部实例名，并按关键词检索出目标——后续所有课程定位物体的方式。

## 准备
同第 01 课。

## 运行
```bash
python -m examples.euler.beginner.lesson_06_identify_object.run
```

## 关键代码
「探针」连接后读取场景全部 body 名（只读，不改场景）：
```python
body_names = probe_body_names(args.addr)
matches = [n for n in body_names if SEARCH_KEYWORD.lower() in n.lower()]
```
这一步回答的问题是：**我想操作的东西，在场景里叫什么名字？**

## 动手改
把 `SEARCH_KEYWORD` 从 `"block"` 改成 `"ball"`，重跑检索球。

## 解释结果
名称发现是层 3「场景无关」的基础：不管场景是你 spawn 的还是手动拖的，
只要按名字能找到，脚本就能对它操作——这就是第 13 课开始的工作方式。

## 小挑战
试试用 `"table"` 检索：命中几个？为什么 `table_1` 和 `ball_1` 都在
同一个 body 名单里？（提示：它们都是场景里的实体。）

## 常见问题
| 现象 | 修复 |
|---|---|
| 检索无命中 | 检查关键词拼写；先看打印的完整名单里目标叫什么 |
| 名单很长 | 一个资产可能带多个子 body（如机器人关节），属正常 |

## 下一课
[第 07 课：移动物体](../lesson_07_move_object/README.md) — 层 2 开始：你来改配方。
