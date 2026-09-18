# 第 01 课：Hello OrcaGym

## 本课效果
OrcaLab 视口出现地面和一个方块，终端显示连接与 spawn 成功。

## 准备
- OrcaLab 已启动（含 Studio 视口）
- ✅ 已在 OrcaLab 资产库中订阅 **OrcaPlaygroundAssets** 资产包
- 首次订阅后需重启 OrcaLab（引擎侧资产表为一次性填充）

### 怎么订阅资产包（第一次来的同学看这里）
1. 打开资产库 [https://simassets.orca3d.cn/](https://simassets.orca3d.cn/)，用你的
   OrcaLab 账号登录（OrcaLab 客户端底部资产栏的【打开资产库】按钮也能跳过去）
2. 搜索 **OrcaPlaygroundAssets**，进入资产包详情页，点击【订阅】
3. **重启 OrcaLab**——订阅不会立即生效：客户端只在启动时同步已订阅的资产包，
   右下角【资产同步】按钮可以查看下载进度
4. 验证：OrcaLab 底部【资产栏】里能看到 OrcaPlaygroundAssets 的资产，即可运行本课

> 更多资产操作（以图搜资产、分类浏览、取消订阅）见官方文档：
> [如何搜索和订阅资产](https://docs.orca3d.cn/#/FAQ-list/045-如何搜索和订阅资产.md) ·
> [订阅资产后如何在 OrcaLab 中使用](https://docs.orca3d.cn/#/FAQ-list/047-订阅资产后如何在orcalab中使用.md) ·
> [OrcaLab 基础操作指南](https://docs.orca3d.cn/#/操作指南/OrcaLab基础操作指南_v1.0.md)

### 官方"拖拽"与本课"脚本"的关系
官方入门路径是"订阅 → 在资产栏把资产**拖拽**进视口"；本课脚本做的是同一件事的
代码版——`asset_path` 指向的就是你订阅的那批资产，只是摆放由
`ActorSpec(position=...)` 完成。两条路用的是同一批资产；第 13 课起你会亲手走
"拖拽"那条路。

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
| 报"资产包未订阅" | 按「准备」节四步走：订阅 → 重启 OrcaLab → 等资产同步完成 → 重跑 |
| 资产栏里找不到资产 | 订阅只在 OrcaLab 重启时同步；点右下角【资产同步】确认进度走完 |
| 视口看不到方块 | 检查视口是否聚焦原点；确认终端出现"场景已就绪" |

## 下一课
[第 02 课：导入一个物体](../lesson_02_load_object/README.md) — 替换另一种物体。
