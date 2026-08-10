# 第 2 课：加载 USD 物体资产（时序 spawn 三张桌子）

> 场景构建子系统第 2 课。本课通过时序 spawn 三张桌子，验证 `append_scene()` 增量 spawn
> 范式在非机器人资产（USD 物体）上的适用性。

---

## 1. 课程目标

验证 `append_scene()` 对 USD 物体资产的增量 spawn 能力：

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 第 1 张桌子 spawn | `add_actor + append_scene` | t=0s 视口出现 desk_1（左侧） |
| 2 | 第 2 张桌子 spawn | `add_actor + append_scene` | t=3s 视口出现 desk_1 + desk（中间） |
| 3 | 第 3 张桌子 spawn | `add_actor + append_scene` | t=6s 视口出现 desk_1 + desk + desk_2（右侧） |
| 4 | 前序桌子保留 | `append_scene` 不销毁 | 三桌子并排，间距 2.0m |

> **与第 1 课的区别**：资产类型从机器人（XML）变为 USD 物体，但 spawn 范式完全一致。
> `append_scene()` 对任何 spawnable 资产都适用。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaStudio/OrcaLab 已启动并监听 `--addr`
- ✅ 已订阅 desk 系列资产包（desk / desk_1 / desk_2）
- ✅ 加载一个空关卡并点击运行

---

## 3. 目录结构（自包含）

```
examples/scene_building/01_assets/02_load_usd_object/
├── 02_load_usd_object.md   ← 本教程
├── load_usd_object.py      ← 核心逻辑（DeskSpec、load_usd_object）
└── run_load_usd_object.py  ← 脚本入口（argparse + sceneinfo + 主循环）
```

**资产路径**：desk 系列路径硬编码在 `load_usd_object.py` 顶部常量。

**依赖**：`orca_gym.scene.orca_gym_scene` + `orca_gym.utils.rotations` + `orca_gym.log`。

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认：间距 3.0m，间隔 3s
python examples/scene_building/01_assets/02_load_usd_object/run_load_usd_object.py

# 自定义间距与间隔
python examples/scene_building/01_assets/02_load_usd_object/run_load_usd_object.py --spacing 4.0 --interval 5
```

> **注意**：推荐使用 `run_load_usd_object.py` 入口（含 sceneinfo + 主循环 + 异常捕获）。
> `load_usd_object.py` 自带 `main()` 也可直接运行，但不含 sceneinfo 阶段报告。

---

## 5. 预期输出

```
加载 USD 物体（三张桌子）@ localhost:50051（间距 2.00m，间隔 3.0s）
加载场景中
清空现有场景...
场景已清空
[1/3] 开始 spawn: desk_1
已经添加 desk_1 @ (-2.0, 0.0, 0.0)
本轮添加完毕，当前场景共 1 张桌子
等待 3.0s 后 spawn 下一个...
[2/3] 开始 spawn: desk
已经添加 desk @ (0.0, 0.0, 0.0)
本轮添加完毕，当前场景共 2 张桌子
等待 3.0s 后 spawn 下一个...
[3/3] 开始 spawn: desk_2
已经添加 desk_2 @ (2.0, 0.0, 0.0)
本轮添加完毕，当前场景共 3 张桌子
本次添加完毕所有模型，如需退出请在当前终端中断或者在OrcaLab退出运行时模式
加载完成
spawn 完成，保持场景运行，按 Ctrl+C 退出
```

**通过条件**：
- ✅ 视口按时序出现三桌子，前序不被销毁
- ✅ 三桌子并排，间距 2.0m
- ✅ 日志输出"已经添加...本次添加完毕..."

---

## 6. 本课概念

| 概念 | 说明 |
|------|------|
| `Actor` | spawnable 资产规格，与第 1 课完全一致 |
| `append_scene()` | 增量 spawn，对 USD 物体同样适用 |
| `DeskSpec` | 桌子规格数据类，类比 `RobotSpec` |

### 代码解析

```python
# DeskSpec → Actor 转换
def _make_actor(spec: DeskSpec) -> Actor:
    return Actor(
        name=spec.name,
        asset_path=spec.asset_path,      # USD 物体路径
        position=np.array(spec.pos, dtype=np.float64),
        rotation=rotations.euler2quat(np.array([0.0, 0.0, 0.0])),
        scale=1.0,
    )
```

- USD 物体与机器人 spawn 流程完全一致，`Actor` 统一接口。
- 差异仅在资产路径（`assets/b819e2ae5bc79b02/...` vs `assets/e071469a36d3c8aa/...`）。

---

## 7. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio gRPC 地址 |
| `--spacing` | `2.0` | 桌子间距（米，沿 x 轴） |
| `--interval` | `3.0` | spawn 间隔（秒） |

---

## 8. 故障排查

### Q1：桌子未出现

**原因**：未订阅 desk 资产包，或资产路径错误。

**解决**：
1. 确认 OrcaStudio 已订阅 desk / desk_1 / desk_2 资产包
2. 确认资产路径 `assets/b819e2ae5bc79b02/default_projectsim/prefabs/desk_*_usda` 正确

### Q2：桌子重叠

**原因**：`--spacing` 设置过小。

**解决**：桌子比机器人占地大，建议 `--spacing ≥ 1.5`。

---

## 9. 参见

- 设计文档：`03_示例开发计划.md §2.1.2`
- 第 1 课：`01_load_mjcf_robot/01_load_mjcf_robot.md`（spawn 范式相同）
