# 第 1 课：加载机器人 spawnable 资产（时序 spawn）

> 场景构建子系统第 1 课。本课通过时序 spawn 三个机器人，验证 `OrcaGymScene.add_actor`
> + `append_scene()` 增量 spawn 范式。

---

## 1. 课程目标

验证 `append_scene()` 增量 spawn 的时序布局能力：

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 第 1 个机器人 spawn | `add_actor + append_scene` | t=0s 视口出现 g1_omnipicker（左侧） |
| 2 | 第 2 个机器人 spawn | `add_actor + append_scene` | t=5s 视口出现 g1_omnipicker + go2（中间） |
| 3 | 第 3 个机器人 spawn | `add_actor + append_scene` | t=10s 视口出现 g1_omnipicker + go2 + h1（右侧） |
| 4 | 前序机器人保留 | `append_scene` 不销毁 | 三机器人以原点为中心并排，go2 居中 |

> **核心范式**：`append_scene()` 跳过 `OnDestroyScene`，仅处理 `m_addActorMap` 队列。
> 与 `publish_scene()`（销毁 → 重建 → 清空）不同，`append_scene()` 是增量 spawn。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaLab 已启动并监听 `--addr`
- ✅ 已订阅 **PlayGroundScene**和**OrcaPlaygroundAssets**资产包
- ✅ 加载一个空关卡并点击运行

---

## 3. 目录结构（自包含）

```
examples/scene_building/01_assets/01_load_mjcf_robot/
├── 01_load_mjcf_robot.md   ← 本教程
├── load_mjcf_robot.py      ← 核心逻辑（RobotSpec、spawn_robot_sequence）
└── run_load_mjcf_robot.py  ← 脚本入口（argparse + sceneinfo + 主循环）
```

**资产路径**：机器人 spawnable 路径硬编码在 `load_mjcf_robot.py` 顶部常量。

**依赖**：`orca_gym.scene.orca_gym_scene` + `orca_gym.utils.rotations` + `orca_gym.log`。

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认：间距 1.0m，间隔 5s
python examples/scene_building/01_assets/01_load_mjcf_robot/run_load_mjcf_robot.py

# 自定义间距与间隔
python examples/scene_building/01_assets/01_load_mjcf_robot/run_load_mjcf_robot.py --spacing 1.5 --interval 3

# 指定 Studio 地址
python examples/scene_building/01_assets/01_load_mjcf_robot/run_load_mjcf_robot.py --addr 192.168.1.100:50051
```

---

## 5. 预期输出

```
时序 spawn 三机器人 @ localhost:50051（间距 1.00m，间隔 5.0s）
加载场景中
清空现有场景...
场景已清空
[1/3] 开始 spawn: g1
已经添加 g1@ (-1.0, 0.0, 0.0)
本轮添加完毕，当前场景共 1 个机器人
等待 5.0s 后 spawn 下一个...
[2/3] 开始 spawn: go2
已经添加 go2 @ (0.0, 0.0, 0.0)
本轮添加完毕，当前场景共 2 个机器人
等待 5.0s 后 spawn 下一个...
[3/3] 开始 spawn: h1
已经添加 h1 @ (1.0, 0.0, 0.0)
本轮添加完毕，当前场景共 3 个机器人
本次添加完毕所有模型，如需退出请在当前终端中断或者在OrcaLab退出运行时模式
加载完成
spawn 完成，保持场景运行，按 Ctrl+C 退出
```

**通过条件**：
- ✅ 视口按时序出现三机器人，前序不被销毁
- ✅ 三机器人以原点为中心并排，go2 居中
- ✅ 日志输出"已经添加...本次添加完毕..."

---

## 6. 本课概念

| 概念 | 说明 |
|------|------|
| `OrcaGymScene` | 场景构建公共 API，提供 `add_actor` / `append_scene` / `publish_scene` |
| `Actor` | spawnable 资产规格数据类（name / asset_path / position / rotation / scale） |
| `append_scene()` | 增量 spawn：不销毁已 spawn 实体，仅处理 `m_addActorMap` 队列 |
| `publish_scene()` | 全量重建：销毁全部 → 重建 → 清空（对比用，本课不使用） |

### 代码解析

```python
# 时序 spawn 核心循环
for i, spec in enumerate(all_specs):
    scene.add_actor(_make_actor(spec))  # 加入 m_addActorMap 队列
    scene.append_scene()                # 增量 spawn，不销毁前序
    if i < len(all_specs) - 1:
        time.sleep(interval)            # 等待下一时序点
```

- `add_actor(actor)`：将 Actor 规格加入 `m_addActorMap`，不立即 spawn。
- `append_scene()`：触发 gRPC `AppendScene` 调用，C++ 侧仅执行 `OnCreateScene` 处理队列，
  跳过 `OnDestroyScene`，实现增量 spawn。

---

## 7. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaLab gRPC 地址 |
| `--spacing` | `1.0` | 机器人间距（米，沿 x 轴） |
| `--interval` | `5.0` | spawn 间隔（秒） |

---

## 8. 故障排查

### Q1：程序运行无任何日志输出

**原因**：运行了 `load_mjcf_robot.py`（模块文件，只有函数定义），而非 `run_load_mjcf_robot.py`（入口）。

**解决**：

```bash
# 正确：运行 run_load_mjcf_robot.py
python examples/scene_building/01_assets/01_load_mjcf_robot/run_load_mjcf_robot.py
```

### Q2：前序机器人消失

**原因**：误用 `publish_scene()` 替代 `append_scene()`。`publish_scene` 会销毁全部已 spawn 实体后重建。

**解决**：确认 `load_mjcf_robot.py` 中使用 `scene.append_scene()` 而非 `scene.publish_scene()`。

### Q3：gRPC 连接失败

**原因**：OrcaLab 未启动，或地址端口错误。

**解决**：
1. 确认 OrcaLab 已启动
2. 确认监听端口（默认 50051）
3. 加载一个空关卡并点击运行

---

## 8. 参见

- 设计文档：`03_示例开发计划.md §2.1.1`
- append_scene C++ 实现：`OrcaEngine/Gems/Mujoco/Code/Source/Runtime/MjGlobalSettingsComponent.cpp`
