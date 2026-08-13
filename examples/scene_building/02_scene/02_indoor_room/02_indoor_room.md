# 第 2 课：程序化室内房间（墙体/地板/家具 spawnable）

> 场景构建子系统第 2 课（02_scene 子系统）。本课程序化 spawn 室内房间：地板 + 3 面墙 + 桌子，
> 演示批量布局、几何参数化与时序 spawn。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 地板 spawn | `add_actor + append_scene` | 视口出现地板 |
| 2 | 3 面墙围成开放房间 | `add_actor × 3` + 旋转 | 西/东/南三面墙围成 U 形 |
| 3 | 桌子位于 floor 上方 | `add_actor` | 桌子贴在地板上 |
| 4 | 时序 spawn | `append_scene` 增量发布 | 依次出现，间隔 1s |

> **核心范式**：`add_actor` 收集规格 → `append_scene` 增量发布（前序 actor 不被销毁）。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaStudio/OrcaLab 已启动并监听 `--addr`
- ✅ 加载一个空关卡并点击运行

### 资产说明

本样例使用以下已验证 spawnable 资产：

| 资产 | spawnable 路径 | 用途 |
|------|----------------|------|
| wall | `assets/816f95ce16021282/default_project/prefabs/wall_usda` | 墙体 |
| floor | `assets/816f95ce16021282/default_project/prefabs/floor_usda` | 地板 |
| desk_2 | `assets/b819e2ae5bc79b02/default_projectsim/prefabs/desk_2_usda` | 桌子 |

> 注：spawnable name = 文件路径去掉扩展名。OrcaStudio/OrcaLab 只识别 `.spawnable` 后缀资产。

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/02_indoor_room/
├── 02_indoor_room.md      ← 本教程
├── indoor_room.py         ← 核心逻辑（build_indoor_room）
└── run_indoor_room.py     ← 脚本入口（argparse + sceneinfo + 主流程）
```

**依赖**：`orca_gym.scene.orca_gym_scene` + `orca_gym.log`

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认：5m × 5m，墙高 5m，间隔 1s
python examples/scene_building/02_scene/02_indoor_room/run_indoor_room.py

# 自定义尺寸
python examples/scene_building/02_scene/02_indoor_room/run_indoor_room.py --width 6 --height 4

# 指定 Studio 地址
python examples/scene_building/02_scene/02_indoor_room/run_indoor_room.py --addr 192.168.1.100:50051
```

---

## 5. 预期输出

```
构建室内房间 @ localhost:50051（5.0m × 5.0m，墙高 5.0m，间隔 1.0s）
加载场景中
清空现有场景...
构建室内房间：5.0m × 5.0m，墙高 5.0m，共 5 个 actor
[1/5] spawn: floor @ (0.00, 0.00, 0.01)
已经添加 floor
等待 1.0s 后 spawn 下一个...
[2/5] spawn: wall_west @ (-2.50, 0.00, 0.00)
已经添加 wall_west
等待 1.0s 后 spawn 下一个...
[3/5] spawn: wall_east @ (2.50, 0.00, 0.00)
已经添加 wall_east
等待 1.0s 后 spawn 下一个...
[4/5] spawn: wall_south @ (0.00, -2.50, 0.00)
已经添加 wall_south
等待 1.0s 后 spawn 下一个...
[5/5] spawn: desk @ (0.00, 0.00, 0.01)
已经添加 desk
本次添加完毕所有模型，如需退出请在当前终端中断或者在OrcaLab退出运行时模式
加载完成
spawn 完成
```

**通过条件**：
- ✅ 视口依次出现 5 个 actor（地板 + 3 墙 + 桌子）
- ✅ 西/东/南三面墙围成 U 形开放房间（北面开放便于观察）
- ✅ 桌子贴在地板上方

---

## 6. 本课概念

### 布局设计

```
     北面开放（无墙，便于观察）
     ┌─────────────┐
     │             │
     │   [桌子]    │   ← 位于房间中央，贴 floor 上方
     │             │
西墙 │             │ 东墙  ← 沿 y 轴，无旋转
(x=  │             │ (x=
-2.5)│             │ +2.5)
     │             │
     │             │
     └─────────────┘
     南墙 (y=-2.5，绕 z 轴 90°)

     地板铺满原点（z=0.01，微抬避免 Z-fighting）
```

### 墙体旋转

wall 默认沿 y 轴展开，因此：

| 墙 | 位置 | 旋转 | 说明 |
|----|------|------|------|
| 西墙 | (-2.5, 0, 0) | 无 | 沿 y 轴，无需旋转 |
| 东墙 | (2.5, 0, 0) | 无 | 沿 y 轴，无需旋转 |
| 南墙 | (0, -2.5, 0) | 绕 z 轴 90° | 改为沿 x 轴 |

> 绕 z 轴 90° 的四元数：`(cos(π/4), 0, 0, sin(π/4))` ≈ `(0.7071, 0, 0, 0.7071)`

### 时序 spawn 范式

使用 `append_scene` 增量发布，前序 actor 不被销毁：

```python
scene.add_actor(actor1)
scene.append_scene()   # actor1 出现
time.sleep(interval)

scene.add_actor(actor2)
scene.append_scene()   # actor2 出现，actor1 保留
```

> `publish_scene` 是销毁-重建语义（先销毁全部 spawned actor，再用队列重建），
> 不适合时序 spawn。`append_scene` 仅处理新增 actor。

### 坐标系说明

- **spawn pos 是 body 原点**，不是几何中心
- wall/floor prefab 内部 geom 有局部偏移（geom 中心在 body 上方）
- 因此 body z=0 时，wall 底部贴地、floor 顶面贴地
- floor 额外微抬 `FLOOR_Z_OFFSET=0.01` 避免与场景自带地面 Z-fighting 闪烁

---

## 7. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio/OrcaLab gRPC 地址 |
| `--width` | `5.0` | 房间宽度（米，墙长度） |
| `--height` | `5.0` | 房间高度（米，墙高度） |
| `--interval` | `1.0` | spawn 间隔（秒） |

---

## 8. 故障排查

### Q1：墙体朝向不对

**原因**：wall 资产默认朝向可能不是沿 y 轴。

**解决**：调整 `indoor_room.py` 中 `_QUAT_Z90` 常量。

### Q2：桌子位置不对

**原因**：desk_2 资产的默认尺寸或朝向可能与预期不同。

**解决**：调整 `indoor_room.py` 中 desk 的 pos 参数。

### Q3：Spawnable name not found

**原因**：资产路径错误或未注册到 OrcaStudio/OrcaLab。

**解决**：
1. 确认 spawnable 文件后缀为 `.spawnable`（不是 `.prefab`）
2. 确认路径格式为 `assets/<hash>/<project>/prefabs/<name>`（去掉扩展名）
3. 在 OrcaStudio/OrcaLab 中刷新资产目录

### Q4：前序 actor 在新 actor spawn 时消失

**原因**：`orca_gym` 版本较旧，不含 `append_scene` 方法，降级为 `publish_scene`（销毁-重建语义）。

**解决**：升级 `orca_gym` 到开发版（`pip install -e /path/to/OrcaGym`）。

---
