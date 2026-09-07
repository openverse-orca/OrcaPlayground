# 第 3 课：从场景 JSON 批量加载（d12.json）

> 场景构建子系统第 3 课。本课从 OrcaLab 场景导出的 JSON 文件批量加载多 actor 场景，
> 验证 `append_scene()` 对复杂场景的适用性。

---

## 1. 课程目标

验证从 JSON 场景描述文件批量加载的能力：

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | JSON 解析 | `parse_scene_json(json_path)` | 解析出 5 个 AssetActor |
| 2 | 递归遍历 | `_collect_asset_actors(node, out)` | 处理 GroupActor 的 children |
| 3 | 批量 spawn | `add_actor + append_scene` 循环 | 5 个 actor 依次 spawn |
| 4 | 前序保留 | `append_scene` 不销毁 | 5 个 actor 全部保留 |

> **核心能力**：从 OrcaLab 场景导出格式（GroupActor + AssetActor 树）加载场景，
> 支持 transform 的 position / rotation / scale 完整字段。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaLab 已启动
- ✅ 已在 OrcaLab 资产库中订阅 **d12_openpi_pick** 和 **d12_waist** 资产包
- ✅ 在 OrcaLab 中加载 **orcalab_day** 关卡
---

## 3. 目录结构（自包含）

```
examples/scene_building/01_assets/03_load_usd_scene/
├── 03_load_usd_scene.md   ← 本教程
├── load_usd_scene.py      ← 核心逻辑（JSON 解析 + 批量 spawn）
├── run_load_usd_scene.py  ← 脚本入口（argparse + sceneinfo + 主循环）
└── d12.json               ← 场景 JSON 示例（5 个 AssetActor）
```

**资产路径**：由 `d12.json` 中的 `asset_path` 字段定义。

**依赖**：`orca_gym.scene.orca_gym_scene` + `orca_gym.utils.rotations` + `orca_gym.log` + `json` + `pathlib`。

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认：加载同目录 d12.json
python examples/scene_building/01_assets/03_load_usd_scene/run_load_usd_scene.py

# 指定 JSON 文件
python examples/scene_building/01_assets/03_load_usd_scene/run_load_usd_scene.py --json /path/to/scene.json

# 自定义间隔
python examples/scene_building/01_assets/03_load_usd_scene/run_load_usd_scene.py --interval 2.0
```

> **注意**：推荐使用 `run_load_usd_scene.py` 入口（含 sceneinfo + 主循环 + 异常捕获）。
> `load_usd_scene.py` 自带 `main()` 也可直接运行，但不含 sceneinfo 阶段报告。

---

## 5. 预期输出

```
加载 USD 场景 @ localhost:50051（JSON: d12.json (默认), 间隔 1.0s）
加载场景中
从 JSON 加载到 5 个 AssetActor:
  - d12_waist_usda_1 @ (0.72, -0.17, 0.0) scale=1.0
  - table_green_03_1 @ (0.0, 0.0, 0.0) scale=1.4
  - cardboardbox_01_1 @ (0.24, -0.17, 1.07) scale=0.8
  - barcode_01_1 @ (0.25, 0.21, 1.09) scale=1.0
  - c12c_1 @ (0.21, -0.56, 1.07) scale=1.0
清空现有场景...
场景已清空
[1/5] 开始 spawn: d12_waist_usda_1
已经添加 d12_waist_usda_1 @ (0.72, -0.17, 0.0)
本轮添加完毕，当前场景共 1 个 actor
等待 1.0s 后 spawn 下一个...
[2/5] 开始 spawn: table_green_03_1
已经添加 table_green_03_1 @ (0.0, 0.0, 0.0)
本轮添加完毕，当前场景共 2 个 actor
...
[5/5] 开始 spawn: c12c_1
已经添加 c12c_1 @ (0.21, -0.56, 1.07)
本轮添加完毕，当前场景共 5 个 actor
本次添加完毕所有模型，如需退出请在当前终端中断或者在OrcaLab退出运行时模式
加载完成
spawn 完成，保持场景运行，按 Ctrl+C 退出
```

**通过条件**：
- ✅ JSON 正确解析出 5 个 AssetActor
- ✅ 5 个 actor 依次 spawn，前序不被销毁
- ✅ transform 的 position / rotation / scale 正确应用

---

## 6. 本课概念

### JSON 格式

OrcaLab 场景导出格式（GroupActor + AssetActor 树）：

```json
{
  "type": "GroupActor",
  "children": [
    {
      "type": "AssetActor",
      "name": "xxx",
      "asset_path": "assets/...",
      "transform": {
        "position": "[x,y,z]",
        "rotation": "[w,x,y,z]",   // 四元数
        "scale": 1.0
      }
    }
  ]
}
```

### 核心函数

| 函数 | 说明 |
|------|------|
| `parse_scene_json(json_path)` | 从 JSON 解析 AssetActor 列表，递归遍历 GroupActor |
| `_collect_asset_actors(node, out)` | 递归收集 AssetActor，处理嵌套 GroupActor |
| `_parse_vec(s, n)` | 解析 `"[a,b,c]"` 字符串为 tuple |
| `_make_actor(spec)` | `SceneActorSpec → Actor` 转换 |

### 代码解析

```python
# JSON 解析 + 批量 spawn
all_specs = parse_scene_json(json_path)  # 解析 JSON
clear_scene(addr)                         # 清空场景
scene = OrcaGymScene(addr)
for spec in all_specs:
    scene.add_actor(_make_actor(spec))   # 加入队列
    scene.append_scene()                 # 增量 spawn
    time.sleep(interval)                 # 等待下一时序点
```

---

## 7. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaLab gRPC 地址 |
| `--json` | `d12.json`（同目录） | 场景 JSON 文件路径 |
| `--interval` | `1.0` | spawn 间隔（秒） |

---

## 8. 故障排查

### Q1：`FileNotFoundError: 场景 JSON 文件不存在`

**原因**：`--json` 路径错误，或默认 d12.json 被移动。

**解决**：
1. 确认 `d12.json` 在 `03_load_usd_scene/` 目录下
2. 使用 `--json /absolute/path/to/scene.json` 指定绝对路径

### Q2：JSON 解析失败

**原因**：JSON 格式不符合 OrcaLab 场景导出格式。

**解决**：
1. 确认 JSON 顶层 `type` 为 `"GroupActor"`
2. 确认每个 actor 有 `transform.position` / `transform.rotation` / `transform.scale` 字段
3. 确认 position/rotation 为字符串格式 `"[a,b,c]"`

### Q3：部分资产未出现

**原因**：未订阅对应资产包。

**解决**：确认 OrcaLab 已订阅 d12.json 中所有 `asset_path` 引用的资产包。

---

## 9. 参见

- 设计文档：`03_示例开发计划.md §2.1.3`
- d12.json 原始位置：`examples/d12/demo/d12.json`
- 第 1 课：`01_load_mjcf_robot/01_load_mjcf_robot.md`（spawn 范式相同）
