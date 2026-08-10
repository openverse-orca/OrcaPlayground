# 第 7 课：光照系统配置（spawn + 光源）

> 场景构建子系统第 7 课（02_scene 子系统）。本课演示 HDR 环境光、点光源、阴影配置。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 不同光源效果可见 | `add_light` × 3 | 点光源/方向光/环境光 |
| 2 | 阴影方向正确 | 方向光位置 | 阴影朝向正确 |
| 3 | 环境光亮度变化 | `intensity` 参数 | 整体亮度变化 |

> **当前状态**：骨架（TODO 未实现）。演示物体路径 `_DEMO_OBJECT_SPAWNABLE_PATH` 待确认。
> OrcaGym 光源 API 可能缺失，降级为 Studio UI 手动配置光照。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaStudio/OrcaLab 已启动
- ✅ 已订阅演示物体 spawnable 资产包

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/07_lighting_setup/
├── 07_lighting_setup.md  ← 本教程
├── lighting_setup.py     ← 核心逻辑（build_lighting_scene + query_lighting_state）
└── run_lighting_setup.py ← 脚本入口（骨架，argparse + TODO）
```

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

python examples/scene_building/02_scene/07_lighting_setup/run_lighting_setup.py
```

---

## 5. 本课概念

| 光源类型 | 参数 | 说明 |
|----------|------|------|
| point | pos + intensity | 点光源，全向照射 |
| directional | pos + intensity | 方向光，模拟太阳光 |
| ambient | pos + intensity | 环境光，整体填充 |

### 代码解析

```python
# 三种光源
collector.add_light(name="point_light",       light_type="point",       pos=(2, 2, 3), intensity=100)
collector.add_light(name="directional_light", light_type="directional", pos=(0, 0, 10), intensity=50)
collector.add_light(name="ambient_light",     light_type="ambient",     pos=(0, 0, 0),  intensity=20)
```

---

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio gRPC 地址 |

---

## 7. API 缺口

- `OrcaGym` 光源配置 API 可能缺失
- 降级方案：脚本引导用户在 Studio UI 手动配置光照，脚本仅 spawn 演示物体并读取打印当前光照状态
- `query_lighting_state` 待 OrcaGym 扩展光源查询 API 后实现

---

## 8. 参见

- 设计文档：`03_示例开发计划.md §2.2.2 (7)`
