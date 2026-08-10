# 第 4 课：户外地形（高度图 spawnable）

> 场景构建子系统第 4 课（02_scene 子系统）。本课 spawn 高度图地形，演示纹理混合与斜坡物体。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 地形 spawnable 加载 | `add_terrain` | 视口出现起伏地形 |
| 2 | 地形起伏可见 | 高度图数据 | 视口可见山丘/谷地 |
| 3 | 物体在斜坡滑动 | 物理引擎 | 物体沿斜坡滑落 |

> **当前状态**：骨架（TODO 未实现）。资产路径 `_TERRAIN_SPAWNABLE_PATH` 待确认。
> 地形 spawnable 资产需预先生成高度图数据并打包。

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaStudio/OrcaLab 已启动
- ✅ 已订阅地形 spawnable 资产包（含高度图数据）

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/04_outdoor_terrain/
├── 04_outdoor_terrain.md  ← 本教程
├── outdoor_terrain.py     ← 核心逻辑（build_outdoor_terrain）
└── run_outdoor_terrain.py ← 脚本入口（骨架，argparse + TODO）
```

---

## 4. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

python examples/scene_building/02_scene/04_outdoor_terrain/run_outdoor_terrain.py --size 20 --resolution 64
```

---

## 5. 本课概念

| 概念 | 说明 |
|------|------|
| 高度图 | 2D 网格存储地形高程数据 |
| `add_terrain` | ActorCollector 的地形添加方法（spawnable + size） |
| 纹理混合 | 多层纹理根据高度/坡度混合（需资产支持） |

---

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio gRPC 地址 |
| `--size` | `20.0` | 地形尺寸（米） |
| `--resolution` | `64` | 高度图分辨率（待确认） |

---

## 7. 参见

- 设计文档：`03_示例开发计划.md §2.2.2 (4)`
