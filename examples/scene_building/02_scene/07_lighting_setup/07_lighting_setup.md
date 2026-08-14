# 第 7 课：光照系统配置（spawn spotlight + set_light_info + 动态光照）

> 场景构建子系统第 7 课（02_scene 子系统）。本课 spawn 多盏 spotlight 资产并通过
> `set_light_info` 配置光源颜色/强度，env 步进时持续旋转光源 body 并刷新 light info，
> 实现动态光照效果。

---

## 1. 课程目标

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | spotlight 资产正确 spawn | `scene.add_actor(Actor(...))` | 视口可见聚光灯模型 |
| 2 | publish_scene 触发 MJCF 重建 | `scene.publish_scene()` | Studio 视口刷新 |
| 3 | 光源颜色/强度应用 | `scene.set_light_info(name, LightInfo)` | 桌面物体被照亮 |
| 4 | 多光源组合照明 | 10 盏随机色 spotlight | 彩色光照混合 |
| 5 | 动态光照 | `env.step` 内旋转 body + 刷新 light info | 光源持续旋转/变色 |

---

## 2. 前置条件

- conda `orca` 环境可用
- OrcaLab 已启动并监听 `--addr`
- 已订阅资产包：**OrcaPlaygroundAssets + run_light_night**
  - 资产订阅地址：https://simassets.orca3d.cn/
  - 是否需要手动拖动到布局中：**否**

---

## 3. 目录结构（自包含）

```
examples/scene_building/02_scene/07_lighting_setup/
├── 07_lighting_setup.md   ← 本教程
├── lighting_setup.py      ← 核心逻辑（build_lighting_scene + LightsEnv）
└── run_lighting_setup.py  ← 脚本入口（spawn + 光源配置 + 仿真）
```

---

## 4. 资产说明

本课使用以下已验证 spawnable 资产（同 `examples/replicator/run_lights.py`）：

| 资产 | spawnable 路径 | 用途 |
|------|----------------|------|
| cup_of_coffee | `assets/e071469a36d3c8aa/default_project/prefabs/cup_of_coffee_usda` | 演示物体（杯子） |
| office_desk | `assets/e071469a36d3c8aa/default_project/prefabs/office_desk_7_mb_usda` | 桌子（被照亮对象） |
| spotlight | `assets/e071469a36d3c8aa/default_project/prefabs/spotlight` | 聚光灯（×10） |

> 注：spawnable name = 文件路径去掉扩展名。OrcaLab 只识别 `.spawnable` 后缀资产。

---

## 5. 运行步骤

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认 10 盏 spotlight + 启动仿真
python examples/scene_building/02_scene/07_lighting_setup/run_lighting_setup.py

# 自定义光源数量
python examples/scene_building/02_scene/07_lighting_setup/run_lighting_setup.py --light-count 5

# 自定义 OrcaLab 地址
python examples/scene_building/02_scene/07_lighting_setup/run_lighting_setup.py --addr localhost:50051
```

**流程**：
1. spawn 桌子 + 杯子 + N 盏 spotlight 到 Studio（编辑模式）
2. `set_light_info` 设置每盏光源的随机颜色和强度
3. 在 OrcaLab 中点击「运行」按钮进入运行模式（MuJoCo 初始化）
4. 脚本自动重试创建 Euler env，拉取已 spawn 的场景 MJCF
5. `env.set_scene_runtime` 注入 OrcaGymSceneRuntime（供 env 刷新 light info）
6. `env.step(action)` 循环步进物理，`env.render()` 推送视口
7. step() 内每帧旋转光源 body + 分批刷新 light info（动态光照）
8. Ctrl+C 退出

---

## 6. 本课概念

### 6.1 光源 API

| API | 说明 |
|-----|------|
| `scene.add_actor(Actor(...))` | spawn spotlight 资产（与普通 actor 相同） |
| `scene.publish_scene()` | 触发 MJCF 重建（光源实体化） |
| `scene.set_light_info(name, LightInfo)` | 设置光源颜色和强度 |
| `env.set_mocap_pos_and_quat(updates)` | 回写光源 body transform 到 Studio（每帧） |
| `scene_runtime.set_light_info(name, LightInfo)` | 仿真中动态刷新光源颜色/强度 |

### 6.2 LightInfo

```python
from orca_gym.scene.orca_gym_scene import LightInfo

light_info = LightInfo(
    color=np.array([1.0, 0.85, 0.6]),  # RGB，0~1
    intensity=800.0,                    # 强度
)
scene.set_light_info("light_warm", light_info)
```

### 6.3 动态光照原理

```
build_lighting_scene()
  ↓ scene.add_actor × (桌子 + 杯子 + N 盏 spotlight)
  ↓ scene.publish_scene()                ← MJCF 重建
  ↓ scene.set_light_info × N            ← 初始颜色/强度
  ↓
env = gym.make("Lights")
  ↓ env.set_scene_runtime(scene_runtime) ← 注入 runtime 供 env 调用 set_light_info
  ↓ env.step() 循环:
      ├─ _rotate_light_bodies_in_batch()  ← set_mocap_pos_and_quat 回写 body transform
      └─ _update_light_info_group()       ← set_light_info 分批刷新颜色/强度
```

> **关键**：`env.reset()` 后 MuJoCo mocap body 从 MJCF 重载 transform，但 O3DE 光源实体的
> transform 可能未同步。每帧通过 `set_mocap_pos_and_quat` 回写 body transform 确保光源
> 位置/朝向正确，这是动态光照可见的核心。

### 6.4 光源 body transform 同步

spotlight 资产 spawn 后，Studio 端的 O3DE 光源实体与 MuJoCo mocap body 是分离的：
- MuJoCo mocap body：由 MJCF 定义，`env.reset()` 后从 MJCF 重载
- O3DE 光源实体：由 spotlight prefab 创建，需通过 `set_mocap_pos_and_quat` 主动同步

`LightsEnv._rotate_light_bodies_in_batch` 每帧读取 mocap body 的当前 xpos/xquat，
通过 `set_mocap_pos_and_quat` 回写到 Studio，触发 O3DE 光源实体的 transform 同步。

### 6.5 分批刷新 light info

10 盏光源每帧全量刷新会导致 250 RPC/sec（60fps × 10），可能触发 OrcaGymSceneRuntime
过载保护。`_update_light_info_group` 将光源分为 5 组，每帧只刷新 1 组（2 盏），
所有光源在 5 帧内轮询一次，RPC 负载降至 ~120 RPC/sec。

### 6.6 与 replicator 的关系

本课光照逻辑照搬自 `examples/replicator/run_lights.py` + `lights_env.py`：

| 来源 | 内容 |
|------|------|
| `examples/replicator/run_lights.py` | `create_scene` → `build_lighting_scene` |
| `examples/replicator/lights_env.py` | `LightsEnv` 类（光源动画逻辑） |
| `examples/replicator/run_simulation.py` | `run_simulation` → `_run_simulation` |

---

## 7. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaLab gRPC 地址 |
| `--light-count` | `10` | 光源数量 |

