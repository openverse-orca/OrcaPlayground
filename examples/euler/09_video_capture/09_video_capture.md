# 第 9 课：Studio 视频录制与截帧 — G1 行走录制在线验证

> 阶段四（P4）在线端到端验证第 9 课。本课验证 Studio 视频/帧/时间戳采集 API 在 G1 行走过程中
> 运行正确，复用 Lesson 8 的行走控制链路驱动行走。
>
> **前置依赖**：本课依赖 **Lesson 8 行走控制已验证**。若 Lesson 8 行走不稳定（瘫倒/乱踹），
> 本课视频中 G1 也会出现同样问题，应先修复 Lesson 8。

---

## 1. 课程目标

验证 `OrcaGymEulerEnv` 提供的 Studio 交互层 API 在 G1 行走过程中运行正确，
覆盖数值判定 + 人工观察：

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 摄像头使能 | `get_current_frame` | frame_idx >= 0 |
| 2 | 帧索引递增 | `get_next_frame`（每 50 步） | cur > prev |
| 3 | PNG 截帧格式 | `get_frame_png` | PNG magic + PIL 解码通过 |
| 4 | 时间戳查询 | `get_camera_time_stamp` | 返回含 `camera_head_*` 的 dict |
| 5 | mp4 文件生成 | `begin_save_video` + `stop_save_video` | mp4 文件存在 |

> **行走控制**：本课复用 Lesson 8 的 `g1_locomotion.py`（含内联 `HistoryHandler` + PD 控制器），
> 由 `G1Locomotion` 封装观测组装 + ONNX 推理 + 动作后处理 + PD 转力矩，`compute_ctrl` 每周期
> 返回 29 维位置目标 q_target，`do_simulation` 闭环 PD 步进转 tau。

---

## 2. 前置条件

### 2.1 环境准备

1. **OrcaStudio/OrcaLab 运行中**：启动 OrcaStudio/OrcaLab
2. **加载空关卡**：加载一个**空关卡**（场景中**不要**预先放置 G1，脚本会通过 `add_actor` 动态加载）
3. **点击运行**：在 OrcaStudio/OrcaLab 中点击「运行」按钮，使关卡进入仿真状态
4. **orca conda 环境**：所有命令在 `orca` 环境下执行

> **为什么用空关卡？** Euler 体系走 `LoadLocalEnv` 路径，不会填充 Studio 端
> `m_spawnedEntities`，导致 `SetCameraSensorInfo` 找不到 actor、摄像头流无法激活。
> 本课通过 `OrcaGymScene.add_actor` + `publish_scene` 走 AddActor 路径加载 G1，
> spawn 时填充 `m_spawnedEntities` 并激活 `CameraCaptureComponent`，随后
> `EulerEnv` 的 `LoadLocalEnv` 从场景生成 MJCF 用于仿真控制。

### 2.2 资产文件

```
OrcaPlayground/examples/euler/assets/g1/g1_29dof_camera.xml          # G1 模型（含 camera_head）
OrcaPlayground/examples/euler/assets/g1/config/g1_29dof_hist.yaml    # 配置（PD 参数、默认姿态）
OrcaPlayground/examples/euler/assets/g1/models/dec_loco/model_6600.onnx  # 行走策略 ONNX
```

### 2.3 障碍物 spawnable（可选，默认启用）

脚本会自动通过 `add_actor` 加载 50 个障碍物（环形分布在 5–10m 区域）。所需的 spawnable
资产由 **Euler_asset 资产包** 提供，需先在 OrcaStudio/OrcaLab 中订阅该资产包
（订阅步骤见 [TUTORIAL.md → 订阅 Euler_asset 资产包](../TUTORIAL.md)）。

脚本自动按顺序尝试两种 spawnable 路径，任一可用即可：

| 资产 | 候选路径 1（OrcaStudio 缓存） | 候选路径 2（OrcaLab Euler_asset） |
|------|------------------------------|-----------------------------------|
| box | `assets/prefabs/obstacle_box_usda` | `assets/e071469a36d3c8aa/default_project/prefabs/obstacle_box_usda` |
| capsule | `assets/prefabs/obstacle_capsule_usda` | `assets/e071469a36d3c8aa/default_project/prefabs/obstacle_capsule_usda` |
| cylinder | `assets/prefabs/obstacle_cylinder_usda` | `assets/e071469a36d3c8aa/default_project/prefabs/obstacle_cylinder_usda` |
| sphere | `assets/prefabs/obstacle_sphere_usda` | `assets/e071469a36d3c8aa/default_project/prefabs/obstacle_sphere_usda` |

> 若两种候选路径均不可用，脚本会抛出错误提醒并列出已尝试路径，请确认 Euler_asset
> 已订阅并导入。

> 若仅想快速测试（不加载障碍物），可加 `--no-obstacles` 参数。

### 2.4 前置依赖

**Lesson 8 行走控制已通过验收**。本课复用 `g1_locomotion.py` 驱动行走，若 Lesson 8 行走
不稳定，本课视频中 G1 也会瘫倒/乱踹。

---

## 3. 目录结构

本目录自包含，不依赖外部 `envs/` 目录：

```
09_video_capture/
├── 09_video_capture.md       ← 本教程文档
├── video_capture.py          ← 脚本入口（spawn_scene + 创建 Env）
├── video_capture_env.py      ← Env 子类（视频采集验证 + 残留文件清理）
├── g1_base_env.py            ← G1 基类（资产路径指向 examples/euler/assets/g1）
├── scene_scanner.py          ← 场景扫描（探针 OrcaGymEulerEnv）
├── g1_locomotion.py          ← ONNX 行走封装（复用 Lesson 8，含 PD 控制器）
├── obstacle_spawner.py       # 障碍物布局生成 + 颜色设置
└── online_verifier.py        ← 在线验证器
```

---

## 4. 用户操作步骤

### 步骤 1（人工）：启动 OrcaStudio/OrcaLab 并加载空关卡

1. 启动 OrcaStudio/OrcaLab
2. 加载一个**空关卡**（无需预先放置 G1，脚本会动态 spawn）
3. （可选）按 §2.3 导入障碍物 mjcf 生成 spawnable
4. 点击「运行」按钮，使关卡进入仿真状态

### 步骤 2（人工）：运行脚本

```bash
cd OrcaPlayground
conda activate orca

# 默认连接 127.0.0.1:50051，加载 G1 + 50 个障碍物
python examples/euler/09_video_capture/video_capture.py

# 指定 Studio 地址
python examples/euler/09_video_capture/video_capture.py --addr 192.168.1.100:50051

# GPU 后端（Euler.SolverMujoco，需 CUDA 可用）
python examples/euler/09_video_capture/video_capture.py --device cuda:0

# 不加载障碍物（仅 G1，空场景，快速测试）
python examples/euler/09_video_capture/video_capture.py --no-obstacles

# 自定义步数（默认 450 = 9 秒仿真）
python examples/euler/09_video_capture/video_capture.py --num-steps 300
```

### 步骤 3（人工）：按提示在 Studio 视口确认场景

脚本 spawn 完 G1 + 障碍物后会暂停并提示：

```
[INFO] 等待 3 秒，让 Studio 端 MuJoCo grpc server 完成重启...
  [PAUSE] 请在 OrcaStudio/OrcaLab 视口中确认 G1 机器人 + 障碍物已出现在场景中（按 Space 键继续）
```

在 Studio 视口确认 G1 + 障碍物已出现后，**按 Space 键**继续。

### 步骤 4（自动）：脚本驱动三阶段行走录制

脚本通过 `G1BaseEnv.run_lesson` 框架（RTF=1.0 限速）步进 450 个控制周期
（每周期 `G1_FRAME_SKIP` 物理步，共 9 秒仿真，按墙钟对齐 RTF=1.0）：

| 阶段 | 步数 | 动作 | 截帧 |
|------|------|------|------|
| 1 前进 | 0–149 | `lin_vel=(0.3, 0.0)` | step 0, 100 |
| 2 转弯 | 150–299 | `ang_vel=0.5` | step 200 |
| 3 横移 | 300–449 | `lin_vel=(0.0, 0.3)` | step 300, 400 |

每个控制周期：`compute_ctrl` ONNX 推理 → `do_simulation` 闭环 PD 步进 →
`verify_step`（每 50 步查帧索引）→ `observe_step`（周期截帧 + 阶段切换）→ `render`。

### 步骤 5（人工）：观察 Studio 视口

| 时刻 | 观察提示 | 预期画面 |
|------|---------|---------|
| step 0 | `recording_started` | G1 开始前进，正在录制视频（阶段 1/3） |
| step 150 | `phase_turn` | G1 开始转弯（阶段 2/3） |
| step 300 | `phase_strafe` | G1 开始横移（阶段 3/3） |

### 步骤 6（自动）：脚本输出判定报告

脚本运行结束后：
- 控制台打印数值判定（`[PASS]`/`[FAIL]`）+ 人工观察提示（`[OBSERVE]`）
- JSON 报告写入 `/tmp/euler_Lesson_8__视频输出_report.json`
- 产出物：`/tmp/g1_walk_video/**/*.mp4`（录制视频）、`/tmp/g1_frames/**/*.png`（截帧）
- 退出码：全部通过返回 0，有失败返回 1

---

## 5. 预期结果

### 5.1 控制台输出示例

```
[INFO] 已注册 G1 + 50 个障碍物到场景
[INFO] 已为 50 个障碍物设置随机颜色
[INFO] 等待 3 秒，让 Studio 端 MuJoCo grpc server 完成重启...
  [PAUSE] 请在 OrcaStudio/OrcaLab 视口中确认 G1 机器人 + 障碍物已出现在场景中（按 Space 键继续）
  [OBSERVE] start: 请在 Studio 视口观察 G1 初始姿态：应站立在地面上
[DEBUG] agent_name='g1_29dof_camera_usda'
[DEBUG] set_camera_sensor_info succeeded (all streams + recording)
  [PASS] camera_enabled: actual=0, expected=>=0 摄像头使能检查
  [OBSERVE] recording_started: Studio 视口：G1 开始前进，正在录制视频（阶段 1/3：前进 3 秒）
  [PASS] png_file_valid_format: actual=/tmp/g1_frames/color/camera_head_color_*.png ...
  [PASS] frame_index_increasing_50: ...
  ...
  [PASS] timestamp_returned: actual=['camera_head_color', ...] ...
  [PASS] mp4_file_generated: actual=['/tmp/g1_walk_video/video/...mp4'] ...

============================================================
判定报告: Lesson 9: 视频输出
数值判定: 7/7 passed
人工观察: 10 项（请在上方 [OBSERVE] 提示处确认）
总结: ALL PASS
============================================================
```

### 5.2 产出物

| 产出物 | 路径 | 说明 |
|--------|------|------|
| 录制视频 | `/tmp/g1_walk_video/video/*.mp4` | RGB 彩色视频 |
| 录制视频 | `/tmp/g1_walk_video/depth/*.mp4` | 深度视频 |
| PNG 截帧 | `/tmp/g1_frames/color/*.png` | 彩色图截帧 |
| PNG 截帧 | `/tmp/g1_frames/normal/*.png` | 法线图截帧 |
| PNG 截帧 | `/tmp/g1_frames/object_color/*.png` | 实例分割色标图截帧 |
| JSON 报告 | `/tmp/euler_Lesson_8__视频输出_report.json` | 判定报告 |

> **启动时自动清理**：每次运行 `before_loop` 会先清除 `_VIDEO_DIR` 下的 mp4 与
> `_FRAME_DIR` 下的 png（递归子目录），避免本次判定读到上一次运行的残留产物。

---

## 6. 常见问题

### Q1：`camera_enabled` 失败（frame_idx = -1）

**原因**：录制未真正开始，或 `set_camera_sensor_info` 未成功激活摄像头。

**解决**：
1. 确认 OrcaStudio/OrcaLab 中关卡已点击「运行」按钮
2. 确认走的是 **AddActor 路径**（空关卡 + 脚本 `spawn_scene`），而非预先放好 G1
3. 检查日志是否出现 `set_camera_sensor_info succeeded`；若 failed 检查 gRPC 地址

### Q2：`frame_index_increasing` 失败（帧号不递增）

**原因**：录制未真正开始，或 Studio 端 `m_isBeginSaving==false`。

**解决**：
1. 确认 `begin_save_video` 已在 `set_camera_sensor_info` 之后调用（顺序约束）
2. 确认 Studio 视口在播放（非暂停状态）
3. 检查 `/tmp/g1_walk_video/` 目录是否创建

### Q3：`png_file_valid_format` 失败（PNG 未生成或格式不对）

**原因**：`get_frame_png` 异步写文件未完成，或摄像头流未激活。

**解决**：
1. 确认 `/tmp/g1_frames/` 目录可写
2. 脚本已轮询 10s（0.5s × 20 次），若仍失败可能是 Studio 写盘慢，可增大轮询次数
3. 检查 `camera_head` 摄像头是否存在（G1 模型含 `camera_head`）

### Q4：`mp4_file_generated` 失败（mp4 未生成）

**原因**：`stop_save_video` 未完成编码，或录制路径错误。

**解决**：
1. 确认 `stop_save_video` 已调用（在 `after_loop` 中）
2. 检查 `/tmp/g1_walk_video/` 目录权限
3. Studio 编码 mp4 可能需要几秒，脚本已轮询 5s

### Q5：`SetCameraSensorInfo failed` / actor not found

**原因**：Studio 端 `m_spawnedEntities` 未填充该 actor。

**解决**：
1. 确认走空关卡 + `spawn_scene` 路径（不要预先放 G1）
2. 确认 `publish_scene` 后等待 3 秒让 MuJoCo grpc server 重启完成
3. 确认按 Space 键继续后才创建 Env

### Q6：视频中 G1 行走不稳定

**原因**：Lesson 8 行走控制未通过，行走链路有问题（瘫倒/乱踹）。

**解决**：
1. 先修复 Lesson 8（参见 `08_locomotion.md` §6 常见问题）
2. 本课不修改行走控制逻辑，仅验证视频采集 API
3. 行走稳定后重新运行本课

---

## 7. Studio 采集 API

本课用到两类 API：**摄像头传感器激活** 与 **视频/帧/时间戳采集**。

### 7.1 摄像头传感器激活

| API | 作用 | 调用时机 |
|-----|------|---------|
| `set_camera_sensor_info(actor_name, capture_rgb, capture_depth, save_mp4_file, use_dds, **kwargs)` | 激活/配置摄像头传感器流（gRPC `SetCameraSensorInfo`） | `before_loop` |

Studio 端 MuJoCo `<camera>` 默认不推送 WebSocket RGB/Depth 流，必须通过本方法显式激活。
`begin_save_video` 只控制 MP4 文件录制，与本方法**正交**——两者都需调用。

**关键扩展参数**（keyword-only，`None` 表示不修改 server 现有值，对应 proto3 optional 语义）：

| 参数 | 作用 |
|------|------|
| `capture_normal` | 是否捕获法线图 |
| `capture_object_color` | 是否捕获实例分割色标图 |
| `is_recording` | 是否正在录制（触发 `InitCameraSensor` 创建传感器 + 注册到 syncManager） |
| `random_object_color` | 是否随机分配物体颜色（object_color 图更易区分） |
| `use_nvenc` / `nvenc_gpu_index` | NvEnc 硬件编码开关与 GPU 索引 |
| `width` / `height` / `vertical_fov` / `near_clip` / `far_clip` / `gamma` | 相机成像参数 |
| `color_port` / `depth_port` | RGB/Depth 流 WebSocket 端口 |
| `dds_topic` / `dds_stream_id` | DDS 传输主题与流 ID |

> **时序约束**：`is_recording=True` 会在 Studio 端 `SetIsRecording` → `InitCameraSensor`
> 中根据 `color/depth/normal/object_color` 开关决定是否创建 `CameraSensor` 并注册到
> `CameraSyncManager`。因此 `is_recording` 必须在所有 `capture_*` 开关设置之后由
> server 端统一触发（本课一次性传入所有参数，server 端保证顺序）。

### 7.2 视频 / 帧 / 时间戳采集

| API | 作用 | 调用时机 |
|-----|------|---------|
| `begin_save_video(dir, capture_mode=0)` | 开始录制到目录（设置 `m_isBeginSaving=true`） | `before_loop`（`set_camera_sensor_info` 之后） |
| `get_current_frame` | 查询当前帧索引 | `before_loop` 使能检查 + `verify_step` |
| `get_next_frame` | 带轮询获取下一帧索引 | `verify_step` 每 50 步 |
| `get_frame_png(dir)` | 异步写 PNG 到目录 | `observe_step`（step 0/100/200/300/400） |
| `get_camera_time_stamp(last_frame)` | 查询相机时间戳（`camera_head_*` 键） | `after_loop`（`stop_save_video` 之前） |
| `stop_save_video` | 停止录制并编码 mp4 | `after_loop` |

> **顺序约束**：`begin_save_video` 必须在 `get_current_frame` 之前调用。
> Studio 端 `CameraSyncManager::GetCurrentFrameIndex` 会 `WaitBeginSave(30ms)`，
> 若 `m_isBeginSaving==false` 则超时返回 -1，导致 `camera_enabled` 误判失败。
>
> **CaptureMode**：`ASYNC=0` 异步录制，不阻塞仿真步进；`SYNC=1` 会阻塞直到帧写入完成。
> 本课用 `capture_mode=0`（异步）。

### 7.3 产出物目录结构

Studio 端 `CameraSensor` 对 4 类相机通道独立保存，mp4 子目录名与 channel 名**不一致**：

```
/tmp/g1_walk_video/
├── video/      # color 通道 mp4（CameraSensor.cpp 硬编码 video/ 子目录）
├── depth/      # depth 通道 mp4
├── normal/     # normal 通道 mp4
└── object_color/  # object_color 通道 mp4

/tmp/g1_frames/
├── color/      # RGB 彩色图 PNG
├── depth/      # 深度图 .npy（浮点数组）
├── normal/     # 法线图 PNG
└── object_color/  # 实例分割色标图 PNG
```

> 通道是否输出取决于 `set_camera_sensor_info` 中 `capture_rgb/capture_depth/
> capture_normal/capture_object_color` 开关。本课全部开启。

---

## 8. 下一步

- **Lesson 10**：体操作与拖拽判定（行走中拖拽 G1、锚定后释放、观察恢复行走）
- 本课产出的 mp4 可用于 Lesson 10 的行走基线对比
