# 第 9 课：Studio 视频录制与截帧 — G1 行走录制在线验证

> 阶段四（P4）在线端到端验证第 9 课。本课验证 OrcaGym 新相机 API（客户端
> PyAV remux 录制）在 G1 行走过程中运行正确，复用 Lesson 8 的行走控制链路
> 驱动行走。
>
> **前置依赖**：本课依赖 **Lesson 8 行走控制已验证**。若 Lesson 8 行走不稳定（瘫倒/乱踹），
> 本课视频中 G1 也会出现同样问题，应先修复 Lesson 8。
>
> **API 迁移说明**：OrcaGym 已删除引擎侧 MP4 录制 RPC（`set_camera_sensor_info`
> 已移除，`begin_save_video` / `stop_save_video` / `get_camera_time_stamp` /
> `get_current_frame` 均已废弃为 no-op）。本课改用客户端 PyAV remux 路径：
> `start_streaming` → `render(simulate_index=...)` → `save_streaming`。

---

## 1. 课程目标

验证 `OrcaGymEulerEnv` 提供的新相机 API 在 G1 行走过程中运行正确，
覆盖数值判定 + 人工观察：

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 相机注册 | `get_camera_names` | 非空列表（含 `camera_head`） |
| 2 | 推流启动 | `start_streaming`（RGB+Depth） | 无异常返回 |
| 3 | 帧索引递增 | `get_recorder_manager().get_latest_frame_simulate_index`（每 50 步） | cur > prev |
| 4 | PNG 截帧格式 | `get_frame_png` | PNG magic + PIL 解码通过 |
| 5 | mp4 文件生成 | `save_streaming`（客户端 PyAV remux） | mp4 文件存在且非空 |
| 6 | 时间戳返回 | `RemuxResult.timestamps_ns` | 非空（每帧纳秒时间戳） |

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
> `m_spawnedEntities`，导致相机注册与截帧 RPC 找不到 actor、摄像头流无法激活。
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
├── video_capture_env.py      ← Env 子类（推流录制验证 + 残留文件清理）
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
`verify_step`（每 50 步查帧索引）→ `observe_step`（周期截帧 + 阶段切换）→
`render(simulate_index=...)`（Env 子类自动注入递增索引驱动帧对齐）。

### 步骤 5（人工）：观察 Studio 视口

| 时刻 | 观察提示 | 预期画面 |
|------|---------|---------|
| step 0 | `recording_started` | G1 开始前进，正在推流录制（阶段 1/3） |
| step 150 | `phase_turn` | G1 开始转弯（阶段 2/3） |
| step 300 | `phase_strafe` | G1 开始横移（阶段 3/3） |

### 步骤 6（自动）：脚本输出判定报告

脚本运行结束后：
- 控制台打印数值判定（`[PASS]`/`[FAIL]`）+ 人工观察提示（`[OBSERVE]`）
- JSON 报告写入 `/tmp/euler_Lesson_8__视频输出_report.json`
- 产出物：`/tmp/g1_walk_video/*.mp4`（客户端 remux 录制视频）、
  `/tmp/g1_frames/**/*.png`（截帧）
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
[DEBUG] agent_name='g1_29dof_camera_usda', camera_name='camera_head_...'
[DEBUG] start_streaming succeeded (RGB + Depth)
  [PASS] camera_registered: actual=['camera_head_...'] ...
  [PASS] streaming_started: actual='camera_head_...' ...
  [OBSERVE] recording_started: Studio 视口：G1 开始前进，正在推流录制（阶段 1/3：前进 3 秒）
  [PASS] png_file_valid_format: actual=/tmp/g1_frames/color/camera_head_color_*.png ...
  [PASS] frame_index_increasing_50: ...
  ...
  [PASS] mp4_file_generated: actual=/tmp/g1_walk_video/g1_walk_color.mp4 ...
  [PASS] timestamp_returned: actual=270 ...

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
| 录制视频 | `/tmp/g1_walk_video/g1_walk_color.mp4` | RGB 彩色视频（客户端 PyAV remux） |
| 录制视频 | `/tmp/g1_walk_video/g1_walk_depth.mp4` | 深度视频（客户端 PyAV remux） |
| PNG 截帧 | `/tmp/g1_frames/color/*.png` | 彩色图截帧 |
| PNG 截帧 | `/tmp/g1_frames/normal/*.png` | 法线图截帧 |
| PNG 截帧 | `/tmp/g1_frames/object_color/*.png` | 实例分割色标图截帧 |
| JSON 报告 | `/tmp/euler_Lesson_8__视频输出_report.json` | 判定报告 |

> **启动时自动清理**：每次运行 `before_loop` 会先清除 `_VIDEO_DIR` 下的 mp4 与
> `_FRAME_DIR` 下的 png/npy（递归子目录），避免本次判定读到上一次运行的残留产物。

---

## 6. 常见问题

### Q1：`camera_registered` 失败（get_camera_names 返回空列表）

**原因**：相机未被 Studio 端注册，通常因 G1 未走 AddActor spawn 路径加载。

**解决**：
1. 确认 OrcaStudio/OrcaLab 中关卡已点击「运行」按钮
2. 确认走的是 **AddActor 路径**（空关卡 + 脚本 `spawn_scene`），而非预先放好 G1
3. 检查日志是否出现 `[DEBUG] agent_name=...`（spawn 成功的标志）

### Q2：`streaming_started` 失败（start_streaming 抛异常）

**原因**：相机不存在或未注册（`ValueError`），或 WebSocket 连接超时（`ConnectionError`）。

**解决**：
1. 确认 `camera_registered` 已通过（相机列表非空）
2. 检查 gRPC 地址（`--addr`）是否正确
3. 检查端口 7070/7071 是否被占用（其他进程的旧推流会话）

### Q3：`frame_index_increasing` 失败（cur_frame 为 None 或不递增）

**原因**：推流已启动但客户端录制器未收到帧（WebSocket 未连通，或 render 未驱动）。

**解决**：
1. 确认 Studio 视口在播放（非暂停状态）
2. 确认 `streaming_started` 已通过
3. 检查本机到 Studio 主机的 7070/7071 端口连通性（远程 Studio 需放行 WebSocket 端口）

### Q4：`png_file_valid_format` 失败（PNG 未生成或格式不对）

**原因**：`get_frame_png` 异步写文件未完成，或摄像头流未激活。

**解决**：
1. 确认 `/tmp/g1_frames/` 目录可写
2. 脚本已轮询 10s（0.5s × 20 次），若仍失败可能是 Studio 写盘慢，可增大轮询次数
3. 检查 `camera_head` 摄像头是否存在（G1 模型含 `camera_head`）

### Q5：`mp4_file_generated` 失败（mp4 未生成）

**原因**：`save_streaming` 客户端 remux 失败，或录制器缓冲中无该区间的帧。

**解决**：
1. 确认 `streaming_started` 已通过（无帧则无法 remux）
2. 检查 `/tmp/g1_walk_video/` 目录权限
3. 检查日志中 `[WARN] save_streaming(...)` 的异常信息
4. 确认 OrcaGym 版本包含 `orca_gym.recorder` 模块（客户端 PyAV remux 依赖 PyAV）

### Q6：视频中 G1 行走不稳定

**原因**：Lesson 8 行走控制未通过，行走链路有问题（瘫倒/乱踹）。

**解决**：
1. 先修复 Lesson 8（参见 `08_locomotion.md` §6 常见问题）
2. 本课不修改行走控制逻辑，仅验证视频采集 API
3. 行走稳定后重新运行本课

---

## 7. Studio 采集 API

本课用到两类 API：**相机推流与客户端录制** 与 **帧截取**。
均为 OrcaGym 新相机 API（客户端 PyAV remux，替代已删除的引擎侧 MP4 录制 RPC）。

### 7.1 相机枚举与推流

| API | 作用 | 调用时机 |
|-----|------|---------|
| `get_camera_names()` | 枚举所有已注册相机名称 | `before_loop` |
| `start_streaming(camera_name, **kwargs)` | 一键配置相机属性 + 启动推流与录制器 | `before_loop` |

`start_streaming` 关键参数（未提供的键保留相机当前属性）：

| 参数 | 作用 |
|------|------|
| `capture_rgb` / `capture_depth` | 是否输出 RGB / 深度流（决定创建哪些录制器） |
| `color_port` / `depth_port` | RGB / Depth 流 WebSocket 端口（本课 7070/7071，与 G1 模型 `camera user="7070 7071"` 一致） |
| `width` / `height` / `vertical_fov` | 分辨率与视场角 |
| `near_clip` / `far_clip` / `gamma` | 裁剪面与深度校正 |
| `use_nvenc` / `nvenc_gpu_index` | NvEnc 硬件编码开关与 GPU 索引 |

### 7.2 帧对齐驱动与客户端录制

| API | 作用 | 调用时机 |
|-----|------|---------|
| `render(simulate_index=..., request_idr=...)` | 渲染并携带物理步索引（帧对齐依据，必须递增） | 每控制周期（本课由 Env 子类 `render` 重写自动注入） |
| `get_recorder_manager().get_latest_frame_simulate_index(camera)` | 查询客户端录制器最新已到达帧的索引 | `verify_step` 每 50 步 |
| `save_streaming(camera, type, file_path, start, end)` | 将 `[start, end]` 区间 H.264 流客户端 remux 为 mp4（非阻塞，返回 `Future[RemuxResult]`） | `after_loop` |
| `get_frame_png(dir)` | 异步写 PNG 到目录 | `observe_step`（step 0/100/200/300/400） |

**`RemuxResult`** 字段：`file_path`（输出路径）、`frame_count`（帧数）、
`frame_indices`（每帧的 simulate_index）、`timestamps_ns`（每帧纳秒时间戳）。

> **时序约束**：
> 1. `start_streaming` 必须在 `save_streaming` 之前（录制器需已启动）
> 2. `render(simulate_index=...)` 的索引必须递增，录制器按此索引对齐帧
> 3. 录制段首帧 `request_idr=True`（本课由 Env 子类自动处理），配合
>    `save_streaming` 内部前向截断保证 mp4 起点可播放
> 4. `env.close()` 会自动 `stop_all_and_save` 保存未完成的录制任务

### 7.3 产出物目录结构

```
/tmp/g1_walk_video/            # 客户端 remux 的 mp4（路径由 save_streaming 显式指定）
├── g1_walk_color.mp4          # RGB 彩色视频
└── g1_walk_depth.mp4          # 深度视频

/tmp/g1_frames/                # get_frame_png 的截帧（Studio 端按通道写入）
├── color/                     # RGB 彩色图 PNG
├── depth/                     # 深度图 .npy（浮点数组）
├── normal/                    # 法线图 PNG
└── object_color/              # 实例分割色标图 PNG
```

> 截帧通道是否输出取决于 Studio 端 `CameraCaptureComponent` 的开关。

---

## 8. 下一步

- **Lesson 10**：体操作与拖拽判定（行走中拖拽 G1、锚定后释放、观察恢复行走）
- 本课产出的 mp4 可用于 Lesson 10 的行走基线对比
