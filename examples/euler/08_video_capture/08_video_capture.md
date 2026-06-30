# 第 8 课：Studio 视频录制与截帧 — G1 行走录制在线验证

> 阶段四（P4）在线端到端验证第 8 课。本课验证 Studio 视频/帧/时间戳采集 API 在 G1 行走过程中
> 运行正确，复用 Lesson 7 的行走控制链路驱动行走。
>
> **前置依赖**：本课依赖 **Lesson 7 行走控制已验证**。若 Lesson 7 行走不稳定（瘫倒/乱踹），
> 本课视频中 G1 也会出现同样问题，应先修复 Lesson 7。

---

## 1. 课程目标

验证 `OrcaGymEulerEnv` 提供的 Studio 交互层 API 在 G1 行走过程中运行正确，
覆盖 5 项数值判定 + 1 项人工观察：

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 摄像头使能 | `get_current_frame` | frame_idx >= 0 |
| 2 | 帧索引递增 | `get_next_frame`（每 50 步） | cur > prev |
| 3 | PNG 截帧生成 | `get_frame_png` | 文件存在且 size > 100 |
| 4 | 时间戳查询 | `get_camera_time_stamp` | 返回含 `camera_head` 的 dict |
| 5 | mp4 文件生成 | `begin_save_video` + `stop_save_video` | mp4 文件存在 |
| - | g1_walking_in_video（人工） | Studio 视口 | 录制视频中 G1 行走画面正常 |

> **行走控制**：本课复用 Lesson 7 的 `g1_locomotion.py`（含内联 `HistoryHandler` + PD 控制器），
> 由 `G1Locomotion` 封装观测组装 + ONNX 推理 + 动作后处理 + PD 转力矩，`compute_ctrl` 每周期
> 返回 29 维力矩 tau。行走过程中录制视频、截帧、查询时间戳，验证采集 API 在动态场景下运行正确。

> **录制生命周期**：`before_loop` 调 `begin_save_video`（CaptureMode.ASYNC 异步），
> 循环 500 步（10 秒）后 `after_loop` 调 `stop_save_video`，覆盖完整行走过程。

---

## 2. 前置条件

### 2.1 环境准备

1. **OrcaStudio 运行中**：启动 OrcaStudio，加载含 1 个 G1 的关卡（`g1_29dof_camera.xml`）
2. **关卡已运行**：在 OrcaStudio 中点击「运行」按钮，使关卡进入仿真状态
3. **orca conda 环境**：所有命令在 `orca` 环境下执行

### 2.2 资产文件

```
OrcaPlayground/assets/g1/g1_29dof_camera.xml          # G1 模型（含 camera_head）
OrcaPlayground/assets/g1/config/g1_29dof_hist.yaml    # 配置（PD 参数、默认姿态）
OrcaPlayground/assets/g1/models/dec_loco/model_6600.onnx  # 行走策略 ONNX
```

### 2.3 前置依赖

**Lesson 7 行走控制已通过验收**。本课复用 `g1_locomotion.py` 驱动行走，若 Lesson 7 行走
不稳定，本课视频中 G1 也会瘫倒/乱踹。

---

## 3. 目录结构

本目录自包含，不依赖外部 `envs/` 目录：

```
08_video_capture/
├── 08_video_capture.md       ← 本教程文档
├── video_capture.py          ← 脚本入口
├── video_capture_env.py      ← Env 子类（视频采集验证）
├── g1_base_env.py            ← G1 基类（资产路径指向 assets/g1）
├── scene_scanner.py          ← 场景扫描（探针 OrcaGymEulerEnv）
├── g1_locomotion.py          ← ONNX 行走封装（复用 Lesson 7，含 PD 控制器）
└── online_verifier.py        ← 在线验证器
```

---

## 4. 运行步骤

### 步骤 1（人工）：启动 OrcaStudio 并加载 G1 关卡

1. 启动 OrcaStudio
2. 加载含 1 个 G1 的关卡（场景含 `g1_29dof_camera.xml`，内置 `camera_head`）
3. 点击「运行」按钮，使关卡进入仿真状态

### 步骤 2（自动）：运行 Lesson 8 脚本

```bash
cd OrcaPlayground
conda activate orca

# 默认连接 127.0.0.1:50051
python examples/euler/08_video_capture/video_capture.py

# 指定 Studio 地址
python examples/euler/08_video_capture/video_capture.py --addr 192.168.1.100:50051
```

### 步骤 3（自动）：脚本驱动 VideoCaptureEnv 步进 500 帧

脚本通过 `G1BaseEnv.run_lesson` 框架（RTF=1.0 限速）：
- `reset()` → G1 站立到 keyframe 初始姿态
- `before_loop`：摄像头使能检查 + `begin_save_video`（开始录制）
- 循环 500 个控制周期（每周期 20 物理步 = 20ms，共 10 秒仿真，按墙钟对齐 RTF=1.0）
  - `compute_ctrl(step)`：`G1Locomotion.compute_action` ONNX 推理 + PD 控制器 → 29 维 tau
  - `do_simulation(ctrl, frame_skip)`：步进物理仿真
  - `verify_step`（每 50 步）：`get_next_frame` 检查帧索引递增
  - `observe_step`（step 0/250）：录制中行走观察提示
  - `render()`：同步到 Studio 视口
- `after_loop`：`get_frame_png` 截帧 + `get_camera_time_stamp` 时间戳 + `stop_save_video` + mp4 检查

### 步骤 4（人工）：观察 Studio 视口

| 时刻 | 观察提示 | 预期画面 |
|------|---------|---------|
| step 0 | `g1_walking_in_video` | G1 在 ONNX 策略 + PD 控制器驱动下开始行走（录制中） |
| step 250 | `walking_stable_in_video` | 录制中段画面应正常（依赖 Lesson 7 行走已跑通） |

> **RTF=1.0 实时性**：脚本按墙钟对齐每个控制周期 20ms，确保视口动作与真实时间
> 同步，便于人工观察行走姿态。10 秒录制覆盖多个步态周期（`GAIT_PERIOD=0.9s`）。

### 步骤 5（自动）：脚本输出判定报告

脚本运行结束后：
- 控制台打印 5 项 `[PASS]`/`[FAIL]` 数值判定 + 2 项 `[OBSERVE]` 观察提示
- JSON 报告写入 `/tmp/euler_Lesson_8__视频录制_report.json`
- 产出物：`/tmp/g1_walk_video/*.mp4`（录制视频）、`/tmp/g1_frames/color/*.png`（截帧）
- 退出码：全部通过返回 0，有失败返回 1

---

## 5. 预期结果

### 5.1 控制台输出示例

```
[OBSERVE] start: 请在 Studio 视口观察 G1 初始姿态：应站立在地面上
  [PASS] camera_enabled: actual=42 expected=>=0 摄像头使能检查
  [OBSERVE] recording_started: Studio 视口：G1 开始行走，正在录制视频
  [OBSERVE] g1_walking_in_video: Studio 视口：G1 应在策略控制下行走（录制中）
  [PASS] frame_index_increasing_50: actual=92 expected=>42 帧索引递增
  [PASS] frame_index_increasing_100: ...
  ...
  [PASS] frame_index_increasing_450: ...
  [OBSERVE] walking_stable_in_video: Studio 视口：录制中段画面应正常（依赖 Lesson 7 行走已跑通）
  [PASS] png_file_generated: actual=/tmp/g1_frames/color/camera_head_color_42.png expected=exists & size>100 PNG 截帧文件生成
  [PASS] timestamp_returned: actual=['camera_head'] expected=['camera_head'] 时间戳查询返回
  [PASS] mp4_file_generated: actual=['/tmp/g1_walk_video/g1_walk_42.mp4'] expected=non-empty 录制完成后 mp4 文件生成

============================================================
判定报告: Lesson 8: 视频录制
数值判定: 5/5 passed
人工观察: 2 项（请在上方 [OBSERVE] 提示处确认）
总结: ALL PASS
============================================================
```

### 5.2 产出物

| 产出物 | 路径 | 说明 |
|--------|------|------|
| 录制视频 | `/tmp/g1_walk_video/*.mp4` | G1 行走 10 秒视频 |
| PNG 截帧 | `/tmp/g1_frames/color/camera_head_color_*.png` | 单帧彩色图 |
| JSON 报告 | `/tmp/euler_Lesson_8__视频录制_report.json` | 判定报告 |

---

## 6. 常见问题

### Q1：`camera_enabled` 失败（frame_idx = -1）

**原因**：OrcaStudio 关卡未运行，或摄像头未使能。

**解决**：
1. 确认 OrcaStudio 中关卡已点击「运行」按钮
2. 确认关卡含摄像头（`g1_29dof_camera.xml` 内置 `camera_head`）
3. 检查 gRPC 地址是否正确（`--addr`）

### Q2：`frame_index_increasing` 失败（帧号不递增）

**原因**：录制未真正开始，或 `get_next_frame` 轮询超时。

**解决**：
1. 确认 `begin_save_video` 已调用（检查 `recording_started` 观察提示）
2. 确认 Studio 视口在播放（非暂停状态）
3. 检查 `/tmp/g1_walk_video/` 目录是否创建

### Q3：`png_file_generated` 失败（PNG 未生成或 size ≤ 100）

**原因**：`get_frame_png` 异步写文件未完成，或路径错误。

**解决**：
1. 确认 `/tmp/g1_frames/` 目录可写
2. 脚本已轮询 10s（0.5s × 20 次），若仍失败可能是 Studio 写盘慢，可增大轮询次数
3. 检查 `camera_head` 摄像头是否存在

### Q4：`mp4_file_generated` 失败（mp4 未生成）

**原因**：`stop_save_video` 未完成编码，或录制路径错误。

**解决**：
1. 确认 `stop_save_video` 已调用（在 `after_loop` 中）
2. 检查 `/tmp/g1_walk_video/` 目录权限
3. Studio 编码 mp4 可能需要几秒，若立即检查可能未完成

### Q5：视频中 G1 行走不稳定（g1_walking_in_video 失败）

**原因**：Lesson 7 行走控制未通过，行走链路有问题（瘫倒/乱踹）。

**解决**：
1. 先修复 Lesson 7（参见 `07_locomotion.md` §6 常见问题）
2. 本课不修改行走控制逻辑，仅验证视频采集 API
3. 行走稳定后重新运行本课

---

## 7. Studio 采集 API

| API | 作用 | 调用时机 |
|-----|------|---------|
| `begin_save_video(dir, capture_mode=0)` | 开始异步录制到目录 | `before_loop` |
| `get_current_frame` / `get_next_frame` | 查询当前/下一帧索引 | `verify_step` 每 50 步 |
| `get_frame_png(dir)` | 异步写 PNG 到目录 | `after_loop` |
| `get_camera_time_stamp(last_frame)` | 查询相机时间戳 | `after_loop` |
| `stop_save_video` | 停止录制并编码 mp4 | `after_loop` |

> **CaptureMode.ASYNC=0**：异步录制，不阻塞仿真步进；`SYNC=1` 会阻塞直到帧写入完成。

---

## 8. 下一步

- **Lesson 9**：体操作与拖拽判定（行走中拖拽 G1、锚定后释放、观察恢复行走）
- 本课产出的 mp4 可用于 Lesson 9 的行走基线对比
