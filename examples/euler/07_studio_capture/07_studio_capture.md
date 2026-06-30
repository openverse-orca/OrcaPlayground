# 第 7 课：Studio 视频录制与截帧 — G1 行走录制在线验证

> 阶段四（P4）在线端到端验证第 4 课。本课验证 G1 人形机器人在 OrcaStudio 真实仿真中
> 的 Studio 视频/帧/时间戳采集 API 运行正确，同时复现 G1 ONNX 行走策略控制。

---

## 1. 课程目标

验证 `OrcaGymEulerEnv` 提供的 Studio 交互层 API 在 G1 行走过程中运行正确，
覆盖 5 项验证点，贯穿录制生命周期（before_loop → verify_step → after_loop）：

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | 摄像头使能 | `get_current_frame` | frame_idx >= 0 |
| 2 | 帧索引递增 | `get_next_frame`（每 50 步） | cur > prev |
| 3 | PNG 截帧生成 | `get_frame_png` | 文件存在且 size > 100 |
| 4 | 时间戳查询 | `get_camera_time_stamp` | 返回含 `camera_head` 的 dict |
| 5 | mp4 文件生成 | `begin_save_video` + `stop_save_video` | mp4 文件存在 |

> **行走控制**：本课运行 G1 行走控制程序（基于 ONNX 策略 `model_6600.onnx`），
> 由 `G1Locomotion` 封装观测组装 + ONNX 推理 + 动作后处理，`compute_ctrl` 每周期
> 返回 29 维 q_target。行走过程中录制视频、截帧、查询时间戳，验证采集 API 在
> 动态场景下运行正确。

> **录制生命周期**：`before_loop` 调 `begin_save_video`（CaptureMode.ASYNC 异步），
> 循环 500 步（10 秒）后 `after_loop` 调 `stop_save_video`，覆盖完整行走过程。

---

## 2. 前置条件

### 2.1 环境准备

1. **OrcaStudio 运行中**：启动 OrcaStudio，加载含 1 个 G1 的关卡（`g1_29dof_camera.xml`）
2. **关卡已运行**：在 OrcaStudio 中点击「运行」按钮，使关卡进入仿真状态
3. **orca conda 环境**：所有命令在 `orca` 环境下执行

### 2.2 ONNX 策略文件

本课依赖 G1 行走策略 ONNX 文件，已随仓库提供：

```
OrcaPlayground/assets/g1/models/dec_loco/model_6600.onnx
```

配置文件：`OrcaPlayground/assets/g1/config/g1_29dof_hist.yaml`

---

## 3. 运行步骤

### 步骤 1（人工）：启动 OrcaStudio 并加载 G1 关卡

1. 启动 OrcaStudio
2. 加载含 1 个 G1 的关卡（场景含 `g1_29dof_camera.xml`）
3. 点击「运行」按钮，使关卡进入仿真状态

### 步骤 2（自动）：运行 Lesson 7 脚本

```bash
cd OrcaPlayground
conda activate orca

# 默认连接 127.0.0.1:50051
python examples/euler/07_studio_capture/studio_capture.py

# 指定 Studio 地址
python examples/euler/07_studio_capture/studio_capture.py --addr 192.168.1.100:50051
```

### 步骤 3（自动）：脚本驱动 StudioCaptureEnv 步进 500 帧

脚本通过 `G1BaseEnv.run_lesson` 框架（RTF=1.0 限速）：
- `reset()` → G1 站立到 keyframe 初始姿态
- `before_loop`：摄像头使能检查 + `begin_save_video`（开始录制）
- 循环 500 个控制周期（每周期 20 物理步 = 20ms，共 10 秒仿真，按墙钟对齐 RTF=1.0）
  - `compute_ctrl(step)`：`G1Locomotion.compute_action` ONNX 推理 → 29 维 q_target
  - `do_simulation(ctrl, frame_skip)`：步进物理仿真
  - `verify_step`（每 50 步）：`get_next_frame` 检查帧索引递增
  - `observe_step`（step 0/250）：行走观察提示
  - `render()`：同步到 Studio 视口
- `after_loop`：`get_frame_png` 截帧 + `get_camera_time_stamp` 时间戳 + `stop_save_video` + mp4 检查

### 步骤 4（人工）：观察 Studio 视口

| 时刻 | 观察提示 | 预期画面 |
|------|---------|---------|
| step 0 | `g1_walking` | G1 在 ONNX 策略控制下开始行走 |
| step 250 | `walking_stable` | G1 行走稳定，录制中段画面正常 |

> **RTF=1.0 实时性**：脚本按墙钟对齐每个控制周期 20ms，确保视口动作与真实时间
> 同步，便于人工观察行走姿态。10 秒录制覆盖多个步态周期（`GAIT_PERIOD=0.9s`）。

### 步骤 5（自动）：脚本输出判定报告

脚本运行结束后：
- 控制台打印 5 项 `[PASS]`/`[FAIL]` 数值判定 + 2 项 `[OBSERVE]` 观察提示
- JSON 报告写入 `/tmp/euler_Lesson_7__视频录制_report.json`
- 产出物：`/tmp/g1_walk_video/*.mp4`（录制视频）、`/tmp/g1_frames/color/*.png`（截帧）
- 退出码：全部通过返回 0，有失败返回 1

---

## 4. 预期结果

### 4.1 控制台输出示例

```
[OBSERVE] start: 请在 Studio 视口观察 G1 初始姿态：应站立在地面上
  [PASS] camera_enabled: actual=42 expected=>=0 摄像头使能检查
  [OBSERVE] recording_started: Studio 视口：G1 开始行走，正在录制视频
  [OBSERVE] g1_walking: Studio 视口：G1 应在策略控制下行走
  [PASS] frame_index_increasing_50: actual=92 expected=>42 帧索引递增
  [PASS] frame_index_increasing_100: ...
  ...
  [PASS] frame_index_increasing_450: ...
  [OBSERVE] walking_stable: Studio 视口：G1 行走应稳定，录制中段画面正常
  [PASS] png_file_generated: actual=/tmp/g1_frames/color/camera_head_color_42.png expected=exists & size>100 PNG 截帧文件生成
  [PASS] timestamp_returned: actual=['camera_head'] expected=['camera_head'] 时间戳查询返回
  [PASS] mp4_file_generated: actual=['/tmp/g1_walk_video/g1_walk_42.mp4'] expected=non-empty 录制完成后 mp4 文件生成

============================================================
判定报告: Lesson 7: 视频录制
数值判定: 5/5 passed
人工观察: 2 项（请在上方 [OBSERVE] 提示处确认）
总结: ALL PASS
============================================================
```

### 4.2 产出物

| 产出物 | 路径 | 说明 |
|--------|------|------|
| 录制视频 | `/tmp/g1_walk_video/*.mp4` | G1 行走 10 秒视频 |
| PNG 截帧 | `/tmp/g1_frames/color/camera_head_color_*.png` | 单帧彩色图 |
| JSON 报告 | `/tmp/euler_Lesson_7__视频录制_report.json` | 判定报告 |

---

## 5. 常见问题

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

### Q5：G1 行走不稳定或摔倒

**原因**：ONNX 策略观测组装错误，或 G1 初始姿态与策略预期不符。

**解决**：
1. 确认 `G1Locomotion.reset()` 已调用（在 `initialize_simulation` 中创建时自动调用）
2. 确认 `g1_29dof_camera.xml` 的 keyframe 与策略训练时一致
3. 检查 `g1_29dof_hist.yaml` 的 `DEFAULT_DOF_ANGLES` 是否匹配

---

## 6. 实现说明

### 6.1 G1Locomotion 行走控制

`G1Locomotion`（本目录内 `g1_locomotion.py`，含内联 `HistoryHandler`）封装 `model_6600.onnx` 推理：

1. **状态读取**（公共 API）：`env.data.qpos[3:7]`（基座四元数）、`env.data.qvel[3:6]`
   （基座角速度）、`env.query_joint_qpos` / `env.query_joint_qvel`（29 关节状态）
2. **观测组装**：13 项拼接（last_action、base_ang_vel、指令、相位、dof 状态、history、
   projected_gravity、ref_upper、sin/cos_phase），匹配 `decoupled_locomotion_stand_height` 布局
3. **ONNX 推理**：`session.run` → 12 维 lower-body action
4. **后处理**：`scaled = action * 0.25` → concat(ref_upper 17 维) → `+ default_dof_angles`
   → clip 到关节限位 → 29 维 q_target

### 6.2 Studio 采集 API

| API | 作用 | 调用时机 |
|-----|------|---------|
| `begin_save_video(dir, capture_mode=0)` | 开始异步录制到目录 | `before_loop` |
| `get_current_frame` / `get_next_frame` | 查询当前/下一帧索引 | `verify_step` 每 50 步 |
| `get_frame_png(dir)` | 异步写 PNG 到目录 | `after_loop` |
| `get_camera_time_stamp(last_frame)` | 查询相机时间戳 | `after_loop` |
| `stop_save_video` | 停止录制并编码 mp4 | `after_loop` |

> **CaptureMode.ASYNC=0**：异步录制，不阻塞仿真步进；`SYNC=1` 会阻塞直到帧写入完成。

### 6.3 录制生命周期与 run_lesson 框架

```
run_lesson(500):
  reset()
  before_loop:  camera_enabled + begin_save_video
  for step in 0..499:
    compute_ctrl:    G1Locomotion ONNX 推理
    do_simulation:   步进物理
    verify_step:     每 50 步 frame_index_increasing
    observe_step:    step 0/250 行走观察
    render:          同步视口
    RTF 限速:        对齐 20ms 墙钟
  after_loop:    get_frame_png + get_camera_time_stamp + stop_save_video + mp4 检查
  verify_final:  （空）
  report:        5 项判定 + 2 项观察
```

---

## 7. 下一步

- **Lesson 8**：体操作与拖拽判定（在行走中拖拽 G1、锚定后释放、观察恢复行走）
- 本课产出的 mp4 可用于 Lesson 8 的行走基线对比
