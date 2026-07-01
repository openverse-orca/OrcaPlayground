"""VideoCaptureEnv — Lesson 8：视频输出验证（截帧 + 录制）。

验证视频输出功能:

1. 截帧功能: get_frame_png 保存 PNG，校验格式（PNG magic + PIL 解码）
2. 录制功能: begin_save_video 启动录制，运动过程中帧索引递增，
   stop_save_video 后生成 mp4 文件
3. 时间戳查询: get_camera_time_stamp 返回 camera_head_* 键

阶段序列（num_steps=450，共 9 秒仿真）:
    before_loop:  激活摄像头流 + 使能检查 + 开始录制
    steps 0–149:  G1 前进 3 秒（lin_vel=0.3）+ 周期截帧
    steps 150–299: G1 转弯 3 秒（ang_vel=0.5）+ 周期截帧
    steps 300–449: G1 横移 3 秒（lin_vel=(0,0.3)）+ 周期截帧
    after_loop:   停止录制 + mp4 检查 + 时间戳查询 + 提示查看文件

> **前置依赖**：本课依赖 Lesson 7 行走控制已验证（复用 ``g1_locomotion.py`` 驱动行走）。
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np
from g1_base_env import G1BaseEnv, OnlineVerifier
from g1_locomotion import G1Locomotion
from PIL import Image

# 视频输出目录与截帧目录
_VIDEO_DIR = "/tmp/g1_walk_video"
_FRAME_DIR = "/tmp/g1_frames"

# PNG magic bytes: \x89PNG\r\n\x1a\n
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# 动作阶段划分（每阶段 150 步 = 3 秒仿真）
_PHASE_FORWARD_END = 150    # steps 0–149：前进
_PHASE_TURN_END = 300       # steps 150–299：转弯
_PHASE_STRAFE_END = 450     # steps 300–449：横移
# 周期截帧间隔
_FRAME_CAPTURE_INTERVAL = 100


class VideoCaptureEnv(G1BaseEnv):
    """Lesson 8 Env 子类：截帧 + 录制验证。

    重写钩子:
        - initialize_simulation: 创建 G1Locomotion 实例
        - compute_ctrl: ONNX 推理 → q_target
        - do_simulation: 闭环 PD 步进
        - before_loop: 摄像头使能 + 开始录制 + 启动前进
        - verify_step: 每 50 步检查帧索引递增
        - observe_step: 周期截帧 + 阶段动作切换
        - after_loop: 停止录制 + mp4 检查 + 时间戳 + 提示查看文件
    """

    def initialize_simulation(self):
        """初始化仿真 + 创建 G1Locomotion 行走策略封装。"""
        super().initialize_simulation()
        self.locomotion = G1Locomotion(agent_name=self.agent_name)
        self._prev_frame: int = 0
        self._q_target: np.ndarray = np.zeros(self.model.nu, dtype=np.float64)

    def compute_ctrl(self, step: int) -> np.ndarray:
        """ONNX 策略推理 → 位置目标 q_target（29 维）。"""
        q_target = self.locomotion.compute_q_target(self)
        self._q_target = q_target
        return q_target

    def do_simulation(self, ctrl: np.ndarray, n_frames: int) -> None:
        """闭环 PD 步进：每物理步重读 obs 重算 tau。"""
        q_target = ctrl
        for _ in range(n_frames):
            dof_pos, dof_vel = self.locomotion.read_joint_state(self)
            tau = self.locomotion.compute_tau(q_target, dof_pos, dof_vel)
            self.set_ctrl(tau)
            self.mj_step(1)

    def before_loop(self, verifier: OnlineVerifier) -> None:
        """循环前：清除残留文件 + 激活摄像头流 + 开始录制 + 使能检查 + 启动前进。

        顺序约束：begin_save_video 必须在 get_current_frame 之前调用。
        Studio 端 CameraSyncManager::GetCurrentFrameIndex 会 WaitBeginSave(30ms)，
        若 m_isBeginSaving==false 则超时返回 -1，导致 camera_enabled 误判失败。
        """
        # 0. 清除上一次运行残留的 mp4 / png，避免本次判定读到旧产物
        self._clean_previous_outputs()

        # 1. 设置摄像头传感器开关（gRPC SetCameraSensorInfo，扩展参数）
        #    通过新扩展的 optional 字段一次性开启所有流和录制，无需 Editor 手工操作。
        #    扩展参数（None 表示不修改 server 现有值，对应 proto3 optional 语义）：
        #      capture_normal / capture_object_color / is_recording / ...
        print(f"[DEBUG] agent_name={self.agent_name!r}")
        try:
            self.set_camera_sensor_info(
                actor_name=self.agent_name,
                capture_rgb=True,
                capture_depth=True,          # 开启深度流
                save_mp4_file=True,          # 开启 MP4 录制
                use_dds=False,
                # 扩展参数：开启所有相机流 + 录制
                capture_normal=True,         # 法线图
                capture_object_color=True,   # 实例分割色标图
                is_recording=True,           # 激活录制（替代手工勾选 IsRecording）
                random_object_color=True,    # 随机分配物体颜色（object_color 图更易区分）
            )
            print("[DEBUG] set_camera_sensor_info succeeded (all streams + recording)")
        except RuntimeError as e:
            print(f"[DEBUG] set_camera_sensor_info failed: {e}")

        # 2. 开始录制（设置 m_isBeginSaving=true，后续 get_current_frame 才能返回有效帧号）
        os.makedirs(_VIDEO_DIR, exist_ok=True)
        self.begin_save_video(_VIDEO_DIR, capture_mode=0)

        # 3. 摄像头使能检查（begin_save_video 之后，m_isBeginSaving==true）
        frame_idx = self.get_current_frame()
        verifier.check(
            "camera_enabled",
            frame_idx >= 0,
            frame_idx,
            ">=0",
            "摄像头使能检查",
        )

        # 4. 启动前进（阶段 1：steps 0–149）
        self.locomotion.set_commands(stand=1, lin_vel=(0.3, 0.0), ang_vel=0.0)
        verifier.observe(
            "recording_started",
            "Studio 视口：G1 开始前进，正在录制视频（阶段 1/3：前进 3 秒）",
        )

        self._prev_frame = frame_idx

    def verify_step(self, step: int, verifier: OnlineVerifier) -> None:
        """循环中：每 50 步检查帧索引递增。"""
        if step % 50 == 0 and step > 0:
            cur_frame = self.get_next_frame()
            verifier.check(
                f"frame_index_increasing_{step}",
                cur_frame > self._prev_frame,
                cur_frame,
                f">{self._prev_frame}",
                "帧索引递增",
            )
            self._prev_frame = cur_frame

    def observe_step(self, step: int, verifier: OnlineVerifier) -> None:
        """循环中：阶段动作切换 + 周期截帧。"""
        # 阶段动作切换
        if step == _PHASE_FORWARD_END:
            self.locomotion.set_commands(stand=1, lin_vel=(0.0, 0.0), ang_vel=0.5)
            verifier.observe(
                "phase_turn",
                "Studio 视口：G1 开始转弯（阶段 2/3：转弯 3 秒）",
            )
        elif step == _PHASE_TURN_END:
            self.locomotion.set_commands(stand=1, lin_vel=(0.0, 0.3), ang_vel=0.0)
            verifier.observe(
                "phase_strafe",
                "Studio 视口：G1 开始横移（阶段 3/3：横移 3 秒）",
            )

        # 周期截帧（step 0, 100, 200, 300, 400）
        if step % _FRAME_CAPTURE_INTERVAL == 0:
            self._capture_and_validate_frame(verifier, step)

    # --- 残留文件清理 ---

    def _clean_previous_outputs(self) -> None:
        """清除 _VIDEO_DIR 下的 mp4 与 _FRAME_DIR 下的 png（递归子目录）。

        Studio 端 mp4 落在 ${_VIDEO_DIR}/video/、${_VIDEO_DIR}/depth/ 等子目录；
        PNG 落在 ${_FRAME_DIR}/color/、${_FRAME_DIR}/normal/ 等子目录。
        启动时清空这些残留文件，避免本次运行读到上次的产物导致判定误报。
        """
        for pattern in (
            f"{_VIDEO_DIR}/**/*.mp4",
            f"{_FRAME_DIR}/**/*.png",
        ):
            for path in glob.glob(pattern, recursive=True):
                try:
                    os.remove(path)
                except OSError:
                    pass

    # --- 截帧格式校验 ---

    def _capture_and_validate_frame(self, verifier: OnlineVerifier, step: int) -> None:
        """截帧并校验 PNG 文件格式。

        调用 get_frame_png 保存 PNG → 等待文件生成 → 校验格式：
        1. 文件存在且 size > 100
        2. PNG magic bytes（\\x89PNG\\r\\n\\x1a\\n）
        3. PIL 解码校验文件完整性
        """
        os.makedirs(_FRAME_DIR, exist_ok=True)
        self.get_frame_png(_FRAME_DIR)

        # 轮询等待最新的 PNG 文件生成
        png_path: str | None = None
        for _ in range(20):  # max_wait 0.5s × 20 = 10s
            pngs = glob.glob(f"{_FRAME_DIR}/color/camera_head_color_*.png")
            if pngs:
                png_path = sorted(pngs)[-1]
                if os.path.getsize(png_path) > 100:
                    break
            time.sleep(0.5)

        # 校验文件格式
        format_valid = False
        fail_reason = "文件未生成"
        if png_path and os.path.exists(png_path):
            with open(png_path, "rb") as f:
                header = f.read(8)
            if header != _PNG_MAGIC:
                fail_reason = f"PNG magic 不匹配: {header!r}"
            else:
                try:
                    with Image.open(png_path) as img:
                        img.verify()
                    format_valid = True
                except Exception as e:
                    fail_reason = f"PIL 解码失败: {e}"
        elif png_path:
            fail_reason = f"文件过小: {os.path.getsize(png_path)} bytes"

        # 仅在 step 0 做正式判定（避免单个截帧失败淹没主流程）
        if step == 0:
            verifier.check(
                "png_file_valid_format",
                format_valid,
                png_path,
                "valid PNG (magic + PIL decode)",
                f"PNG 截帧文件格式校验{'通过' if format_valid else '失败: ' + fail_reason}",
            )
        verifier.observe(
            f"png_capture_step_{step}",
            f"step {step} 截帧：{png_path}"
            f"（{'PASS' if format_valid else 'FAIL: ' + fail_reason}）",
        )

    def after_loop(self, verifier: OnlineVerifier) -> None:
        """循环后：停止录制 + mp4 检查 + 时间戳 + 提示查看文件。"""
        # 停止行走
        self.locomotion.set_commands(stand=1, lin_vel=(0.0, 0.0), ang_vel=0.0)

        # 时间戳查询（在 stop_save_video 之前，此时 m_syncTimeStamp 仍有数据）
        # Studio 端 SYNC 模式下 key 格式为 entityName + "_color"/"_depth"，
        # 即 camera_head_color / camera_head_depth（而非裸 camera_head）。
        timestamps = self.get_camera_time_stamp(last_frame_index=self._prev_frame)
        ts_keys = list(timestamps.keys())
        verifier.check(
            "timestamp_returned",
            any(k.startswith("camera_head") for k in ts_keys),
            ts_keys,
            "contains camera_head_*",
            "时间戳查询返回",
        )

        # 停止录制
        self.stop_save_video()

        # mp4 文件生成检查
        # Studio 端 CameraSensor::BeginSaveMp4File 实际路径：
        #   ${filePath}/video/${entityName}_color.mp4
        #   ${filePath}/depth/${entityName}_depth.mp4
        # 即在传入路径下创建 video/depth 子目录。
        mp4s: list[str] = []
        for _ in range(10):  # 轮询等待 mp4 写入完成
            mp4s = glob.glob(f"{_VIDEO_DIR}/video/*.mp4") + glob.glob(f"{_VIDEO_DIR}/depth/*.mp4")
            if mp4s:
                break
            time.sleep(0.5)
        verifier.check(
            "mp4_file_generated",
            len(mp4s) > 0,
            mp4s,
            "non-empty",
            "录制完成后 mp4 文件生成",
        )

        # 收集所有输出文件
        # Studio 端 CameraSensor 对 4 类相机通道独立保存：
        #   color       — RGB 彩色图（PNG / MP4）
        #   depth       — 深度图（.npy 浮点数组 / MP4）
        #   normal      — 法线图（PNG / MP4）
        #   object_color — 实例分割色标图（PNG / MP4）
        # 通道是否输出取决于 Studio 端 CameraCaptureComponent 的开关
        # （ColorCamera/DepthCamera/NormalCamera/ObjectColorCamera）。
        #
        # 注意 mp4 子目录名与 channel 名不一致：
        #   color 的 mp4 在 ${filePath}/video/ 下（CameraSensor.cpp:582）
        #   其余通道 mp4 在 ${filePath}/${channel}/ 下

        file_categories = [
            ("color",        "video",        "RGB 彩色图",   "可见光画面，人眼可直观查看"),
            ("depth",        "depth",        "深度图",       "每像素为相机坐标系下距离（米），记录场景几何结构"),
            ("normal",       "normal",       "法线图",       "每像素为表面法线向量（RGB 编码），反映表面朝向"),
            ("object_color", "object_color", "实例分割色标图", "每个物体实例分配唯一颜色，用于实例分割训练/评估"),
        ]

        print()
        print("=" * 70)
        print("视频输出已生成，请自行查看：")
        print()

        total_mp4 = 0
        total_frames = 0

        for channel, mp4_dir, cn_name, desc in file_categories:
            mp4_list = sorted(glob.glob(f"{_VIDEO_DIR}/{mp4_dir}/*.mp4"))
            png_list = sorted(glob.glob(f"{_FRAME_DIR}/{channel}/*.png"))
            npy_list = sorted(glob.glob(f"{_FRAME_DIR}/{channel}/*.npy"))
            frame_list = png_list + npy_list
            total_mp4 += len(mp4_list)
            total_frames += len(frame_list)

            print(f"  [{cn_name}]（{channel}）")
            print(f"    用途：{desc}")
            if mp4_list:
                print(f"    MP4 视频（{len(mp4_list)} 个）：")
                for p in mp4_list:
                    print(f"      {p}")
            if frame_list:
                print(f"    截帧（{len(frame_list)} 个）：")
                for p in frame_list:
                    print(f"      {p}")
            if not mp4_list and not frame_list:
                print("    （本通道未生成文件，Studio 端该通道开关可能未开启）")
            print()

        print("-" * 70)
        print("查看提示：")
        print("  MP4 播放（Linux 需安装播放器）：")
        print("    sudo apt install mpv        # 或 vlc / ffmpeg")
        print("    mpv <文件路径>              # 播放 mp4")
        print()
        print("  PNG 查看（彩色/法线/分割图）：")
        print("    xdg-open <文件路径>         # 系统默认图片查看器")
        print()
        print("  深度图 .npy 查看（NumPy 浮点数组，无法用图片查看器打开）：")
        print("    python -c \"import numpy as np; a=np.load('<文件>');")
        print("      print(a.shape, a.dtype, a.min(), a.max())\"")
        print("    # 可视化：python -c \"import numpy as np, cv2;")
        print("      a=np.load('<文件>'); cv2.imshow('d', a/a.max()); cv2.waitKey(0)\"")
        print("=" * 70)
        verifier.observe(
            "output_files_ready",
            f"生成 {total_mp4} 个 mp4 + {total_frames} 个截帧，请查看上方文件列表",
        )
