"""VideoCaptureEnv — Lesson 9：视频输出验证（截帧 + 录制）。

验证视频输出功能（OrcaGym 新相机 API，客户端 PyAV remux 录制）：

1. 截帧功能: get_frame_png 保存 PNG，校验格式（PNG magic + PIL 解码）
2. 推流与录制: start_streaming 启动 RGB+Depth 推流；render(simulate_index=...)
   驱动帧对齐；save_streaming 将区间 H.264 流客户端 remux 为 mp4
3. 时间戳查询: save_streaming 返回的 RemuxResult.timestamps_ns 携带每帧
   纳秒时间戳（替代已废弃的 get_camera_time_stamp）

阶段序列（num_steps=450，共 9 秒仿真）:
    before_loop:  枚举相机 + 启动 RGB/Depth 推流 + 启动前进
    steps 0–149:  G1 前进 3 秒（lin_vel=0.3）+ 周期截帧
    steps 150–299: G1 转弯 3 秒（ang_vel=0.5）+ 周期截帧
    steps 300–449: G1 横移 3 秒（lin_vel=(0,0.3)）+ 周期截帧
    after_loop:   save_streaming 生成 color/depth mp4 + 时间戳检查 + 提示查看文件

> **前置依赖**：本课依赖 Lesson 8 行走控制已验证（复用 ``g1_locomotion.py`` 驱动行走）。
> **API 迁移说明**：OrcaGym 已删除引擎侧 MP4 录制 RPC（``set_camera_sensor_info``
> 已移除，``begin_save_video`` / ``stop_save_video`` / ``get_camera_time_stamp`` /
> ``get_current_frame`` 均已废弃为 no-op）。本课改用客户端 PyAV remux 路径：
> ``start_streaming`` → ``render(simulate_index=...)`` → ``save_streaming``。
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

# 相机推流端口（与 g1_29dof_camera.xml 中 camera user="7070 7071" 一致）
_COLOR_PORT = 7070
_DEPTH_PORT = 7071

# 动作阶段划分（每阶段 150 步 = 3 秒仿真）
_PHASE_FORWARD_END = 150    # steps 0–149：前进
_PHASE_TURN_END = 300       # steps 150–299：转弯
_PHASE_STRAFE_END = 450     # steps 300–449：横移
# 周期截帧间隔
_FRAME_CAPTURE_INTERVAL = 100


class VideoCaptureEnv(G1BaseEnv):
    """Lesson 9 Env 子类：截帧 + 录制验证。

    重写钩子:
        - initialize_simulation: 创建 G1Locomotion 实例
        - compute_ctrl: ONNX 推理 → q_target
        - _pd_controller: 闭环 PD 单步（架构 §6.4 S6）
        - render: 注入递增 simulate_index 驱动客户端录制帧对齐
        - before_loop: 枚举相机 + 启动推流 + 启动前进
        - verify_step: 每 50 步检查已到达帧 simulate_index 递增
        - observe_step: 周期截帧 + 阶段动作切换
        - after_loop: save_streaming 生成 mp4 + 时间戳 + 提示查看文件
    """

    def initialize_simulation(self):
        """初始化仿真 + 创建 G1Locomotion 行走策略封装。"""
        super().initialize_simulation()
        self.locomotion = G1Locomotion(agent_name=self.agent_name)
        self._prev_frame: int = -1
        self._q_target: np.ndarray = np.zeros(self.model.nu, dtype=np.float64)
        # 相机推流状态（before_loop 中填充）
        self._camera_name: str | None = None
        self._render_idx: int = 0
        self._idr_pending: bool = False

    def compute_ctrl(self, step: int) -> np.ndarray:
        """ONNX 策略推理 → 位置目标 q_target（29 维）。"""
        q_target = self.locomotion.compute_q_target(self)
        self._q_target = q_target
        return q_target

    def _pd_controller(self, target: np.ndarray) -> np.ndarray:
        """闭环 PD 单步 hook（架构 §6.4 S6）：重读 obs 重算 tau。

        由父类 step() 在 frame_skip 循环内每物理步调用一次，
        返回 tau 后由父类 step() → do_simulation(tau, 1) 执行单步仿真。

        Args:
            target: q_target（29,）（由 compute_ctrl 返回），位置目标，非力矩。

        Returns:
            tau (29,): PD 力矩，供 do_simulation(tau, 1) 执行。
        """
        q_target = target
        dof_pos, dof_vel = self.locomotion.read_joint_state(self)
        tau = self.locomotion.compute_tau(q_target, dof_pos, dof_vel)
        return tau

    def render(self, simulate_index: int = -1, request_idr: bool = False):
        """注入递增 simulate_index 驱动客户端录制帧对齐（新相机 API）。

        基类 run_lesson 每控制周期调用一次 render()；客户端录制器按
        simulate_index 区间提取帧，因此这里自动维护递增计数器。推流启动后
        的首帧请求 IDR 关键帧，配合 save_streaming 的前向截断保证 mp4
        起点可正常播放。

        Args:
            simulate_index: 物理仿真步索引；-1 表示自动递增（默认）。
            request_idr: 是否请求引擎输出 IDR 关键帧。
        """
        if simulate_index < 0:
            self._render_idx += 1
            simulate_index = self._render_idx
            if self._idr_pending:
                request_idr = True
                self._idr_pending = False
        return super().render(simulate_index=simulate_index, request_idr=request_idr)

    def before_loop(self, verifier: OnlineVerifier) -> None:
        """循环前：清除残留文件 + 枚举相机 + 启动 RGB/Depth 推流 + 启动前进。

        新相机 API 无需显式"开始录制"——客户端录制器在 start_streaming 时
        已启动并缓存码流，after_loop 中通过 save_streaming 按区间提取。
        """
        # 0. 清除上一次运行残留的 mp4 / png，避免本次判定读到旧产物
        self._clean_previous_outputs()

        # 1. 枚举已注册相机
        camera_names = self.get_camera_names()
        verifier.check(
            "camera_registered",
            len(camera_names) > 0,
            camera_names,
            "non-empty",
            "相机注册检查（get_camera_names）",
        )
        # 优先选头部相机（G1 模型中为 camera_head），否则取第一个
        self._camera_name = next(
            (n for n in camera_names if "head" in n.lower()),
            camera_names[0] if camera_names else None,
        )
        print(
            f"[DEBUG] agent_name={self.agent_name!r}, "
            f"camera_name={self._camera_name!r}"
        )

        # 2. 启动推流与录制器（替代旧 set_camera_sensor_info）
        #    RGB+Depth 双流；端口与 g1_29dof_camera.xml 的 user="7070 7071" 一致
        try:
            self.start_streaming(
                self._camera_name,
                capture_rgb=True,
                capture_depth=True,
                color_port=_COLOR_PORT,
                depth_port=_DEPTH_PORT,
            )
            print("[DEBUG] start_streaming succeeded (RGB + Depth)")
            verifier.check(
                "streaming_started",
                True,
                self._camera_name,
                self._camera_name,
                "相机推流启动（start_streaming RGB+Depth）",
            )
        except (ValueError, ConnectionError) as e:
            verifier.check(
                "streaming_started",
                False,
                str(e),
                "started",
                "相机推流启动（start_streaming RGB+Depth）",
            )

        # 3. 录制帧对齐：后续 render() 自动携带递增 simulate_index（首帧 IDR）
        self._render_idx = 0
        self._idr_pending = True
        self._prev_frame = -1

        # 4. 启动前进（阶段 1：steps 0–149）
        self.locomotion.set_commands(stand=1, lin_vel=(0.3, 0.0), ang_vel=0.0)
        verifier.observe(
            "recording_started",
            "Studio 视口：G1 开始前进，正在推流录制（阶段 1/3：前进 3 秒）",
        )

    def verify_step(self, step: int, verifier: OnlineVerifier) -> None:
        """循环中：每 50 步检查已到达帧的 simulate_index 递增。"""
        if step % 50 == 0 and step > 0:
            cur_frame = self.get_recorder_manager().get_latest_frame_simulate_index(
                self._camera_name
            )
            verifier.check(
                f"frame_index_increasing_{step}",
                cur_frame is not None and cur_frame > self._prev_frame,
                cur_frame,
                f">{self._prev_frame}",
                "帧索引递增",
            )
            if cur_frame is not None:
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
        """清除 _VIDEO_DIR 下的 mp4 与 _FRAME_DIR 下的 png/npy（递归子目录）。

        客户端 remux 的 mp4 由本课显式指定路径写在 _VIDEO_DIR 下；
        get_frame_png 的 PNG 落在 _FRAME_DIR/color/ 等子目录。
        启动时清空这些残留文件，避免本次运行读到上次的产物导致判定误报。
        """
        for pattern in (
            f"{_VIDEO_DIR}/**/*.mp4",
            f"{_FRAME_DIR}/**/*.png",
            f"{_FRAME_DIR}/**/*.npy",
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
        """循环后：save_streaming 生成 mp4 + 时间戳检查 + 提示查看文件。"""
        # 停止行走
        self.locomotion.set_commands(stand=1, lin_vel=(0.0, 0.0), ang_vel=0.0)

        # 客户端 PyAV remux：将 [0, _render_idx] 区间的 H.264 流保存为 mp4。
        # 非阻塞提交 + future.result() 阻塞等待；帧号/时间戳由 RemuxResult 返回
        # （替代已废弃的 get_camera_time_stamp / stop_save_video）。
        end_idx = max(self._render_idx - 1, 0)
        mp4_specs: list[tuple[str, str]] = [
            ("color", os.path.join(_VIDEO_DIR, "g1_walk_color.mp4")),
            ("depth", os.path.join(_VIDEO_DIR, "g1_walk_depth.mp4")),
        ]
        results: dict[str, object] = {}
        for stream_kind, path in mp4_specs:
            try:
                future = self.save_streaming(
                    camera_name=self._camera_name,
                    camera_type=stream_kind,
                    file_path=path,
                    start_simulate_index=0,
                    end_simulate_index=end_idx,
                )
                results[stream_kind] = future.result()
            except Exception as e:
                print(f"[WARN] save_streaming({stream_kind}) 失败: {e}")

        # mp4 文件生成检查（以 color 流为主判定对象）
        color_result = results.get("color")
        mp4_ok = (
            color_result is not None
            and os.path.exists(color_result.file_path)
            and os.path.getsize(color_result.file_path) > 0
        )
        verifier.check(
            "mp4_file_generated",
            mp4_ok,
            color_result.file_path if color_result is not None else "未生成",
            "existing mp4",
            "客户端 remux 生成 mp4 文件",
        )

        # 时间戳检查（RemuxResult.timestamps_ns，每帧纳秒时间戳）
        ts_count = len(color_result.timestamps_ns) if color_result is not None else 0
        verifier.check(
            "timestamp_returned",
            ts_count > 0,
            ts_count,
            ">0 timestamps",
            "录制结果时间戳返回（RemuxResult.timestamps_ns）",
        )

        # 收集所有输出文件
        # 截帧（get_frame_png）由 Studio 端 CameraSensor 按 4 类相机通道独立保存：
        #   color        — RGB 彩色图（PNG）
        #   depth        — 深度图（.npy 浮点数组）
        #   normal       — 法线图（PNG）
        #   object_color — 实例分割色标图（PNG）
        # 通道是否输出取决于 Studio 端 CameraCaptureComponent 的开关。

        print()
        print("=" * 70)
        print("视频输出已生成，请自行查看：")
        print()

        mp4_list = sorted(glob.glob(f"{_VIDEO_DIR}/*.mp4"))
        frame_categories = [
            ("color",        "RGB 彩色图",   "可见光画面，人眼可直观查看"),
            ("depth",        "深度图",       "每像素为相机坐标系下距离（米），记录场景几何结构"),
            ("normal",       "法线图",       "每像素为表面法线向量（RGB 编码），反映表面朝向"),
            ("object_color", "实例分割色标图", "每个物体实例分配唯一颜色，用于实例分割训练/评估"),
        ]

        total_frames = 0
        print("  [MP4 视频]（客户端 PyAV remux）")
        if mp4_list:
            for p in mp4_list:
                print(f"    {p}")
        else:
            print("    （未生成 mp4）")
        print()

        for channel, cn_name, desc in frame_categories:
            png_list = sorted(glob.glob(f"{_FRAME_DIR}/{channel}/*.png"))
            npy_list = sorted(glob.glob(f"{_FRAME_DIR}/{channel}/*.npy"))
            frame_list = png_list + npy_list
            total_frames += len(frame_list)

            print(f"  [{cn_name}]（{channel}）")
            print(f"    用途：{desc}")
            if frame_list:
                print(f"    截帧（{len(frame_list)} 个）：")
                for p in frame_list:
                    print(f"      {p}")
            else:
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
            f"生成 {len(mp4_list)} 个 mp4 + {total_frames} 个截帧，请查看上方文件列表",
        )
