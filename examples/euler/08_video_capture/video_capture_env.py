"""VideoCaptureEnv — Lesson 8：Studio 视频录制与截帧验证（G1 行走录制）。

在阶段四在线模式下，连接 OrcaStudio 加载 G1 关卡，验证 Studio 视频/帧/时间戳
采集 API 在 G1 行走过程中运行正确。脚本通过 G1BaseEnv.run_lesson 框架步进 500 帧，
在 G1Locomotion ONNX 策略 + PD 控制器驱动行走的同时录制视频、截帧、查询时间戳。

> **前置依赖**：本课依赖 Lesson 7 行走控制已验证（复用 `g1_locomotion.py` 驱动行走）。
> 若 Lesson 7 行走不稳定，本课视频中 G1 也会出现瘫倒/乱踹，应先修复 Lesson 7。

验证 API（Studio 交互层）:
    - begin_save_video / stop_save_video（视频录制）
    - get_current_frame / get_next_frame（帧索引查询）
    - get_frame_png（PNG 截帧）
    - get_camera_time_stamp（相机时间戳）

验证点（5 项数值判定 + 1 项人工观察）:
    before_loop:
    1. camera_enabled: get_current_frame() >= 0（摄像头使能）
    verify_step（每 50 步）:
    2. frame_index_increasing_{step}: 帧索引递增
    after_loop:
    3. png_file_generated: get_frame_png 生成 PNG 文件（size > 100）
    4. timestamp_returned: get_camera_time_stamp 返回 camera_head 键
    5. mp4_file_generated: stop_save_video 后 mp4 文件生成
    - g1_walking_in_video: 录制视频中 G1 行走画面正常（依赖 Lesson 7 行走已跑通）

行走控制链路（复用 Lesson 7 的 g1_locomotion.py）:
    ONNX 策略输出位置目标 q_target (29,)
    → PD 控制器: tau = Kp*(q_target - q) + Kd*(0 - qd)
    → clip 到 motor_effort_limit
    → 传给 motor 执行器（G1 执行器是力矩控制）

参见 docs/design/development/orca_gym_euler_phase4_directory_restructure.md §3.7
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np
from g1_base_env import G1BaseEnv, OnlineVerifier
from g1_locomotion import G1Locomotion

# 视频输出目录与截帧目录（after_loop 复用）
_VIDEO_DIR = "/tmp/g1_walk_video"
_FRAME_DIR = "/tmp/g1_frames"


class VideoCaptureEnv(G1BaseEnv):
    """Lesson 8 Env 子类：G1 行走 + Studio 视频采集验证。

    重写钩子:
        - initialize_simulation: 创建 G1Locomotion 实例（含 PD 控制器）
        - compute_ctrl: 调用 G1Locomotion.compute_q_target 返回位置目标（29 维）
        - do_simulation: 闭环 PD 步进（每物理步重读 obs 重算 tau，与原版一致）
        - before_loop: 摄像头使能检查 + begin_save_video
        - verify_step: 每 50 步检查帧索引递增
        - observe_step: 行走观察提示
        - after_loop: 截帧 + 时间戳 + stop_save_video + mp4 检查
    """

    def initialize_simulation(self):
        """初始化仿真 + 创建 G1Locomotion 行走策略封装（含 PD 控制器）。"""
        super().initialize_simulation()
        self.locomotion = G1Locomotion(agent_name=self.agent_name)
        self._prev_frame: int = 0
        # q_target 缓存（compute_ctrl 写入，do_simulation 读取）
        self._q_target: np.ndarray = np.zeros(self.model.nu, dtype=np.float64)

    def compute_ctrl(self, step: int) -> np.ndarray:
        """ONNX 策略推理 → 位置目标 q_target（29 维，rad）。

        本方法只做 ONNX 推理生成 q_target，不做 PD 转力矩。
        PD 转力矩在 do_simulation 中每物理步闭环重算（与原版 g1_env.py
        每 mj_step 重读 obs 重算 PD 一致），避免开环累积误差导致失稳。

        Returns:
            q_target (29,): 关节位置目标（rad）。
        """
        q_target = self.locomotion.compute_q_target(self)
        self._q_target = q_target
        return q_target

    def do_simulation(self, ctrl: np.ndarray, n_frames: int) -> None:
        """闭环 PD 步进：每物理步重读 obs 重算 tau（与原版 g1_env.py 一致）。

        原版 g1_env.step 在 frame_skip 循环内每步重读 qpos/qvel 重算 PD 力矩，
        否则单次 tau 跑 20 步会因开环累积误差导致 G1 失稳侧翻。
        本方法复刻该闭环结构。视口同步由 run_lesson 循环中的 render() 负责。

        Args:
            ctrl: 此处为 q_target（29,）（由 compute_ctrl 返回），不是力矩。
            n_frames: 物理步数（frame_skip=20）。
        """
        q_target = ctrl  # compute_ctrl 返回的是 q_target，不是 tau
        for _ in range(n_frames):
            dof_pos, dof_vel = self.locomotion.read_joint_state(self)
            tau = self.locomotion.compute_tau(q_target, dof_pos, dof_vel)
            self.set_ctrl(tau)
            self.mj_step(1)

    def before_loop(self, verifier: OnlineVerifier) -> None:
        """循环前：摄像头使能检查 + 开始录制。"""
        # 1. 摄像头使能检查
        frame_idx = self.get_current_frame()
        verifier.check(
            "camera_enabled",
            frame_idx >= 0,
            frame_idx,
            ">=0",
            "摄像头使能检查",
        )

        # 2. 开始录制（path 是目录，CaptureMode.ASYNC=0 异步）
        os.makedirs(_VIDEO_DIR, exist_ok=True)
        self.begin_save_video(_VIDEO_DIR, capture_mode=0)
        verifier.observe(
            "recording_started",
            "Studio 视口：G1 开始行走，正在录制视频",
        )

        # 记录初始帧索引，供 verify_step 帧索引递增检查
        self._prev_frame = frame_idx

    def verify_step(self, step: int, verifier: OnlineVerifier) -> None:
        """循环中：每 50 步检查帧索引递增。"""
        if step % 50 == 0:
            cur_frame = self.get_next_frame()
            if step > 0:
                verifier.check(
                    f"frame_index_increasing_{step}",
                    cur_frame > self._prev_frame,
                    cur_frame,
                    f">{self._prev_frame}",
                    "帧索引递增",
                )
            self._prev_frame = cur_frame

    def observe_step(self, step: int, verifier: OnlineVerifier) -> None:
        """循环中：阶段性人工观察提示。"""
        if step == 0:
            verifier.observe(
                "g1_walking_in_video",
                "Studio 视口：G1 应在策略控制下行走（录制中）",
            )
        elif step == 250:
            verifier.observe(
                "walking_stable_in_video",
                "Studio 视口：录制中段画面应正常（依赖 Lesson 7 行走已跑通）",
            )

    def after_loop(self, verifier: OnlineVerifier) -> None:
        """循环后：截帧 + 时间戳 + 停止录制 + mp4 检查。"""
        # 3. 视频截帧（get_frame_png 异步写 PNG 到目录，返回 None）
        os.makedirs(_FRAME_DIR, exist_ok=True)
        self.get_frame_png(_FRAME_DIR)

        # 轮询 PNG 文件生成
        png_path: str | None = None
        for _ in range(20):  # max_wait 0.5s × 20 = 10s
            pngs = glob.glob(f"{_FRAME_DIR}/color/camera_head_color_*.png")
            if pngs:
                png_path = pngs[0]
                if os.path.getsize(png_path) > 100:  # 文件大小稳定
                    break
            time.sleep(0.5)
        verifier.check(
            "png_file_generated",
            png_path is not None and os.path.getsize(png_path) > 100,
            png_path,
            "exists & size>100",
            "PNG 截帧文件生成",
        )

        # 4. 时间戳查询
        timestamps = self.get_camera_time_stamp(last_frame_index=self._prev_frame)
        verifier.check(
            "timestamp_returned",
            "camera_head" in timestamps,
            list(timestamps.keys()),
            ["camera_head"],
            "时间戳查询返回",
        )

        # 5. 停止录制
        self.stop_save_video()

        # 6. mp4 文件生成检查
        mp4s = glob.glob(f"{_VIDEO_DIR}/*.mp4")
        verifier.check(
            "mp4_file_generated",
            len(mp4s) > 0,
            mp4s,
            "non-empty",
            "录制完成后 mp4 文件生成",
        )
