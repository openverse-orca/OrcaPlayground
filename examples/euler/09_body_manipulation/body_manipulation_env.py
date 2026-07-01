"""BodyManipulationEnv — Lesson 9：体操作与交互式驱动（锚定/mocap 驱动/equality）。

交互式菜单驱动模式：机器人在无绑定状态下向前行走 3 秒，然后暂停询问用户选择
操作（1/2/3/4/5），根据选择绑定 mocap 并在 3 秒内周期性移动机器人，或取消
绑定继续自主行走，或退出。每次暂停前读取 pelvis 位姿等数据做检查。

菜单选项:
    1: 绑定 mocap，提升 0.5 米（z+0.5）
    2: 绑定 mocap，向前移动 1 米（x+1）
    3: 绑定 mocap，向左移动 1 米（y+1）
    4: 取消绑定，机器人继续自主向前移动 3 秒
    5: 结束退出示例程序

验证 API（体操作层，公共 API）:
    - anchor_actor / release_body_anchored：程序化锚定/释放 G1 pelvis
    - set_mocap_pos_and_quat：驱动 mocap body
    - anchored_actor：查询当前锚定的 actor（判断绑定状态）
    - get_body_xpos_xmat_xquat：读取位姿

> **前置依赖**：本课依赖 Lesson 7 行走控制已验证（复用 ``g1_locomotion.py`` 驱动行走）。
> G1 模型 ``g1_29dof_camera.xml`` 已内置 mocap body ``TestMocapAnchor``。
>
> **命名空间**：OrcaStudio 加载模型时会给所有 body 加命名空间前缀（如
> ``g1_29dof_camera_usda_``），本课在 ``initialize_simulation`` 中通过后缀扫描
> 动态解析实际 body 名，不硬编码前缀。
>
> **交互式 UI 操作**（Studio 鼠标拖拽）属于基础能力，不在本课展示。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np
from g1_base_env import G1BaseEnv, OnlineVerifier
from g1_locomotion import G1Locomotion

# XML 中定义的 body 后缀（无命名空间前缀，实际名由 initialize_simulation 动态解析）
# 命名为 TestMocapAnchor 以避免与 Studio 关卡自带的 ActorManipulator 重名
_MOCAP_BODY_SUFFIX = "TestMocapAnchor"
# 行走参数
_WALK_LIN_VEL = 0.3  # 前进速度 m/s
# 每个交互阶段的步数（3 秒 = 3 / (frame_skip * time_step) = 3 / 0.02 = 150）
_PHASE_STEPS = 150
# mocap 移动目标位移（m）
_DELTA_UP = np.array([0.0, 0.0, 0.5])      # 提升 0.5 米
_DELTA_FORWARD = np.array([1.0, 0.0, 0.0])  # 向前 1 米
_DELTA_LEFT = np.array([0.0, 1.0, 0.0])    # 向左 1 米
# 锚定后位置稳定性容差（m）
_ANCHOR_ATOL = 0.1


class BodyManipulationEnv(G1BaseEnv):
    """Lesson 9 Env 子类：交互式体操作验证。

    重写钩子:
        - initialize_simulation: 创建 G1Locomotion + 动态解析 mocap body 名
        - run_interactive: 交互式菜单循环（替代 run_lesson 固定步数循环）
    """

    def initialize_simulation(self):
        """初始化仿真 + 创建 G1Locomotion 行走策略封装 + 解析场景 body 名。"""
        super().initialize_simulation()
        self.locomotion = G1Locomotion(agent_name=self.agent_name)
        self._q_target: np.ndarray = np.zeros(self.model.nu, dtype=np.float64)
        # 动态解析 XML 内置 mocap body 的实际场景名（含命名空间前缀）
        self._mocap_body_name = self._resolve_body_by_suffix(_MOCAP_BODY_SUFFIX)
        # 交互循环计数（用于判定项命名，确保多次循环判定名唯一）
        self._cycle_count = 0
        self._check_count = 0

    def _resolve_body_by_suffix(self, suffix: str) -> str:
        """通过后缀扫描场景 body 名表，返回完整名（含命名空间前缀）。

        OrcaStudio 加载模型时给所有 body 加命名空间前缀（如
        ``g1_29dof_camera_usda_``），硬编码 XML 中的 body 名会因缺少前缀而
        在 ``body_name2id`` 中 KeyError。本方法扫描 ``model.get_body_names()``
        找到以 ``_{suffix}`` 结尾的完整名。

        Args:
            suffix: XML 中定义的 body 名（无前缀，如 ``TestMocapAnchor``）。

        Returns:
            实际场景中的完整 body 名。

        Raises:
            ValueError: 场景中找不到匹配的 body。
        """
        for name in self.model.get_body_names():
            if name == suffix or name.endswith(f"_{suffix}"):
                return name
        raise ValueError(
            f"场景中找不到后缀为 {suffix!r} 的 body，"
            f"可用 body: {list(self.model.get_body_names())}"
        )

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

    # --- 交互式循环（替代 run_lesson）---

    def run_interactive(self, verifier: OnlineVerifier) -> dict[str, Any]:
        """交互式菜单循环：行走 3 秒 → 暂停询问 → 执行操作 → 循环。

        流程:
            1. reset → 启动行走
            2. 循环:
                a. 无绑定向前行走 3 秒（150 步）
                b. 暂停前读取 pelvis 位姿做检查
                c. 显示菜单，等待用户输入 1/2/3/4/5
                d. 根据选择执行：
                    - 1/2/3: 绑定 mocap + 3 秒内周期性移动到目标
                    - 4: 取消绑定 + 自主行走 3 秒
                    - 5: 退出循环
            3. 输出报告

        Args:
            verifier: 在线判定器实例。

        Returns:
            判定报告字典。
        """
        self.reset()
        verifier.observe(
            "start",
            "Studio 视口：G1 初始站立姿态，即将开始交互式体操作演示",
            step=0,
        )

        # 启动行走
        self.locomotion.set_commands(stand=1, lin_vel=(_WALK_LIN_VEL, 0.0), ang_vel=0.0)
        verifier.observe(
            "g1_walking_start",
            "Studio 视口：G1 开始前进行走（无绑定状态）",
        )

        cycle_target = self.frame_skip * self._time_step

        while True:
            self._cycle_count += 1
            # 阶段 1：无绑定向前行走 3 秒
            self._run_phase(verifier, phase_name="free_walk", bound=False, delta=None,
                            cycle_target=cycle_target)

            # 阶段 2：暂停前读取数据做检查
            self._check_state_before_pause(verifier)

            # 阶段 3：显示菜单，等待用户输入
            choice = self._prompt_user_choice()

            if choice == "5":
                verifier.observe(
                    "user_exit",
                    "用户选择退出，演示结束",
                )
                break

            if choice == "4":
                # 取消绑定，继续自主向前移动 3 秒
                self._ensure_released(verifier)
                self._run_phase(verifier, phase_name="free_walk_after_release",
                                bound=False, delta=None, cycle_target=cycle_target)
            else:
                # 绑定 + 周期性移动 3 秒
                delta = {
                    "1": _DELTA_UP,
                    "2": _DELTA_FORWARD,
                    "3": _DELTA_LEFT,
                }[choice]
                phase_name = {
                    "1": "bound_up",
                    "2": "bound_forward",
                    "3": "bound_left",
                }[choice]
                self._run_phase(verifier, phase_name=phase_name, bound=True,
                                delta=delta, cycle_target=cycle_target)

        # 结束
        self.locomotion.set_commands(stand=0, lin_vel=(0.0, 0.0), ang_vel=0.0)
        verifier.observe(
            "lesson9_done",
            "Lesson 9 交互式体操作演示结束",
        )
        return verifier.report()

    def _run_phase(
        self,
        verifier: OnlineVerifier,
        phase_name: str,
        bound: bool,
        delta: np.ndarray | None,
        cycle_target: float,
    ) -> None:
        """执行一个 3 秒阶段（150 步）。

        Args:
            verifier: 判定器（用于观察提示）。
            phase_name: 阶段名称（用于日志）。
            bound: 是否绑定 mocap。True 表示绑定 + 周期性移动；False 表示自主行走。
            delta: 绑定模式下的目标位移（3,）；自主行走模式为 None。
            cycle_target: 每控制周期目标墙钟时长（RTF=1.0 限速）。
        """
        agent = self.agent_name
        pelvis_name = f"{agent}_pelvis"

        if bound:
            # 程序化绑定 pelvis（weld）：使用 G1 预制的 TestMocapAnchor + anchor_box_weld，
            # 不走 anchor_actor（那是 Studio UI 系统自带 ActorManipulator 的入口）。
            # 通过 _anchor_mocap_name 缓存指定程序化 mocap body，
            # update_anchor_equality_constraints 会据此匹配 anchor_box_weld 槽位。
            # 1. 记录起始位姿（用于周期性移动插值与位移校验）
            start_pose = self.get_body_xpos_xmat_xquat([pelvis_name])[pelvis_name]
            start_pos = start_pose["xpos"].copy()
            # 2. 设置 mocap 到 pelvis 当前位姿（避免绑定瞬间跳变）
            self.set_mocap_pos_and_quat({
                self._mocap_body_name: {
                    "pos": start_pos.tolist(),
                    "quat": start_pose["xquat"].tolist(),
                }
            })
            # 3. 建立 weld 约束（TestMocapAnchor ↔ pelvis）
            #    _anchor_mocap_name 指定程序化 mocap body，供 release_body_anchored 释放
            self._anchor_mocap_name = self._mocap_body_name
            self.update_anchor_equality_constraints(pelvis_name, anchor_type="weld")
            self._anchored_actor = pelvis_name
            self._anchor_type = "weld"
            # 4. 绑定后停止行走（stand=0），机器人跟随 mocap
            self.locomotion.set_commands(stand=0, lin_vel=(0.0, 0.0), ang_vel=0.0)
            verifier.observe(
                f"{phase_name}_start",
                f"Studio 视口：G1 pelvis 已绑定，3 秒内移动 {delta.tolist()}",
            )
        else:
            # 自主行走
            self.locomotion.set_commands(stand=1, lin_vel=(_WALK_LIN_VEL, 0.0), ang_vel=0.0)

        for step in range(_PHASE_STEPS):
            cycle_start = time.perf_counter()
            ctrl = self.compute_ctrl(step)
            self.do_simulation(ctrl, self.frame_skip)

            if bound and delta is not None:
                # 周期性移动 mocap：线性插值，3 秒内移动 delta
                progress = (step + 1) / _PHASE_STEPS
                target_pos = start_pos + delta * progress
                self.set_mocap_pos_and_quat({
                    self._mocap_body_name: {
                        "pos": target_pos.tolist(),
                        "quat": [1, 0, 0, 0],
                    }
                })

            self.render()
            # RTF=1.0 限速
            remaining = cycle_target - (time.perf_counter() - cycle_start)
            if remaining > 0:
                time.sleep(remaining)

        if bound:
            # 阶段结束：验证位移是否到位
            end_pose = self.get_body_xpos_xmat_xquat([pelvis_name])[pelvis_name]
            end_pos = end_pose["xpos"]
            actual_delta = end_pos - start_pos
            verifier.check_allclose(
                f"{phase_name}_displacement_c{self._cycle_count}",
                actual_delta,
                delta,
                atol=_ANCHOR_ATOL,
                detail=f"{phase_name} 位移到位（atol={_ANCHOR_ATOL}）",
            )

    def _check_state_before_pause(self, verifier: OnlineVerifier) -> None:
        """暂停前读取状态做检查（pelvis 位姿 + 绑定状态）。"""
        self._check_count += 1
        agent = self.agent_name
        pelvis_name = f"{agent}_pelvis"
        pelvis = self.get_body_xpos_xmat_xquat([pelvis_name])[pelvis_name]
        verifier.check(
            f"pelvis_pose_finite_{self._check_count}",
            bool(np.all(np.isfinite(pelvis["xpos"]))),
            pelvis["xpos"].tolist(),
            "finite",
            f"暂停前 pelvis 位姿有限（第 {self._check_count} 次检查）",
        )

    def _ensure_released(self, verifier: OnlineVerifier) -> None:
        """确保已释放绑定（若当前已绑定则释放）。"""
        if self.anchored_actor is not None:
            self.release_body_anchored()
            verifier.observe(
                "release_anchor",
                "Studio 视口：释放绑定，G1 恢复物理仿真与行走",
            )

    def _prompt_user_choice(self) -> str:
        """显示交互菜单，等待用户输入 1/2/3/4/5。

        Returns:
            用户选择的字符（"1"-"5"）。
        """
        print("\n" + "=" * 60)
        print("请选择操作（输入数字键）：")
        print("  1: 绑定 mocap，提升 0.5 米")
        print("  2: 绑定 mocap，向前移动 1 米")
        print("  3: 绑定 mocap，向左移动 1 米")
        print("  4: 取消绑定，机器人继续自主向前移动 3 秒")
        print("  5: 结束退出示例程序")
        print("=" * 60)
        sys.stdout.flush()

        if not sys.stdin.isatty():
            # 非交互终端：默认退出
            print("[非交互终端，自动选择 5 退出]")
            return "5"

        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                rlist, _, _ = select.select([fd], [], [], 1.0 / 30.0)
                if rlist:
                    ch = os.read(fd, 1).decode(errors="ignore")
                    if ch in ("1", "2", "3", "4", "5"):
                        sys.stdout.write("\r\n")
                        sys.stdout.flush()
                        print(f"已选择: {ch}")
                        return ch
                # 等待期间持续 render 刷新视口
                self.render()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
