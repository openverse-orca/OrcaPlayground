"""BodyManipulationEnv — Lesson 10：体操作与交互式驱动（锚定/mocap 驱动/equality）。

交互式菜单驱动模式：机器人在无绑定状态下向前行走 3 秒，然后暂停询问用户选择
操作（1/2/3/4/5），根据选择绑定 mocap 并在 3 秒内周期性移动机器人，或取消
绑定继续自主行走，或退出。每次暂停前读取 pelvis 位姿等数据做检查。

菜单选项:
    1: 绑定 mocap，提升 0.5 米（z+0.5）
    2: 绑定 mocap，向前移动 1 米（x+1）
    3: 绑定 mocap，向左移动 1 米（y+1）
    4: 取消绑定，机器人继续自主向前移动 3 秒
    5: 结束退出示例程序

验证 API（体操作层，公共原语，消费者自管编排）:
    - equality_find_slot_by_body / equality_constraint：定位槽位 + 保存快照
    - equality_update：写入/恢复约束（绑定/释放均走此原语）
    - set_mocap_pos_and_quat：对齐 mocap 位姿 + 周期性驱动 mocap body
    - get_body_xpos_xmat_xquat：读取位姿

> **前置依赖**：本课依赖 Lesson 8 行走控制已验证（复用 ``g1_locomotion.py`` 驱动行走）。
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
from common.g1_locomotion import G1Locomotion

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
    """Lesson 10 Env 子类：交互式体操作验证。

    重写钩子:
        - initialize_simulation: 创建 G1Locomotion + 动态解析 mocap body 名
        - compute_ctrl: ONNX 推理 → q_target
        - _pd_controller: 闭环 PD 单步（架构 §6.4 S6）
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
        # 程序化绑定状态（自管快照，不污染 UI 抓取的 _anchor_* 字段）
        self._bound_slot: int | None = None
        self._original_eq_snapshot: dict | None = None

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
            "Lesson 10 交互式体操作演示结束",
        )
        return verifier.report()

    def _bind_mocap_to_pelvis(self, pelvis_name: str) -> None:
        """程序化绑定：把自备 mocap 绑定到 pelvis（仿照 _anchor_actor 编排模式）。

        使用公共无状态原语组合，业务状态自管（_bound_slot / _original_eq_snapshot）。
        不依赖已删除的 equality_bind_mocap，不触碰 UI 抓取的 _anchor_* 字段。

        编排:
            1. 幂等保护：已绑定时不重复绑定，避免覆盖快照（§4.5 bug 修正）
            2. equality_find_slot_by_body 查找含 mocap 的槽位
            3. equality_constraint 保存原始约束快照（释放时恢复）
            4. set_mocap_pos_and_quat 对齐 mocap 位姿到 pelvis（避免下一帧拉扯）
            5. equality_update 写入 weld 约束（内部 mj_forward）

        Args:
            pelvis_name: pelvis 完整名（含 agent 前缀，如 "g1_pelvis"）。
        """
        import mujoco

        # 幂等保护：已绑定时不重复绑定，避免覆盖快照
        if self._bound_slot is not None:
            return

        mocap_name = self._mocap_body_name
        # 1. 查找含自备 mocap 的槽位（公共原语）
        slot = self.equality_find_slot_by_body(mocap_name)
        if slot == -1:
            raise ValueError(
                f"模型中无含 {mocap_name} 的 equality 槽位，"
                f"请检查关卡 XML"
            )
        # 2. 保存原始约束快照（消费者自管业务状态）
        self._original_eq_snapshot = self.equality_constraint(slot)
        self._bound_slot = slot
        # 3. 对齐 mocap 位姿到 pelvis 当前位姿（避免下一帧拉扯）
        mocap_id = self.model.body_name2id(mocap_name)
        pelvis_pose = self.get_body_xpos_xmat_xquat([pelvis_name])[pelvis_name]
        self.set_mocap_pos_and_quat({
            mocap_name: {
                "pos": pelvis_pose["xpos"],
                "quat": pelvis_pose["xquat"],
            }
        })
        # 4. 确定改 obj1 还是 obj2（mocap 一端保持，另一端改为 pelvis）
        if self._original_eq_snapshot["obj1_id"] == mocap_id:
            new_obj1_name = mocap_name
            new_obj2_name = pelvis_name
        else:
            new_obj1_name = pelvis_name
            new_obj2_name = mocap_name
        # 5. 写入约束（公共原语，内部 mj_forward）
        self.equality_update(
            slot,
            eq_type=mujoco.mjtEq.mjEQ_WELD,
            obj1_name=new_obj1_name,
            obj2_name=new_obj2_name,
        )

    def _release_mocap(self) -> None:
        """程序化释放：从快照恢复原始约束（仿照 _release_body_anchored 编排模式）。

        不依赖已删除的 equality_release，不穿墙 self._gym.update_equality_constraints。
        用 equality_update 从快照恢复原始约束（id→name 反查），内部 mj_forward。
        未绑定时 no-op。
        """
        if self._bound_slot is None or self._original_eq_snapshot is None:
            return
        # 用快照恢复原始约束（公共原语，内部 mj_forward）
        self.equality_update(
            self._bound_slot,
            eq_type=self._original_eq_snapshot["type"],
            obj1_name=self.model.body_id2name(self._original_eq_snapshot["obj1_id"]),
            obj2_name=self.model.body_id2name(self._original_eq_snapshot["obj2_id"]),
            data=self._original_eq_snapshot["data"],
        )
        self._bound_slot = None
        self._original_eq_snapshot = None

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
            # 记录起始位姿（用于周期性移动插值与位移校验）
            start_pose = self.get_body_xpos_xmat_xquat([pelvis_name])[pelvis_name]
            start_pos = start_pose["xpos"].copy()
            # 程序化绑定 pelvis（weld）：消费者自管编排，使用公共原语
            # （仿照 _anchor_actor 编排模式，不走已删除的 equality_bind_mocap）
            self._bind_mocap_to_pelvis(pelvis_name)
            # 绑定后停止行走（stand=0），机器人跟随 mocap
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
            if bound and delta is not None:
                # 绑定阶段：零力矩，让 WELD 约束无对抗地拖动机器人
                action = np.zeros(self.model.nu)
                self.step(action)
            else:
                action = self.compute_ctrl(step)
                self.step(action)

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
        """确保已释放绑定（若当前已绑定则释放）。

        使用 _release_mocap 从快照恢复原始约束（公共原语 equality_update），
        不依赖已删除的 equality_release，不触碰 UI 抓取的 _anchor_* 字段。
        """
        was_bound = self._bound_slot is not None
        self._release_mocap()
        if was_bound:
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

        if sys.platform == "win32":
            # Windows: msvcrt 非阻塞按键检测
            import msvcrt

            while True:
                if msvcrt.kbhit():
                    ch = msvcrt.getch().decode(errors="ignore")
                    if ch in ("1", "2", "3", "4", "5"):
                        sys.stdout.write("\r\n")
                        sys.stdout.flush()
                        print(f"已选择: {ch}")
                        return ch
                # 等待期间持续 render 刷新视口
                self.render()
                time.sleep(0.01)
        else:
            # Unix: raw 模式 + select 非阻塞轮询
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
