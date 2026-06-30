"""ForceApplyEnv — Lesson 5 外力应用与状态设置验证环境（阶段四在线验证）。

验证 G1 外力应用与状态设置 R 类 API 在线运行正确，覆盖：
- apply_body_force / clear_body_force / clear_all_forces（外力应用与清除）
- set_geom_friction（geom 摩擦系数设置）
- query_contact_force（接触力查询）
- set_mocap_pos_and_quat（mocap 位姿写入，经 weld 约束驱动 manipulation_box）

验证逻辑分阶段（run_lesson 框架自动步进，零控下 G1 会瘫倒）：
- step 0（初始直立）：接触力查询验证（G1 站立触地，法向力显著为正）
- step 10（施力前）：记录 pelvis z，对 torso 施加 200N 向上力
- step 30（施力后）：pelvis z 应上升、xfrc_applied 记录力值，随后清力
- step 35（清力后）：xfrc_applied 归零
- step 50：clear_all_forces 全清 + set_geom_friction 烟雾测试
- step 70：set_mocap_pos_and_quat 写入 + 回读一致
- step 90：weld 约束驱动 manipulation_box 跟随 mocap 目标位置

架构合规：
- 继承 G1BaseEnv，仅重写 verify_step / observe_step 钩子
- 状态访问全部通过 env.data / env.apply_body_force / env.clear_body_force /
  env.clear_all_forces / env.set_geom_friction / env.query_contact_simple /
  env.query_contact_force / env.set_mocap_pos_and_quat / env.get_body_xpos_xmat_xquat
  公共 API
- 不触 _gym / _stub / _channel 等私有属性
- 参见 docs/design/development/orca_gym_euler_phase4_online_validation_development.md §4.3.2
"""

from __future__ import annotations

import numpy as np

from envs.euler.g1_base_env import G1BaseEnv
from envs.euler.online_verifier import OnlineVerifier

# 施加的向上力（N）。G1 整机质量 ~35kg（重 ~343N），零控下 pelvis 自由下落，
# 500N 净向上 ~157N，足以在 20 控制周期（0.4s）内克服下落动量并使 pelvis 上升。
# 力施加在 pelvis 上（而非 torso_link）：零控下腰部关节无力矩，torso 施力难以
# 经由松弛关节传递到 pelvis，直接对 pelvis 施力可可靠验证 apply_body_force。
_LIFT_FORCE_N = 500.0
_LIFT_BODY = "pelvis"  # 施力目标 body（与测量 body 一致，确保力直接驱动被测部位）

# mocap 目标位姿（manipulation_box 经 weld 跟随到此位置）
_MOCAP_TARGET_POS = np.array([0.7, 0.0, 0.5])
_MOCAP_TARGET_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


class ForceApplyEnv(G1BaseEnv):
    """Lesson 5 外力应用与状态设置验证环境。

    step 0  验证 query_contact_force（G1 站立触地，法向力显著为正）；
    step 10 记录 pelvis z 并对 pelvis 施加向上力；
    step 30 验证 pelvis 上升 + xfrc 记录，随后 clear_body_force；
    step 35 验证 xfrc 清零；
    step 50 验证 clear_all_forces 全清 + set_geom_friction 烟雾测试；
    step 70 验证 set_mocap_pos_and_quat 写入回读一致；
    step 90 验证 weld 约束驱动 manipulation_box 跟随 mocap。
    """

    def verify_step(self, step: int, verifier: OnlineVerifier) -> None:
        """每控制周期数值判定。"""
        agent = self.agent_name

        if step == 0:
            self._verify_contact_force(step, verifier, agent)
        elif step == 10:
            self._apply_lift_force(step, verifier, agent)
        elif step == 30:
            self._verify_lift_and_clear(step, verifier, agent)
        elif step == 35:
            self._verify_force_cleared(step, verifier, agent)
        elif step == 50:
            self._verify_clear_all_and_friction(step, verifier, agent)
        elif step == 70:
            self._verify_mocap_writeback(step, verifier, agent)
        elif step == 90:
            self._verify_mocap_drives_box(step, verifier, agent)

    # --- 阶段 1：接触力查询（step 0，G1 直立足部触地）---

    def _verify_contact_force(
        self, step: int, verifier: OnlineVerifier, agent: str
    ) -> None:
        """step 0：query_contact_force 返回显著法向力（G1 站立触地）。"""
        contacts = self.query_contact_simple()
        verifier.check(
            "contact_exists",
            len(contacts) >= 1,
            actual=len(contacts),
            expected=">=1",
            detail="G1 站立时与地面有接触",
        )
        if not contacts:
            return
        contact_ids = list(range(len(contacts)))
        forces = self.query_contact_force(contact_ids)
        # 接触力前 3 分量在接触坐标系下，第 1 分量为法向力
        max_normal = max(abs(f[0]) for f in forces.values())
        verifier.check(
            "contact_normal_force",
            max_normal > 50.0,
            actual=f"{max_normal:.1f}N",
            expected="> 50N",
            detail="query_contact_force 返回显著法向力（G1 足部触地）",
        )

    # --- 阶段 2：施加外力（step 10）---

    def _apply_lift_force(
        self, step: int, verifier: OnlineVerifier, agent: str
    ) -> None:
        """step 10：记录 pelvis z，对 pelvis 施加向上力。"""
        pelvis = self.get_body_xpos_xmat_xquat([f"{agent}_pelvis"])
        self._z_before = float(pelvis[f"{agent}_pelvis"]["xpos"][2])
        self.apply_body_force(
            f"{agent}_{_LIFT_BODY}",
            np.array([0.0, 0.0, _LIFT_FORCE_N]),
            np.array([0.0, 0.0, 0.0]),
        )

    # --- 阶段 3：验证抬起 + xfrc 记录 + 清力（step 30）---

    def _verify_lift_and_clear(
        self, step: int, verifier: OnlineVerifier, agent: str
    ) -> None:
        """step 30：pelvis z 上升、xfrc 记录力值，随后 clear_body_force。"""
        pelvis = self.get_body_xpos_xmat_xquat([f"{agent}_pelvis"])
        z_after = float(pelvis[f"{agent}_pelvis"]["xpos"][2])
        z_before = getattr(self, "_z_before", z_after)
        verifier.check(
            "force_lift_pelvis",
            z_after > z_before + 0.01,
            actual=f"{z_after:.4f} (before={z_before:.4f})",
            expected=f">{z_before + 0.01:.4f}",
            detail="施力后 pelvis 上升 > 1cm",
        )

        # xfrc_applied 可读到力值（DataView 只读视图，按 body_id 索引）
        body_id = self.model.body_name2id(f"{agent}_{_LIFT_BODY}")
        xfrc = self.data.xfrc_applied[body_id, :3]
        verifier.check(
            "xfrc_recorded",
            bool(np.any(xfrc != 0)),
            actual=xfrc.tolist(),
            expected="non-zero",
            detail="xfrc_applied 记录了施加的力",
        )

        # 清力
        self.clear_body_force(f"{agent}_{_LIFT_BODY}")

    # --- 阶段 4：验证清力（step 35）---

    def _verify_force_cleared(
        self, step: int, verifier: OnlineVerifier, agent: str
    ) -> None:
        """step 35：clear_body_force 后 xfrc 归零。"""
        body_id = self.model.body_name2id(f"{agent}_{_LIFT_BODY}")
        xfrc = self.data.xfrc_applied[body_id, :3]
        verifier.check(
            "xfrc_cleared",
            bool(np.all(xfrc == 0)),
            actual=xfrc.tolist(),
            expected="zeros",
            detail="clear_body_force 后 xfrc 归零",
        )

    # --- 阶段 5：clear_all_forces + set_geom_friction（step 50）---

    def _verify_clear_all_and_friction(
        self, step: int, verifier: OnlineVerifier, agent: str
    ) -> None:
        """step 50：clear_all_forces 全清 + set_geom_friction 烟雾测试。"""
        self.clear_all_forces()
        all_zero = bool(np.all(self.data.xfrc_applied == 0))
        verifier.check(
            "clear_all_forces",
            all_zero,
            actual="all zero" if all_zero else "non-zero remains",
            expected="all zero",
            detail="clear_all_forces 清除全部 body 外力",
        )

        # set_geom_friction 烟雾测试：取一个 G1 geom 设置摩擦系数
        geom_dict = self.model.get_geom_dict()
        g1_geom = next(
            (name for name in geom_dict if name.startswith(f"{agent}_")),
            None,
        )
        if g1_geom is not None:
            self.set_geom_friction(
                {g1_geom: np.array([0.8, 0.005, 0.0001])}
            )
            verifier.check(
                "set_geom_friction_ok",
                True,
                actual=g1_geom,
                expected="no error",
                detail="set_geom_friction 调用成功（写入 geom_friction）",
            )
        else:
            verifier.check(
                "set_geom_friction_ok",
                False,
                actual="no g1 geom found",
                expected="g1 geom",
                detail="set_geom_friction 烟雾测试：未找到 G1 geom",
            )

    # --- 阶段 6：mocap 写入回读（step 70）---

    def _verify_mocap_writeback(
        self, step: int, verifier: OnlineVerifier, agent: str
    ) -> None:
        """step 70：set_mocap_pos_and_quat 写入后回读一致。"""
        anchor = f"{agent}_ActorManipulator_Anchor"
        self.set_mocap_pos_and_quat(
            {anchor: {"pos": _MOCAP_TARGET_POS, "quat": _MOCAP_TARGET_QUAT}}
        )
        read_pos = self.data.mocap_pos(anchor)
        read_quat = self.data.mocap_quat(anchor)
        verifier.check_allclose(
            "mocap_pos_readback",
            read_pos,
            _MOCAP_TARGET_POS,
            atol=1e-6,
            detail="set_mocap_pos_and_quat 写入位置后回读一致",
        )
        verifier.check_allclose(
            "mocap_quat_readback",
            read_quat,
            _MOCAP_TARGET_QUAT,
            atol=1e-6,
            detail="set_mocap_pos_and_quat 写入四元数后回读一致",
        )
        # 记录供 step 90 box 跟随验证使用
        self._mocap_box_name = f"{agent}_manipulation_box"

    # --- 阶段 7：weld 驱动 box 跟随（step 90）---

    def _verify_mocap_drives_box(
        self, step: int, verifier: OnlineVerifier, agent: str
    ) -> None:
        """step 90：weld 约束驱动 manipulation_box 跟随 mocap 目标位置。"""
        box_name = getattr(self, "_mocap_box_name", f"{agent}_manipulation_box")
        box = self.get_body_xpos_xmat_xquat([box_name])
        box_pos = box[box_name]["xpos"]
        verifier.check_allclose(
            "mocap_drives_box_via_weld",
            np.asarray(box_pos),
            _MOCAP_TARGET_POS,
            atol=0.05,
            detail="weld 约束驱动 manipulation_box 跟随 mocap 目标位置",
        )

    def observe_step(self, step: int, verifier: OnlineVerifier) -> None:
        """阶段性人工观察提示。"""
        if step == 10:
            verifier.observe(
                "force_applied",
                f"Studio 视口：G1 应被向上抬起（pelvis 施加 {_LIFT_FORCE_N:.0f}N 向上力）",
                step=step,
            )
        elif step == 30:
            verifier.observe(
                "force_cleared",
                "Studio 视口：清力后 G1 应自由落体回落",
                step=step,
            )
        elif step == 70:
            verifier.observe(
                "mocap_box_follow",
                "Studio 视口：manipulation_box 应跟随 mocap 移动到目标位置 [0.7, 0, 0.5]",
                step=step,
            )
