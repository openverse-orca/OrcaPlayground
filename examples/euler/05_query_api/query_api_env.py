"""QueryApiEnv — Lesson 5 状态查询验证环境（阶段四在线验证）。

验证 G1 全套状态查询 API 在线运行正确，覆盖：
- 关节状态查询：query_joint_qpos / query_joint_qvel / query_joint_qacc
- Body 位姿查询：get_body_xpos_xmat_xquat
- 传感器查询：query_sensor_data
- 执行器力矩查询：query_actuator_torques
- 接触查询：query_contact_simple
- 质量查询：body_subtree_mass
- 基座坐标系变换：query_position_body_B

验证逻辑分两阶段：
- step 0（初始直立）：集中验证 9 项查询 API，并记录初始 pelvis 高度
- step 50（瘫倒验证）：零控下力控 motor 无法保持站立，G1 瘫倒，
  pelvis/torso 高度显著下降

架构合规：
- 继承 G1BaseEnv，仅重写 verify_step / observe_step 钩子
- 状态访问全部通过 env.data / env.query_* / env.get_body_xpos_xmat_xquat 公共 API
- 不触 _gym / _stub / _channel 等私有属性
- 参见 docs/design/development/orca_gym_euler_phase4_online_validation_development.md §4.3.1
"""

from __future__ import annotations

import numpy as np
from g1_base_env import (
    G1_ACTUATOR_SUFFIXES,
    G1_ROT_JOINT_SUFFIXES,
    G1BaseEnv,
)
from online_verifier import OnlineVerifier

# step 0 验证初始直立，step 50 验证瘫倒（零控下力控 motor 无法保持站立）
_COLLAPSE_CHECK_STEP = 50


class QueryApiEnv(G1BaseEnv):
    """Lesson 5 状态查询验证环境。

    step 0 集中执行 9 项查询验证（初始直立状态：关节维度/qpos 一致性/pelvis 高度/
    imu 维度/torso 质量/torso 相对 pelvis 位置/执行器力矩/site 查询/接触查询），
    并记录初始 pelvis 高度。
    step 50 验证瘫倒：零控下 G1 瘫倒，pelvis 高度较初始显著下降。
    """

    def verify_step(self, step: int, verifier: OnlineVerifier) -> None:
        """每控制周期数值判定。"""
        agent = self.agent_name

        if step == 0:
            self._verify_initial_upright(step, verifier, agent)
        elif step == _COLLAPSE_CHECK_STEP:
            self._verify_collapse(step, verifier, agent)

    def _verify_initial_upright(
        self, step: int, verifier: OnlineVerifier, agent: str
    ) -> None:
        """step 0：验证初始直立状态（9 项查询 API）。"""
        # 1. 关节 qpos 维度（29 个 hinge joint，每个长度 1）
        joint_names = [f"{agent}_{s}" for s in G1_ROT_JOINT_SUFFIXES]
        qpos = self.query_joint_qpos(joint_names)
        verifier.check(
            "joint_qpos_dim",
            len(qpos) == 29,
            actual=len(qpos),
            expected=29,
            detail="29 个 hinge joint qpos",
        )

        # 2. qpos 与 env.data.qpos 切片一致（按各关节 qpos 地址逐段拼接比较）
        # 场景含多 body（Toys/Manipulator），不可用 data.qpos[7:] 整段比较；
        # 通过 jnt_qposadr 取每个关节的 qpos 起始地址，构造期望值逐段比对。
        expected_segs = [
            self.data.qpos[self.jnt_qposadr(j): self.jnt_qposadr(j) + 1]
            for j in joint_names
        ]
        expected = np.concatenate(expected_segs)
        qpos_arr = np.concatenate([qpos[j] for j in joint_names])
        verifier.check_allclose(
            "joint_qpos_vs_data",
            qpos_arr,
            expected,
            atol=1e-9,
            detail="query_joint_qpos 与 data.qpos 按关节地址切片一致",
        )

        # 3. pelvis 初始高度（keyframe 站立姿态，记录供瘫倒验证比较）
        pelvis = self.get_body_xpos_xmat_xquat([f"{agent}_pelvis"])
        pelvis_z = float(pelvis[f"{agent}_pelvis"]["xpos"][2])
        self._initial_pelvis_z = pelvis_z
        # 同时记录 torso 初始高度，供瘫倒验证比较
        torso_init = self.get_body_xpos_xmat_xquat([f"{agent}_torso_link"])
        self._initial_torso_z = float(torso_init[f"{agent}_torso_link"]["xpos"][2])
        verifier.check_range(
            "pelvis_initial_height",
            pelvis_z,
            0.70,
            0.95,
            "G1 站立初始高度（keyframe 站立姿态）",
        )

        # 4. IMU sensor 维度（imu_quat 长度 4）
        imu_quat = self.query_sensor_data([f"{agent}_imu_quat"])
        imu_dim = len(imu_quat[f"{agent}_imu_quat"])
        verifier.check(
            "imu_quat_dim",
            imu_dim == 4,
            actual=imu_dim,
            expected=4,
            detail="imu_quat sensor 维度",
        )

        # 5. body_subtree_mass 为正（torso 子树质量）
        torso_mass = self.body_subtree_mass(f"{agent}_torso_link")
        verifier.check(
            "torso_subtree_mass_positive",
            torso_mass > 0,
            actual=float(torso_mass),
            expected=">0",
            detail="torso 子树质量为正",
        )

        # 6. 基座坐标系变换：torso 相对 pelvis 的位置（z 分量为正，躯干在骨盆上方）
        torso_B = self.query_position_body_B(
            f"{agent}_torso_link", f"{agent}_pelvis"
        )
        verifier.check_range(
            "torso_rel_pelvis_z",
            float(torso_B[2]),
            0.0,
            0.2,
            "躯干在骨盆上方（基座系 z）",
        )

        # 7. 执行器力矩查询（29 个 motor）
        actuator_names = [f"{agent}_{s}" for s in G1_ACTUATOR_SUFFIXES]
        torques = self.query_actuator_torques(actuator_names)
        verifier.check(
            "actuator_torque_dim",
            len(torques) == 29,
            actual=len(torques),
            expected=29,
            detail="29 个 motor 力矩",
        )

        # 8. Site 查询（imu site xpos/xmat 维度）
        imu_site = self.query_site_pos_and_mat([f"{agent}_imu"])
        site_pos = imu_site[f"{agent}_imu"]["xpos"]
        verifier.check(
            "site_pos_dim",
            len(site_pos) == 3,
            actual=len(site_pos),
            expected=3,
            detail="imu site xpos 维度",
        )

        # 9. 接触查询（G1 站立时双脚与地面应有接触，contact 数 ≥ 1）
        contacts = self.query_contact_simple()
        verifier.check(
            "contact_count",
            len(contacts) >= 1,
            actual=len(contacts),
            expected=">=1",
            detail="G1 站立时与地面接触数",
        )

    def _verify_collapse(
        self, step: int, verifier: OnlineVerifier, agent: str
    ) -> None:
        """step 50：验证零控下 G1 瘫倒（pelvis 高度显著下降）。

        G1 采用力控 motor 执行器，ctrl=0 时关节无力矩输出，重力作用下机器人
        瘫倒在地。验证 pelvis 高度较初始显著下降（drop > 0.1m），
        且 torso 高度也同步下降。
        """
        pelvis = self.get_body_xpos_xmat_xquat([f"{agent}_pelvis"])
        current_pelvis_z = float(pelvis[f"{agent}_pelvis"]["xpos"][2])
        initial_z = getattr(self, "_initial_pelvis_z", current_pelvis_z)
        drop = initial_z - current_pelvis_z

        verifier.check(
            "g1_collapsed_pelvis_drop",
            drop > 0.1,
            actual=f"drop={drop:.3f}m (initial={initial_z:.3f}, now={current_pelvis_z:.3f})",
            expected="drop > 0.1m",
            detail="零控下 G1 瘫倒：pelvis 高度较初始显著下降",
        )

        # torso 高度也应下降（躯干随骨盆一起下落）
        torso = self.get_body_xpos_xmat_xquat([f"{agent}_torso_link"])
        current_torso_z = float(torso[f"{agent}_torso_link"]["xpos"][2])
        initial_torso_z = getattr(self, "_initial_torso_z", current_torso_z)
        torso_drop = initial_torso_z - current_torso_z
        verifier.check(
            "g1_collapsed_torso_drop",
            torso_drop > 0.05,
            actual=f"drop={torso_drop:.3f}m (initial={initial_torso_z:.3f}, now={current_torso_z:.3f})",
            expected="drop > 0.05m",
            detail="零控下 G1 瘫倒：torso 高度较初始下降",
        )

    def observe_step(self, step: int, verifier: OnlineVerifier) -> None:
        """阶段性人工观察提示。"""
        if step == 0:
            verifier.observe(
                "g1_standing",
                "Studio 视口：G1 初始应站立在地面上，双臂自然下垂",
                step=step,
            )
        elif step == _COLLAPSE_CHECK_STEP:
            verifier.observe(
                "g1_collapsed",
                "Studio 视口：G1 应已瘫倒在地（零控下力控 motor 无法保持站立）",
                step=step,
            )
