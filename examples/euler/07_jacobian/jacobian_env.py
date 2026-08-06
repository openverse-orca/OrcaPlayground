"""JacobianEnv — Lesson 7 雅可比与 IK 验证环境（阶段四在线验证）。

验证 G1 雅可比计算与 IK 迭代 API 在线运行正确，覆盖：
- mj_jacBody（body 平移/旋转雅可比）
- mj_jacSite（site 雅可比）
- query_site_xvalp_xvalr（site 速度查询，与 jacp @ qvel 一致性）
- set_joint_qpos + mj_forward（合规状态写入 + 前向更新）
- 阻尼最小二乘 IK 迭代收敛

验证逻辑全部在 step 0 集中执行（run_lesson 框架在 verify_step 前已 step
一次，在线模式需经一次步进同步远端数据后 xpos/jacp 才非零）：
- pelvis 雅可比形状 (3, nv)，nv ≥ 35（G1 自身 6 free + 29 关节）
- imu site 速度 = jacp_site @ data.qvel（atol=1e-4）
- 阻尼最小二乘 IK 迭代 80 次，左脚到达目标位置（抬高约 10cm，atol=0.02）

架构合规：
- 继承 G1BaseEnv，仅重写 verify_step / observe_step 钩子
- 雅可比通过 env.mj_jacBody / env.mj_jacSite / env.query_site_xvalp_xvalr 公共 API
- 状态写入通过 env.set_joint_qpos（W1，复制 qpos → 修改 G1 关节段 → set_joint_qpos），
  不直接写 data.qpos；修改后调 env.mj_forward 更新派生量
- 不触 _gym / _stub / _channel 等私有属性
- 参见 docs/design/development/orca_gym_euler_phase4_online_validation_development.md §4.3.3

与开发文档代码片段的偏离（多 body 场景必需）：
- 形状检查用 self.model.nv（场景 83）而非硬编码 35；另校验 nv ≥ 35
- IK dof 列用 G1 关节 dofadr 范围 [v_min, v_max] 而非 [:, 7:]（场景 G1 关节起于 dof 48）
- IK qpos 更新按各关节 jnt_qposadr 逐个写入，而非 qpos[7:]（避免污染其他 body）
- IK 用阻尼最小二乘（λ=0.05）替代伪逆，80 次迭代替代 50 次，提升收敛稳健性
"""

from __future__ import annotations

import numpy as np
from g1_base_env import G1_ROT_JOINT_SUFFIXES, G1BaseEnv
from online_verifier import OnlineVerifier

# IK 参数：阻尼最小二乘 + 步长，经探测在 G1 多 body 场景下 80 次迭代收敛（err < 0.02）
_IK_DAMPING = 0.05
_IK_STEP = 0.05
_IK_ITERS = 80
# 左脚目标偏移：y +0.05m，z +0.10m（抬高约 10cm）
_IK_TARGET_OFFSET = np.array([0.0, 0.05, 0.10])
_IK_FOOT_SUFFIX = "left_ankle_roll_link"
_IK_ATOL = 0.02


class JacobianEnv(G1BaseEnv):
    """Lesson 7 雅可比与 IK 验证环境。

    step 0 集中执行 3 项验证：pelvis 雅可比形状、imu site 速度一致性、
    阻尼最小二乘 IK 迭代收敛（左脚抬高约 10cm 到达目标）。
    """

    def verify_step(self, step: int, verifier: OnlineVerifier) -> None:
        """每控制周期数值判定（全部在 step 0 集中验证）。"""
        if step != 0:
            return
        agent = self.agent_name
        self._verify_pelvis_jac_shape(step, verifier, agent)
        self._verify_imu_site_vel(step, verifier, agent)
        self._verify_ik_convergence(step, verifier, agent)

    # --- 验证 1：pelvis 雅可比形状 ---

    def _verify_pelvis_jac_shape(
        self, step: int, verifier: OnlineVerifier, agent: str
    ) -> None:
        """step 0：mj_jacBody 返回 (3, nv)，nv ≥ 35（G1 自身 6 free + 29 关节）。"""
        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        self.mj_jacBody(jacp, jacr, body_name=f"{agent}_pelvis")
        verifier.check(
            "jac_shape",
            jacp.shape == (3, nv) and nv >= 35,
            actual=f"{jacp.shape} (nv={nv})",
            expected="(3, nv) & nv>=35",
            detail="pelvis 雅可比形状 (3, nv)，nv ≥ 35（G1 6 free + 29 关节）",
        )
        # 记录 G1 关节 dof 列范围，供 IK 使用
        joint_names = [f"{agent}_{s}" for s in G1_ROT_JOINT_SUFFIXES]
        self._g1_dof_adrs = [self.jnt_dofadr(jn) for jn in joint_names]
        self._g1_qpos_adrs = [self.jnt_qposadr(jn) for jn in joint_names]
        self._g1_joint_names = joint_names

    # --- 验证 2：imu site 速度与 jacp @ qvel 一致 ---

    def _verify_imu_site_vel(
        self, step: int, verifier: OnlineVerifier, agent: str
    ) -> None:
        """step 0：query_site_xvalp_xvalr 与 jacp_site @ qvel 一致（atol=1e-4）。"""
        site_name = f"{agent}_imu"
        xvalp, _ = self.query_site_xvalp_xvalr([site_name])
        jacp_site = np.zeros((3, self.model.nv))
        jacr_site = np.zeros((3, self.model.nv))
        self.mj_jacSite(jacp_site, jacr_site, site_name=site_name)
        expected_vel = jacp_site @ self.data.qvel
        verifier.check_allclose(
            "site_vel_vs_jac",
            xvalp[site_name],
            expected_vel,
            atol=1e-4,
            detail="imu site 速度 = jacp_site @ data.qvel",
        )

    # --- 验证 3：阻尼最小二乘 IK 迭代收敛（两阶段）---

    def _verify_ik_convergence(
        self, step: int, verifier: OnlineVerifier, agent: str
    ) -> None:
        """step 0：阻尼最小二乘 + 关节限位 clamp 的 IK，左脚到达目标位置（atol=0.02）。

        分两阶段执行，每阶段后暂停等待用户观察：

        **阶段 1 — 预设微蹲姿态**：
        G1 默认 qpos=0 时膝盖完全伸直，纯 DLS 会朝「后弯」（限位负方向）走以抬高
        脚部——这是反关节路径。预设膝盖前弯 + 髋前屈 + 踝背屈（补偿使脚底水平）
        后，IK 从已弯曲状态继续正向弯曲抬脚，路径自然且在限位内收敛。

        **阶段 2 — IK 抬左脚**：
        每次 IK 迭代后将关节角 clamp 到 ``jnt_range``（公共 API
        ``model.get_joint_dict()`` 返回的 Range/Limited 字段），强制遵守限位。
        dq = J⁺·Δx  （DLS 伪逆，λ 阻尼）
        q ← clamp(q + dq·step, jnt_range)
        """
        foot_body = f"{agent}_{_IK_FOOT_SUFFIX}"

        # G1 关节 dof 列范围（验证 1 已记录）
        v_adrs = self._g1_dof_adrs
        v_min, v_max = min(v_adrs), max(v_adrs)
        g1_joint_cols = slice(v_min, v_max + 1)

        # 读取 G1 各关节限位（jnt_range），用于每次迭代后 clamp
        jdict = self.model.get_joint_dict()
        jnt_lo = np.array(
            [
                jdict[jn]["Range"][0] if jdict[jn]["Limited"] else -np.inf
                for jn in self._g1_joint_names
            ]
        )
        jnt_hi = np.array(
            [
                jdict[jn]["Range"][1] if jdict[jn]["Limited"] else np.inf
                for jn in self._g1_joint_names
            ]
        )

        # === 阶段 1：预设微蹲姿态 ===
        # 膝盖前弯 +0.6 rad（≈34°）、髋前屈 -0.3 rad、踝背屈 -0.3 rad 补偿
        # （踝背屈使脚底法向 z=1.0，完全水平贴地）。双腿对称。
        preset = {
            f"{agent}_left_knee_joint": 0.6,
            f"{agent}_left_hip_pitch_joint": -0.3,
            f"{agent}_left_ankle_pitch_joint": -0.3,
            f"{agent}_right_knee_joint": 0.6,
            f"{agent}_right_hip_pitch_joint": -0.3,
            f"{agent}_right_ankle_pitch_joint": -0.3,
        }
        qpos_preset = self.data.qpos.copy()
        for jn, val in preset.items():
            qpos_preset[self.jnt_qposadr(jn)] = val
        self.set_joint_qpos(qpos_preset)
        self.mj_forward()
        verifier.observe(
            "preset_squat",
            "Studio 视口：G1 进入微蹲姿态（双膝前弯，双脚底水平贴地）。"
            "此为 IK 起始姿态——从已弯曲状态出发，IK 可沿自然方向继续弯曲抬脚，"
            "避免从伸直状态走反关节路径。",
            step=step,
        )
        # 暂停阶段 1：让用户观察微蹲姿态后再继续 IK
        self.wait_for_keypress("阶段 1/2：已预设微蹲姿态（双脚底水平），请观察")

        # === 阶段 2：IK 抬左脚 ===
        # 重新读取脚部初始位置（预设姿态后脚位已变）
        foot_pos = self.get_body_xpos_xmat_xquat([foot_body])[foot_body]["xpos"]
        target = foot_pos + _IK_TARGET_OFFSET

        jacr = np.zeros((3, self.model.nv))
        for _ in range(_IK_ITERS):
            jacp_foot = np.zeros((3, self.model.nv))
            self.mj_jacBody(jacp_foot, jacr, body_name=foot_body)
            cur = self.get_body_xpos_xmat_xquat([foot_body])[foot_body]["xpos"]
            delta = target - cur
            jac_leg = jacp_foot[:, g1_joint_cols]
            # 阻尼最小二乘：dq = J^T (J J^T + λ²I)^-1 delta
            dq = jac_leg.T @ np.linalg.inv(
                jac_leg @ jac_leg.T + _IK_DAMPING**2 * np.eye(3)
            ) @ delta
            # 合规写入：复制 qpos → 仅改 G1 关节段 → clamp 到限位 → set_joint_qpos（W1）
            qpos = self.data.qpos.copy()
            for j, qadr in enumerate(self._g1_qpos_adrs):
                qpos[qadr] = np.clip(qpos[qadr] + dq[j] * _IK_STEP, jnt_lo[j], jnt_hi[j])
            self.set_joint_qpos(qpos)
            self.mj_forward()

        final = self.get_body_xpos_xmat_xquat([foot_body])[foot_body]["xpos"]
        verifier.check_allclose(
            "ik_foot_target",
            final,
            target,
            atol=_IK_ATOL,
            detail=f"IK 迭代 {_IK_ITERS} 次后左脚到达目标位置（抬高约 10cm）",
        )
        verifier.observe(
            "ik_foot_movement",
            "Studio 视口：左脚应自然前弯抬高约 10cm（膝盖正向弯曲，非反关节）。",
            step=step,
        )
        # 暂停阶段 2：让用户观察 IK 抬脚姿态后再继续循环
        self.wait_for_keypress("阶段 2/2：IK 已抬起左脚，请观察")
