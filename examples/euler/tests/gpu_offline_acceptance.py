"""OrcaPlayground Euler GPU 离线验收脚本（Phase D，需 CUDA）。

在线（gRPC + 渲染）可视化验收由用户手工执行；本脚本用 **GPU 后端 + 离线模型**
做确定性的数值与接线验收，覆盖 Phase D 的 4 个离线可验证点：

1. **偏移解析**：G1 首驱动关节 qpos 偏移 == 7、dof 偏移 == 6，且 29 个被驱动
   关节在 qpos/dof 上连续（验证 ``jnt_qposadr``/``jnt_dofadr`` 运行时解析结果，
   而非硬编码 7/6）。
2. **pd_kernel 数值一致性**：device 侧 ``pd_kernel`` 输出的 tau 与 host numpy
   参考公式逐元素一致（含限幅、以及按偏移读取正确槽位）。
3. **register_pid_controller 接线**：GPU 上注册 + update_target + step 推进后
   qpos/qacc 有限，且「PD 开启(kp>0)」与「PD 关闭(kp=0)」产生不同 qacc，
   证明前步 kernel 确实在 step 中被执行（模式 3 生效）。
4. **RTF 冒烟**：以模式 3 粒度（每控制周期 update_target + step(20)）连跑并计时，
   断言无 NaN 且 RTF >= 1.0（完整渲染管线的 RTF 验收留待在线）。

运行方式（需 whitelist，直接解释器调起，勿加 shell 管道）:
    <conda-base>/envs/OrcaFlow_Flow/bin/python -m unittest \
        examples.euler.tests.gpu_offline_acceptance -v

无 CUDA 时（沙箱内 CPU 环境），GPU 用例自动 skip。
"""

from __future__ import annotations

import os
import time
import unittest

import mujoco
import numpy as np

_DEVICE = "cuda:0"

# 就地引用 assets 下的 G1 离线模型（与 common/g1_base_env.py 一致）
_EULER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_G1_XML = os.path.join(_EULER_ROOT, "assets", "g1", "g1_29dof_camera.xml")


def _load_g1_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(_G1_XML)


def _actuated_joint_names(model: mujoco.MjModel) -> list[str]:
    """返回按 actuator（ctrl）顺序排列的被驱动关节名列表（长度 nu）。

    pd_kernel 依赖不变式（见 orca_gym.core.euler.controller.pd_kernel docstring）：
    ctrl 的执行器顺序与被驱动关节顺序一致。
    """
    names: list[str] = []
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        assert name is not None
        names.append(name)
    return names


def _actuator_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        assert name is not None
        names.append(name)
    return names


def _dict_to_flat(result: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """把 ``query_joint_*`` 返回的 {joint_name: np.ndarray} 按 names 顺序拼接成 1-D。"""
    return np.concatenate([np.asarray(result[n], dtype=np.float64) for n in names])


class TestG1ModelOffsetResolution(unittest.TestCase):
    """偏移解析（CPU，离线）：验证「运行时解析」而非硬编码 7/6。"""

    def test_first_actuated_joint_offset_is_7_6(self):
        model = _load_g1_model()
        names = _actuated_joint_names(model)
        first = names[0]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, first)
        self.assertEqual(int(model.jnt_qposadr[jid]), 7)
        self.assertEqual(int(model.jnt_dofadr[jid]), 6)

    def test_actuated_joints_contiguous(self):
        model = _load_g1_model()
        names = _actuated_joint_names(model)
        self.assertEqual(len(names), model.nu)
        qadrs = [
            int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in names
        ]
        dadrs = [
            int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in names
        ]
        self.assertEqual(qadrs, list(range(qadrs[0], qadrs[0] + len(qadrs))))
        self.assertEqual(dadrs, list(range(dadrs[0], dadrs[0] + len(dadrs))))

    def test_actuators_are_joint_transmissions(self):
        """执行器为关节力矩传动（trntype==0），与 PD 力矩控制兼容。"""
        model = _load_g1_model()
        for trn in model.actuator_trntype:
            self.assertEqual(int(trn), int(mujoco.mjtTrn.mjTRN_JOINT))


def _cuda_available() -> bool:
    try:
        import orca.flow as flow

        return bool(flow.is_cuda_available())
    except Exception:
        return False


_HAS_CUDA = _cuda_available()


@unittest.skipUnless(_HAS_CUDA, "需要 CUDA")
class TestPdKernelNumerics(unittest.TestCase):
    """pd_kernel 数值一致性（GPU）：device tau vs host 参考公式。"""

    @classmethod
    def setUpClass(cls) -> None:
        import orca.flow as flow

        cls.flow = flow

    @staticmethod
    def _host_reference(
        q_target: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        motor_limit: np.ndarray,
        qpos_offset: int,
        qvel_offset: int,
    ) -> np.ndarray:
        nu = q_target.shape[0]
        tau = np.empty(nu, dtype=np.float64)
        for i in range(nu):
            raw = kp[i] * (q_target[i] - qpos[qpos_offset + i]) - kd[i] * qvel[qvel_offset + i]
            tau[i] = np.clip(raw, -motor_limit[i], motor_limit[i])
        return tau

    def _run_device_pd(
        self,
        q_target: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        motor_limit: np.ndarray,
        qpos_offset: int,
        qvel_offset: int,
    ) -> np.ndarray:
        from orca_gym.core.euler.controller import pd_kernel

        flow = self.flow
        nu = q_target.shape[0]
        with flow.ScopedDevice(_DEVICE):
            q_target_dev = flow.array(q_target.reshape(1, nu), dtype=flow.float32)
            qpos_dev = flow.array(qpos.reshape(1, -1), dtype=flow.float32)
            qvel_dev = flow.array(qvel.reshape(1, -1), dtype=flow.float32)
            kp_dev = flow.array(kp, dtype=flow.float32)
            kd_dev = flow.array(kd, dtype=flow.float32)
            limit_dev = flow.array(motor_limit, dtype=flow.float32)
            ctrl_dev = flow.zeros((1, nu), dtype=flow.float32)
            flow.launch(
                pd_kernel,
                dim=nu,
                inputs=[
                    q_target_dev,
                    qpos_dev,
                    qvel_dev,
                    kp_dev,
                    kd_dev,
                    limit_dev,
                    ctrl_dev,
                    qpos_offset,
                    qvel_offset,
                ],
                device=_DEVICE,
            )
            return np.asarray(ctrl_dev.numpy(), dtype=np.float64).reshape(nu)

    def test_g1_offsets_match_host(self):
        rng = np.random.default_rng(0)
        nu = 29  # G1 驱动关节数
        nq, nv = 43, 41
        q_target = rng.uniform(-0.5, 0.5, nu)
        qpos = rng.uniform(-0.5, 0.5, nq)
        qvel = rng.uniform(-1.0, 1.0, nv)
        kp = rng.uniform(10.0, 150.0, nu)
        kd = rng.uniform(0.1, 5.0, nu)
        motor_limit = rng.uniform(100.0, 500.0, nu)

        expected = self._host_reference(q_target, qpos, qvel, kp, kd, motor_limit, 7, 6)
        actual = self._run_device_pd(q_target, qpos, qvel, kp, kd, motor_limit, 7, 6)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_offset_resolution_generalizes(self):
        """非 G1 偏移（5, 4）也应被正确解析，证明偏移是运行时参数而非硬编码。"""
        rng = np.random.default_rng(1)
        nu = 4
        nq, nv = 12, 11
        qpos_offset, qvel_offset = 5, 4
        q_target = rng.uniform(-0.5, 0.5, nu)
        qpos = rng.uniform(-0.5, 0.5, nq)
        qvel = rng.uniform(-1.0, 1.0, nv)
        kp = rng.uniform(10.0, 150.0, nu)
        kd = rng.uniform(0.1, 5.0, nu)
        motor_limit = rng.uniform(10.0, 50.0, nu)

        expected = self._host_reference(
            q_target, qpos, qvel, kp, kd, motor_limit, qpos_offset, qvel_offset
        )
        actual = self._run_device_pd(
            q_target, qpos, qvel, kp, kd, motor_limit, qpos_offset, qvel_offset
        )
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_clamp_saturates(self):
        """大误差 + 小限幅 => tau 被限幅到 ±motor_limit。"""
        nu = 3
        nq, nv = 10, 9
        q_target = np.array([5.0, -5.0, 0.0])  # 远大于当前 qpos=0
        qpos = np.zeros(nq)
        qvel = np.zeros(nv)
        kp = np.full(nu, 1000.0)
        kd = np.full(nu, 0.0)
        motor_limit = np.array([10.0, 20.0, 30.0])

        actual = self._run_device_pd(q_target, qpos, qvel, kp, kd, motor_limit, 7, 6)
        np.testing.assert_allclose(actual, [10.0, -20.0, 0.0], atol=1e-6)


def _fresh_solver_and_pid(
    kp_value: float,
):
    """创建 solver、注册 PD 并返回 (solver_instance, pid, joint_names, actuator_names)。"""
    from orca_gym.core.euler.mujoco_sim_core_euler import MuJoCoSimCoreEuler

    core = MuJoCoSimCoreEuler()
    core.init_simulation(_G1_XML, device=_DEVICE, timestep=0.001)
    names = _actuated_joint_names(core.mj_model)
    nu = len(names)
    kp = np.full(nu, kp_value, dtype=np.float64)
    kd = np.full(nu, 1.0, dtype=np.float64)
    motor_limits = np.full(nu, 500.0, dtype=np.float64)
    pid = core.register_pid_controller(
        "pd", kp=kp, kd=kd, motor_limits=motor_limits, joint_names=names
    )
    return core, pid, names, _actuator_names(core.mj_model)


@unittest.skipUnless(_HAS_CUDA, "需要 CUDA")
class TestRegisterPidControllerGpu(unittest.TestCase):
    """register_pid_controller 接线（GPU）：模式 3（GPU-Native PD）在 step 中生效。"""

    def test_register_step_finite_and_pd_active(self):
        # 目标 = 各驱动关节 qpos0 + 0.4，保证每个关节都有非零 PD 误差
        core_on, pid_on, names, _ = _fresh_solver_and_pid(kp_value=1000.0)
        qpos0 = core_on.mj_model.qpos0.copy()
        qpos_offset = core_on.jnt_qposadr(names[0])
        target = qpos0[qpos_offset:qpos_offset + len(names)] + 0.4
        pid_on.update_target(target)
        core_on.set_qpos_qvel(qpos0, np.zeros(core_on.nv))
        core_on.step(1)
        qacc_on = _dict_to_flat(core_on.query_joint_qacc(names), names)

        self.assertTrue(np.isfinite(qacc_on).all(), "PD 开启时 qacc 应有限")

        # PD 关闭：kp=0 => ctrl=0，仅重力/被动动力学
        core_off, pid_off, _, _ = _fresh_solver_and_pid(kp_value=0.0)
        pid_off.update_target(target)
        core_off.set_qpos_qvel(core_off.mj_model.qpos0.copy(), np.zeros(core_off.nv))
        core_off.step(1)
        qacc_off = _dict_to_flat(core_off.query_joint_qacc(names), names)

        # 两种情形 qacc 不同 => 前步 kernel（ctrl）确实参与了动力学
        self.assertFalse(
            np.allclose(qacc_on, qacc_off, atol=1e-5),
            "PD 开启/关闭应产生不同加速度，说明设备侧 PD 未生效",
        )
        self.assertTrue(np.isfinite(qacc_off).all())


@unittest.skipUnless(_HAS_CUDA, "需要 CUDA")
class TestRtfSmoke(unittest.TestCase):
    """RTF 冒烟（GPU）：模式 3 粒度连跑，无 NaN 且 RTF >= 1.0。"""

    def test_rtf_above_1(self):
        core, pid, names, _ = _fresh_solver_and_pid(kp_value=100.0)
        nu = len(names)
        target = np.zeros(nu, dtype=np.float64)
        core.set_qpos_qvel(core.mj_model.qpos0.copy(), np.zeros(core.nv))

        frame_skip = 20
        dt = 0.001
        # 预热（含首次编译/JIT）
        for _ in range(20):
            pid.update_target(target)
            core.step(frame_skip)

        n_cycles = 100
        t0 = time.perf_counter()
        for _ in range(n_cycles):
            pid.update_target(target)
            core.step(frame_skip)
        wall_s = time.perf_counter() - t0

        sim_s = n_cycles * frame_skip * dt
        rtf = sim_s / wall_s
        qpos = _dict_to_flat(core.query_joint_qpos(names), names)
        self.assertTrue(np.isfinite(qpos).all(), "连续推进后 qpos 出现 NaN")
        print(f"\n[RTF] sim={sim_s:.3f}s wall={wall_s:.3f}s RTF={rtf:.2f}")
        self.assertGreaterEqual(rtf, 1.0, "离线纯物理推进 RTF 应 >= 1.0")


if __name__ == "__main__":
    unittest.main()
