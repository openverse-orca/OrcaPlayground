"""OrcaPlayground Euler 公共库离线测试（Phase D 验收，CPU-only）。

覆盖范围（无 GPU、无 gRPC）：
- scoped_name 纯函数：在线带前缀 / 离线空前缀拼接
- G1BaseEnv CPU 离线构造（skip_grpc_load=True）：agent_name 置空、
  _device_pd 恒 None、后端 MUJOCO、模型维度正确
- step() 模式分派与时间推进：模式 1（host 批量，_pd_controller 1 次）、
  模式 2（host 逐子步闭环，_pd_controller frame_skip 次）
- _register_device_pd 在 CPU 后端下为 no-op（回退模式 2，不触 flow/GPU）

运行方式（沙箱内 CPU-only，无需 whitelist）:
    <conda-base>/envs/orca/bin/python -m unittest \
        examples/euler/tests/test_common_offline.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

# 就地引用 common（非打包模块）：把 examples/euler 加入 sys.path
_EULER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EULER_ROOT not in sys.path:
    sys.path.insert(0, _EULER_ROOT)

from common.g1_base_env import G1BaseEnv, G1_TIME_STEP, G1_FRAME_SKIP, scoped_name  # noqa: E402
from orca_gym.core.euler.sim_config import SimBackend  # noqa: E402


class TestScopedName(unittest.TestCase):
    """scoped_name 纯函数拼接语义（在线带前缀 / 离线空前缀）。"""

    def test_online_with_prefix(self):
        self.assertEqual(
            scoped_name("g1", "left_hip_pitch_joint"), "g1_left_hip_pitch_joint"
        )

    def test_offline_empty_string_prefix(self):
        self.assertEqual(scoped_name("", "left_hip_pitch_joint"), "left_hip_pitch_joint")

    def test_offline_none_prefix(self):
        self.assertEqual(scoped_name(None, "left_hip_pitch_joint"), "left_hip_pitch_joint")


class _CountingEnv(G1BaseEnv):
    """记录 _pd_controller 调用次数的 G1 子类（验证模式 1/2 分派）。"""

    def __init__(self, *args, **kwargs):
        self.pd_calls = 0
        super().__init__(*args, **kwargs)

    def _pd_controller(self, target: np.ndarray) -> np.ndarray:
        self.pd_calls += 1
        return np.zeros(self.model.nu, dtype=np.float32)


class _DevicePdEnv(G1BaseEnv):
    """声明需要 device PD 的子类（无 locomotion 属性，验证 CPU 回退）。"""

    _requires_device_pd = True


class TestG1BaseEnvOfflineConstruction(unittest.TestCase):
    """G1BaseEnv CPU 离线构造验收（skip_grpc_load=True）。"""

    def test_offline_construction(self):
        env = G1BaseEnv(skip_grpc_load=True)
        try:
            # agent_name 离线模式置空（scoped_name 直接用后缀）
            self.assertEqual(env.agent_name, "")
            # device PD 未构造（CPU 后端不注册，step 走模式 1/2）
            self.assertIsNone(env._device_pd)
            # 默认 host 批量（模式 1）
            self.assertFalse(env._per_substep_ctrl)
            # 后端为 MUJOCO（CPU）
            self.assertEqual(env.sim_config.backend, SimBackend.MUJOCO)
            # 模型维度正确（G1 29 旋转关节 + 1 free base）
            self.assertEqual(env.model.nu, 29)
            self.assertGreater(env.model.nq, 0)
            # 时间步与 frame_skip 常量
            self.assertEqual(env.frame_skip, G1_FRAME_SKIP)
            self.assertEqual(env._time_step, G1_TIME_STEP)
        finally:
            env.close()


class TestStepModeDispatch(unittest.TestCase):
    """step() 模式 1/2 分派与时间推进验收（CPU 离线）。"""

    def test_mode1_host_batch_single_pd_call(self):
        env = _CountingEnv(skip_grpc_load=True)
        try:
            env._per_substep_ctrl = False
            t0 = float(env.data.time)
            env.step(np.zeros(env.model.nu, dtype=np.float32))
            # 模式 1：_pd_controller 仅调用 1 次，批量步进 frame_skip
            self.assertEqual(env.pd_calls, 1)
            self.assertAlmostEqual(
                float(env.data.time) - t0, env.frame_skip * env._time_step, places=9
            )
        finally:
            env.close()

    def test_mode2_host_per_substep_pd_calls_frame_skip_times(self):
        env = _CountingEnv(skip_grpc_load=True)
        try:
            env._per_substep_ctrl = True
            t0 = float(env.data.time)
            env.step(np.zeros(env.model.nu, dtype=np.float32))
            # 模式 2：每物理步重算 ctrl，_pd_controller 调用 frame_skip 次
            self.assertEqual(env.pd_calls, env.frame_skip)
            self.assertAlmostEqual(
                float(env.data.time) - t0, env.frame_skip * env._time_step, places=9
            )
        finally:
            env.close()

    def test_device_pd_none_never_triggers_mode3(self):
        """_device_pd 为 None 时 step 恒不触 GPU；计数型子的 _device_pd 应保持 None。"""
        env = _CountingEnv(skip_grpc_load=True)
        try:
            self.assertIsNone(env._device_pd)
            t0 = float(env.data.time)
            env.step(np.zeros(env.model.nu, dtype=np.float32))
            self.assertEqual(env.pd_calls, 1)  # 走模式 1，而非模式 3
            self.assertGreater(float(env.data.time), t0)
        finally:
            env.close()


class TestRegisterDevicePdCpuFallback(unittest.TestCase):
    """_register_device_pd 在 CPU 后端下的 no-op 回退验证。"""

    def test_register_device_pd_noop_on_cpu(self):
        env = _DevicePdEnv(skip_grpc_load=True)
        try:
            # 后端为 MUJOCO，_register_device_pd 直接跳过（不触 flow/GPU）
            env._register_device_pd()
            self.assertIsNone(env._device_pd)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()