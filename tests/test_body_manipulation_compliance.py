"""阶段3: 09 课 BodyManipulationEnv 消费者自管编排合规测试。

验证 09 课 body_manipulation_env.py 改造后：
- _bind_mocap_to_pelvis / _release_mocap 使用公共无状态原语编排
- 不调用已删除的 equality_bind_mocap / equality_release / equality_snapshot
- 不穿墙 _gym._sim / _mjData / _gym.update_equality_constraints
- 绑定入口有幂等保护（if self._bound_slot is not None: return）
- _run_phase / _ensure_released 委托到 _bind_mocap_to_pelvis / _release_mocap

对应 docs/design/development/orca_gym_euler_anchor_equality_refactor_development.md
阶段3 验收标准。

运行方式:
    <conda-base>/envs/orca/bin/python -m unittest tests.test_body_manipulation_compliance
"""

import pathlib
import re
import unittest

# body_manipulation_env.py 位于
# OrcaPlayground/examples/euler/09_body_manipulation/body_manipulation_env.py
# __file__ = OrcaPlayground/tests/test_body_manipulation_compliance.py
# parents[0] = OrcaPlayground/tests
# parents[1] = OrcaPlayground
_ENV_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples" / "euler" / "09_body_manipulation"
    / "body_manipulation_env.py"
)


def _read_env_source() -> str:
    """读取 body_manipulation_env.py 源码（用于源码审查）。"""
    return _ENV_PATH.read_text(encoding="utf-8")


def _extract_method(source: str, method_name: str) -> str:
    """提取指定方法体（从 def 到下一个同级 def / class / EOF）。

    Args:
        source: 源码文本。
        method_name: 方法名（如 "_bind_mocap_to_pelvis"）。

    Returns:
        方法体文本（含 def 行）。未找到返回空字符串。
    """
    pattern = rf"    def {re.escape(method_name)}\((.*?)(?=\n    def |\nclass |\Z)"
    match = re.search(pattern, source, re.DOTALL)
    return match.group(0) if match else ""


def _strip_comments_and_docstrings(source: str) -> str:
    """移除注释和 docstring，仅保留可执行代码（用于 grep 穿墙检测）。"""
    # 移除 docstring（"""..."""）
    exec_source = re.sub(r'"""[\s\S]*?"""', '', source)
    # 移除单行注释
    exec_source = re.sub(r'#.*', '', exec_source)
    return exec_source


class TestBindMocapToPelvisCompliance(unittest.TestCase):
    """_bind_mocap_to_pelvis 编排合规性测试。"""

    def setUp(self):
        self.source = _read_env_source()
        self.block = _extract_method(self.source, "_bind_mocap_to_pelvis")
        self.assertTrue(self.block, "_bind_mocap_to_pelvis 方法未找到")

    def test_uses_public_primitives(self):
        """K1: _bind_mocap_to_pelvis 使用公共原语编排。"""
        block = self.block
        self.assertIn("self.equality_find_slot_by_body", block)
        self.assertIn("self.equality_constraint", block)
        self.assertIn("self.set_mocap_pos_and_quat", block)
        self.assertIn("self.equality_update", block)

    def test_idempotency_guard(self):
        """§4.5 bug 修正：绑定入口有幂等保护。"""
        block = self.block
        # 幂等保护：已绑定时不重复绑定
        self.assertIn("self._bound_slot is not None", block)

    def test_no_deleted_api_calls(self):
        """不调用已删除的 equality_bind_mocap / equality_release / equality_snapshot。"""
        exec_block = _strip_comments_and_docstrings(self.block)
        for deleted in [
            "self.equality_bind_mocap",
            "self.equality_release",
            "self.equality_snapshot",
        ]:
            self.assertNotIn(
                deleted, exec_block,
                f"_bind_mocap_to_pelvis 调用已删除的 {deleted}",
            )

    def test_no_private_access(self):
        """K4: 不穿墙 _gym._sim / _mjData / _gym.update_equality_constraints。"""
        exec_block = _strip_comments_and_docstrings(self.block)
        for forbidden in [
            "self._gym._sim",
            "_mjData",
            "_mjModel",
            "self._gym.update_equality_constraints",
            "self._gym._studio",
        ]:
            self.assertNotIn(
                forbidden, exec_block,
                f"_bind_mocap_to_pelvis 穿墙访问 {forbidden}",
            )

    def test_uses_weld_type(self):
        """绑定类型为 weld（6 自由度全锁定）。"""
        block = self.block
        self.assertIn("mjEQ_WELD", block)


class TestReleaseMocapCompliance(unittest.TestCase):
    """_release_mocap 编排合规性测试。"""

    def setUp(self):
        self.source = _read_env_source()
        self.block = _extract_method(self.source, "_release_mocap")
        self.assertTrue(self.block, "_release_mocap 方法未找到")

    def test_uses_equality_update(self):
        """K1: _release_mocap 用 equality_update 从快照恢复（不依赖 equality_release）。"""
        block = self.block
        self.assertIn("self.equality_update", block)
        # id→name 反查
        self.assertIn("body_id2name", block)

    def test_no_deleted_api_calls(self):
        """不调用已删除的 equality_release / equality_snapshot / equality_bind_mocap。"""
        exec_block = _strip_comments_and_docstrings(self.block)
        for deleted in [
            "self.equality_release",
            "self.equality_snapshot",
            "self.equality_bind_mocap",
        ]:
            self.assertNotIn(
                deleted, exec_block,
                f"_release_mocap 调用已删除的 {deleted}",
            )

    def test_no_private_access(self):
        """K4: 不穿墙 _gym._sim / _mjData / _gym.update_equality_constraints。"""
        exec_block = _strip_comments_and_docstrings(self.block)
        for forbidden in [
            "self._gym._sim",
            "_mjData",
            "_mjModel",
            "self._gym.update_equality_constraints",
            "self._gym._studio",
        ]:
            self.assertNotIn(
                forbidden, exec_block,
                f"_release_mocap 穿墙访问 {forbidden}",
            )

    def test_noop_when_not_bound(self):
        """未绑定时 no-op（_bound_slot is None 或 _original_eq_snapshot is None）。"""
        block = self.block
        self.assertIn("self._bound_slot is None", block)
        self.assertIn("self._original_eq_snapshot is None", block)

    def test_clears_state_after_release(self):
        """释放后清除业务状态。"""
        block = self.block
        self.assertIn("self._bound_slot = None", block)
        self.assertIn("self._original_eq_snapshot = None", block)


class TestRunPhaseDelegatesToBind(unittest.TestCase):
    """_run_phase bound=True 委托到 _bind_mocap_to_pelvis。"""

    def setUp(self):
        self.source = _read_env_source()
        self.block = _extract_method(self.source, "_run_phase")
        self.assertTrue(self.block, "_run_phase 方法未找到")

    def test_calls_bind_mocap_to_pelvis(self):
        """_run_phase bound 分支委托 _bind_mocap_to_pelvis（不再内联 bind 逻辑）。"""
        block = self.block
        self.assertIn("self._bind_mocap_to_pelvis", block)

    def test_no_direct_deleted_api_in_run_phase(self):
        """_run_phase 不直接调用已删除的 equality_bind_mocap / equality_release。"""
        exec_block = _strip_comments_and_docstrings(self.block)
        for deleted in [
            "self.equality_bind_mocap",
            "self.equality_release",
            "self.equality_snapshot",
        ]:
            self.assertNotIn(
                deleted, exec_block,
                f"_run_phase 直接调用已删除的 {deleted}",
            )

    def test_uses_set_mocap_for_movement(self):
        """移动阶段仍用 set_mocap_pos_and_quat 周期性驱动 mocap。"""
        block = self.block
        self.assertIn("self.set_mocap_pos_and_quat", block)


class TestEnsureReleasedDelegatesToRelease(unittest.TestCase):
    """_ensure_released 委托到 _release_mocap。"""

    def setUp(self):
        self.source = _read_env_source()
        self.block = _extract_method(self.source, "_ensure_released")
        self.assertTrue(self.block, "_ensure_released 方法未找到")

    def test_calls_release_mocap(self):
        """_ensure_released 委托 _release_mocap（不再直接调 equality_release）。"""
        block = self.block
        self.assertIn("self._release_mocap", block)

    def test_no_direct_deleted_api(self):
        """_ensure_released 不直接调用已删除的 equality_release。"""
        exec_block = _strip_comments_and_docstrings(self.block)
        for deleted in [
            "self.equality_release",
            "self.equality_snapshot",
            "self.equality_bind_mocap",
            "self._gym.update_equality_constraints",
        ]:
            self.assertNotIn(
                deleted, exec_block,
                f"_ensure_released 直接调用已删除的 {deleted}",
            )


class TestGlobalSourceCompliance(unittest.TestCase):
    """全局源码合规性测试。"""

    def setUp(self):
        self.source = _read_env_source()
        self.exec_source = _strip_comments_and_docstrings(self.source)

    def test_no_deleted_api_in_executable_code(self):
        """可执行代码中不含已删除 API 的调用。"""
        for deleted in [
            "self.equality_bind_mocap",
            "self.equality_release",
            "self.equality_snapshot",
        ]:
            self.assertNotIn(
                deleted, self.exec_source,
                f"可执行代码中含已删除的 {deleted} 调用",
            )

    def test_no_private_tunnel_access(self):
        """K4: 全局不穿墙 _gym._sim / _mjData / _mjModel。"""
        for forbidden in [
            "self._gym._sim",
            "self._gym._mjData",
            "self._gym._mjModel",
            "self._gym._studio",
            "self._gym.update_equality_constraints",
        ]:
            self.assertNotIn(
                forbidden, self.exec_source,
                f"全局穿墙访问 {forbidden}",
            )

    def test_state_fields_self_managed(self):
        """消费者自管业务状态字段存在。"""
        source = self.source
        self.assertIn("self._bound_slot", source)
        self.assertIn("self._original_eq_snapshot", source)

    def test_no_anchor_field_pollution(self):
        """不触碰 UI 抓取的 _anchor_* 字段。"""
        exec_source = self.exec_source
        for anchor_field in [
            "self._anchored_actor",
            "self._anchor_type",
            "self._anchor_original_eq",
            "self._anchor_mocap_name",
        ]:
            self.assertNotIn(
                anchor_field, exec_source,
                f"09 课污染 UI 抓取字段 {anchor_field}",
            )

    def test_methods_exist(self):
        """阶段3 新增方法存在。"""
        source = self.source
        self.assertIn("def _bind_mocap_to_pelvis(", source)
        self.assertIn("def _release_mocap(", source)


if __name__ == "__main__":
    unittest.main()
