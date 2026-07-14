"""OnlineVerifier — 阶段四在线判定框架。

为 Lesson 4–8 的在线端到端 example 提供统一的数值判定与人工观察收集机制：
- check / check_allclose / check_range：数值判定（运行时读写 env.data，与预期值比较）
- observe：人工观察项（打印 Studio 视口观察提示，引导用户确认视觉效果）
- report：运行结束后输出 JSON 报告 + 控制台摘要

参见 docs/design/development/orca_gym_euler_phase4_online_validation_development.md §4.2。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from typing import Any

import numpy as np


def _safe_filename(name: str) -> str:
    """将 lesson_name 转换为安全的文件名片段。"""
    return re.sub(r"[^A-Za-z0-9_\u4e00-\u9fa5]", "_", name)


class OnlineVerifier:
    """在线判定器：运行中收集判定项，结束后输出报告。

    使用方式:
        verifier = OnlineVerifier("Lesson 4: 状态查询 API")
        verifier.check("joint_qpos_dim", len(qpos) == 29, len(qpos), 29)
        verifier.check_allclose("joint_qpos_vs_data", qpos_arr, expected, atol=1e-6)
        verifier.check_range("pelvis_initial_height", pelvis_z, 0.75, 0.85)
        verifier.observe("g1_standing", "Studio 视口：G1 应站立在地面上")
        report = verifier.report()  # 输出 JSON + 控制台摘要
    """

    def __init__(self, lesson_name: str, report_dir: str | None = None) -> None:
        """初始化判定器。

        Args:
            lesson_name: 课程名称（如 "Lesson 4: 状态查询 API"）。
            report_dir: JSON 报告输出目录，默认系统临时目录。
        """
        self.lesson_name = lesson_name
        self.report_dir = report_dir if report_dir is not None else tempfile.gettempdir()
        self.checks: list[dict[str, Any]] = []
        self.observations: list[dict[str, Any]] = []
        self._observed_names: set[str] = set()

    def check(
        self,
        name: str,
        condition: bool,
        actual: Any = None,
        expected: Any = None,
        detail: str = "",
    ) -> None:
        """数值判定：condition 为 True 则通过。

        Args:
            name: 判定项名称（唯一标识）。
            condition: 判定条件（True 通过，False 失败）。
            actual: 实际值（用于报告记录）。
            expected: 预期值（用于报告记录）。
            detail: 补充说明。
        """
        passed = bool(condition)
        self.checks.append(
            {
                "name": name,
                "passed": passed,
                "actual": _to_serializable(actual),
                "expected": _to_serializable(expected),
                "detail": detail,
            }
        )
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: actual={actual}, expected={expected} {detail}")

    def check_allclose(
        self,
        name: str,
        actual: Any,
        expected: Any,
        atol: float = 1e-3,
        detail: str = "",
    ) -> None:
        """数值近似判定（np.allclose 封装）。

        Args:
            name: 判定项名称。
            actual: 实际值（数组或标量）。
            expected: 预期值（数组或标量）。
            atol: 绝对容差。
            detail: 补充说明。
        """
        actual_arr = np.asarray(actual)
        expected_arr = np.asarray(expected)
        if actual_arr.shape != expected_arr.shape:
            self.check(
                name,
                False,
                actual_arr.shape,
                expected_arr.shape,
                f"shape mismatch {detail}",
            )
            return
        passed = bool(np.allclose(actual_arr, expected_arr, atol=atol))
        self.checks.append(
            {
                "name": name,
                "passed": passed,
                "actual": _to_serializable(actual_arr.tolist()),
                "expected": _to_serializable(expected_arr.tolist()),
                "detail": f"atol={atol} {detail}",
            }
        )
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: atol={atol} {detail}")

    def check_range(
        self,
        name: str,
        value: float,
        low: float,
        high: float,
        detail: str = "",
    ) -> None:
        """范围判定：low <= value <= high。

        Args:
            name: 判定项名称。
            value: 实际值。
            low: 下界（含）。
            high: 上界（含）。
            detail: 补充说明。
        """
        passed = bool(low <= value <= high)
        self.checks.append(
            {
                "name": name,
                "passed": passed,
                "actual": _to_serializable(value),
                "expected": f"[{low}, {high}]",
                "detail": detail,
            }
        )
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: value={value}, expected=[{low}, {high}] {detail}")

    def observe(self, name: str, prompt: str, step: int = 0) -> None:
        """人工观察项：打印提示，等用户在 Studio 视口确认。

        同名观察项只记录一次（去重），避免循环中重复打印。

        Args:
            name: 观察项名称（唯一标识，用于去重）。
            prompt: 观察提示文本（引导用户在 Studio 视口确认）。
            step: 触发该观察的步数（用于报告记录）。
        """
        if name not in self._observed_names:
            self.observations.append(
                {"name": name, "prompt": prompt, "step": step}
            )
            self._observed_names.add(name)
        print(f"  [OBSERVE] {name}: {prompt}")

    def report(self) -> dict[str, Any]:
        """输出判定报告并返回 JSON 字典。

        输出:
        1. 控制台摘要：pass/fail 计数 + 人工观察项数 + ALL PASS / SOME FAILED
        2. JSON 报告文件：写入 {report_dir}/euler_{lesson_name}_report.json

        Returns:
            完整报告字典（含 summary/checks/observations）。
        """
        passed_count = sum(1 for c in self.checks if c["passed"])
        total = len(self.checks)
        all_passed = (passed_count == total) and (total > 0)

        report: dict[str, Any] = {
            "lesson": self.lesson_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": total,
                "passed": passed_count,
                "failed": total - passed_count,
                "all_passed": all_passed,
            },
            "checks": self.checks,
            "observations": self.observations,
        }

        print("\n" + "=" * 60)
        print(f"判定报告: {self.lesson_name}")
        print(f"数值判定: {passed_count}/{total} passed")
        print(
            f"人工观察: {len(self.observations)} 项"
            "（请在上方 [OBSERVE] 提示处确认）"
        )
        print(f"总结: {'ALL PASS' if all_passed else 'SOME FAILED'}")
        print("=" * 60)

        # 写入 JSON 文件
        safe_name = _safe_filename(self.lesson_name)
        report_path = os.path.join(self.report_dir, f"euler_{safe_name}_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报告已写入: {report_path}")

        return report


def _to_serializable(value: Any) -> Any:
    """将 numpy 类型转换为 JSON 可序列化的 Python 原生类型。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
