"""一次性运行 tests/ 目录下全部测试用例。

使用 unittest 标准测试发现机制，自动收集并运行所有 test_*.py。

运行方式:
    <conda-base>/envs/orca/bin/python tests/run_tests.py
    <conda-base>/envs/orca/bin/python tests/run_tests.py -v

也可指定单测模块:
    <conda-base>/envs/orca/bin/python -m unittest tests.test_ruff_config

注意:
- GPU / 在线测试默认 skip，需通过环境变量启用（见各测试文件 docstring）。
- 测试须在 orca conda 环境下运行（AGENTS.md 规则 1）。
"""

import sys
import unittest
from pathlib import Path

# 测试目录（本脚本所在目录）
_TESTS_DIR = Path(__file__).resolve().parent
# 仓库根目录（tests/ 的上一级），用于将仓库根加入 sys.path，
# 使 `tests.test_xxx` 这类模块路径可被 import
_REPO_ROOT = _TESTS_DIR.parent


def main() -> int:
    """发现并运行全部测试，返回退出码（0=成功，非 0=失败）。"""
    # 将仓库根加入 sys.path，保证 `import tests.xxx` / `-m unittest tests.test_xxx` 可用
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    # 命令行 -v 开关控制详细程度
    verbose = "-v" in sys.argv[1:]

    loader = unittest.TestLoader()
    # 从 tests/ 目录发现所有 test_*.py，顶层模块名以 'tests' 为前缀
    suite = loader.discover(
        start_dir=str(_TESTS_DIR),
        pattern="test_*.py",
        top_level_dir=str(_REPO_ROOT),
    )

    runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1,
        buffer=True,
    )
    result = runner.run(suite)

    # unittest.TestResult: failures/errors 非空时返回 1，否则 0
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
