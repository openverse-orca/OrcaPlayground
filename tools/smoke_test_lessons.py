"""课程冒烟测试 — 30 课导入 + example.yaml 结构断言（不连引擎）。

用途：本地自测（阶段开发回归用）。一条命令确认没碰坏既有课程：
  - 每课 run.py 可导入（语法 / import 链路 / _common 依赖完整）
  - 每课 example.yaml 字段齐全（id/title/entry/challenge/challenge_type）

为什么不连引擎真跑：课程核心验收是"实测数字命中理论"（需 OrcaLab
+ 资产订阅，见各课 README ）；本脚本只覆盖
静态可验证的分层，真跑用 tools/run_all_beginner_lessons.sh。

用法:
    python tools/smoke_test_lessons.py            # 全量（默认）
    python tools/smoke_test_lessons.py -v         # 逐课打印 OK 项
    python tools/smoke_test_lessons.py --stage 5  # 只测指定阶段

退出码: 0 全绿 / 1 有失败。
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import sys
from typing import Any

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_LESSON_GLOB = "stage*/lesson_*"

_REQUIRED_FIELDS = ("id", "title", "entry")
_REQUIRED_CHALLENGE_FIELDS = ("challenge", "challenge_type")
_VALID_CHALLENGE_TYPES = ("swap", "combine", "predict")


def discover_lessons(stage: int | None) -> list[pathlib.Path]:
    """发现全部课程 run.py（按目录名排序，即课程编号顺序）。"""
    root = _REPO_ROOT / "examples" / "euler" / "beginner"
    if stage is not None:
        pattern = f"stage{stage}_*/lesson_*/run.py"
    else:
        pattern = "stage*/lesson_*/run.py"
    return sorted(root.glob(pattern))


def smoke_import_module(run_path: pathlib.Path) -> str | None:
    """导入课程模块，返回 None 表示通过，否则返回错误描述。"""
    # 相对仓库根拼模块名（绝对路径 parts 带根 '/'，不可直接 join）
    rel = run_path.relative_to(_REPO_ROOT).with_suffix("")
    module = ".".join(rel.parts)
    try:
        importlib.import_module(module)
        return None
    except Exception as exc:  # noqa: BLE001 — 任意导入失败都需汇总上报
        return f"{type(exc).__name__}: {exc}"


def smoke_check_yaml(run_path: pathlib.Path) -> str | None:
    """校验同目录 example.yaml 的结构，返回 None 表示通过。"""
    yaml_path = run_path.parent / "example.yaml"
    if not yaml_path.exists():
        return "缺少 example.yaml"
    try:
        data: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — 解析失败需汇总上报
        return f"yaml 解析失败 {type(exc).__name__}: {exc}"
    for field in _REQUIRED_FIELDS:
        if not data.get(field):
            return f"缺少基础字段: {field}"
    for field in _REQUIRED_CHALLENGE_FIELDS:
        if field not in data:
            return f"缺少挑战字段: {field}"
    if data["challenge_type"] not in _VALID_CHALLENGE_TYPES:
        return f"挑战类型非法: {data['challenge_type']}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="课程导入 + yaml 冒烟测试")
    parser.add_argument("-v", "--verbose", action="store_true", help="逐课打印通过项")
    parser.add_argument("--stage", type=int, choices=range(1, 9), help="只测指定阶段（1-8）")
    args = parser.parse_args()

    lessons = discover_lessons(args.stage)
    if not lessons:
        print(f"未发现课程（stage={args.stage}）")
        return 1
    print(f"发现 {len(lessons)} 个课程模块\n")

    import_fails: list[tuple[str, str]] = []
    yaml_fails: list[tuple[str, str]] = []
    for run_path in lessons:
        lesson = run_path.parent.name
        err = smoke_import_module(run_path)
        if err is None:
            if args.verbose:
                print(f"[OK  ] 导入 {lesson}")
        else:
            import_fails.append((lesson, err))
            print(f"[FAIL] 导入 {lesson}: {err}")

        err = smoke_check_yaml(run_path)
        if err is None:
            if args.verbose:
                print(f"[OK  ] yaml  {lesson}")
        else:
            yaml_fails.append((lesson, err))
            print(f"[FAIL] yaml  {lesson}: {err}")

    total = len(lessons)
    ok_import = total - len(import_fails)
    ok_yaml = total - len(yaml_fails)
    print(f"\n=== 汇总：导入 {ok_import}/{total} 通过，yaml {ok_yaml}/{total} 通过 ===")
    return 0 if not (import_fails or yaml_fails) else 1


if __name__ == "__main__":
    sys.exit(main())
