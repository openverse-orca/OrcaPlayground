"""The supported sensor entry points describe scenes, measurements and controls."""

import ast
import io
from pathlib import Path
import subprocess
import sys
import tokenize

import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/euler/sensor_provider"
ENTRY_MODULES = ("run", "playground", "dexhand_euler", "dexhand_seven_pad")
SOURCE_MODULES = (*ENTRY_MODULES, "scene_env", "dexhand_control", "sensor_dashboard", "provider_paths")


@pytest.mark.parametrize("module", ENTRY_MODULES)
def test_entry_help_describes_the_example_without_implementation_brand(module):
    result = subprocess.run(
        [sys.executable, "-m", f"examples.euler.sensor_provider.{module}", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MuJoCo" not in result.stdout
    assert "--steps" in result.stdout and "--output" in result.stdout


def test_example_explanatory_text_uses_physics_or_mjcf_terms():
    # Review the submitted example only; local historical tools are independent.
    for module in SOURCE_MODULES:
        path = EXAMPLES / f"{module}.py"
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                assert "mujoco" not in (ast.get_docstring(node) or "").lower(), path.name
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                assert "mujoco" not in token.string.lower(), (path.name, token.start)
