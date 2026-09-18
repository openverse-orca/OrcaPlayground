"""User documentation describes examples and the model format, not implementation details."""

from pathlib import Path
import re


EXAMPLES = Path(__file__).resolve().parents[2] / "examples/euler/sensor_provider"


def test_readme_lists_four_sensor_commands_and_two_hand_presentations():
    text = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    commands = re.findall(r"^python -m examples\.euler\.sensor_provider\.run --example (\w+)$",
                          text, flags=re.MULTILINE)
    assert commands == ["touch_grid", "contact_grid", "rangefinder", "seven_pad", "hand", "hand_grid"]
    assert "四种传感器示例" in text
    assert "不是另外两种传感器" in text


def test_user_docs_only_reference_mujoco_for_format_compatibility():
    for name in ("README.md", "advanced.md", "scenes/dexhand/README.md"):
        text = (EXAMPLES / name).read_text(encoding="utf-8")
        mentions = [line for line in text.splitlines() if "mujoco" in line.lower()]
        assert mentions, name
        assert all("格式兼容 MuJoCo MJCF" in line for line in mentions), name
        assert "native_mjcf" not in text, name
        assert "orca_mjcf" not in text, name
