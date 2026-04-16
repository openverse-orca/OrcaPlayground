"""
OrcaPlayground Setup
独立仓库安装脚本
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取 README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

def read_requirements(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


def discover_example_extras(examples_dir: Path) -> dict[str, list[str]]:
    extras: dict[str, list[str]] = {}
    if not examples_dir.exists():
        return extras

    for child in sorted(examples_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("__"):
            continue
        has_example_markers = (
            (child / "README.md").exists()
            or (child / "requirements.txt").exists()
            or any(child.glob("*.py"))
        )
        if not has_example_markers:
            continue
        extras[child.name] = read_requirements(child / "requirements.txt")

    return extras


def merge_unique_requirements(groups: dict[str, list[str]]) -> list[str]:
    merged: list[str] = []
    for requirements in groups.values():
        for requirement in requirements:
            if requirement not in merged:
                merged.append(requirement)
    return merged


# 读取基础 requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = read_requirements(requirements_file)

example_extras = discover_example_extras(Path(__file__).parent / "examples")
example_extras["all"] = merge_unique_requirements(example_extras)

setup(
    name="orca-playground",
    version="1.0.0",
    description="OrcaGym 示例代码独立仓库",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Orca Team",
    url="https://github.com/openverse-orca/OrcaPlayground",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=requirements,
    python_requires=">=3.9",
    extras_require=example_extras,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)

