"""
OrcaPlayground Setup
独立仓库安装脚本
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取 README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# 读取 requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

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
    extras_require={
        # 注意：ray[rllib] 与 orca-gym 的 gymnasium==1.2.1 存在依赖冲突
        # 如需使用 ray[rllib]，请参考 README 中的说明手动安装
        # "rllib": ["ray[rllib]>=2.0.0"],  # 已注释，存在依赖冲突
    },
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

