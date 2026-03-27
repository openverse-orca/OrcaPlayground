#!/bin/bash
# 安装当前 legged_gym 示例所需的 SB3 依赖

echo "安装基础依赖..."
pip install pydantic

echo "安装 SB3 依赖..."
pip install stable-baselines3>=2.3.2 sb3_contrib>=2.3.0

echo "依赖安装完成！"

