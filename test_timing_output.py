#!/usr/bin/env python3
"""
测试 Timing 输出是否正常工作
"""
import subprocess
import sys
import time
from pathlib import Path

# 创建一个简单的测试场景 - 使用 DamBreak 场景
test_scene = """{
  "Configuration": {
    "particleRadius": 0.025,
    "sim2D": true,
    "stopAt": 0.5,
    "timeStepSize": 0.001
  },
  "Fluid": {
    "id": "Fluid",
    "viscosity": 0.01,
    "surfaceTension": 0.0,
    "density0": 1000.0
  },
  "FluidBlocks": [
    {
      "id": "Fluid",
      "start": [0.0, 0.0, 0.0],
      "end": [1.0, 1.0, 0.1],
      "translation": [0.0, 0.0, 0.0]
    }
  ],
  "RigidBodies": [],
  "Emitters": [],
  "Materials": []
}"""

# 保存测试场景
test_scene_path = Path("/tmp/test_timing_scene.json")
with open(test_scene_path, "w") as f:
    f.write(test_scene)

print("=== 测试 Timing 输出 ===")
print(f"测试场景: {test_scene_path}")

# 运行 SPlisHSPlasH
sph_simulator = Path("/home/superfhwl/repo/SPlisHSPlasH/bin/SPHSimulator")
if not sph_simulator.exists():
    print(f"❌ SPlisHSPlasH 不存在: {sph_simulator}")
    sys.exit(1)

print(f"运行: {sph_simulator}")
print("参数: --performance-stats --no-initial-pause")

# 运行并捕获输出 - 使用现有的 DamBreak 场景
result = subprocess.run(
    [str(sph_simulator), "--performance-stats", "--no-initial-pause", 
     "/home/superfhwl/repo/SPlisHSPlasH/data/Scenes/DamBreakModel.json"],
    capture_output=True,
    text=True,
    timeout=60
)

print(f"\n返回码: {result.returncode}")

# 检查输出中是否有 [TIME_STATS]
if "[TIME_STATS]" in result.stdout:
    print("✅ 在 stdout 中找到 [TIME_STATS] 输出")
    # 提取 [TIME_STATS] 行
    for line in result.stdout.split('\n'):
        if "[TIME_STATS]" in line:
            print(f"  {line}")
elif "[TIME_STATS]" in result.stderr:
    print("✅ 在 stderr 中找到 [TIME_STATS] 输出")
    for line in result.stderr.split('\n'):
        if "[TIME_STATS]" in line:
            print(f"  {line}")
else:
    print("❌ 未找到 [TIME_STATS] 输出")
    print("\nstdout 前 100 行:")
    for i, line in enumerate(result.stdout.split('\n')[:100]):
        if line.strip():
            print(f"  {line}")

# 清理
test_scene_path.unlink()
print("\n=== 测试完成 ===")
