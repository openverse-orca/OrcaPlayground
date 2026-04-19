#!/bin/bash
# 验证性能统计功能是否正常工作

echo "=== 验证性能统计功能 ==="

# 1. 检查 SPlisHSPlasH 是否编译了最新的代码
echo ""
echo "1. 检查 SPlisHSPlasH 编译时间..."
ls -la /home/superfhwl/repo/SPlisHSPlasH/bin/SPHSimulator

# 2. 检查 TimeStepDFSPH_CUDA.cpp 的修改时间
echo ""
echo "2. 检查 TimeStepDFSPH_CUDA.cpp 修改时间..."
ls -la /home/superfhwl/repo/SPlisHSPlasH/Orca/GPUSPHSolver/DFSPH/TimeStepDFSPH_CUDA.cpp

# 3. 运行简单的 SPlisHSPlasH 测试
echo ""
echo "3. 运行 SPlisHSPlasH 性能统计测试..."
cd /home/superfhwl/repo/SPlisHSPlasH/bin

# 创建一个简单的测试场景
cat > /tmp/test_performance.json << 'EOF'
{
  "Configuration": {
    "particleRadius": 0.025,
    "sim2D": true,
    "stopAt": 0.5
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
      "end": [0.5, 0.5, 0.1],
      "translation": [0.0, 0.0, 0.0]
    }
  ],
  "RigidBodies": [],
  "Emitters": [],
  "Materials": []
}
EOF

# 运行测试
./SPHSimulator --performance-stats --no-initial-pause /tmp/test_performance.json 2>&1 | head -100

echo ""
echo "=== 验证完成 ==="