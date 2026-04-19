#!/usr/bin/env python3
"""
测试性能统计查看器的启动
"""
import subprocess
import sys
from pathlib import Path
import time

# 创建一个测试日志文件
log_content = """
[TIME_STATS] initializeCPUDataEachTimeStep=1.234 initializeGPUDataEachTimeStep=0.567
[TIME_STATS] initializeCPUDataEachTimeStep=1.123 initializeGPUDataEachTimeStep=0.456
[TIME_STATS] initializeCPUDataEachTimeStep=1.345 initializeGPUDataEachTimeStep=0.678
"""

# 创建测试日志文件
test_log_path = Path(".") / "test_performance_log.txt"
with open(test_log_path, "w") as f:
    f.write(log_content)

# 构建性能统计查看器的路径
stats_viewer_path = Path(".") / "envs" / "fluid_stats" / "performance_stats_viewer.py"

print(f"测试日志文件: {test_log_path}")
print(f"性能统计查看器路径: {stats_viewer_path}")

if stats_viewer_path.exists():
    print("启动性能统计查看器...")
    # 启动独立的进程运行性能统计查看器
    process = subprocess.Popen([
        sys.executable,
        str(stats_viewer_path),
        str(test_log_path)
    ])
    
    print("性能统计查看器已启动，进程ID:", process.pid)
    print("按任意键退出...")
    input()
    
    # 终止进程
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
else:
    print(f"性能统计查看器脚本不存在: {stats_viewer_path}")

# 清理测试文件
test_log_path.unlink()
print("测试完成")