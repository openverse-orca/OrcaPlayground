#!/usr/bin/env python3
"""
测试性能统计查看器的启动逻辑
"""
import subprocess
import sys
from pathlib import Path
import time

# 模拟日志文件
log_content = """
[TIME_STATS] initializeCPUDataEachTimeStep=1.234 initializeGPUDataEachTimeStep=0.567
[TIME_STATS] initializeCPUDataEachTimeStep=1.123 initializeGPUDataEachTimeStep=0.456
[TIME_STATS] initializeCPUDataEachTimeStep=1.345 initializeGPUDataEachTimeStep=0.678
"""

# 创建测试日志文件
test_log_path = Path(".") / "test_performance_log.txt"
with open(test_log_path, "w") as f:
    f.write(log_content)

# 模拟run_fluid_sim.py中的路径构建逻辑
print("模拟run_fluid_sim.py中的路径构建逻辑...")
current_file = Path(__file__)
print(f"当前文件路径: {current_file}")
print(f"当前文件父目录: {current_file.parent}")
print(f"当前文件父父目录: {current_file.parent.parent}")
print(f"当前文件父父父目录: {current_file.parent.parent.parent}")

# 构建性能统计查看器的路径
project_root = current_file.parent.resolve()
stats_viewer_path = project_root / "envs" / "fluid_stats" / "performance_stats_viewer.py"

print(f"项目根目录: {project_root}")
print(f"性能统计查看器路径: {stats_viewer_path}")
print(f"路径是否存在: {stats_viewer_path.exists()}")

if stats_viewer_path.exists():
    # 启动独立的进程运行性能统计查看器
    print(f"启动性能统计查看器: {stats_viewer_path}")
    print(f"监控日志文件: {test_log_path}")
    # 确保使用正确的Python解释器
    python_exe = sys.executable
    print(f"使用Python解释器: {python_exe}")
    
    # 启动进程
    process = subprocess.Popen([
        python_exe,
        str(stats_viewer_path),
        str(test_log_path)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    print("性能统计查看器已启动，进程ID:", process.pid)
    
    # 等待一段时间，然后检查进程状态
    time.sleep(2)
    
    # 检查进程是否仍在运行
    if process.poll() is None:
        print("性能统计查看器进程正在运行")
    else:
        print(f"性能统计查看器进程已退出，退出码: {process.returncode}")
        # 打印错误信息
        stdout, stderr = process.communicate()
        if stdout:
            print("标准输出:", stdout)
        if stderr:
            print("标准错误:", stderr)
    
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