#!/usr/bin/env python3
"""
测试性能统计系统的完整流程
"""
import subprocess
import sys
import time
from pathlib import Path

# 1. 创建测试日志文件（模拟 SPlisHSPlasH 输出）
test_log_content = """[TIME_STATS] initializeCPUDataEachTimeStep=1.234 initializeGPUDataEachTimeStep=0.567 step=0
[TIME_STATS] initializeCPUDataEachTimeStep=1.123 initializeGPUDataEachTimeStep=0.456 step=1
[TIME_STATS] initializeCPUDataEachTimeStep=1.345 initializeGPUDataEachTimeStep=0.678 step=2
[TIME_STATS] initializeCPUDataEachTimeStep=1.256 initializeGPUDataEachTimeStep=0.589 step=3
[TIME_STATS] initializeCPUDataEachTimeStep=1.189 initializeGPUDataEachTimeStep=0.612 step=4
"""

test_log_path = Path(".") / "test_performance_system.log"
with open(test_log_path, "w") as f:
    f.write(test_log_content)

print(f"✅ 创建测试日志文件: {test_log_path}")
print(f"日志内容预览:")
print(test_log_content[:200] + "...")

# 2. 测试解析器
print("\n2. 测试性能统计解析器...")
sys.path.insert(0, str(Path(".") / "envs" / "fluid_stats"))
from performance_stats_parser import load_all_performance_records_from_file

records = load_all_performance_records_from_file(test_log_path)
print(f"✅ 解析到 {len(records)} 条记录")
if records:
    print(f"第一条记录: {records[0]}")
    print(f"所有键: {list(records[0].keys())}")

# 3. 启动图表查看器
print("\n3. 启动性能统计图表查看器...")
stats_viewer_path = Path(".") / "envs" / "fluid_stats" / "performance_stats_viewer.py"

if stats_viewer_path.exists():
    print(f"启动: {stats_viewer_path}")
    print(f"监控: {test_log_path}")
    
    process = subprocess.Popen([
        sys.executable,
        str(stats_viewer_path),
        str(test_log_path),
        "--interval", "0.5"
    ])
    
    print(f"✅ 图表查看器已启动，进程ID: {process.pid}")
    print("\n图表应该显示以下内容：")
    print("- X轴: Step (步骤序号)")
    print("- Y轴: Time (ms) (时间，毫秒)")
    print("- 曲线: initializeCPUDataEachTimeStep, initializeGPUDataEachTimeStep")
    print("\n按 Ctrl+C 退出...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
else:
    print(f"❌ 图表查看器不存在: {stats_viewer_path}")

# 清理
test_log_path.unlink()
print("✅ 测试完成")
