"""ros2_bootstrap — ROS2 环境自动注入模块（12_slam 示例公共组件）。

在导入 rclpy 之前 `import ros2_bootstrap`，自动完成：
1. 扫描 /opt/ros/ 下已安装的 ROS2 发行版（如 humble）
2. 将其 Python 包路径（dist-packages / site-packages）加入 PYTHONPATH，
   将 $ROS2_ROOT/lib 加入 LD_LIBRARY_PATH
3. 因 LD_LIBRARY_PATH 运行时修改不影响当前进程的 dlopen，
   通过 os.execv 重启自身使新环境变量生效（仅重启一次，幂等防循环）

本模块逻辑内联自 OrcaGym 仓库 orca_gym/tools/lidar_ros2_bridge.py 的
检测/注入逻辑（L38-77），适用于 ros2_bridge conda 环境
（Python 3.10 与 Humble 的 rclpy C 扩展 ABI 匹配）。

用法（必须是脚本中第一个本地 import，位于任何 rclpy 导入之前）:
    import ros2_bootstrap  # noqa: F401  (副作用模块)
    import rclpy
"""

import glob
import os
import sys


def _detect_ros2_root():
    """扫描 /opt/ros/ 下已安装的 ROS2 发行版，按字母序取首个有效安装。"""
    for candidate in sorted(glob.glob("/opt/ros/*")):
        if os.path.isfile(os.path.join(candidate, "setup.bash")):
            return candidate
    return None


def _detect_py_version(ros2_root):
    """从 ROS2 安装目录探测 Python 版本目录名（如 python3.10）。"""
    for pattern in ("local/lib/python3.*", "lib/python3.*"):
        matches = sorted(glob.glob(os.path.join(ros2_root, pattern)))
        if matches:
            return os.path.basename(matches[0])
    # 兜底：使用当前解释器版本
    return f"python{sys.version_info.major}.{sys.version_info.minor}"


def ensure_ros2():
    """注入 ROS2 环境变量；若需要则 execv 重启自身（仅一次）。"""
    ros2_root = _detect_ros2_root()
    if ros2_root is None:
        print("[ERROR] 未找到 ROS2 安装（/opt/ros/<distro>/setup.bash）。")
        print("  Ubuntu 22.04: sudo apt install -y ros-humble-ros-base "
              "ros-humble-geometry-msgs ros-humble-nav-msgs")
        sys.exit(1)

    py_ver = _detect_py_version(ros2_root)
    # Humble 布局：local/lib/python3.10/dist-packages + lib/python3.10/site-packages
    py_candidates = [
        f"{ros2_root}/local/lib/{py_ver}/dist-packages",
        f"{ros2_root}/lib/{py_ver}/site-packages",
    ]
    ros2_py_path = ":".join(p for p in py_candidates if os.path.isdir(p))

    need_restart = False
    if ros2_py_path and ros2_py_path not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = ros2_py_path + ":" + os.environ.get("PYTHONPATH", "")
        need_restart = True
    if ros2_root + "/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = (
            ros2_root + "/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
        )
        need_restart = True

    # LD_LIBRARY_PATH 在进程启动时被 ld.so 缓存，运行时修改 os.environ 不影响
    # 当前进程的 dlopen。通过 os.execv 重启自身，让新进程以正确环境变量启动。
    if need_restart and "_SLAM12_ROS2_BOOTSTRAPPED" not in os.environ:
        os.environ["_SLAM12_ROS2_BOOTSTRAPPED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)


ensure_ros2()
