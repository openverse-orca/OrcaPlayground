# 12_slam：G1 人形机器人 LiDAR 建图 + A* 自主导航

双脚本示例：G1 行走策略接收速度指令行走，LiDAR 扫描建图，A* 全局规划导航到目标点。
两个脚本通过 **ROS2 topic** 通信，可分别独立重启。

```
OrcaStudio (G1 + LiDAR + 目标点)
   │ gRPC
   ▼
脚本 A  run_policy_service.py          脚本 B  lidar_slam_nav.py
  订阅 /cmd_vel → G1 行走策略            订阅 /odom（真值位姿）+ /goal（目标）
  发布 /odom（pelvis 位姿）              LiDAR 建图 + A* 规划 + 纯跟踪
  发布 /goal（目标点世界坐标）    ──────▶ 发布 /cmd_vel（速度指令）
```

## 环境配置

### 1. ROS2 Humble（系统级，apt 安装）

```bash
sudo apt install -y ros-humble-ros-base ros-humble-geometry-msgs ros-humble-nav-msgs
```

> **为什么必须 Humble + Python 3.10**：`rclpy` 含 C 扩展，与 Python 版本 ABI 强绑定。
> Humble 对应 Python 3.10，conda 环境必须用 3.10 创建，否则 rclpy 导入崩溃。

### 2. conda 环境 `ros2_bridge`

两个脚本都在此环境运行（既有 OrcaGym 仿真客户端，又有 rclpy）：

```bash
conda create -n ros2_bridge python=3.10 -y
conda activate ros2_bridge

# OrcaGym 发行包（提供 orca_gym.protos 的 pb2 模块与 gRPC 依赖）
pip install orca-gym

# 本示例依赖（BreezySLAM 不在 PyPI，从 GitHub 安装，见 requirements.txt 说明）
pip install onnxruntime matplotlib numpy
pip install "git+https://github.com/simondlevy/BreezySLAM#subdirectory=python"
```

### 3. ROS2 环境变量：无需 source

脚本里 `import ros2_bootstrap` 会**自动**完成 ROS2 环境注入（扫 `/opt/ros/*`、
设 `PYTHONPATH`/`LD_LIBRARY_PATH`、必要时 execv 重启自身一次），
所以**不需要** `source /opt/ros/humble/setup.bash`，直接 `python` 运行即可。

## 资产准备（OrcaStudio 关卡）

1. 场景中有 **1 台 G1**（沿用 `../08_locomotion` 的策略 ONNX 与配置）
2. G1 上附着 **LiDAR 实体**，实体名默认 `LiDAR`（可用 `--entity` 改）
3. 场景中放一个**目标标记点** site，名 `site1_site1`（脚本 A 自动查其世界坐标发布到 /goal；
   也可用 `--goal x,y` 手动指定，无需放标记点）

## 运行

```bash
conda activate ros2_bridge
cd examples/euler/12_slam

# 1. OrcaStudio 加载关卡并启动仿真后——
# 2. 脚本 A：行走策略服务（先启动）
python run_policy_service.py          # Ctrl+C 随时退出
                                      # 可选: --goal-site <site名> --addr HOST:PORT

# 3. 脚本 B：LiDAR 建图 + 导航（另开终端）
python lidar_slam_nav.py

# 常用可选参数
python lidar_slam_nav.py --goal 5,0 --map-meters 50 --no-viz
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--goal x,y` | /goal topic | 目标点（MuJoCo 世界坐标，米），手动指定优先 |
| `--entity NAME` | LiDAR | LiDAR 实体名 |
| `--addr HOST:PORT` | 127.0.0.1:50051 | OrcaStudio gRPC 地址 |
| `--map-meters N` | 50 | 地图边长（米），目标距起点超出半径会 WARN |
| `--no-viz` | 关 | 关闭 matplotlib 实时地图 |
| `--self-test` | - | 离线自测纯函数（无需 Studio） |

## ROS2 Topics

| Topic | 类型 | 方向 | 内容 |
|-------|------|------|------|
| `/cmd_vel` | geometry_msgs/Twist | B → A | 速度指令（linear.x 前进 m/s、angular.z 左转 rad/s） |
| `/odom` | nav_msgs/Odometry | A → B | pelvis 位姿 x/y/yaw（MuJoCo 真值） |
| `/goal` | geometry_msgs/PointStamped | A → B | 目标点世界坐标（latched，晚启动也能收到） |

## 导航原理（30 秒版）

1. **建图**：用 /odom 真值位姿把 LiDAR 扫描累积成占用栅格图（无定位漂移）；
   每束取**所有垂直层中最远回波**——天然丢弃打地面的回波伪环
2. **规划**：A*（障碍膨胀 0.45m）每 2s 重规划；前方受阻 0.5s 内立即重规划；
   未探明区域视为可通行 → 撞墙后自动绕行（规划式探索）
3. **跟踪**：纯跟踪沿路径走，瞄准路径前方 1.2m 处的目标点；
   到达目标 0.3m 内发零速（站立）

> BreezySLAM 仅用于可视化地图与漂移对比（`slam-odom=` 字段），不参与导航控制。
> 其坐标约定有多个反直觉的坑（FOV/2 偏移、1° 降采样、dtheta 符号），详见
> [lidar_slam_nav.py](lidar_slam_nav.py) 模块 docstring，**勿凭直觉修改**。

## 故障排查

| 现象 | 原因 / 处理 |
|------|------------|
| `ModuleNotFoundError: rclpy` | conda 环境不是 Python 3.10，或未装 ROS2 Humble（见环境配置） |
| `未找到 ROS2 安装` | `/opt/ros/<distro>/setup.bash` 不存在，先 apt 安装 |
| `LiDAR entity not found` | 场景中 LiDAR 实体名与 `--entity` 不符 |
| 长时间 `wait_goal` | 脚本 A 未启动，或场景中无 `site1_site1` 标记点（改用 `--goal`） |
| 目标超出地图半径 WARN | 加大 `--map-meters` |
| 一直 `no_path` 站立 | 目标被墙封死或地图未建好；观察可视化中黄色扫描点是否正常 |
| 导航乱走、可视化一圈伪墙 | 垂直层选错（打地面）；确认日志"垂直层选择"一行，A* 建图已用跨层融合免疫 |
