"""12_slam 脚本 B：LiDAR 建图 + A* 全局导航 + 实时地图（教程示例）。

独立进程，作为 ROS2 节点 "lidar_slam_nav"：
- 订阅 /odom：策略服务发布的 pelvis 位姿（导航控制用其直推真值；
  BreezySLAM 融合位姿仅供建图可视化与漂移对比）
- 订阅 /goal：策略服务查询场景标记点后 latched 发布的世界坐标
- 发布 /cmd_vel：速度指令（linear.x=前进、angular.z=左转）

导航架构（规划式探索）:
    1. /odom 真值位姿 + LiDAR 跨垂直层融合（每束取最远回波，天然丢弃
       地面回波伪环）累积占用栅格图；未探明区域视为可通行
    2. A*（障碍膨胀 0.45m）周期性/前方受阻时重规划全局路径
    3. 纯跟踪沿路径走；到达目标（<0.3m）发全零指令（站立）

BreezySLAM 坐标约定（经 coreslam.c 源码分析 + 合成实验验证，勿改）:
    - 内部为标准数学系（y 上、theta CCW 正）；getpos() 返回相对地图
      左上角的 mm 坐标，转物理系需减 half_m
    - pose_change 的 dtheta 为 CCW 正（与 odom yaw 同号，不取负）
    - interpolate_scan 假定内部束 k 角度 = k 度，而 scan_update_xy 按
      -FOV/2+k 放置 → 传 (物理角 + FOV/2) % FOV
    - 该"k 度"网格仅在束间隔≈1° 时自洽：高分辨率 LiDAR（3600 bins
      × 0.1°）必须降采样到 1°/束，否则伪墙圆环毁掉地图

运行环境: ros2_bridge conda 环境（ros2_bootstrap 自动注入 Humble 路径）。

用法:
    # 前置：OrcaStudio 运行含 G1 + LiDAR 实体 + 标记点 site1_site1 的关卡，
    # 策略服务（run_policy_service.py）已启动并发布 /goal
    /home/orca/miniconda3/envs/ros2_bridge/bin/python \\
        examples/euler/12_slam/lidar_slam_nav.py

    # 手动指定目标（MuJoCo 世界坐标，米，覆盖 /goal）
    ... lidar_slam_nav.py --goal 5,0

    # 离线自测（无 Studio，测纯函数）
    ... lidar_slam_nav.py --self-test
"""

from __future__ import annotations

import argparse
import heapq
import math
import sys
import time

import ros2_bootstrap  # noqa: F401  ROS2 环境注入（必须先于 rclpy 导入）

import numpy as np

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

# --- 导航参数 ---
HZ = 10  # 主循环频率（Hz）
GOAL_TOL = 0.3  # 到达判定距离（m）
STOP_DIST = 0.8  # 前方受阻停车距离（m）
INFLATE = 0.45  # A* 障碍膨胀半径（m，机器人半径+余量）
REPLAN_S = 2.0  # 周期重规划间隔（s）；受阻时 0.5s 节流内立即重规划
WARMUP_S = 2.0  # 启动后站立建图时间（s）
MAP_PIXELS = 800  # 地图分辨率（像素）
SIGMA_XY_MM, SIGMA_TH_DEG = 50, 8  # RMHC 位姿搜索噪声（odom 可靠，取小）

# 长回波判定距离（m）：近水平层能看到远墙（长回波多），打地层回波短
_LONG_ECHO_M = 3.0


# ---------------------------------------------------------------------------
# gRPC LiDAR 查询（与 orca_gym/tools/lidar_ros2_bridge.py 同模式）
# ---------------------------------------------------------------------------

def query_lidar(stub, entity_name):
    """查询 LiDAR 点云，返回 dict（失败返回 None）。

    返回结构:
        bin_count, vertical_layers, angular_resolution(rad/束),
        max_h_angle(rad), min_range/max_range(m),
        ranges (bin_count, vertical_layers)（-1 无效）
        # 注意：角度字段单位是弧度，main 中用 math.degrees 换算为度
    """
    from orca_gym.protos import mjc_message_pb2

    request = mjc_message_pb2.LiDARPointCloudRequest(entity_name=entity_name)
    try:
        response = stub.QueryLiDARPointCloud(request, timeout=2.0)
    except Exception as e:  # grpc.RpcError
        print(f"[WARN] gRPC query failed: {e}")
        return None

    if response.status == mjc_message_pb2.LiDARPointCloudResponse.ENTITY_NOT_FOUND:
        print(f"[ERROR] LiDAR entity not found: {entity_name}")
        return None
    if response.status == mjc_message_pb2.LiDARPointCloudResponse.NO_DATA:
        return None

    result = {
        "bin_count": response.bin_count,
        "vertical_layers": response.vertical_layers,
        "angular_resolution": response.angular_resolution,
        "max_h_angle": response.max_h_angle,
        "min_range": response.min_range,
        "max_range": response.max_range,
    }

    if response.range_data:
        ranges = np.frombuffer(response.range_data, dtype=np.float32).copy()
        result["ranges"] = ranges.reshape(
            response.bin_count, response.vertical_layers
        )
    else:
        result["ranges"] = np.full(
            (response.bin_count, response.vertical_layers), -1.0, dtype=np.float32
        )
    return result


def select_scan_layer(ranges: np.ndarray) -> int:
    """选择用于 2D SLAM 的垂直层：长回波最多的层（近水平、能看到远墙）。

    最低层（index 0）通常下俯打向地面：地面回波在 2D 地图上构成圆环
    伪墙。注意开阔场景该启发式可靠，长廊内各层长回波都少时会退化；
    A* 建图不受影响（用跨层融合，见 fuse_vertical_layers）。
    """
    return int(np.argmax((ranges > _LONG_ECHO_M).sum(axis=0)))


def make_valid_ranges(ranges: np.ndarray, min_range: float, max_range: float) -> np.ndarray:
    """把 LiDAR ranges 转为有效距离数组（无效点为 inf）。"""
    out = np.asarray(ranges, dtype=np.float64).copy()
    out[(out < min_range) | (out > max_range)] = math.inf
    return out


def fuse_vertical_layers(ranges: np.ndarray) -> np.ndarray:
    """跨垂直层融合：每束取最远回波 — A* 建图防地面伪环（纯函数）。

    墙面回波(远)保留、地面回波(近)丢弃。单层选择在长廊内会因长回波
    稀少而退化为最低层（下俯打地面），地面环会封死 A*；取最远回波
    天然选中近水平层的墙面距离，不依赖层选择。

    ranges 形状 (束数, 垂直层)，无效为 -1；全层无效的束返回 -1。
    """
    r = np.asarray(ranges, dtype=np.float64)
    if r.ndim == 2 and r.shape[1] > 1:
        return np.max(r, axis=1)
    return r.ravel()


# ---------------------------------------------------------------------------
# 全局规划：真值建图 + A* + 纯跟踪
# ---------------------------------------------------------------------------

class OccupancyGridMap:
    """LiDAR 扫描累积占用栅格地图（odom 真值位姿锚定，供 A* 规划）。

    与 BreezySLAM 的估计位姿建图不同：本地图用 /odom 直推位姿（MuJoCo
    真值）累积扫描，地图与导航控制天然同帧、无定位漂移。坐标为物理系
    （原点=启动位置，x 前 y 左）。

    counts: int16 栅格，命中 +3、射线穿越 -1，clip [-4, +4]；>= OCC_TH
    视为占用。未知（0）视为可通行——配合"前方受阻即重规划"形成
    规划式探索：A* 可穿过未探明区域规划，撞墙后地图更新、自动绕行。
    """

    HIT = 3
    MISS = -1
    CLIP = 4
    OCC_TH = 2

    def __init__(self, size_meters: float, resolution: float = 0.1):
        self.size = size_meters
        self.res = resolution
        self.n = int(round(size_meters / resolution))
        self.half = size_meters / 2.0
        self.counts = np.zeros((self.n, self.n), dtype=np.int16)

    def world_cell(self, xy: tuple[float, float]) -> tuple[int, int]:
        """物理系坐标 → 栅格 (iy, ix)。"""
        return (
            int((xy[1] + self.half) / self.res),
            int((xy[0] + self.half) / self.res),
        )

    def cell_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        """栅格 (iy, ix) 中心 → 物理系坐标。"""
        return (
            (cell[1] + 0.5) * self.res - self.half,
            (cell[0] + 0.5) * self.res - self.half,
        )

    def _cells(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        ix = ((np.asarray(xs) + self.half) / self.res).astype(np.int64)
        iy = ((np.asarray(ys) + self.half) / self.res).astype(np.int64)
        ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        return np.stack([iy[ok], ix[ok]], axis=1)  # (m, 2)

    def update(
        self,
        pose: tuple[float, float, float],
        ranges_m: np.ndarray,
        angles_deg: np.ndarray,
    ) -> None:
        """用一帧扫描更新地图：射线穿越 -1，命中端点 +3。"""
        x, y, th = pose
        finite = np.isfinite(ranges_m)
        if not finite.any():
            return
        r = ranges_m[finite]
        a = np.radians(angles_deg[finite]) + th
        ca, sa = np.cos(a), np.sin(a)
        # 射线穿越打空闲点（每束 20 个采样，不含端点）
        fr = np.linspace(0.05, 0.95, 20)
        px = x + np.outer(r, fr) * ca[:, None]
        py = y + np.outer(r, fr) * sa[:, None]
        cells = self._cells(px.ravel(), py.ravel())
        np.add.at(self.counts, (cells[:, 0], cells[:, 1]), self.MISS)
        # 命中端点
        hcells = self._cells(x + r * ca, y + r * sa)
        np.add.at(self.counts, (hcells[:, 0], hcells[:, 1]), self.HIT)
        np.clip(self.counts, -self.CLIP, self.CLIP, out=self.counts)

    def occupancy(self) -> np.ndarray:
        """占用栅格（bool）。"""
        return self.counts >= self.OCC_TH

    def inflated(self, radius_m: float) -> np.ndarray:
        """占用栅格膨胀（机器人半径+安全边距），用于 A*。"""
        occ = self.occupancy()
        k = int(round(radius_m / self.res))
        if k <= 0:
            return occ
        out = occ.copy()
        h, w = occ.shape
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                if di == 0 and dj == 0:
                    continue
                shifted = np.zeros_like(occ)
                dst_y = slice(max(-di, 0), h - max(di, 0))
                src_y = slice(max(di, 0), h - max(-di, 0))
                dst_x = slice(max(-dj, 0), w - max(dj, 0))
                src_x = slice(max(dj, 0), w - max(-dj, 0))
                shifted[dst_y, dst_x] = occ[src_y, src_x]
                np.bitwise_or(out, shifted, out=out)
        return out


def astar_path(
    occ: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    max_expand: int = 40000,
) -> list[tuple[int, int]] | None:
    """8 邻域 A*（octile 启发，对角不穿角）。occ[iy, ix] True = 不可通行。

    对角移动要求两侧正交格均可行（禁止从两个对角障碍间斜切穿过）。
    返回 (iy, ix) 栅格路径（含首尾），不可达 / 起终点占用返回 None。
    max_expand 封顶扩张量：目标不可达时快速失败（≈120ms），
    防止全图扩张（160k 格 ≈ 0.5s）拖慢 10Hz 主循环。
    """
    h, w = occ.shape
    if not (
        0 <= start[0] < h and 0 <= start[1] < w
        and 0 <= goal[0] < h and 0 <= goal[1] < w
    ):
        return None
    if occ[start] or occ[goal]:
        return None
    sqrt2 = math.sqrt(2.0)

    def heur(a: tuple[int, int], b: tuple[int, int]) -> float:
        dx = abs(a[1] - b[1])
        dy = abs(a[0] - b[0])
        return (dx + dy) + (1.0 - sqrt2) * min(dx, dy)

    open_heap: list[tuple[float, float, tuple[int, int]]] = [
        (heur(start, goal), 0.0, start)
    ]
    g_score = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    closed = np.zeros_like(occ, dtype=bool)
    neighbors = (
        (-1, -1, sqrt2), (-1, 0, 1.0), (-1, 1, sqrt2),
        (0, -1, 1.0), (0, 1, 1.0),
        (1, -1, sqrt2), (1, 0, 1.0), (1, 1, sqrt2),
    )
    expanded = 0
    while open_heap and expanded < max_expand:
        _, g_cur, cur = heapq.heappop(open_heap)
        if closed[cur]:
            continue
        closed[cur] = True
        expanded += 1
        if cur == goal:
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = cur
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]
        for di, dj, cost in neighbors:
            nxt = (cur[0] + di, cur[1] + dj)
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                continue
            if occ[nxt] or closed[nxt]:
                continue
            if di != 0 and dj != 0 and (
                occ[cur[0] + di, cur[1]] or occ[cur[0], cur[1] + dj]
            ):
                continue  # 对角穿角禁止：两侧正交格须可行
            ng = g_cur + cost
            if ng < g_score.get(nxt, math.inf):
                g_score[nxt] = ng
                parent[nxt] = cur
                heapq.heappush(open_heap, (ng + heur(nxt, goal), ng, nxt))
    return None


def nearest_free(
    occ: np.ndarray,
    cell: tuple[int, int],
    max_r: int = 8,
) -> tuple[int, int] | None:
    """找 cell 周围最近的可行栅格（起终点落在膨胀区时用）。"""
    h, w = occ.shape
    for r in range(max_r + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                c = (cell[0] + di, cell[1] + dj)
                if 0 <= c[0] < h and 0 <= c[1] < w and not occ[c]:
                    return c
    return None


def pure_pursuit_cmd(
    pose: tuple[float, float, float],
    path: list[tuple[float, float]],
    lookahead: float = 1.2,
    cruise: float = 0.4,
    turn_gain: float = 2.0,
    max_w: float = 0.5,
    align_slow_rad: float = 0.5,
) -> tuple[float, float, float]:
    """沿路径纯跟踪：瞄准沿路径前方 ~lookahead 弧长处的点（纯函数）。

    目标点选择：先找路径上离机器人最近的点，再沿路径**向前**积累弧长
    到 lookahead 处。不能从头扫第一个 ≥lookahead 的点——机器人过弯
    冲过路径时，那会是身后的点（方位 ≈180°）→ vx 压 0 纯原地转
    （G1 行走策略对纯转向弱响应）→ 卡死振荡。

    path 为物理系 (x, y) 序列（从近到远），返回 (vx, vy, w)。
    """
    x, y, th = pose
    i_near = 0
    d_near = math.inf
    for i, (px, py) in enumerate(path):
        d = math.hypot(px - x, py - y)
        if d < d_near:
            d_near = d
            i_near = i
    target = path[-1]
    acc = 0.0
    for j in range(i_near + 1, len(path)):
        acc += math.hypot(path[j][0] - path[j - 1][0], path[j][1] - path[j - 1][1])
        if acc >= lookahead:
            target = path[j]
            break
    hd = math.atan2(target[1] - y, target[0] - x) - th
    hd = (hd + math.pi) % (2 * math.pi) - math.pi
    w = max(-max_w, min(max_w, turn_gain * hd))
    vx = cruise if abs(hd) < align_slow_rad else cruise * 0.3
    return vx, 0.0, w


def front_min_distance(
    ranges_m: np.ndarray,
    angles_deg: np.ndarray,
    half_angle: float = 30.0,
) -> float:
    """前方扇区最小有效距离（m），无有效束返回 inf。"""
    ang_norm = (angles_deg + 180.0) % 360.0 - 180.0
    mask = np.isfinite(ranges_m) & (np.abs(ang_norm) <= half_angle)
    return float(np.min(ranges_m[mask])) if mask.any() else math.inf


def world_goal_to_slam(
    goal_xy: tuple[float, float],
    odom_init: tuple[float, float, float],
) -> tuple[float, float]:
    """MuJoCo 世界坐标目标点 → 物理系坐标（纯函数）。

    物理系定义：原点=启动时机器人位置，x 轴=初始朝向，y 轴=初始朝向
    左侧（右手系）。初始朝向由首帧 /odom 的 yaw0 给出。
    """
    x0, y0, yaw0 = odom_init
    dx = goal_xy[0] - x0
    dy = goal_xy[1] - y0
    cos0, sin0 = math.cos(yaw0), math.sin(yaw0)
    # 沿初始朝向的分量 → u；沿初始朝向左侧的分量 → v
    return (dx * cos0 + dy * sin0, -dx * sin0 + dy * cos0)


# ---------------------------------------------------------------------------
# matplotlib 实时地图
# ---------------------------------------------------------------------------

class SlamMapVisualizer:
    """BreezySLAM 占用栅格地图实时可视化（matplotlib ion）。

    显示坐标系：物理系（m），原点=启动时机器人位置（画面中心）。
    SLAM 地图字节 row 0 = y 最小（数学系底部），origin='lower' 直接
    正确显示，无需翻转。
    """

    def __init__(self, map_pixels: int, map_meters: float):
        import matplotlib.pyplot as plt

        self.n = map_pixels
        self.size = map_meters
        self.plt = plt
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.img_artist = None
        (self.traj_artist,) = self.ax.plot([], [], "b-", lw=1.0, label="trajectory")
        (self.robot_artist,) = self.ax.plot(
            [], [], "go", ms=8, label="robot"
        )
        (self.goal_artist,) = self.ax.plot(
            [], [], "r*", ms=14, label="goal"
        )
        (self.scan_artist,) = self.ax.plot([], [], "y.", ms=1, label="scan")
        (self.path_artist,) = self.ax.plot(
            [], [], "c-", lw=2.0, alpha=0.7, label="A* path"
        )
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.legend(loc="upper right", fontsize=8)
        self.fig.canvas.manager.set_window_title("12_slam map")

    def update(
        self,
        mapbytes: bytearray,
        traj: list[tuple[float, float]],
        pose: tuple[float, float, float],
        goal: tuple[float, float] | None,
        scan_pts_xy: np.ndarray | None,
        path: list[tuple[float, float]] | None = None,
    ) -> None:
        """刷新地图、轨迹、机器人、目标点、当前扫描点、规划路径。"""

        def set_pts(art, pts) -> None:
            """把 (N,2) 点序列刷到 artist（空则清空）。"""
            if pts is not None and len(pts) > 0:
                p = np.asarray(pts)
                art.set_data(p[:, 0], p[:, 1])
            else:
                art.set_data([], [])

        arr = np.frombuffer(bytes(mapbytes), dtype=np.uint8).reshape(self.n, self.n)
        half = self.size / 2.0
        if self.img_artist is None:
            self.img_artist = self.ax.imshow(
                arr,
                cmap="gray",
                vmin=0,
                vmax=255,
                extent=(-half, half, -half, half),
                origin="lower",
            )
            self.ax.set_xlim(-half, half)
            self.ax.set_ylim(-half, half)
        else:
            self.img_artist.set_data(arr)

        set_pts(self.traj_artist, traj)
        self.robot_artist.set_data([pose[0]], [pose[1]])
        set_pts(self.goal_artist, [goal] if goal is not None else None)
        set_pts(self.scan_artist, scan_pts_xy)
        set_pts(self.path_artist, path)
        self.ax.set_title(
            f"pose=({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f}deg)"
        )
        self.plt.pause(0.001)

    def close(self) -> None:
        self.plt.close(self.fig)


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="12_slam 脚本 B: LiDAR 建图 + A* 全局导航"
    )
    parser.add_argument(
        "--addr", default="127.0.0.1:50051", help="OrcaStudio gRPC 地址"
    )
    parser.add_argument(
        "--entity", default="LiDAR", help="LiDAR 实体名（OrcaLab 场景中的 Entity）"
    )

    def check_goal(value: str) -> tuple[float, float]:
        """--goal 参数解析并校验格式（必须是 x,y 两个数值）。"""
        values = [float(v) for v in value.split(",")]
        if len(values) != 2:
            raise argparse.ArgumentTypeError(f"格式应为 x,y（收到: {value!r}）")
        return (values[0], values[1])


    parser.add_argument("--goal", type=check_goal, default=None,
                        help="目标点 x,y（MuJoCo 世界坐标，米；默认用 /goal）")
    parser.add_argument(
        "--map-meters", type=float, default=50, help="地图边长（米）"
    )
    parser.add_argument("--no-viz", action="store_true", help="关闭 matplotlib 可视化")
    parser.add_argument(
        "--self-test", action="store_true", help="离线自测纯函数后退出"
    )
    return parser.parse_args()


def run_self_test() -> int:
    """离线自测：模块内全部纯函数。"""
    # 垂直层自动选择 — 层 0 打地（短回波），层 7 近水平（长回波）
    r = np.full((100, 16), -1.0)
    r[:, 0] = 0.5 + 0.1 * np.random.rand(100)
    r[:, 7] = 4.0 + np.random.rand(100) * 6.0
    assert select_scan_layer(r) == 7
    print("case 1 垂直层选择: PASS")

    # 世界坐标 goal 转换 — 朝向 +x 恒等；朝向 +y（yaw0=90°）时 (3,4) → (4,-3)
    got = world_goal_to_slam((3.0, 4.0), (0.0, 0.0, 0.0))
    assert abs(got[0] - 3.0) < 1e-9 and abs(got[1] - 4.0) < 1e-9, got
    got = world_goal_to_slam((3.0, 4.0), (0.0, 0.0, math.pi / 2))
    assert abs(got[0] - 4.0) < 1e-9 and abs(got[1] + 3.0) < 1e-9, got
    print("case 2 世界坐标转换: PASS")

    # 真值建图 — 单束正前 2m 命中 → 端点占用、沿途空闲、膨胀覆盖邻域
    g = OccupancyGridMap(5.0, 0.1)
    g.update(
        (0.0, 0.0, 0.0),
        np.array([math.inf, 2.0, math.inf]),
        np.array([-10.0, 0.0, 10.0]),
    )
    occ = g.occupancy()
    assert occ[g.world_cell((2.0, 0.0))], "endpoint should be occupied"
    assert not occ[g.world_cell((1.0, 0.0))], "ray should be free"
    assert g.inflated(0.3)[g.world_cell((2.0, 0.3))], "inflation covers neighbor"
    assert not g.inflated(0.3)[g.world_cell((0.0, 0.0))], "robot cell stays free"
    print("case 3 真值建图+膨胀: PASS")

    # A* — L 形墙（底部留缺口）→ 绕缺口不穿墙；全墙封死 → None
    occ = np.zeros((40, 40), dtype=bool)
    occ[0:25, 20] = True
    p = astar_path(occ, (5, 5), (5, 35))
    assert p is not None and p[0] == (5, 5) and p[-1] == (5, 35)
    assert not any(occ[c] for c in p), "path crosses wall"
    occ[0:40, 20] = True
    assert astar_path(occ, (5, 5), (5, 35)) is None
    print("case 4 A* 绕障/封死: PASS")

    # 纯跟踪 — 直线正前 → 直行；过弯冲出路径（最近点在身后）→
    # 目标须沿路径前方（左转），不得选身后点（会纯原地转卡死）
    vx, _, w = pure_pursuit_cmd((0, 0, 0), [(1, 0), (2, 0), (3, 0)])
    assert vx > 0 and abs(w) < 0.05, (vx, w)
    path = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (2.0, 4.0)]
    _, _, w = pure_pursuit_cmd((2.5, 0.5, 0.0), path)
    assert w > 0, f"过弯冲出应左转（沿路径前向）: w={w}"
    print("case 5 纯跟踪: PASS")

    # 跨层融合 — 每束取最远回波：地面环(近)丢弃、墙面(远)保留
    rr = np.array([[-1.0, 1.5, 2.8], [-1.0, 1.2, 3.0]])
    f = fuse_vertical_layers(rr)
    assert abs(f[0] - 2.8) < 1e-9 and abs(f[1] - 3.0) < 1e-9, f
    assert (fuse_vertical_layers(np.full((2, 3), -1.0)) == -1.0).all()
    print("case 6 跨垂直层融合: PASS")

    print("self-test all PASS")
    return 0


def main() -> None:
    args = parse_args()

    if args.self_test:
        sys.exit(run_self_test())

    import grpc
    from orca_gym.protos import mjc_message_pb2_grpc
    from breezyslam.algorithms import RMHC_SLAM
    from breezyslam.sensors import Laser

    goal_world: tuple[float, float] | None = args.goal  # check_goal 已校验
    if goal_world is not None:
        print(f"[INFO] 使用手动 --goal（MuJoCo 世界坐标）: {goal_world}")
    half_m = args.map_meters / 2.0

    # --- gRPC 通道（大消息配置）---
    channel = grpc.insecure_channel(
        args.addr,
        options=[
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),
        ],
    )
    stub = mjc_message_pb2_grpc.GrpcServiceStub(channel)

    # --- ROS2 节点 ---
    rclpy.init()
    node = rclpy.create_node("lidar_slam_nav")
    cmd_pub = node.create_publisher(Twist, "/cmd_vel", 10)
    odom_state: dict = {"x": None, "y": None, "yaw": None, "t": None}
    # 首帧 /odom 即机器人初始位姿（世界系），用于 goal 坐标转换
    odom_init: tuple[float, float, float] | None = None
    goal_state: dict = {"world": goal_world, "slam": None}

    def odom_cb(msg: Odometry) -> None:
        nonlocal odom_init
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        odom_state["x"] = msg.pose.pose.position.x
        odom_state["y"] = msg.pose.pose.position.y
        odom_state["yaw"] = yaw
        odom_state["t"] = time.time()
        if odom_init is None:
            odom_init = (odom_state["x"], odom_state["y"], yaw)
            print(f"[INFO] 初始位姿 /odom: {odom_init}")

    def goal_cb(msg: PointStamped) -> None:
        # --goal 参数手动指定的世界坐标优先，不覆盖
        if goal_state["world"] is None:
            goal_state["world"] = (msg.point.x, msg.point.y)
            print(
                f"[INFO] 收到 /goal（MuJoCo 世界坐标）: "
                f"({msg.point.x:.2f}, {msg.point.y:.2f})"
            )

    node.create_subscription(Odometry, "/odom", odom_cb, 10)
    node.create_subscription(
        PointStamped,
        "/goal",
        goal_cb,
        QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
    )

    # --- 等待首帧 LiDAR 数据，动态构建 Laser 模型 ---
    print(f"[INFO] 等待 LiDAR 数据（entity={args.entity}, addr={args.addr}）...")
    data = None
    while data is None:
        data = query_lidar(stub, args.entity)
        if data is None:
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.5)

    # 注意：gRPC 响应中 angular_resolution / max_h_angle 单位为弧度，
    # BreezySLAM 需要度，必须换算
    fov_deg = math.degrees(data["max_h_angle"])
    res_deg = math.degrees(data["angular_resolution"])

    # 2D SLAM 用单垂直层（自动选长回波最多的近水平层）
    v_layer = select_scan_layer(data["ranges"])
    print(
        f"[INFO] 垂直层选择: layer {v_layer}/{data['vertical_layers']} "
        f"(各层长回波数: {(data['ranges'] > _LONG_ECHO_M).sum(axis=0).tolist()})"
    )

    # 降采样到 1°/束（原因见模块 docstring 的 BreezySLAM 坐标约定）
    scan_n = max(1, int(round(fov_deg)))
    sample_step = max(1, int(round(data["bin_count"] / scan_n)))
    scan_n = max(1, data["bin_count"] // sample_step)
    sample_idx = (np.arange(scan_n) * sample_step) % data["bin_count"]

    laser = Laser(
        scan_size=scan_n,
        scan_rate_hz=HZ,
        detection_angle_degrees=fov_deg,
        distance_no_detection_mm=int(data["max_range"] * 1000),
    )
    slam = RMHC_SLAM(
        laser,
        MAP_PIXELS,
        args.map_meters,
        sigma_xy_mm=SIGMA_XY_MM,
        sigma_theta_degrees=SIGMA_TH_DEG,
    )

    # LiDAR 束角度：物理系（0=机器人前方，CCW 正）。bin k 的局部角 = k·res
    # （引擎 shader：lidarDir=(cos h, sin h, sin v)，与机器人系一致）
    angles_phys = np.degrees(sample_idx * data["angular_resolution"])
    # BreezySLAM interpolate_scan 假定内部束 k 角=k 度，而 scan_update_xy
    # 按 -FOV/2+k 放置 → 相差 FOV/2，故传 (物理角+FOV/2)%FOV
    angles_slam = (angles_phys + fov_deg / 2.0) % fov_deg
    no_detect_mm = int(data["max_range"] * 1000)
    min_range, max_range = data["min_range"], data["max_range"]

    print(
        f"[INFO] LiDAR 模型: {data['bin_count']} bins x "
        f"{data['vertical_layers']} layers, FOV {fov_deg:.0f}deg, "
        f"束间隔 {res_deg:.3f}deg, range [{min_range}, {max_range}]m"
    )

    viz = None if args.no_viz else SlamMapVisualizer(MAP_PIXELS, args.map_meters)

    def publish_cmd(vx: float, vy: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = w
        cmd_pub.publish(msg)

    def odom_delta_pose():
        """首帧 /odom 锚定的直推物理系位姿（真值，无漂移）；无 odom 返回 None。"""
        if odom_init is None or odom_state["t"] is None:
            return None
        ddx = odom_state["x"] - odom_init[0]
        ddy = odom_state["y"] - odom_init[1]
        c0, s0 = math.cos(odom_init[2]), math.sin(odom_init[2])
        oth = odom_state["yaw"] - odom_init[2]
        oth = (oth + math.pi) % (2 * math.pi) - math.pi
        return (ddx * c0 + ddy * s0, -ddx * s0 + ddy * c0, oth)

    def should_replan(now_t: float, fmin: float) -> bool:
        """重规划触发：无路径 / 周期到 / 前方受阻（均节流 0.5s，
        防 A* 高频重算卡顿）。"""
        if plan_path is None:
            return now_t - last_plan_t > 0.5
        return (
            now_t - last_plan_t > REPLAN_S
            or (fmin < STOP_DIST and now_t - last_plan_t > 0.5)
        )

    def do_replan(now_t: float) -> None:
        """A* 全局重规划：膨胀图上起终点各找最近可行格后求解。"""
        nonlocal plan_path, last_plan_t
        occ_inf = grid_map.inflated(INFLATE)
        s_cell = nearest_free(
            occ_inf, grid_map.world_cell((nav_pose[0], nav_pose[1]))
        )
        g_cell = nearest_free(occ_inf, grid_map.world_cell(goal_slam))
        cells = (
            astar_path(occ_inf, s_cell, g_cell)
            if s_cell is not None and g_cell is not None
            else None
        )
        plan_path = (
            [grid_map.cell_world(c) for c in cells[::2]]
            + [grid_map.cell_world(cells[-1])]
            if cells
            else None
        )
        last_plan_t = now_t

    # --- SLAM 主循环 ---
    period = 1.0 / HZ
    start_time = time.time()
    prev_odom: tuple[float, float, float, float] | None = None
    traj: list[tuple[float, float]] = []
    mapbytes = bytearray(MAP_PIXELS * MAP_PIXELS)
    arrived = False
    grid_map = OccupancyGridMap(args.map_meters)
    plan_path: list[tuple[float, float]] | None = None
    last_plan_t = 0.0
    frame_count = 0

    print(f"[INFO] SLAM 导航启动: warmup={WARMUP_S}s, 等待 /goal 与首帧 /odom...")
    try:
        while rclpy.ok():
            loop_start = time.monotonic()

            # 处理 /odom、/goal 回调（非阻塞）
            rclpy.spin_once(node, timeout_sec=0.0)

            # goal 就绪判定：世界坐标 + 初始位姿 → 物理系坐标（只算一次）
            goal_slam = goal_state["slam"]
            if (
                goal_slam is None
                and goal_state["world"] is not None
                and odom_init is not None
            ):
                goal_slam = world_goal_to_slam(goal_state["world"], odom_init)
                goal_state["slam"] = goal_slam
                print(f"[INFO] 目标点（物理系）: {goal_slam}")
                if math.hypot(*goal_slam) > half_m:
                    print(
                        f"[WARN] 目标距起点 {math.hypot(*goal_slam):.1f}m "
                        f"超出地图半径 {half_m:.1f}m，请增大 --map-meters"
                    )

            # 查询 LiDAR（所选垂直层）+ 降采样到 1°/束
            data = query_lidar(stub, args.entity)
            if data is None:
                time.sleep(0.1)
                continue
            ranges_2d = data["ranges"][sample_idx, v_layer]

            # 里程计差分 → pose_change（mm, deg, s）
            pose_change = None
            if odom_state["t"] is not None:
                cur_odom = (
                    odom_state["x"], odom_state["y"], odom_state["yaw"],
                    odom_state["t"],
                )
                if prev_odom is not None:
                    dxy_mm = math.hypot(
                        cur_odom[0] - prev_odom[0], cur_odom[1] - prev_odom[1]
                    ) * 1000.0
                    # yaw 差分 wrap 到 [-pi, pi]，避免穿越 ±180° 时跳 360°
                    dyaw = cur_odom[2] - prev_odom[2]
                    dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
                    # BreezySLAM dtheta 为 CCW 正（与 odom yaw 同号，不取负）
                    dt = max(cur_odom[3] - prev_odom[3], 1e-3)
                    pose_change = (dxy_mm, math.degrees(dyaw), dt)
                prev_odom = cur_odom

            # 距离转 mm，无效点置 no_detect（BreezySLAM 忽略）
            valid_ranges = make_valid_ranges(ranges_2d, min_range, max_range)
            dist_mm = [
                int(r * 1000.0) if math.isfinite(r) else no_detect_mm
                for r in valid_ranges
            ]

            slam.update(
                dist_mm, pose_change, scan_angles_degrees=angles_slam.tolist()
            )
            x_mm, y_mm, theta_deg = slam.getpos()
            # getpos() 相对地图左上角（mm）→ 物理系（中心原点，y 上，CCW 正）
            pose = (
                x_mm / 1000.0 - half_m,
                y_mm / 1000.0 - half_m,
                math.radians(theta_deg),
            )

            # 导航控制用 odom 直推位姿（真值，无漂移）；SLAM 位姿仅用于
            # 建图可视化与漂移对比（首帧 /odom 未到时暂用 SLAM 位姿）
            odom_pose = odom_delta_pose()
            nav_pose = odom_pose if odom_pose is not None else pose

            # 导航决策（goal 未就绪 / warmup 内原地零速等待）
            plan_used = False
            if goal_slam is None or time.time() - start_time < WARMUP_S:
                vx, vy, w, arrived = 0.0, 0.0, 0.0, False
            else:
                # 真值位姿建图（与控制同帧）；跨垂直层融合根治地面伪环
                valid_fused = make_valid_ranges(
                    fuse_vertical_layers(data["ranges"][sample_idx, :]),
                    min_range,
                    max_range,
                )
                grid_map.update(nav_pose, valid_fused, angles_phys)
                fmin = front_min_distance(valid_fused, angles_phys)
                if should_replan(time.time(), fmin):
                    do_replan(time.time())
                if plan_path is not None:
                    vx, vy, w = pure_pursuit_cmd(nav_pose, plan_path)
                    arrived = math.hypot(
                        goal_slam[0] - nav_pose[0],
                        goal_slam[1] - nav_pose[1],
                    ) < GOAL_TOL
                    if fmin < STOP_DIST and not arrived:
                        vx = 0.0  # 前方受阻：停下等立即重规划绕行
                    if arrived:
                        vx, vy, w = 0.0, 0.0, 0.0
                    plan_used = True
                else:
                    vx, vy, w, arrived = 0.0, 0.0, 0.0, False
            publish_cmd(vx, vy, w)

            frame_count += 1
            if frame_count % HZ == 0:  # 每秒一条状态
                if goal_slam is None:
                    status = "wait_goal"
                elif arrived:
                    status = "ARRIVED"
                elif plan_used:
                    status = "PLAN"
                else:
                    status = "no_path"  # A* 暂未成功（等待建图/重规划）
                extra = ""
                if plan_path is not None:
                    extra += f" plan={len(plan_path)}wp"
                if goal_slam is not None:
                    brg = math.degrees(math.atan2(
                        goal_slam[1] - nav_pose[1], goal_slam[0] - nav_pose[0]
                    ))
                    dist = math.hypot(
                        goal_slam[0] - nav_pose[0], goal_slam[1] - nav_pose[1]
                    )
                    extra += f" goal_brg={brg:.0f}deg dist={dist:.1f}m"
                if odom_pose is not None:
                    drift = math.hypot(
                        pose[0] - odom_pose[0], pose[1] - odom_pose[1]
                    )
                    extra += (
                        f" odom=({odom_pose[0]:.2f},{odom_pose[1]:.2f},"
                        f"{math.degrees(odom_pose[2]):.0f}deg)"
                        f" slam-odom={drift:.2f}m"
                    )
                print(
                    f"[{status}] pose=({pose[0]:.2f},{pose[1]:.2f},"
                    f"{math.degrees(pose[2]):.0f}deg) cmd=({vx:.2f},{vy:.2f},{w:.2f})"
                    f"{extra}"
                )

            # 可视化（每 5 帧）
            if viz is not None and frame_count % 5 == 0:
                if frame_count % 50 == 1:  # 轨迹降采样
                    traj.append((pose[0], pose[1]))
                # 扫描点转到物理系（供显示）
                finite = np.isfinite(valid_ranges)
                scan_pts = None
                if finite.any():
                    a = np.radians(angles_phys[finite])
                    r = valid_ranges[finite]
                    lx, ly = r * np.cos(a), r * np.sin(a)
                    c, s = math.cos(pose[2]), math.sin(pose[2])
                    scan_pts = np.stack(
                        [pose[0] + c * lx - s * ly, pose[1] + s * lx + c * ly],
                        axis=1,
                    )
                slam.getmap(mapbytes)
                viz.update(mapbytes, traj, pose, goal_slam, scan_pts, plan_path)

            # 周期对齐
            remaining = period - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C，安全停止...")
    finally:
        publish_cmd(0.0, 0.0, 0.0)
        if viz is not None:
            viz.close()
        node.destroy_node()
        rclpy.shutdown()
        channel.close()
        print("[INFO] SLAM 导航退出")


if __name__ == "__main__":
    main()
