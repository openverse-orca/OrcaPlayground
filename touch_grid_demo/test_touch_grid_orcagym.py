"""
走 OrcaGym 链路（而非裸 mujoco）验证 touch_grid 传感器能否读通。

前提：
  1. 引擎已重新编译（包含 plugin 传感器的 <extension>/<config> 导出支持）。
  2. OrcaStudio 正在运行，且当前场景里含有一个 touch_grid plugin 传感器
     （例如把官方 touch_grid.xml 导入成 prefab 放进 Level，或在夹爪 site 上加 sensor）。
  3. OrcaStudio 的 gRPC 地址默认为 localhost:50051。

用法：
    conda run -n orcalab python test_touch_grid_orcagym.py
    conda run -n orcalab python test_touch_grid_orcagym.py --addr localhost:50051 --steps 500

这条链路就是数采时实际用的：
    OrcaStudio 导出场景 MJCF -> orca_gym 本地用 mujoco 加载 -> env.query_sensor_data() 读 sensordata
读通即代表 touch_grid 能进 obs、进而被数采保存。
"""
import argparse
import os
import time

import gymnasium as gym
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ENV_ENTRY_POINT = "orca_gym.scripts.sim_env:SimEnv"
TIME_STEP = 0.001
FRAME_SKIP = 20


def spawn_prefab(addr: str, asset_path: str):
    """把一个已导入的 prefab spawn 进运行中的 OrcaStudio（清空原场景后发布）。"""
    from orca_gym.scene.orca_gym_scene import OrcaGymScene, Actor
    import orca_gym.utils.rotations as rotations

    print(f"清空场景并 spawn prefab: {asset_path}")
    clear = OrcaGymScene(addr)
    clear.publish_scene()  # 发布空场景，清掉旧内容
    time.sleep(1)
    clear.close()
    time.sleep(1)

    scene = OrcaGymScene(addr)
    actor = Actor(
        name="touch_grid_probe",
        asset_path=asset_path,
        position=np.array([0.0, 0.0, 0.0]),
        rotation=rotations.euler2quat(np.array([0.0, 0.0, 0.0])),
        scale=1.0,
    )
    scene.add_actor(actor)
    scene.publish_scene()
    time.sleep(1)
    scene.close()
    print("prefab 已发布。")


def make_env(addr: str, agent_name: str):
    env_id = "TouchGridProbe-OrcaGym-" + addr.replace(":", "-")
    if env_id not in gym.envs.registry:
        gym.register(
            id=env_id,
            entry_point=ENV_ENTRY_POINT,
            kwargs={
                "frame_skip": FRAME_SKIP,
                "orcagym_addr": addr,
                "agent_names": [agent_name],
                "time_step": TIME_STEP,
            },
            max_episode_steps=10**9,
        )
    return gym.make(env_id)


def find_touch_grid_sensors(env):
    """从模型里找出 plugin / touch 类传感器，返回 [(name, dim), ...]。"""
    sensor_dict = env.unwrapped.model.gen_sensor_dict()
    hits = []
    for name, info in sensor_dict.items():
        type_str = info.get("SensorTypeStr", "")
        # 插件传感器在 orca_gym 里类型字符串会是 "unknown"（mjSENS_PLUGIN 未单独映射），
        # 维度通常 > 6（taxel 网格）。用名字 + 维度双重判定，尽量稳。
        is_plugin_like = type_str in ("unknown", "touch") or "touch" in name.lower()
        if is_plugin_like and info["Dim"] >= 1:
            hits.append((name, info["Dim"]))
    return hits, sensor_dict


def maybe_reshape(values: np.ndarray):
    dim = values.size
    # touch_grid: dim = nchannel * nx * ny。常见 7x7=49 的整数倍。
    if dim % 49 == 0 and dim // 49 >= 1:
        return values.reshape(dim // 49, 7, 7)
    return None


def channel_label(channel: int):
    labels = {
        0: "normal/z",
        1: "tangent/x",
        2: "tangent/y",
        3: "torsion/z",
        4: "rolling/x",
        5: "rolling/y",
    }
    return labels.get(channel, f"channel {channel}")


class TouchGridVisualizer:
    """实时显示 touch_grid 的 7x7 压力热力图。"""

    def __init__(self, sensor_name: str, channel: int, vmax: float | None, mode: str):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.channel = channel
        self.vmax = vmax
        self.mode = mode
        self.fig, self.ax = plt.subplots(num=f"touch_grid: {sensor_name}")
        self.image = self.ax.imshow(
            np.zeros((7, 7), dtype=np.float32),
            cmap="inferno",
            vmin=0.0,
            vmax=vmax if vmax and vmax > 0 else 1.0,
            interpolation="nearest",
            origin="lower",
        )
        self.colorbar = self.fig.colorbar(self.image, ax=self.ax)
        self.ax.set_title(f"{sensor_name} {channel_label(channel)} ({mode})")
        self.ax.set_xlabel("x taxel")
        self.ax.set_ylabel("y taxel")
        self.fig.tight_layout()
        plt.ion()
        plt.show(block=False)

    def _display_values(self, channel_values: np.ndarray):
        if self.mode == "pressure":
            # 当前模型里 touch_grid 返回的是沿 site 法向相反方向的力，接触压力表现为负值。
            return np.maximum(-channel_values, 0.0)
        if self.mode == "abs":
            return np.abs(channel_values)
        return channel_values

    def update(self, grid: np.ndarray, step: int):
        if self.channel >= grid.shape[0]:
            raise ValueError(f"--viz-channel={self.channel} 超出通道数 {grid.shape[0]}")

        pressure = self._display_values(grid[self.channel])
        self.image.set_data(pressure)
        if self.vmax is None:
            current_max = float(max(pressure.max(), 1e-6))
            self.image.set_clim(0.0, current_max)
        self.ax.set_title(f"touch_grid {channel_label(self.channel)}  step={step}  max={pressure.max():.4f}")
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="localhost:50051", help="OrcaStudio gRPC 地址")
    parser.add_argument("--agent", default="NoRobot", help="场景里的 actor 名（占位即可）")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--visualize", action="store_true", help="实时显示 touch_grid 压力热力图")
    parser.add_argument("--viz-every", type=int, default=5, help="每多少步刷新一次热力图")
    parser.add_argument("--viz-channel", type=int, default=0, help="显示第几个 touch_grid 通道")
    parser.add_argument("--viz-vmax", type=float, default=None, help="热力图最大值；默认随数据自动缩放")
    parser.add_argument(
        "--viz-mode",
        choices=("pressure", "abs", "raw"),
        default="pressure",
        help="热力图显示方式：pressure=max(-value,0)，abs=绝对值，raw=原始值",
    )
    parser.add_argument("--realtime", action="store_true", help="按仿真时间 sleep，方便在 OrcaStudio 里看运动")
    parser.add_argument(
        "--spawn-asset",
        default=None,
        help="可选：先把这个 prefab spawn 进场景，例如 "
        "assets/e071469a36d3c8aa/default_project/prefabs/touch_grid_usda",
    )
    args = parser.parse_args()

    if args.spawn_asset:
        spawn_prefab(args.addr, args.spawn_asset)

    print(f"连接 OrcaStudio: {args.addr} ...")
    env = make_env(args.addr, args.agent)
    env.reset()
    print("场景加载成功（说明 OrcaStudio 导出的 MJCF 已被 orca_gym 用 mujoco 成功编译）。")

    hits, sensor_dict = find_touch_grid_sensors(env)
    print(f"\n模型里共 {len(sensor_dict)} 个传感器。候选 touch_grid 传感器：")
    if not hits:
        print("  （未找到）。全部传感器列表：")
        for name, info in sensor_dict.items():
            print(f"    {name}: type={info.get('SensorTypeStr')} dim={info['Dim']}")
        print("\n✗ 没有发现 touch_grid 传感器，请检查 OrcaStudio 场景里是否加了该 sensor。")
        env.close()
        return
    for name, dim in hits:
        print(f"    {name}  dim={dim}")

    target = hits[0][0]
    print(f"\n以 '{target}' 为目标，步进 {args.steps} 步读取数据：\n")
    visualizer = TouchGridVisualizer(target, args.viz_channel, args.viz_vmax, args.viz_mode) if args.visualize else None

    for step in range(args.steps):
        env.step(env.action_space.sample())
        env.render()
        if args.realtime:
            time.sleep(TIME_STEP * FRAME_SKIP)
        should_print = (step + 1) % args.print_every == 0
        should_visualize = visualizer is not None and (step + 1) % args.viz_every == 0
        if should_print or should_visualize:
            data = env.unwrapped.query_sensor_data([target])
            values = np.asarray(data[target], dtype=np.float32).flatten()
            grid = maybe_reshape(values)
            if should_visualize and grid is not None:
                visualizer.update(grid, step + 1)
        if (step + 1) % args.print_every == 0:
            pressure = np.maximum(-values, 0.0)
            channel_max = ""
            if grid is not None:
                channel_max = "  channels=" + ", ".join(
                    f"{channel_label(i)}:{np.maximum(-grid[i], 0.0).max():.4f}"
                    for i in range(grid.shape[0])
                )
            line = (
                f"step={step + 1:4d}  dim={values.size}  "
                f"raw_max={values.max():.4f}  raw_min={values.min():.4f}  pressure_max={pressure.max():.4f}"
                f"{channel_max}"
            )
            print(line)
            if grid is not None and pressure.max() > 0:
                print("  pressure[0] 7x7:")
                print(np.array2string(np.maximum(-grid[0], 0.0), precision=3, suppress_small=True, prefix="  "))

    print("\n✓ OrcaGym 链路读取 touch_grid 成功" if hits else "")
    env.close()


if __name__ == "__main__":
    main()
