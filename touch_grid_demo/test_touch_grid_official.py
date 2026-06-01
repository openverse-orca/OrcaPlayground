"""
测试官方 touch_grid 示例 (mujoco/model/plugin/sensor/touch_grid.xml)。

用法：
    # 无头模式：让小球压到高度图上，打印每个通道的 7x7 taxel 力图
    conda run -n orcalab python test_touch_grid_official.py

    # 交互模式：打开 viewer，可以用鼠标拖拽小球压到凹凸地形上观察读数
    conda run -n orcalab python test_touch_grid_official.py --viewer
"""
import argparse
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "touch_grid.xml")


def get_sensor_layout(model):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch")
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    # touch_grid 输出维度 = nchannel * nx * ny。该模型 config: nchannel=3, size=7 7。
    nchannel, nx, ny = 3, 7, 7
    assert dim == nchannel * nx * ny, f"unexpected sensor dim={dim}"
    return sid, adr, dim, (nchannel, ny, nx)


def read_grid(data, adr, dim, shape):
    vals = data.sensordata[adr:adr + dim].copy()
    return vals.reshape(shape)  # (nchannel, ny, nx)


def run_headless(steps):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    sid, adr, dim, shape = get_sensor_layout(model)
    print(f"sensor 'touch': id={sid} adr={adr} dim={dim} -> reshape{shape}")

    # 把小球初始压低，让它落到中央的高度图 geom 上产生接触
    data.qpos[model.joint("z").qposadr[0]] = 0.05
    mujoco.mj_forward(model, data)

    for step in range(steps):
        mujoco.mj_step(model, data)
        if (step + 1) % 200 == 0:
            grid = read_grid(data, adr, dim, shape)
            print(f"step={step + 1:4d}  ncon={data.ncon}  normal_max={grid[0].max():.4f}")

    grid = read_grid(data, adr, dim, shape)
    channel_names = ["normal", "tangent_x", "tangent_y"]
    for c in range(shape[0]):
        print(f"\n=== channel[{c}] ({channel_names[c]}) 7x7 ===")
        print(np.array2string(grid[c], precision=3, suppress_small=True))

    if grid[0].max() > 0:
        print("\n✓ touch_grid 正常工作，法向通道有接触力输出")
    else:
        print("\n✗ 法向通道无输出。当前接触数 ncon =", data.ncon)
        print("  提示：site 朝向决定感应半球，可加 --viewer 手动把小球压到凸起上观察")


def run_viewer():
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    sid, adr, dim, shape = get_sensor_layout(model)
    print(f"sensor 'touch': id={sid} adr={adr} dim={dim} -> reshape{shape}")
    print(
        "操作：双击选中小球 -> 按住 Ctrl+右键拖动，把它压到中央凸起上。\n"
        "看压力的两种方式：\n"
        "  1) 视觉热力图：site 处会自动出现一组彩色方块，颜色/大小=受力大小（touch_grid 插件自带，无需开关）。\n"
        "  2) 数值：下面终端会实时刷新完整的 7x7 法向力矩阵。\n"
        "Ctrl+C 退出。\n"
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 顺便打开接触力箭头，便于对照（可在 viewer 左侧面板里再关掉）
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        step = 0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1
            if step % 50 == 0:
                grid = read_grid(data, adr, dim, shape)
                normal = grid[0]
                # 终端原地刷新整张法向力图
                print("\033[2J\033[H", end="")  # 清屏
                print(f"ncon={data.ncon}  normal_max={normal.max():.4f}\n")
                print("normal force 7x7:")
                print(np.array2string(normal, precision=3, suppress_small=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="启动交互式 viewer")
    parser.add_argument("--steps", type=int, default=1000, help="无头模式仿真步数")
    args = parser.parse_args()

    if args.viewer:
        run_viewer()
    else:
        run_headless(args.steps)


if __name__ == "__main__":
    main()
