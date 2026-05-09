#!/usr/bin/env python3
"""
无 GUI 跑满指定墙钟时长的 MuJoCo–SPH 耦合，在本地 gRPC 接收粒子帧并统计
「当前帧声明的 SPH 流体粒子总数」；按固定间隔重采样，并输出折线图（PNG）。

前提：
- 本脚本会在 particle_grpc_port（默认 50251）上启动**简易 ParticleData 接收服务**。
  若该端口已被 OrcaStudio 等占用，请改用 --particle-grpc-port 51251 等空闲端口
  或先关闭占用进程。

主循环墙钟步长与 envs.fluid.launch.run_simulation 中 REALTIME_STEP 一致：0.02s。
5s 总时长 → max_steps = 250（由 duration / REALTIME_STEP 计算）。

用法（建议在 examples/fluid 下执行）：
  conda activate orca-apr24
  cd /home/hjadmin/OrcaApr24/OrcaPlayground/examples/fluid
  python run_sph_particle_timeline.py
  # 只接受 gRPC 逐帧计数，禁止用 orcasph 日志回退：
  python run_sph_particle_timeline.py --require-grpc-frames
  # 调试前几条 gRPC 收帧：ORCA_PLAYGROUND_DEBUG_GRPC_FRAME=1 python run_sph_particle_timeline.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# 项目根：examples/fluid -> 上两级
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 与 run_simulation 一致（用于推算步数）
REALTIME_STEP = 0.02

from envs.fluid.launch.run_simulation import (  # noqa: E402
    run_simulation_with_config,
)
from envs.fluid.launch.sph_config import setup_python_logging  # noqa: E402
from envs.fluid.particle_grpc_capture import (  # noqa: E402
    resample_to_intervals,
    start_particle_grpc_capture_server,
)

_ORCAGYM_TMP = Path.home() / ".orcagym" / "tmp"


def _parse_initial_fluid_particles_from_orcasph_log(session_ts: str) -> Optional[int]:
    """从 orcasph 标准输出日志中解析「Number of fluid particles: N」。"""
    p = _ORCAGYM_TMP / f"orcasph_{session_ts}.log"
    if not p.is_file():
        return None
    import re

    m = re.search(r"Number of fluid particles:\s*(\d+)", p.read_text(encoding="utf-8", errors="replace"))
    if not m:
        return None
    return int(m.group(1))


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="5s 耦合 + SPH 粒子数时间线（每 0.25s 采样点）")
    ap.add_argument(
        "--config",
        default="fluid_sim_config.json",
        help="fluid 配置（同 run_fluid_sim）",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="墙钟时长的主循环总秒数（默认 5）",
    )
    ap.add_argument(
        "--sample-interval",
        type=float,
        default=0.25,
        help="重采样间隔（秒，默认 0.25）",
    )
    ap.add_argument(
        "--particle-grpc-port",
        type=int,
        default=50251,
        help="本地捕获 gRPC 端口，须与 SPH 推送地址一致；冲突请改此端口",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="折线图输出路径（默认 本目录下 sph_particle_count_timeline.png）",
    )
    ap.add_argument("--use-all-cpu", action="store_true", help="不绑定 OrcaSPH 到 4+ 核")
    ap.add_argument(
        "--require-grpc-frames",
        action="store_true",
        help="必须收到至少一帧 gRPC(Particle/Raw) 的粒子数；未收到则非 0 退出，且不使用日志回退",
    )
    ap.add_argument(
        "--grpc-raw-csv",
        default=None,
        help="将每条 gRPC 原始样本 (t_perf, count, method) 写入 CSV 路径；默认 与 --output 同主名或 sph_particle_grpc_raw.csv",
    )
    ap.add_argument(
        "--grpc-workers",
        type=int,
        default=32,
        metavar="N",
        help="gRPC 服务线程池大小（默认 32，便于并发 unary）",
    )
    args = ap.parse_args()

    config_path = _SCRIPT_DIR / args.config
    if not config_path.is_file():
        print(f"❌ 未找到配置: {config_path}")
        return 1

    out_png = (
        Path(args.output).resolve()
        if args.output
        else (_SCRIPT_DIR / "sph_particle_count_timeline.png")
    )

    try:
        grpc_server, sink = start_particle_grpc_capture_server(
            args.particle_grpc_port, max_workers=int(args.grpc_workers)
        )
    except OSError as e:
        print(
            f"❌ 无法绑定 0.0.0.0:{args.particle_grpc_port}：{e}\n"
            "   请换 --particle-grpc-port 或关闭占用该端口的程序（如 OrcaStudio 粒子 gRPC 接收）。"
        )
        return 1

    # 将 SPH 的粒子推送指到本机捕获服务
    addr = f"127.0.0.1:{args.particle_grpc_port}"
    print(f"✅ 粒子 gRPC 接收服务: {addr}（OrcaSPH 将向此地址发送帧）")

    try:
        config = _load_json(config_path)
        config["particle_render_grpc_override"] = {"server_address": addr, "enabled": True}

        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_steps = int(round(args.duration / REALTIME_STEP))
        if n_steps < 1:
            n_steps = 1

        print(
            f"⏱  墙钟 {args.duration:.2f}s × 步长 {REALTIME_STEP}s ≈ {n_steps} 步；"
            f"每 {args.sample_interval:.2f}s 一个采样点"
        )
        print("🚀 开始 run_simulation_with_config（无 --gui，OrcaSPH 将自动 Connect OrcaLink）…\n")

        setup_python_logging(config)
        ncpu = None
        if not args.use_all_cpu:
            n = os.cpu_count()
            if n is not None and n > 4:
                ncpu = f"4-{n - 1}"

        run_simulation_with_config(
            config,
            session_timestamp=session_ts,
            cpu_affinity=ncpu,
            max_steps=n_steps,
        )

        all_samples = sink.snapshot()
        grpc_n = len(all_samples)
        raw: list = list(all_samples)

        if not raw and args.require_grpc_frames:
            print(
                "\n❌ --require-grpc-frames：未在 Python gRPC 服务中收到任何 ParticleFrame/Raw 帧。"
                " 可检查：50251 无其它进程占用、proto/grpcio 与 SPH 一致、"
                "或设置 ORCA_PLAYGROUND_DEBUG_GRPC_FRAME=1 查看 C++ 是否发帧。"
            )
            return 2

        if not raw:
            n0 = _parse_initial_fluid_particles_from_orcasph_log(session_ts)
            if n0 is not None:
                # 单点：重采样时全程保持为场景初始流体粒子数（不可压缩 DFSPH 下通常恒定）
                raw = [(time.perf_counter(), n0, "orcasph_log_initial")]
                print(
                    f"\n⚠️  gRPC 上未收到帧（见上方 grpc 大消息/多 worker 修复）；使用 orcasph 日志中的"
                    f"「Number of fluid particles」= {n0} 作回退。"
                )
            else:
                print("\n⚠️  无 gRPC 样本且无法从 orcasph 日志解析初始粒子数，图中为 0。")
        elif grpc_n > 0:
            csvg = args.grpc_raw_csv
            if csvg is None:
                csvg = str(out_png.with_name("sph_particle_grpc_raw.csv"))
            cpath = Path(csvg).resolve()
            cpath.parent.mkdir(parents=True, exist_ok=True)
            with open(cpath, "w", encoding="utf-8") as f:
                f.write("t_perf_s,count,method\n")
                for t0, c0, m0 in all_samples:
                    f.write(f"{t0:.9f},{c0},{m0}\n")
            print(f"📄 已写入逐 gRPC 样本: {cpath}")
        t_rel, counts = resample_to_intervals(
            raw,
            interval_s=args.sample_interval,
            duration_s=args.duration,
        )

        print(f"\n📈 gRPC 原始样本数: {grpc_n}；重采样为 {len(counts)} 个时间点")
        for ti, c in zip(t_rel, counts):
            print(f"   t={ti:6.2f}s  N={c}")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
            ax.plot(t_rel, counts, marker="o", linewidth=1.2, markersize=4)
            ax.set_xlabel("Time (s) since first gRPC frame (resampled)")
            ax.set_ylabel("SPH fluid particle count (header total / raw batches sum)")
            ax.set_title("SPH particle count vs time")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_png)
            plt.close(fig)
            print(f"\n✅ 已保存: {out_png}")
        except Exception as e:
            print(f"\n⚠️  matplotlib 作图失败（{e}）；数值已在上方打印。")
            return 0
    finally:
        grpc_server.stop(0)
        print("🛑 粒子 gRPC 接收服务已停止。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
