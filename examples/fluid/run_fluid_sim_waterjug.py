#!/usr/bin/env python3
"""
全链路 WaterJug 预定轨迹仿真（基于 run_fluid_sim.py）。

与 run_fluid_sim_auto.py 区别：
  - 默认 fluid_sim_config_auto.json：force_position + lag_compensated + water_jug_trajectory
  - initial_z_offset_m=0，关闭 sph_scene 竖直偏移，避免 MuJoCo 与 OrcaSPH 初值错位
  - SceneGenerator 按 force_position 跳过 AnchorPoints

【推荐关卡】FluidTest_Hotel_Bar_Fangfang_AutoMove
  export ORCA_LEVEL_NAME=FluidTest_Hotel_Bar_Fangfang_AutoMove
  # OrcaStudio Play 后：
  python run_fluid_sim_waterjug.py --gui

【示例】
  python run_fluid_sim_waterjug.py --gui
  python run_fluid_sim_waterjug.py --mujoco-gui
  python run_fluid_sim_waterjug.py --gui --mujoco-gui --max-steps 200
  python run_fluid_sim_waterjug.py --build-mode release
"""
from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# 与 run_fluid_sim.py 相同：examples/fluid → OrcaPlayground 根目录
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_file_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_file_dir not in sys.path:
    sys.path.insert(0, current_file_dir)

import run_fluid_sim as rf

DEFAULT_CONFIG = "fluid_sim_config_auto.json"
LOG_BASENAME = "run_fluid_sim_waterjug"


def _build_arg_parser():
    """基于 run_fluid_sim 参数表，仅改默认配置与说明。"""
    parser = rf._build_arg_parser()
    parser.description = "全链路 WaterJug 预定轨迹（force_position，MuJoCo/SPH 初值对齐）"
    for action in parser._actions:
        if action.dest == "config":
            action.default = DEFAULT_CONFIG
            action.help = f"配置文件路径 (默认: {DEFAULT_CONFIG})"
    return parser


def _apply_waterjug_alignment_defaults(config: dict) -> None:
    """
    全链路水壶运行守卫：禁止短链「独立模式」竖直初值偏移渗入 SPH scene。

    run_fluid_sim_auto + fluid_sim_config_auto.json 曾默认 initial_z_offset_m=-0.35
    （短链 H−0.35 m），会在 OrcaSPH 启动前改写 sph_scene，而全链路 Studio 场景
    水壶已在正确高度，导致 MuJoCo 与 SPH 初值错位。
    """
    traj = config.setdefault("water_jug_trajectory", {})
    traj.setdefault("enabled", True)
    if float(traj.get("initial_z_offset_m", 0.0)) == -0.35:
        traj["initial_z_offset_m"] = 0.0
        print("ℹ️  waterjug: initial_z_offset_m 已从短链默认 -0.35 改为 0（全链路对齐 Studio）")

    z_off = float(traj.get("initial_z_offset_m", 0.0))
    offset_cfg = config.setdefault("sph_scene_initial_offset", {})
    if abs(z_off) < 1e-12:
        offset_cfg["enabled"] = False
    elif offset_cfg.get("enabled") is not False:
        offset_cfg.setdefault("sync_from_trajectory_z", True)
        offset_cfg.setdefault("offset_fluid_blocks", bool(traj.get("offset_fluid_blocks", True)))


def _print_sph_run_sanity_check(
    orcagym_tmp_dir: Path, session_timestamp: str, max_steps: int
) -> None:
    """与 run_fluid_sim 相同逻辑，但读取本入口日志文件名。"""
    if max_steps <= 0:
        return
    run_log = orcagym_tmp_dir / f"{LOG_BASENAME}_{session_timestamp}.log"
    olink = orcagym_tmp_dir / f"orcalink_{session_timestamp}.log"
    print("\n" + "=" * 60)
    print("SPH 链路粗检（基于日志关键词）")
    print("=" * 60)
    if not run_log.is_file():
        print(f"⚠️  未找到: {run_log}")
        return
    text = run_log.read_text(encoding="utf-8", errors="replace")
    if "SPH 集成已禁用" in text or "Session ready timeout" in text:
        print("❌ MuJoCo↔OrcaLink 未在时限内凑齐双客户端，或 SPH 集成被关闭。")
    elif "Session is ready" in text or "sph_wrapper.connect() returned: True" in text:
        print("✅ OrcaLink 会话已就绪，Python 侧 bridge 已连接。")
    else:
        print("⚠️  请人工打开日志核对 Session ready / connect True。")
    bad = ("NaN", "diverg", "Error: [OrcaLinkBridge]", "FATAL")
    for p in orcagym_tmp_dir.glob(f"orcasph_{session_timestamp}.log*"):
        if not p.is_file():
            continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        hits = [b for b in bad if b in raw]
        if hits:
            print(f"⚠️  {p.name} 含可疑关键词: {', '.join(hits)}")
        else:
            print(f"✅ {p.name} 未命中常见错误关键词")
        break
    else:
        print(f"⚠️  未找到 orcasph_{session_timestamp}.log")
    if olink.is_file():
        ot = olink.read_text(encoding="utf-8", errors="replace")
        if "SPlisHSPlasH" in ot or "splishsplas" in ot.lower():
            print("✅ orcalink 日志中出现 SPH 侧客户端名。")
    print("=" * 60 + "\n")


def main() -> int:
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    orcagym_tmp_dir = Path.home() / ".orcagym" / "tmp"
    orcagym_tmp_dir.mkdir(parents=True, exist_ok=True)
    log_file = orcagym_tmp_dir / f"{LOG_BASENAME}_{session_timestamp}.log"

    log_handle = open(log_file, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = rf.TeeOutput(sys.stdout, log_handle)
    sys.stderr = rf.TeeOutput(sys.stderr, log_handle)

    try:
        print(f"📝 日志文件: {log_file}")
        print("=" * 60)

        parser = _build_arg_parser()
        args = parser.parse_args()

        err = rf._validate_early_args(args)
        if err is not None:
            return err

        cpu_affinity = rf._resolve_cpu_affinity(args.use_all_cpu)

        config_path = Path(__file__).parent / args.config
        if not config_path.exists():
            print(f"❌ 错误: 配置文件不存在: {config_path}")
            return 1

        config = rf.load_config(str(config_path))
        script_dir = Path(__file__).parent

        _apply_waterjug_alignment_defaults(config)

        rf._apply_build_mode_from_args(config, args)
        rf._apply_particle_render_run_config(args, script_dir, session_timestamp, config)
        err = rf._apply_mujoco_trajectory_config(args, script_dir, session_timestamp, config)
        if err is not None:
            return err
        rf._apply_orcasph_gui_from_args(config, args.gui)
        rf._apply_mujoco_gui_from_args(
            config,
            args.mujoco_gui,
            sph_gui=bool(args.gui),
            shutdown_on_close=True if args.mujoco_shutdown_on_close else None,
        )
        rf._apply_performance_stats_from_args(config, args)
        rf._apply_manual_mode_from_args(config, args)

        try:
            if args.enable_performance_stats and args.performance_stats_plot:
                orcasph_log_file = orcagym_tmp_dir / f"orcasph_{session_timestamp}.log"
                print(f"📈 启动实时性能统计图表，监控日志文件: {orcasph_log_file}")
                project_root_resolved = Path(__file__).parent.parent.parent.resolve()
                stats_viewer_path = (
                    project_root_resolved / "envs" / "fluid_stats" / "performance_stats_viewer.py"
                )
                if stats_viewer_path.exists():
                    import subprocess

                    subprocess.Popen(
                        [sys.executable, str(stats_viewer_path), str(orcasph_log_file)]
                    )
                else:
                    print(f"⚠️  性能统计查看器脚本不存在: {stats_viewer_path}")

            if args.mode == "playback":
                rf.setup_python_logging(config)
                rf.run_particle_playback_from_config(config)
            else:
                rf.run_simulation_with_config(
                    config,
                    session_timestamp=session_timestamp,
                    cpu_affinity=cpu_affinity,
                    max_steps=max(0, int(args.max_steps or 0)),
                )
                _print_sph_run_sanity_check(
                    orcagym_tmp_dir, session_timestamp, max(0, int(args.max_steps or 0))
                )
        except KeyboardInterrupt:
            print("\n✅ 仿真已停止")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            traceback.print_exc()
            return 1

        return 0
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    sys.exit(main())
