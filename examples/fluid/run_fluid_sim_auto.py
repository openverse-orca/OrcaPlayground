#!/usr/bin/env python3
"""
Fluid-MuJoCo 全链路 + WaterJug 自动轨迹（Fangfang_AutoMove 等关卡）。

默认配置：fluid_sim_config_auto.json
  - force_position + lag_compensated（与 dev 主配置一致，非 cloth_dev 旧 multi_point_force）
  - water_jug_trajectory：+Z 抬升 → 旋转倒水

与 run_fluid_sim_waterjug.py 区别：
  - 本入口默认 fluid_sim_config_auto.json（历史 auto 命名）
  - waterjug 模块默认与 auto 共用 fluid_sim_config_auto.json

【推荐关卡】FluidTest_Hotel_Bar_Fangfang_AutoMove
  export ORCA_LEVEL_NAME=FluidTest_Hotel_Bar_Fangfang_AutoMove
  # OrcaStudio Play 后：
  python run_fluid_sim_auto.py --gui

【示例】
  python run_fluid_sim_auto.py --gui
  python run_fluid_sim_auto.py --mujoco-gui
  python run_fluid_sim_auto.py --gui --mujoco-gui --max-steps 200
  ./run_fluid_sim_auto.sh --all-gui
"""
from __future__ import annotations

import run_fluid_sim_waterjug as wj

DEFAULT_CONFIG = "fluid_sim_config_auto.json"
LOG_BASENAME = "run_fluid_sim_auto"


def _build_arg_parser():
    parser = wj._build_arg_parser()
    parser.description = (
        "全链路 WaterJug 自动轨迹（fluid_sim_config_auto.json，force_position + lag）"
    )
    for action in parser._actions:
        if action.dest == "config":
            action.default = DEFAULT_CONFIG
            action.help = f"配置文件路径 (默认: {DEFAULT_CONFIG})"
    return parser


def main() -> int:
    import os
    import sys
    import traceback
    from datetime import datetime
    from pathlib import Path

    import run_fluid_sim as rf

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

        wj._apply_waterjug_alignment_defaults(config)

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
                wj._print_sph_run_sanity_check(
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
    import sys

    sys.exit(main())
