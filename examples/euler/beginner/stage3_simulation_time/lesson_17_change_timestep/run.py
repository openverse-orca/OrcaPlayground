"""第 17 课：调整时间步长 — 仿真时间与墙钟时间。

层 3（用户主导）：场景无关设计——连上 OrcaLab 当前场景，通过关键词
发现方块；没摆就加 --default-scene 兜底。

本课新知识：time_step 决定"一步走多细"。同样的 1 秒仿真时间，
步长 0.002s 要 500 步、0.01s 只要 100 步——步数差 5 倍；但大步长
牺牲接触精度，观察落地瞬间的穿透/抖动差异。

用法:
    python -m examples.euler.beginner.stage3_simulation_time.lesson_17_change_timestep.run --default-scene

验证点:
    1. 两组完成同样的仿真时长，步数严格成比例（500 vs 100）
    2. 仿真时间都走到 1.0s，但墙钟耗时不同
    3. 大步长组的落地接触表现更粗糙（数值可比较落地高度差）
"""

from __future__ import annotations

import argparse
import sys
import time as wall_time

from orca_gym.log.orca_log import get_orca_logger

from examples.euler.beginner._common.discovery import find_body
from examples.euler.beginner._common.scene_recipe import (
    FLOOR_Z_OFFSET,
    ActorSpec,
    clear_scene,
    setup_console_logging,
    spawn_recipe,
)
from examples.euler.beginner._common import sim_link

_logger = get_orca_logger()

# 真实 spawnable 资产（本地导入包 345a60e1cced）
# TODO(asset-lib): 资产正式上传资产库后，将 assets/345a60e1cced/ 统一切换为云端正式包地址
_GROUND_PATH = "assets/345a60e1cced/prefabs/floor_usda"
_BLOCK_PATH = "assets/345a60e1cced/prefabs/cube_usda"

_BLOCK_KEYWORD = "cube"  # 关键词对齐资产内部名（cube_usda → ..._cube），与实例名前缀无关

# ======================= 配方区（改这里） =======================
# 对比的物理步长（秒）：粗步长 5 倍于细步长
TIME_STEPS: tuple[float, ...] = (0.002, 0.01)
# 每组推进的仿真时长（秒）
SIM_DURATION: float = 1.0
# ================================================================


def build_default_recipe() -> list[ActorSpec]:
    """默认配方兜底：地面 + 一个方块（抬高一米）。"""
    return [
        ActorSpec(name="ground", asset_path=_GROUND_PATH, position=(0.0, 0.0, FLOOR_Z_OFFSET)),
        ActorSpec(name="cube_1", asset_path=_BLOCK_PATH, position=(0.0, 0.0, 1.0)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="第 17 课：调整时间步长")
    parser.add_argument("--addr", default="localhost:50051", help="OrcaLab gRPC 地址")
    parser.add_argument(
        "--default-scene",
        action="store_true",
        help="用脚本默认配方摆场景（兜底）；不指定则使用你自己的场景",
    )
    args = parser.parse_args()

    setup_console_logging()

    _logger.info("=" * 60)
    _logger.info("第 17 课：调整时间步长 — 仿真时间与墙钟时间")
    _logger.info(f"  对比步长：{TIME_STEPS}，每组推进 {SIM_DURATION}s 仿真时间")
    _logger.info("=" * 60)

    scene = None
    if args.default_scene:
        _logger.info("[兜底] 使用默认配方重建教学场景")
        clear_scene(args.addr)
        scene = spawn_recipe(args.addr, build_default_recipe())

    results: list[tuple[float, int, float, float]] = []
    try:
        # 两组各建一个 env：time_step 是构造级参数，实验讲武德——各自从头连。
        # 演示用实时节拍（视口看得见下落）；"纯算力全速跑"的墙钟对比见文末说明
        for gi, ts in enumerate(TIME_STEPS):
            if gi > 0:
                _logger.info("[组间停顿] 3 秒——回视口准备好再看下一组")
                wall_time.sleep(3.0)
            env = sim_link.connect_simulation_env(args.addr, time_step=ts)
            try:
                block_name = find_body(env, _BLOCK_KEYWORD)
                if block_name is None:
                    _logger.error("未找到方块。拖入一个方块，或加 --default-scene 运行默认配方。")
                    return 1
                ctrl = sim_link.zero_ctrl(env)
                z_start = sim_link.read_height(env, block_name)
                n_steps = int(round(SIM_DURATION / ts))
                _logger.info(
                    f"[步长 {ts}] 开始：起始 z={z_start:.4f}，将推进 {n_steps} 步 × {ts}s"
                    f"= {SIM_DURATION}s 仿真时间（实时节拍，盯住视口）"
                )
                print_every = max(1, n_steps // 2)
                wall_start = wall_time.perf_counter()
                for i in range(n_steps):
                    sim_link.advance_realtime(env, ctrl, 1)
                    if (i + 1) % print_every == 0:
                        _logger.info(
                            f"  步 {i + 1}/{n_steps}  t={float(env.data.time):.3f}s"
                            f"  z={sim_link.read_height(env, block_name):.4f}"
                        )
                wall_cost = wall_time.perf_counter() - wall_start
                z_end = sim_link.read_height(env, block_name)
                t_end = float(env.data.time)
                results.append((ts, n_steps, wall_cost, z_end))
                _logger.info(
                    f"[步长 {ts}] 结束：{n_steps} 步走完 {SIM_DURATION}s 仿真时间，"
                    f"末高度 z={z_end:.4f}"
                )
                if abs(t_end - SIM_DURATION) > 0.05:
                    _logger.warning(
                        f"[步长 {ts}] 仿真时间偏差：预期 {SIM_DURATION}s 实测 {t_end:.3f}s"
                        "——请把本行日志反馈给课程维护者"
                    )
            finally:
                env.close()
        (ts_a, steps_a, _, z_a), (ts_b, steps_b, _, z_b) = results
        _logger.info(
            f"[完成] 同样 {SIM_DURATION}s 仿真时间：步数 {steps_a} vs {steps_b}"
            f"（比例 {steps_a / steps_b:.1f}）；末高度 {z_a:.4f} vs {z_b:.4f} m"
        )
        if z_b > z_a + 0.1:
            _logger.info(
                "[解释] 自由下落段两组轨迹几乎重合（t=0.5s 时高度差仅厘米级），"
                "但落地后大步长组被弹回高处——粗步长下接触求解器对碰撞冲量解算过冲，"
                "把方块'弹飞'了。这就是步长取舍：省 5 倍步数，代价是接触事件不可信"
            )
        else:
            _logger.info("——大步长省算力，但接触事件精度更差（本场景差异较小）")
        _logger.info(
            "[提示] 演示为实时节拍（1 秒仿真 ≈ 1 秒墙钟）。想看'全速跑'的墙钟差异："
            "把 run.py 里的 advance_realtime 改成 advance 再跑一遍，"
            "对比两组的墙钟耗时"
        )
    finally:
        if scene is not None:
            scene.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        import traceback

        _logger.error(f"脚本异常退出: {exc}\n{traceback.format_exc()}")
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
