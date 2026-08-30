"""RTF/性能诊断工具（环境变量 ``ORCA_RTF_LOG=1`` 开启）。

由 Lesson 5 ``g1_base_env.py`` 的调试遗留抽离，供 ``G1BaseEnv.run_lesson`` 挂载
分段计时。累计每个控制周期的 ``step``(物理步进)/``verify``(判定)/``render``(渲染)/
``sleep``(限速) 分段耗时，并记录超时周期计数，结束打印实际 RTF 与瓶颈分析。

默认关闭，普通运行零开销。更细的 GPU 阶段拆分（flush H2D / step GPU / sync D2H /
view）见 solver 侧 ``ORCA_EULER_PERF_LOG=1``。
"""

from __future__ import annotations

import os
import time

_RTF_LOG_CACHED: bool | None = None


def rtf_log_enabled() -> bool:
    """是否启用 RTF 性能诊断（环境变量 ``ORCA_RTF_LOG=1`` 时）。

    首次调用时求值并缓存；关闭时 ``run_lesson`` 不挂载计时器、不调用
    ``perf_counter``、不打印（零开销）。
    """
    global _RTF_LOG_CACHED
    if _RTF_LOG_CACHED is None:
        _RTF_LOG_CACHED = os.environ.get("ORCA_RTF_LOG") in {"1", "true", "True", "yes"}
    return _RTF_LOG_CACHED


class PerfTimer:
    """``run_lesson`` 控制循环分段计时器。

    累计 ``step``/``verify``/``render``/``sleep`` 四个分段的墙钟耗时（秒），并
    计数"超时周期"（单周期计算耗时超出 ``cycle_target``，此时 ``sleep≈0``）。
    ``run_lesson`` 用显式 ``t0/t1/t2/t3`` 时间戳计算各段差值后经 ``add`` 累加，
    避免把上一周期的限速睡眠时长误计入下一周期。
    """

    def __init__(self) -> None:
        self.step = 0.0
        self.verify = 0.0
        self.render = 0.0
        self.sleep = 0.0
        self.overruns = 0
        # 循环起始时间戳（print_rtf_report 计算实际 RTF 用）
        self.loop_start = 0.0

    def start(self) -> None:
        """记录循环起始时间戳。"""
        self.loop_start = time.perf_counter()

    def add(self, section: str, seconds: float) -> None:
        """把某段耗时（秒）累计到指定分段。

        Args:
            section: ``"step"`` / ``"verify"`` / ``"render"`` 之一。
            seconds: 该段墙钟耗时（秒）。
        """
        if section == "step":
            self.step += seconds
        elif section == "verify":
            self.verify += seconds
        elif section == "render":
            self.render += seconds
        else:
            raise ValueError(f"未知分段: {section!r}")

    def record_sleep(self, slept: float) -> None:
        """累计限速睡眠耗时（秒）。"""
        self.sleep += slept

    def record_overrun(self) -> None:
        """计数一次超时周期（计算耗时超出 cycle_target）。"""
        self.overruns += 1


def print_rtf_report(num_steps: int, cycle_target: float, perf: PerfTimer) -> None:
    """打印 ``run_lesson`` 的 RTF/性能诊断报告。

    汇总每个控制周期的 step(物理步进+D2H 同步)/verify(判定)/render(渲染)/
    sleep(限速) 分段耗时，并计算实际 RTF（仿真秒 / 墙钟秒）。当 RTF<1.0 时，
    通常是 step 或 render 的单周期耗时超过 ``cycle_target``（此时 sleep≈0 且
    overruns 上升）；当 RTF 接近 1.0 时，sleep 会补齐剩余时长。

    Args:
        num_steps: 控制周期数。
        cycle_target: 每个控制周期的目标墙钟时长（秒）= ``frame_skip * time_step``。
        perf: 分段计时器（含计时累计与超时计数）。
    """
    wall = time.perf_counter() - perf.loop_start
    sim_total = num_steps * cycle_target  # 仿真推进总秒数
    rtf = sim_total / wall if wall > 0 else float("inf")
    n = num_steps
    ms = 1e3
    overrun_pct = perf.overruns / n * 100.0
    print(
        "\n===== [RTF 诊断] ====="
        f"\n  周期数={n}  cycle_target={cycle_target * ms:.1f}ms"
        f"  仿真总时长={sim_total:.3f}s  墙钟总时长={wall:.3f}s"
        f"\n  实际 RTF={rtf:.3f}（目标 1.0）"
        f"\n  单周期均值: step(物理)={perf.step / n * ms:.2f}ms"
        f"  verify(判定)={perf.verify / n * ms:.2f}ms"
        f"  render(渲染)={perf.render / n * ms:.2f}ms"
        f"  sleep(限速)={perf.sleep / n * ms:.2f}ms"
        f"\n  超时周期(计算>{cycle_target * ms:.1f}ms)="
        f"{perf.overruns} 次 ({overrun_pct:.1f}%)"
        "\n=====================",
        flush=True,
    )
