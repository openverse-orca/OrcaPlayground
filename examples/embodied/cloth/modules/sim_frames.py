"""宏帧计数：MuJoCo 每 20 个子步 +1；XPBD 每 40 个子步 +1（均对应 0.02 s）。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MujocoMacroFrameCounter:
    """每调用一次 on_substep() 表示完成 1 次 mj_step（默认 0.001 s）。"""

    substeps_per_macro_frame: int = 20
    substep_index: int = 0
    macro_frame: int = 0

    def on_substep(self) -> bool:
        self.substep_index += 1
        if self.substep_index < self.substeps_per_macro_frame:
            return False
        self.substep_index = 0
        self.macro_frame += 1
        return True


@dataclass
class XpbdMacroFrameCounter:
    """每调用一次 on_substep() 表示完成 1 次 phys_world_step（默认 sdt=5e-4 s）。"""

    substeps_per_macro_frame: int = 40  # 注意这里40来自于 0.02/0.0005
    substep_index: int = 0
    macro_frame: int = 0

    def on_substep(self) -> bool:
        self.substep_index += 1
        if self.substep_index < self.substeps_per_macro_frame:
            return False
        self.substep_index = 0
        self.macro_frame += 1
        return True
