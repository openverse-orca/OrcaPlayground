"""Fixed-step simulation runner for control execution."""

from __future__ import annotations

from math import ceil
from typing import Callable

import mujoco

from doubleGripper_towel.control.auto_step_controller import AutoStepController


class SimulationRunner:
    """Run model/data for a fixed horizon with optional callback."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        controller: AutoStepController | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.controller = controller

    def run(
        self,
        total_seconds: float,
        post_step: Callable[[int, float], None] | None = None,
    ) -> tuple[int, float]:
        """Run for total_seconds and return (steps, dt)."""
        dt = float(self.model.opt.timestep)
        total_steps = int(ceil(max(total_seconds, 0.0) / dt))
        for step in range(total_steps):
            if self.controller is not None:
                self.controller.apply(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
            if post_step is not None:
                post_step(step, float(self.data.time))
        return total_steps, dt
