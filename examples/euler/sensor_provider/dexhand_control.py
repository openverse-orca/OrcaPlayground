"""Named position targets for the bundled dexterous-hand demonstration.

This is an open-loop pose sequence, not a general-purpose grasp planner. It
only produces actuator controls; physics and sensor providers remain in Env.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np


APPROACH_TIME = 0.01
CLOSE_TIME = 1.0
SMOOTHING_PERIOD = 0.001
SMOOTHING_RATIO = 0.01
TARGET_TOLERANCE = 0.001
PREGRASP_MODES = ("ramp", "step")
DEFAULT_RING_TARGET_SCALE = 1.004


def validate_grasp_options(pregrasp_mode, ring_target_scale):
    """Validate example controls before opening physics or vendor libraries."""
    if pregrasp_mode not in PREGRASP_MODES:
        raise ValueError("pregrasp_mode must be ramp or step")
    if (isinstance(ring_target_scale, bool) or not math.isfinite(ring_target_scale)
            or not .95 <= ring_target_scale <= 1.05):
        raise ValueError("ring_target_scale must be finite and in 0.95..1.05")

APPROACH_TARGETS = {
    "mot_joint1": 0.05,
    "mot_joint2": -1.8508,
    "mot_joint5": 0.28,
    "mot_joint6": 1.5708,
    "act_joint1_1": 1.5708,
}
FINGER_TARGETS = {
    f"act_joint{finger}_{joint}": angle
    for (finger, joint), angle in zip(
        ((finger, joint) for finger in range(1, 6) for joint in range(2, 5)),
        (0.514, 0.686, 0.539, 0.664, 0.948, 0.795, 0.748, 0.997,
         0.847, 0.687, 0.916, 0.762, 0.416, 0.924, 0.770),
        strict=True,
    )
}
REQUIRED_ACTUATORS = (
    *(f"mot_joint{joint}" for joint in range(1, 7)),
    *(f"act_joint{finger}_{joint}" for finger in range(1, 6)
      for joint in range(1, 5)),
)


class DexHandGraspController:
    """Create controls using only ``model.nu`` and ``get_actuator_dict()``.

    Pass elapsed *simulation* seconds since reset to :meth:`action_at`, before
    ``env.step(action)``. Sampling the same time always produces the same target,
    regardless of frame skip, chart FPS, call order, or wall-clock speed. Env
    still holds each returned target throughout that step's physics substeps.

    The sequence starts from zero targets, ramps the approach between 0.01 s
    and 1 s, then closes and holds. Explicit ``step`` mode applies the full
    approach target at 0.01 s for comparison; it can cause severe impacts.
    Closing starts with a 1% increment and uses exponential
    smoothing equivalent to 1% per millisecond. This does not assert that the
    physical joints have reached their targets or that a grasp has succeeded.
    """

    def __init__(self, model, *, pregrasp_mode="ramp",
                 ring_target_scale=DEFAULT_RING_TARGET_SCALE):
        validate_grasp_options(pregrasp_mode, ring_target_scale)
        self._pregrasp_mode = pregrasp_mode
        finger_targets = {name: target * (ring_target_scale if name.startswith("act_joint4_") else 1.0)
                          for name, target in FINGER_TARGETS.items()}
        if isinstance(model.nu, bool) or not isinstance(model.nu, Integral) or model.nu <= 0:
            raise ValueError("model.nu must be a positive integer")
        self._nu = int(model.nu)
        actuators = model.get_actuator_dict()
        missing = sorted(set(REQUIRED_ACTUATORS) - actuators.keys())
        if missing:
            raise ValueError(f"Grasp model is missing actuators: {', '.join(missing)}")
        self._indices = {}
        used_ids = set()
        for name, metadata in actuators.items():
            index = metadata.get("ActuatorId")
            if (isinstance(index, bool) or not isinstance(index, Integral)
                    or not 0 <= index < self._nu or index in used_ids):
                raise ValueError(f"Invalid or duplicate ActuatorId for {name}: {index!r}")
            used_ids.add(index)
            self._indices[name] = int(index)
            values = np.asarray([0.0, APPROACH_TARGETS.get(name, 0.0),
                                 finger_targets.get(name, 0.0)], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite grasp target for {name}")
            if "CtrlLimited" not in metadata:
                raise ValueError(f"Missing CtrlLimited for {name}")
            if metadata["CtrlLimited"]:
                limits = np.asarray(metadata.get("CtrlRange"), dtype=np.float64)
                if (limits.shape != (2,) or not np.all(np.isfinite(limits))
                        or limits[0] > limits[1]):
                    raise ValueError(f"Invalid CtrlRange for {name}")
                if np.any(values < limits[0]) or np.any(values > limits[1]):
                    raise ValueError(f"Grasp target outside CtrlRange for {name}: {limits}")
        if len(used_ids) != self._nu:
            raise ValueError("Actuator metadata must cover every model control index")
        self._approach = np.zeros(self._nu, dtype=np.float64)
        for name, target in APPROACH_TARGETS.items():
            self._approach[self._indices[name]] = target
        self._finger_ids = np.asarray([self._indices[name] for name in FINGER_TARGETS])
        self._finger_targets = np.asarray(list(finger_targets.values()), dtype=np.float64)
        self._hold_time = CLOSE_TIME + SMOOTHING_PERIOD * (
            math.log(TARGET_TOLERANCE / float(self._finger_targets.max()))
            / math.log1p(-SMOOTHING_RATIO)
        )

    @staticmethod
    def _validate_time(elapsed_seconds: float) -> float:
        if isinstance(elapsed_seconds, bool):
            raise ValueError("elapsed_seconds must be finite and non-negative")
        elapsed = float(elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        return elapsed

    def phase_at(self, elapsed_seconds: float) -> str:
        """Return the command phase, not measured grasp success."""
        elapsed = self._validate_time(elapsed_seconds)
        if elapsed < APPROACH_TIME:
            return "settle"
        if elapsed < CLOSE_TIME:
            return "approach"
        return "hold" if elapsed >= self._hold_time else "close"

    def action_at(self, elapsed_seconds: float) -> np.ndarray:
        """Return an owned float64 control array indexed by actuator identity."""
        elapsed = self._validate_time(elapsed_seconds)
        if elapsed < APPROACH_TIME:
            return np.zeros(self._nu, dtype=np.float64)
        control = self._approach.copy()
        if elapsed < CLOSE_TIME and self._pregrasp_mode == "ramp":
            # At a 1 ms cadence, take 990 small increments, reaching the full
            # pose at 0.999 s before finger closure starts at 1 s.
            alpha = min(1.0, (elapsed - APPROACH_TIME + SMOOTHING_PERIOD)
                        / (CLOSE_TIME - APPROACH_TIME))
            control *= alpha
        if elapsed >= CLOSE_TIME:
            control[self._indices["mot_joint1"]] = 0.0
            if elapsed >= self._hold_time:
                control[self._finger_ids] = self._finger_targets
            else:
                periods = (elapsed - CLOSE_TIME) / SMOOTHING_PERIOD + 1.0
                remaining = self._finger_targets * math.exp(
                    periods * math.log1p(-SMOOTHING_RATIO))
                control[self._finger_ids] = np.where(
                    remaining / (1.0 - SMOOTHING_RATIO) <= TARGET_TOLERANCE,
                    self._finger_targets,
                    self._finger_targets - remaining,
                )
        return control
