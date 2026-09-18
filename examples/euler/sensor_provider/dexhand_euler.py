"""Ten provider sensors in OrcaGym observations, stepped by EulerEnv.

Run from the repository root with
``python -m examples.euler.sensor_provider.run --example hand_grid``.
The environment owns the only physics simulation; no standalone sensor runtime
is created here. Reset observations explicitly mark provider placeholders invalid.
"""

import argparse
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from gymnasium import spaces
import numpy as np

from examples.euler.sensor_provider.dexhand_control import (
    DEFAULT_RING_TARGET_SCALE, PREGRASP_MODES, DexHandGraspController,
    validate_grasp_options,
)
from examples.euler.sensor_provider.provider_paths import default_host_path, provider_path
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv
from orca_gym.sensor.providers.custom_scene import read_custom_sensor_instances


DEFAULT_XML = Path(__file__).resolve().parent / "scenes/dexhand/scene.xml"
TOUCH_IDS = tuple(f"touch_f{finger}" for finger in range(1, 6))
RANGE_IDS = tuple(f"range_f{finger}" for finger in range(1, 6))


class DexHandSensorEnv(OrcaGymEulerEnv):
    """Minimal task environment: raw actuator controls and explicit sensor obs.

    This is an integration example, not a trained policy or grasping task. The
    task selects ten expected channels; the base environment does not insert
    vendor outputs into observations automatically.
    """

    def __init__(self, build_dir: Path | None = None, *, xml_path: Path = DEFAULT_XML,
                 host_path: Path | None = None, provider_manifests=None,
                 frame_skip: int = 4, time_step: float = 0.001):
        if type(frame_skip) is not int or frame_skip <= 0:
            raise ValueError("frame_skip must be a positive integer")
        if not math.isfinite(time_step) or time_step <= 0:
            raise ValueError("time_step must be finite and positive")
        xml_path = Path(xml_path).resolve(strict=True)
        declarations = read_custom_sensor_instances(ET.parse(xml_path).getroot()) or ()
        expected = {name: "com.orca.examples.contact_grid" for name in TOUCH_IDS}
        expected.update({name: "com.orca.examples.rangefinder" for name in RANGE_IDS})
        if {spec["instance_id"]: spec["type_id"] for spec in declarations} != expected:
            raise ValueError("This task expects touch_f1..5 and range_f1..5 from the two example providers")
        packages = (list(provider_manifests) if provider_manifests is not None else
                    [provider_path("contact_grid", build_dir),
                     provider_path("rangefinder", build_dir)])
        self._has_provider_sample = False
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr="localhost:50051",
            agent_names=[],
            time_step=time_step,
            model_xml_path=str(xml_path),
            skip_grpc_load=True,
            render_mode="none",
            sensor_host_path=(str(host_path or default_host_path(build_dir))
                              if host_path is not None or build_dir is not None else None),
            sensor_provider_manifests=[str(path) for path in packages],
        )
        try:
            # Public model metadata determines actuator order and actual limits.
            lower = np.full(self.model.nu, -np.inf, dtype=np.float64)
            upper = np.full(self.model.nu, np.inf, dtype=np.float64)
            for actuator in self.model.get_actuator_dict().values():
                if actuator["CtrlLimited"]:
                    lower[actuator["ActuatorId"]], upper[actuator["ActuatorId"]] = actuator["CtrlRange"]
            self.action_space = spaces.Box(lower, upper, dtype=np.float64)
            sensor_spaces = {
                name: spaces.Box(0.0, np.inf, shape=(4, 4), dtype=np.float64)
                for name in TOUCH_IDS
            }
            sensor_spaces.update({
                name: spaces.Box(-1.0, np.inf, shape=(1,), dtype=np.float64)
                for name in RANGE_IDS
            })
            self.observation_space = spaces.Dict({
                "qpos": spaces.Box(-np.inf, np.inf, shape=(self.model.nq,), dtype=np.float64),
                "qvel": spaces.Box(-np.inf, np.inf, shape=(self.model.nv,), dtype=np.float64),
                "provider_sensors": spaces.Dict(sensor_spaces),
                "provider_valid": spaces.MultiBinary(1),
            })
        except BaseException:
            self.close()
            raise

    def reset_model(self) -> tuple[dict, dict]:
        """Restore the model without advancing physics to obtain a sensor value."""
        self._has_provider_sample = False
        self.set_joint_qpos(self.init_qpos.copy())
        self._sync_view()
        self.set_joint_qvel(self.init_qvel.copy())
        self.set_ctrl(np.zeros(self.model.nu, dtype=np.float64))
        self.mj_forward()
        self._sync_view()
        return self._get_obs(), {
            "provider_ready": False,
            "simulation_time": float(self.data.time),
        }

    def _get_obs(self) -> dict:
        if self._has_provider_sample:
            sensor_values = self.query_provider_sensor_data()
        else:
            # -1 here is a placeholder, not a measured miss: check provider_valid.
            sensor_values = {name: np.zeros((4, 4), dtype=np.float64) for name in TOUCH_IDS}
            sensor_values.update({name: np.full((1,), -1.0, dtype=np.float64) for name in RANGE_IDS})
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "provider_sensors": sensor_values,
            "provider_valid": np.array([self._has_provider_sample], dtype=np.int8),
        }

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        control = np.asarray(action, dtype=np.float64)
        if not np.isfinite(control).all() or not self.action_space.contains(control):
            raise ValueError("action must be a finite actuator vector within the declared control limits")
        self._has_provider_sample = False
        self.do_simulation(control, self.frame_skip)
        self._has_provider_sample = True
        observation = self._get_obs()
        info = {"provider_ready": True, "simulation_time": float(self.data.time)}
        # No reward or terminal condition: this example only demonstrates I/O.
        return observation, 0.0, False, False, info


def run(build_dir: Path | None = None, *, xml_path: Path = DEFAULT_XML, steps: int = 5,
        frame_skip: int = 4, time_step: float = 0.001,
        control: str = "neutral", pregrasp_mode="ramp",
        ring_target_scale=DEFAULT_RING_TARGET_SCALE,
        host_path: Path | None = None, provider_manifests=None) -> dict:
    """Run neutral or scripted grasp targets through the existing env.step."""
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if control not in ("neutral", "grasp"):
        raise ValueError("control must be neutral or grasp")
    validate_grasp_options(pregrasp_mode, ring_target_scale)
    env = DexHandSensorEnv(build_dir, xml_path=xml_path, host_path=host_path,
                          provider_manifests=provider_manifests,
                          frame_skip=frame_skip, time_step=time_step)
    try:
        observation, info = env.reset(seed=42)
        if observation["provider_valid"][0] or not env.observation_space.contains(observation):
            raise RuntimeError("Reset must return in-space placeholders marked invalid")
        initial_time = info["simulation_time"]
        controller = (DexHandGraspController(env.model, pregrasp_mode=pregrasp_mode,
                                           ring_target_scale=ring_target_scale)
                      if control == "grasp" else None)
        action = np.zeros(env.model.nu, dtype=np.float64)
        for _ in range(steps):
            if controller is not None:
                action = controller.action_at(info["simulation_time"] - initial_time)
            observation, _reward, _terminated, _truncated, info = env.step(action)
            if not observation["provider_valid"][0] or not env.observation_space.contains(observation):
                raise RuntimeError("Step must return valid provider data matching the observation space")
            if not all(np.isfinite(values).all() for values in observation["provider_sensors"].values()):
                raise RuntimeError("Nonfinite provider output")
        expected_time = initial_time + steps * env.dt
        if not math.isclose(info["simulation_time"], expected_time, rel_tol=1e-10, abs_tol=1e-12):
            raise RuntimeError("Physics must advance once per environment substep")
        return observation
    finally:
        env.close()


def run_dashboard(build_dir: Path | None = None, *, xml_path: Path = DEFAULT_XML, steps: int = 5000,
                  frame_skip: int = 4, time_step: float = 0.001, fps: float = 20,
                  control: str = "neutral", pregrasp_mode="ramp",
                  ring_target_scale=DEFAULT_RING_TARGET_SCALE,
                  interactive: bool = True, output: Path | None = None,
                  host_path: Path | None = None, provider_manifests=None) -> dict:
    """Show actual DLL outputs without adding a physics runtime or extra steps.

    Display refresh is decimated; the existing env.step still computes every
    provider at every physics substep. No synthetic force is inserted to make
    the chart interesting. Grasp mode moves the actuators to generate contacts;
    neutral mode may have no contact or ray hit. Commands are sampled once per
    Gym step and held over its physics substeps; chart FPS does not set timing.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if control not in ("neutral", "grasp"):
        raise ValueError("control must be neutral or grasp")
    validate_grasp_options(pregrasp_mode, ring_target_scale)
    if isinstance(fps, bool) or not math.isfinite(fps) or not 1 <= fps <= 60:
        raise ValueError("fps must be a finite value in 1..60")
    if output is not None:
        output = Path(output)
        if output.suffix.lower() != ".png":
            raise ValueError("Dashboard output must be a .png path")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Dashboard output already exists: {output}")
    # Lazy import keeps the console example usable without matplotlib/GUI.
    from examples.euler.sensor_provider.sensor_dashboard import SensorDashboard

    dashboard = SensorDashboard(interactive=interactive)
    env = None
    try:
        env = DexHandSensorEnv(build_dir, xml_path=xml_path, host_path=host_path,
                              provider_manifests=provider_manifests,
                              frame_skip=frame_skip, time_step=time_step)
        observation, info = env.reset(seed=42)
        initial_time = float(info["simulation_time"])
        controller = (DexHandGraspController(env.model, pregrasp_mode=pregrasp_mode,
                                           ring_target_scale=ring_target_scale)
                      if control == "grasp" else None)
        dashboard.update(observation, simulation_time=initial_time, source_time=None)
        start_wall = time.monotonic()
        next_draw = initial_time + 1 / fps
        action = np.zeros(env.model.nu, dtype=np.float64)
        for index in range(steps):
            if interactive and not dashboard.is_open():
                break
            if controller is not None:
                action = controller.action_at(info["simulation_time"] - initial_time)
            observation, _reward, _terminated, _truncated, info = env.step(action)
            simulation_time = float(info["simulation_time"])
            if simulation_time + 1e-12 >= next_draw or index == steps - 1:
                # This sample path uses Euler/implicit source-state sampling:
                # the last returned measurement precedes integrated time by dt.
                dashboard.update(observation, simulation_time=simulation_time,
                                 source_time=simulation_time - env.sim_config.timestep)
                next_draw = simulation_time + 1 / fps
                if interactive:
                    remaining = simulation_time - initial_time - (time.monotonic() - start_wall)
                    dashboard.poll(max(0.001, min(1 / fps, remaining)))
        if output is not None:
            dashboard.save(output)
        # Holding a completed plot does not need a live physics/DLL instance.
        env.close()
        env = None
        if interactive:
            while dashboard.is_open():
                dashboard.poll(0.05)
        return observation
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            dashboard.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--steps", type=int, default=None,
                        help="Gym steps (default: 5 neutral console; 5000 grasp/charts)")
    parser.add_argument("--control", choices=("neutral", "grasp"), default="neutral",
                        help="Neutral targets or scripted approach/close/hold (no grasp-success claim)")
    parser.add_argument("--pregrasp-mode", choices=PREGRASP_MODES, default="ramp",
                        help="Grasp approach: ramp (default), or step for high-impact comparison only")
    parser.add_argument("--ring-target-scale", type=float, default=DEFAULT_RING_TARGET_SCALE,
                        help="Ring finger closure target multiplier, 0.95..1.05 (default: 1.004)")
    parser.add_argument("--frame-skip", type=int, default=4, help="Physics substeps per env.step")
    parser.add_argument("--time-step", type=float, default=0.001, help="Seconds per physics substep")
    parser.add_argument("--host", type=Path)
    parser.add_argument("--provider", type=Path, action="append", help="Trusted provider.json; repeat per package")
    parser.add_argument("--plot", action="store_true", help="Show a live sensor dashboard; close the window to stop")
    parser.add_argument("--output", type=Path, help="Save a new PNG (headless unless --plot is also supplied)")
    parser.add_argument("--fps", type=float, default=20, help="Dashboard refresh rate, 1..60 (not physics frequency)")
    args = parser.parse_args()
    charts = args.plot or args.output is not None
    steps = args.steps if args.steps is not None else (5000 if charts or args.control == "grasp" else 5)
    options = dict(xml_path=args.xml, steps=steps, frame_skip=args.frame_skip,
                   time_step=args.time_step, control=args.control, host_path=args.host,
                   provider_manifests=args.provider, pregrasp_mode=args.pregrasp_mode,
                   ring_target_scale=args.ring_target_scale)
    if charts:
        observation = run_dashboard(args.build_dir, **options, fps=args.fps,
                                    interactive=args.plot, output=args.output)
    else:
        observation = run(args.build_dir, **options)
    for name, values in observation["provider_sensors"].items():
        print(f"{name}: shape={values.shape}, dtype={values.dtype}, values={values.tolist()}")
    if charts:
        if args.output is not None:
            print(f"Dashboard: {args.output.resolve()}")
        print(f"Dashboard finished; provider_valid={int(observation['provider_valid'][0])}")
    else:
        print(f"PASS: {steps} Gym steps, {steps * args.frame_skip} physics substeps; "
              "ten provider channels in task-defined observations, provider_valid=1")


if __name__ == "__main__":
    main()
