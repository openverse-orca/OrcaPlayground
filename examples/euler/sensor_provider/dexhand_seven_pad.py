"""Five seven-pad DLL instances bound to the complete hand's existing surfaces.

Each input row belongs to one of seven real pad bodies. One Euler environment
owns the physics and the five DLL instances. A temporary scene changes only
sensor declarations and resource paths; the distributed MJCF and meshes remain
unchanged. No native plugin configuration is interpreted or converted.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
import xml.etree.ElementTree as ET

import numpy as np

from examples.euler.sensor_provider.dexhand_control import (
    DEFAULT_RING_TARGET_SCALE, PREGRASP_MODES, DexHandGraspController,
    validate_grasp_options,
)
from examples.euler.sensor_provider.provider_paths import default_host_path, provider_path
from examples.euler.sensor_provider.scene_env import ProviderSceneEnv


DEFAULT_XML = Path(__file__).resolve().parent / "scenes/dexhand/scene.xml"
TYPE_ID = "com.orca.examples.seven_pad"


@dataclass(frozen=True)
class HandMount:
    instance_id: str
    pads: tuple[str, ...]
    range_frame: str


def prepare_hand_xml(xml: str, asset_base: Path, *, noise_scale=.01) -> tuple[str, tuple[HandMount, ...]]:
    """Replace only custom bindings and resource lookup in a temporary copy.

    Each exact pad body owns its direct geometry, including unnamed geoms.
    The remaining finger and robot are not inferred as sensor ownership.
    """
    validate_run_options(1, "neutral", noise_scale)
    root = ET.fromstring(xml, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    if root.tag != "mujoco" or root.find(".//include") is not None:
        raise ValueError("The complete hand example requires one expanded MJCF document")
    if root.find(".//plugin") is not None:
        raise ValueError("Use the distributed hand asset without native plugin attachments")
    mountings = []
    for finger in range(1, 6):
        frame = f"site_l_f_link{finger}_4"
        if sum(node.get("name") == frame for node in root.findall("worldbody//site")) != 1:
            raise ValueError(f"Expected one exact fingertip site: {frame}")
        pads = tuple(f"force{finger}_f{pad}" for pad in range(1, 8))
        for pad in pads:
            bodies = [node for node in root.findall("worldbody//body") if node.get("name") == pad]
            if len(bodies) != 1 or not bodies[0].findall("geom"):
                raise ValueError(f"Expected one exact pad body with direct geometry: {pad}")
        mountings.append(HandMount(f"seven_pad_f{finger}", pads, frame))

    custom_blocks = root.findall("custom")
    if len(custom_blocks) != 1:
        raise ValueError("The hand scene requires one custom metadata block")
    custom = custom_blocks[0]
    for node in list(custom):
        if node.get("name", "").startswith(("orca.sensor.v", "orca.sensor.instances")):
            custom.remove(node)
    for finger, mounting in enumerate(mountings, 1):
        prefix = f"orca.sensor.v1/{mounting.instance_id}/"
        ET.SubElement(custom, "text", name=prefix + "plugin", data=TYPE_ID)
        for pad, name in enumerate(mounting.pads, 1):
            binding = ET.SubElement(custom, "tuple", name=prefix + f"object/force_f{pad}")
            ET.SubElement(binding, "element", objtype="body", objname=name)
        binding = ET.SubElement(custom, "tuple", name=prefix + "object/range_frame")
        ET.SubElement(binding, "element", objtype="site", objname=mounting.range_frame)
        ET.SubElement(custom, "numeric", name=prefix + "config/noise_scale", data=str(noise_scale))
        ET.SubElement(custom, "text", name=prefix + "seed", data=str(finger))
    ET.indent(custom, space="    ", level=1)
    custom.tail = None
    xml = re.sub(r"<custom\b[^>]*>.*?</custom>",
                 lambda _: ET.tostring(custom, encoding="unicode"), xml, count=1, flags=re.S)
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("The hand scene requires compiler resource paths")
    base = Path(asset_base).resolve()
    shared = compiler.get("assetdir", "")
    for key in ("meshdir", "texturedir"):
        compiler.set(key, str((base / compiler.get(key, shared)).resolve()))
    compiler.tail = None
    xml = re.sub(r"<compiler\b[^>]*(?:/>|>\s*</compiler>)",
                 lambda _: ET.tostring(compiler, encoding="unicode"), xml, count=1)
    # The hand's assets are meshes/textures; their file attributes stay intact.
    if any(asset.get("file") and asset.tag not in {"mesh", "texture"}
           for asset in root.findall("asset/*")):
        raise ValueError("The complete hand scene expects only mesh and texture file assets")
    return xml, tuple(mountings)


@contextmanager
def open_hand(xml_path=DEFAULT_XML, *, build_dir=None, host_path=None, package=None, noise_scale=.01):
    """Bind the five providers to existing geometry in a single EulerEnv."""
    path = Path(xml_path).resolve(strict=True)
    xml, mountings = prepare_hand_xml(path.read_text(encoding="utf-8"), path.parent,
                                     noise_scale=noise_scale)
    with TemporaryDirectory(prefix="orca-seven-pad-hand-") as directory:
        prepared = Path(directory) / "scene.xml"
        prepared.write_text(xml, encoding="utf-8")
        env = ProviderSceneEnv(
            prepared, [package or provider_path("seven_pad", build_dir)],
            sensor_shapes={mount.instance_id: (11,) for mount in mountings}, frame_skip=1,
            host_path=(host_path or default_host_path(build_dir))
            if host_path is not None or build_dir is not None else None,
        )
        try:
            env.reset()
            yield env
        finally:
            env.close()


def validate_run_options(steps, control, noise_scale):
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if control not in ("neutral", "grasp"):
        raise ValueError("control must be neutral or grasp")
    if (isinstance(noise_scale, bool) or not math.isfinite(noise_scale)
            or not 0 <= noise_scale <= 1_000_000):
        raise ValueError("noise_scale must be finite and in 0..1000000")


def hand_samples(env: ProviderSceneEnv, *, steps: int, control: str,
                 pregrasp_mode="ramp", ring_target_scale=DEFAULT_RING_TARGET_SCALE):
    """Yield one real DLL result per physics step, with both time meanings.

    Control and data recording do not depend on chart refresh. The Euler
    environment owns the only physics simulation and provider lifecycle.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if control not in ("neutral", "grasp"):
        raise ValueError("control must be neutral or grasp")
    validate_grasp_options(pregrasp_mode, ring_target_scale)
    controller = (DexHandGraspController(env.model, pregrasp_mode=pregrasp_mode,
                                       ring_target_scale=ring_target_scale)
                  if control == "grasp" else None)
    start_time = env.data.time
    for _ in range(steps):
        source_time = env.data.time
        action = (controller.action_at(source_time - start_time) if controller is not None
                  else np.zeros(env.model.nu, dtype=np.float64))
        outputs = env.step(action)[0]["provider_sensors"]
        if not math.isfinite(env.data.time) or env.data.time <= source_time:
            raise RuntimeError("Physics time did not advance; possible instability/reset. Stop the run.")
        for name, values in outputs.items():
            if values.shape != (11,) or values.dtype != np.dtype("float64") or not np.isfinite(values).all():
                raise RuntimeError(f"Invalid seven-pad DLL output: {name}")
        yield outputs, env.data.time, source_time


def run(build_dir: Path | None = None, *, xml_path=DEFAULT_XML, steps=20, control="neutral",
        host_path: Path | None = None, package: Path | None = None,
        noise_scale=.01, pregrasp_mode="ramp",
        ring_target_scale=DEFAULT_RING_TARGET_SCALE) -> dict[str, np.ndarray]:
    validate_run_options(steps, control, noise_scale)
    validate_grasp_options(pregrasp_mode, ring_target_scale)
    with open_hand(xml_path, build_dir=build_dir, host_path=host_path, package=package,
                   noise_scale=noise_scale) as env:
        for outputs, _simulation_time, _source_time in hand_samples(
                env, steps=steps, control=control,
                pregrasp_mode=pregrasp_mode, ring_target_scale=ring_target_scale):
            pass
    return outputs


def run_dashboard(build_dir: Path | None = None, *, xml_path=DEFAULT_XML, steps=10000,
                  control="grasp", noise_scale=.01, fps=20, history_size=10000,
                  pregrasp_mode="ramp", ring_target_scale=DEFAULT_RING_TARGET_SCALE,
                  interactive=True, output: Path | None = None,
                  host_path: Path | None = None, package: Path | None = None) -> dict[str, np.ndarray]:
    """Record every physics sample; refresh only the ten-curve force display.

    History keeps the last ``history_size`` physics samples, independently of
    FPS. The default window covers the entire default 10 s / 1 ms run. Closing
    a live window stops early; native resources close before the final plot is
    held open. No graph output is fabricated or computed outside the DLL.
    """
    validate_run_options(steps, control, noise_scale)
    validate_grasp_options(pregrasp_mode, ring_target_scale)
    if isinstance(fps, bool) or not math.isfinite(fps) or not 0 < fps <= 240:
        raise ValueError("fps must be finite and in (0, 240]")
    if output is not None:
        output = Path(output)
        if output.suffix.lower() != ".png":
            raise ValueError("output must have a .png suffix")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite chart: {output}")
    from examples.euler.sensor_provider.sensor_dashboard import HandForceDashboard

    dashboard = HandForceDashboard(interactive=interactive, history_size=history_size,
                                   noise_scale=noise_scale)
    outputs = {}
    try:
        with open_hand(xml_path, build_dir=build_dir, host_path=host_path, package=package,
                       noise_scale=noise_scale) as env:
            samples = hand_samples(env, steps=steps, control=control,
                                   pregrasp_mode=pregrasp_mode, ring_target_scale=ring_target_scale)
            initial_time, initial_wall = env.data.time, time.monotonic()
            next_refresh = initial_time + 1 / fps
            next_event_poll = initial_wall
            for index in range(steps):
                # A low drawing FPS must not starve the GUI close event.
                if interactive and time.monotonic() >= next_event_poll:
                    dashboard.poll(0)
                    next_event_poll = time.monotonic() + .02
                if interactive and not dashboard.is_open():
                    break
                outputs, simulation_time, source_time = next(samples)
                # Keep brief contact peaks, even between GUI refreshes.
                dashboard.record(outputs, simulation_time=simulation_time, source_time=source_time)
                if simulation_time + 1e-12 >= next_refresh or index == steps - 1:
                    dashboard.refresh()
                    next_refresh = simulation_time + 1 / fps
                    if interactive:
                        dashboard.poll(0)
                        deadline = initial_wall + simulation_time - initial_time
                        while dashboard.is_open() and time.monotonic() < deadline:
                            dashboard.poll(min(.01, max(0, deadline - time.monotonic())))
        if output is not None:
            dashboard.save(output)
        elif dashboard.is_open():
            dashboard.refresh()
        if interactive:
            while dashboard.is_open():
                dashboard.poll(.05)
        return outputs
    finally:
        dashboard.close()


def print_outputs(outputs):
    for name, values in outputs.items():
        print(f"{name}: shape={values.shape}, dtype={values.dtype}")
        print(f"  distance[m], Fn[N], Ft[N], direction[rad]: {values[:4]}")
        print(f"  seven synthetic response channels (not calibrated capacitance): {values[4:]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--steps", type=int, default=None,
                        help="Physics steps (default: 10000 for charts/grasp, otherwise 20)")
    parser.add_argument("--host", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--noise-scale", type=float, default=.01)
    parser.add_argument("--control", choices=("neutral", "grasp"), default=None,
                        help="Default: grasp with charts, neutral for the console smoke test")
    parser.add_argument("--pregrasp-mode", choices=PREGRASP_MODES, default="ramp",
                        help="Grasp approach: ramp (default), or step for high-impact comparison only")
    parser.add_argument("--ring-target-scale", type=float, default=DEFAULT_RING_TARGET_SCALE,
                        help="Ring finger closure target multiplier, 0.95..1.05 (default: 1.004)")
    parser.add_argument("--plot", action="store_true", help="Show five normal/tangential force pairs")
    parser.add_argument("--output", type=Path, help="Save a new PNG; headless unless --plot is also given")
    parser.add_argument("--fps", type=float, default=20, help="Chart refresh only, not recording/control rate")
    parser.add_argument("--history-size", type=int, default=10000, help="Number of physics samples retained")
    args = parser.parse_args()
    charts = args.plot or args.output is not None
    control = args.control or ("grasp" if charts else "neutral")
    steps = args.steps if args.steps is not None else (10000 if charts or control == "grasp" else 20)
    options = dict(xml_path=args.xml, steps=steps, control=control, host_path=args.host,
                   package=args.package, noise_scale=args.noise_scale,
                   pregrasp_mode=args.pregrasp_mode, ring_target_scale=args.ring_target_scale)
    if charts:
        outputs = run_dashboard(args.build_dir, **options, interactive=args.plot,
                                output=args.output, fps=args.fps, history_size=args.history_size)
    else:
        outputs = run(args.build_dir, **options)
    print_outputs(outputs)
    if args.output is not None:
        print(f"Force chart: {args.output.resolve()}")
    print("Five explicit seven-surface instances; the distributed scene and resources were not changed.")
    print("Open-loop targets do not establish grasp success. Synthetic channels are not product calibration.")


if __name__ == "__main__":
    main()
