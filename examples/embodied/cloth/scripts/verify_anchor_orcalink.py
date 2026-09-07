#!/usr/bin/env python3
"""
Gate-0 校验：MuJoCo 每宏步发布一次，OrcaLink 订阅端逐 unit 对比发送/接收是否一致。

用法:
  python scripts/verify_anchor_orcalink.py
  python scripts/verify_anchor_orcalink.py --macro-frames 20 --no-server
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
sys.path.insert(0, str(ROOT))

from modules.anchor_frame import collect_anchor_frame  # noqa: E402
from modules.anchor_debug_export import open_debug_csv_writer  # noqa: E402
from modules.anchor_publish import frame_to_units  # noqa: E402
from modules.anchor_verify import compare_unit_dicts, flatten_units  # noqa: E402
from modules.body_map import load_body_map, validate_body_map  # noqa: E402
from modules.phase1_trajectory import compute_ctrl, trajectory_duration  # noqa: E402
from modules.orcalink_settings import resolve_orcalink_port  # noqa: E402
from modules.sim_frames import MujocoMacroFrameCounter  # noqa: E402


def _find_server_bin() -> Path | None:
    for rel in (
        "OrcaLink/bin/orcalink",
        "OrcaLink/build/Server/orcalink",
    ):
        p = (REPO / rel).resolve()
        if p.is_file():
            return p
    return None


def _build_client_cfg(
    *,
    host: str,
    port: int,
    session_id: int,
    client_name: str,
    channel_id: int,
    publish: bool,
    subscribe: bool,
    expected_clients: int,
    control_mode: str = "async",
) -> dict:
    return {
        "orcalink_client": {
            "enabled": True,
            "server_address": f"{host}:{port}",
            "session_id": session_id,
            "client_name": client_name,
            "update_rate_hz": 50,
            "session": {
                "control_mode": control_mode,
                "expected_clients": expected_clients,
                "ready_timeout_sec": 30.0,
            },
        },
        "orcalink_bridge": {
            "coupling_mode": "force_position",
            "force_position": {
                "channels": {
                    "force": {"channel_id": 1, "publish": False, "subscribe": False},
                    "position": {
                        "channel_id": channel_id,
                        "publish": publish,
                        "subscribe": subscribe,
                    },
                },
            },
        },
    }


def _publisher_thread(
    cfg: dict,
    host: str,
    port: int,
    session_id: int,
    channel_id: int,
    macro_frames: int,
    expected_by_seq: dict[int, dict[str, tuple[str, ...]]],
    done_evt: threading.Event,
    err_box: list[str],
) -> None:
    from orcalink_client import OrcaLinkClient
    from orcalink_client.config_loader import _build_orcalink_config_from_dict

    try:
        mj_cfg = cfg["mujoco"]
        scene = (ROOT / mj_cfg["model_path"]).resolve()
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        entries = load_body_map(model, cfg)
        frame_skip = int(mj_cfg.get("frame_skip", 20))
        traj_fn = compute_ctrl
        counter = MujocoMacroFrameCounter(substeps_per_macro_frame=frame_skip)

        pub_cfg = _build_client_cfg(
            host=host,
            port=port,
            session_id=session_id,
            client_name="anchor_verify_pub",
            channel_id=channel_id,
            publish=True,
            subscribe=False,
            expected_clients=2,
            control_mode="async",
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        pub = OrcaLinkClient(_build_orcalink_config_from_dict(pub_cfg))
        if not loop.run_until_complete(pub.initialize()):
            err_box.append("发布端 initialize 失败")
            return

        csv_writer = open_debug_csv_writer(cfg, ROOT)

        # 等订阅端就绪
        time.sleep(0.5)
        sent = 0
        while sent < macro_frames and data.time < trajectory_duration() + 0.5:
            frame = collect_anchor_frame(model, data, entries, counter.macro_frame)
            units = frame_to_units(frame)
            if csv_writer:
                csv_writer.write_macro_frame(frame, units)
            expected_by_seq[counter.macro_frame] = flatten_units(units)
            ok = loop.run_until_complete(
                pub.publish_anchor_frame(units, counter.macro_frame, frame.sim_time)
            )
            if not ok:
                err_box.append(f"macro_frame={counter.macro_frame} publish 失败")
                break
            sent += 1
            data.ctrl[:] = traj_fn(data.time)
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
                counter.on_substep()

        if csv_writer:
            csv_writer.close()
        loop.run_until_complete(pub.shutdown())
        loop.close()
        if sent < macro_frames:
            err_box.append(f"仅发布 {sent}/{macro_frames} 宏步")
    except Exception as e:
        err_box.append(f"发布线程异常: {e}")
    finally:
        done_evt.set()


async def _subscriber_loop(
    host: str,
    port: int,
    session_id: int,
    channel_id: int,
    macro_frames: int,
    expected_by_seq: dict[int, dict[str, tuple[str, ...]]],
    publisher_done: threading.Event,
    results: list[tuple[int, bool, list[str]]],
) -> None:
    from orcalink_client import OrcaLinkClient
    from orcalink_client.config_loader import _build_orcalink_config_from_dict

    sub_cfg = _build_client_cfg(
        host=host,
        port=port,
        session_id=session_id,
        client_name="anchor_verify_sub",
        channel_id=channel_id,
        publish=False,
        subscribe=True,
        expected_clients=2,
        control_mode="async",
    )
    sub = OrcaLinkClient(_build_orcalink_config_from_dict(sub_cfg))
    if not await sub.initialize():
        results.append((-1, False, ["订阅端 initialize 失败"]))
        return

    received = 0
    deadline = time.time() + 60.0
    while received < macro_frames and time.time() < deadline:
        frames = await sub.subscribe_anchor_frames(max_count=1)
        if not frames:
            if publisher_done.is_set():
                await asyncio.sleep(0.05)
                continue
            await asyncio.sleep(0.02)
            continue
        for frame in frames:
            seq = int(frame.sequence)
            recv_flat = flatten_units(list(frame.units))
            exp_flat = expected_by_seq.get(seq)
            if exp_flat is None:
                for _ in range(100):
                    await asyncio.sleep(0.01)
                    exp_flat = expected_by_seq.get(seq)
                    if exp_flat is not None:
                        break
            if exp_flat is None:
                results.append((seq, False, [f"macro_frame={seq} 尚无本地期望值（发布顺序问题）"]))
            else:
                errs = compare_unit_dicts(exp_flat, recv_flat)
                results.append((seq, len(errs) == 0, errs))
            received += 1
            if received >= macro_frames:
                break

    await sub.shutdown()
    if received < macro_frames:
        results.append((-2, False, [f"仅收到 {received}/{macro_frames} 宏步"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 MuJoCo 锚点宏步 vs OrcaLink 接收")
    parser.add_argument("--config", type=Path, default=ROOT / "cloth_sim_config.phase1_slide.json")
    parser.add_argument("--macro-frames", type=int, default=10, help="校验宏步数量")
    parser.add_argument("--no-server", action="store_true", help="不自动启动 OrcaLink Server")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    ol = cfg["orcalink"]
    host = ol.get("host", "localhost")
    port = resolve_orcalink_port(ol, args.port)
    session_id = int(ol["client"].get("session_id", 101))
    channel_id = int(ol["client"]["channels"]["position"]["channel_id"])

    scene = (ROOT / cfg["mujoco"]["model_path"]).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    entries = load_body_map(model, cfg)
    errs = validate_body_map(model, entries)
    if errs:
        print("body_map 校验失败:")
        for e in errs:
            print(" ", e)
        return 1

    units_per_frame = len(entries) * 10
    print(f"刚体 {len(entries)} 个，每宏步 {units_per_frame} 个 DataUnit，计划校验 {args.macro_frames} 宏步")

    server_proc = None
    if not args.no_server:
        server_bin = _find_server_bin()
        if server_bin is None:
            print("未找到 orcalink 可执行文件，请手动启动 Server 或使用 --no-server", file=sys.stderr)
            return 1
        server_proc = subprocess.Popen(
            [str(server_bin), "--port", str(port)],
            cwd=str(server_bin.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        print(f"已启动 OrcaLink Server pid={server_proc.pid} :{port}")

    expected_by_seq: dict[int, dict[str, tuple[str, ...]]] = {}
    pub_done = threading.Event()
    pub_err: list[str] = []
    results: list[tuple[int, bool, list[str]]] = []

    pub_thread = threading.Thread(
        target=_publisher_thread,
        args=(cfg, host, port, session_id, channel_id, args.macro_frames, expected_by_seq, pub_done, pub_err),
        daemon=True,
    )
    pub_thread.start()

    try:
        asyncio.run(
            _subscriber_loop(
                host, port, session_id, channel_id, args.macro_frames, expected_by_seq, pub_done, results
            )
        )
    finally:
        pub_thread.join(timeout=30)
        if server_proc:
            server_proc.terminate()

    passed = sum(1 for _, ok, _ in results if ok)
    failed = [r for r in results if not r[1]]

    print("\n========== 校验结果 ==========")
    for seq, ok, err_list in sorted(results, key=lambda x: x[0]):
        if ok:
            n = len(expected_by_seq.get(seq, {}))
            print(f"  macro_frame={seq:4d}  PASS  units={n}")
        else:
            print(f"  macro_frame={seq:4d}  FAIL")
            for e in err_list[:5]:
                print(f"    - {e}")
            if len(err_list) > 5:
                print(f"    ... 另有 {len(err_list) - 5} 条差异")

    if pub_err:
        print("\n发布端错误:")
        for e in pub_err:
            print(" ", e)

    total_checks = len([r for r in results if r[0] >= 0])
    print(f"\n合计: {passed}/{total_checks} 宏步通过, DataUnit/帧={units_per_frame}")
    if failed or pub_err or passed < args.macro_frames:
        return 1
    print("全部宏步发送与 OrcaLink 接收数据一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
