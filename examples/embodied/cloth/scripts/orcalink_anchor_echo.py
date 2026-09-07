#!/usr/bin/env python3
"""OrcaLink 订阅端 echo：打印 Server DEQUEUE_SUBSCRIBE 对应的接收数据（调试用）。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def _run(config_path: Path, max_frames: int) -> int:
    from orcalink_client import OrcaLinkClient
    from orcalink_client.config_loader import _build_orcalink_config_from_dict

    from modules.orcalink_settings import require_orcalink_port  # noqa: E402

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ol = cfg["orcalink"]
    port = require_orcalink_port(ol)
    client_cfg = ol["client"]
    session_cfg = dict(client_cfg.get("session", {}))
    session_cfg["expected_clients"] = 1
    pos_ch = client_cfg.get("channels", {}).get("position", {})
    pos_id = int(pos_ch.get("channel_id", 20))
    data = {
        "orcalink_client": {
            "enabled": True,
            "server_address": f"{ol.get('host', 'localhost')}:{port}",
            "session_id": client_cfg.get("session_id", 101),
            "client_name": "anchor_echo_debug",
            "update_rate_hz": client_cfg.get("update_rate_hz", 50),
            "session": session_cfg,
        },
        "orcalink_bridge": {
            "coupling_mode": "force_position",
            "force_position": {
                "channels": {
                    "force": {"channel_id": 1, "publish": False, "subscribe": False},
                    "position": {
                        "channel_id": pos_id,
                        "publish": False,
                        "subscribe": True,
                    },
                },
            },
        },
    }
    config = _build_orcalink_config_from_dict(data)
    client = OrcaLinkClient(config)
    if not await client.initialize():
        print("连接失败", file=sys.stderr)
        return 1

    os.environ.setdefault("ORCALINK_DEBUG_ANCHOR", "1")
    got = 0
    while got < max_frames:
        frames = await client.subscribe_positions(max_count=1)
        if not frames:
            await asyncio.sleep(0.02)
            continue
        got += 1
        print(f"[ECHO SUBSCRIBE] batch={got} (legacy RigidBodyPosition count={len(frames)})")
    await client.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "cloth_sim_config.phase1_slide.json")
    parser.add_argument("--max-frames", type=int, default=5)
    args = parser.parse_args()
    return asyncio.run(_run(args.config, args.max_frames))


if __name__ == "__main__":
    sys.exit(main())
