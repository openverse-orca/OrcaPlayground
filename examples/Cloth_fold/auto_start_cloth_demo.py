"""OrcaEditor：打开 cloth_demo 并 Play（启动 ParticleRender gRPC 50251）。

与 examples/fluid 无依赖；仅供 XPBD/PBD 布料粒子回放联调。
"""
import os
import sys

import azlmbr.legacy.general as gen

LEVEL_NAME = os.environ.get("ORCA_LEVEL_NAME", "cloth_demo")
EDITOR_INIT_WAIT_SEC = 8
SCENE_LOAD_WAIT_SEC = 15
GAME_MODE_SETTLE_SEC = 8

print("[CLOTH_FOLD] === cloth_demo auto start (PBD, not fluid) ===")
print(f"[CLOTH_FOLD] Level: {LEVEL_NAME}")

gen.idle_wait(float(EDITOR_INIT_WAIT_SEC))

print(f"[CLOTH_FOLD] Opening level '{LEVEL_NAME}'...")
if not gen.open_level_no_prompt(LEVEL_NAME):
    print(f"[CLOTH_FOLD] ERROR: open_level_no_prompt failed for '{LEVEL_NAME}'")
    sys.exit(1)

gen.idle_wait(float(SCENE_LOAD_WAIT_SEC))

print("[CLOTH_FOLD] Entering game mode (Play)...")
gen.enter_game_mode()
gen.idle_wait(float(GAME_MODE_SETTLE_SEC))

print("[CLOTH_FOLD] Done. Expect gRPC 0.0.0.0:50251 — verify: ss -lntp | grep 50251")
print("[CLOTH_FOLD] Then run: OrcaPlayground/examples/Cloth_fold/run_pbd_particle_replay.sh")
