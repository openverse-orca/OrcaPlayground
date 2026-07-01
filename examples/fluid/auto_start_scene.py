"""OrcaEditor：按 ORCA_LEVEL_NAME 打开关卡并进入 Game/Play。

供 ``run_cloth_studio.sh`` / ``AUTO_START_STUDIO=1`` 调用，等待 OrcaGym :50051 与 PBDRender :50261。
"""
import os
import sys

import azlmbr.legacy.general as gen

LEVEL_NAME = os.environ.get("ORCA_LEVEL_NAME", "test20260508")
EDITOR_INIT_WAIT_SEC = 5
SCENE_LOAD_WAIT_SEC = 10
GAME_MODE_SETTLE_SEC = 10

print("[AUTO_START] === OrcaStudio Auto-Start Script ===")
print(f"[AUTO_START] Target level: {LEVEL_NAME}")

print(f"[AUTO_START] Step 1: Waiting {EDITOR_INIT_WAIT_SEC}s for editor initialization...")
gen.idle_wait(EDITOR_INIT_WAIT_SEC)

print(f"[AUTO_START] Step 2: Opening level '{LEVEL_NAME}'...")
result = gen.open_level_no_prompt(LEVEL_NAME)
if not result:
    print(f"[AUTO_START] ERROR: Failed to open level '{LEVEL_NAME}'")
    sys.exit(1)
print("[AUTO_START] Level open request sent successfully")

print(f"[AUTO_START] Step 3: Waiting {SCENE_LOAD_WAIT_SEC}s for scene to load...")
gen.idle_wait(SCENE_LOAD_WAIT_SEC)

print("[AUTO_START] Step 4: Entering game mode (Run -> Start)...")
gen.enter_game_mode()
print("[AUTO_START] Game mode request sent")

print(f"[AUTO_START] Step 5: Waiting {GAME_MODE_SETTLE_SEC}s for MuJoCo to initialize...")
gen.idle_wait(GAME_MODE_SETTLE_SEC)

print("[AUTO_START] === Auto-start complete ===")
print("[AUTO_START] OrcaGym :50051 and PBDRender :50261 should be listening")
