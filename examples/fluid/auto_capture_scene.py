import os
import sys
import time
import threading

import azlmbr.legacy.general as gen
import azlmbr.atom
import azlmbr.bus

LEVEL_NAME = os.environ.get("ORCA_LEVEL_NAME", "water_example")
CAPTURE_DIR = os.environ.get("ORCA_CAPTURE_DIR", "/tmp/orca_captures")
CAPTURE_FPS = int(os.environ.get("ORCA_CAPTURE_FPS", "10"))
CAPTURE_DURATION = int(os.environ.get("ORCA_CAPTURE_DURATION", "6"))
EDITOR_INIT_WAIT_SEC = 5
SCENE_LOAD_WAIT_SEC = 10
GAME_MODE_SETTLE_SEC = 15

class ScreenshotCapture:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.frame_index = 0
        self.done = False
        self.handler = None

    def capture(self, filepath):
        self.done = False
        frame_capture_id = azlmbr.atom.FrameCaptureRequestBus(
            azlmbr.bus.Broadcast, "CaptureScreenshot", filepath
        )
        if frame_capture_id != -1:
            self.handler = azlmbr.atom.FrameCaptureNotificationBusHandler()
            self.handler.connect(frame_capture_id)
            self.handler.add_callback('OnCaptureFinished', self._on_captured)
            waited = 0
            while not self.done and waited < 120:
                gen.idle_wait_frames(1)
                waited += 1
            if self.handler:
                self.handler.disconnect()
            return True
        else:
            print(f"[CAPTURE] Screenshot failed for {filepath}")
            return False

    def _on_captured(self, parameters):
        if parameters[0] == azlmbr.atom.FrameCaptureResult_Success:
            pass
        else:
            print(f"[CAPTURE] Screenshot save failed: {parameters[1]}")
        self.done = True

def main():
    print("[AUTO_CAPTURE] === OrcaStudio Auto-Capture Script ===")
    print(f"[AUTO_CAPTURE] Level: {LEVEL_NAME}")
    print(f"[AUTO_CAPTURE] Capture dir: {CAPTURE_DIR}")
    print(f"[AUTO_CAPTURE] Capture FPS: {CAPTURE_FPS}")
    print(f"[AUTO_CAPTURE] Capture duration: {CAPTURE_DURATION}s")

    print(f"[AUTO_CAPTURE] Step 1: Waiting {EDITOR_INIT_WAIT_SEC}s for editor initialization...")
    gen.idle_wait(float(EDITOR_INIT_WAIT_SEC))

    print(f"[AUTO_CAPTURE] Step 2: Opening level '{LEVEL_NAME}'...")
    result = gen.open_level_no_prompt(LEVEL_NAME)
    if not result:
        print(f"[AUTO_CAPTURE] ERROR: Failed to open level '{LEVEL_NAME}'")
        sys.exit(1)
    print(f"[AUTO_CAPTURE] Level open request sent successfully")

    print(f"[AUTO_CAPTURE] Step 3: Waiting {SCENE_LOAD_WAIT_SEC}s for scene to load...")
    gen.idle_wait(float(SCENE_LOAD_WAIT_SEC))

    print(f"[AUTO_CAPTURE] Step 4: Entering game mode (Run -> Start)...")
    gen.enter_game_mode()
    print(f"[AUTO_CAPTURE] Game mode request sent")

    print(f"[AUTO_CAPTURE] Step 5: Waiting {GAME_MODE_SETTLE_SEC}s for MuJoCo to initialize...")
    gen.idle_wait(float(GAME_MODE_SETTLE_SEC))

    print(f"[AUTO_CAPTURE] Step 6: Starting frame capture for {CAPTURE_DURATION}s at {CAPTURE_FPS} FPS...")
    capturer = ScreenshotCapture(CAPTURE_DIR)
    frame_interval = 1.0 / CAPTURE_FPS
    total_frames = CAPTURE_DURATION * CAPTURE_FPS
    start_time = time.time()

    for i in range(total_frames):
        elapsed = time.time() - start_time
        if elapsed >= CAPTURE_DURATION:
            print(f"[AUTO_CAPTURE] Duration reached after {i} frames")
            break

        filepath = os.path.join(CAPTURE_DIR, f"frame_{i:05d}.png")
        success = capturer.capture(filepath)
        if success:
            if i % 10 == 0:
                print(f"[AUTO_CAPTURE] Captured frame {i}/{total_frames} ({elapsed:.1f}s)")
        else:
            print(f"[AUTO_CAPTURE] Failed to capture frame {i}")

        remaining = frame_interval - (time.time() - start_time - i * frame_interval)
        if remaining > 0:
            gen.idle_wait_frames(max(1, int(remaining * 60)))

    actual_elapsed = time.time() - start_time
    print(f"[AUTO_CAPTURE] === Capture complete ===")
    print(f"[AUTO_CAPTURE] Captured {total_frames} frames in {actual_elapsed:.1f}s")
    print(f"[AUTO_CAPTURE] Frames saved to: {CAPTURE_DIR}")

main()
