from __future__ import annotations

from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parents[1]
DEFAULT_XML_PATH = REPO_ROOT / "assets" / "so101" / "so101_new_calib.xml"
DEFAULT_ORCAGYM_ADDR = "localhost:50051"
DEFAULT_POLICY_HOST = "localhost"
DEFAULT_POLICY_PORT = 8000
DEFAULT_MONITOR_PORTS = [7070, 7090]
DEFAULT_FPS = 30
SIM_TIME_STEP = 0.001
AGENT_NAME = "Group_so101_new_calib_usda"
ACTION_DIM = 12
ARM_OFFSET = 6
GRIPPER_INDEX = 11
