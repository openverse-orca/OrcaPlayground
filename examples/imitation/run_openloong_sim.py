import argparse
import os
import sys

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import orca_gym.scripts.openloong_manipulation as openloong_manipulation

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenLoong manipulation with imitation learning.")
    parser.add_argument("--orcagym_address", type=str, default="localhost:50051", help="The gRPC addresses to connect to")
    parser.add_argument("--agent_name", type=str, default="OpenLoongHand", help="The agent name to control")
    parser.add_argument("--run_mode", type=str, default="teleoperation", help="The run mode (teleoperation / playback / imitation / rollout / augmentation)")
    parser.add_argument("--action_type", type=str, default="end_effector", help="The action type (end_effector / joint_pos)")
    parser.add_argument("--action_step", type=int, default=5, help="How many simulation steps to take for each action")
    parser.add_argument("--prompt", type=str, default="Do something.", help="The task instruction")
    parser.add_argument("--task_config", type=str, help="The task config file to load")
    parser.add_argument("--algo", type=str, default="bc", help="The algorithm to use for training")
    parser.add_argument("--dataset", type=str, help="The file path to save the record")
    parser.add_argument("--model_file", type=str, help="The model file to load for rollout")
    parser.add_argument("--record_length", type=int, default=1200, help="The time length in seconds to record")
    parser.add_argument("--ctrl_device", type=str, default="vr", help="The control device to use")
    parser.add_argument("--playback_mode", type=str, default="random", help="The playback mode (loop or random)")
    parser.add_argument("--rollout_times", type=int, default=10, help="The times to rollout the policy")
    parser.add_argument("--augmented_scale", type=float, default=0.01, help="The scale to augment the dataset")
    parser.add_argument("--augmented_rounds", type=int, default=3, help="The times to augment the dataset")
    parser.add_argument("--teleoperation_rounds", type=int, default=20, help="The rounds to do teleoperation")
    parser.add_argument("--sample_range", type=float, default=0.0, help="The area range to sample positions")
    parser.add_argument("--realtime_playback", type=bool, default=True, help="Enable real-time playback or rollout")

    args = parser.parse_args()

    openloong_manipulation.run_openloong_sim(args, project_root, current_file_path)
