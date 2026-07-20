import numpy as np

FrankaEnvConfig = {
    "TIME_STEP": 0.002,
    "FRAME_SKIP_SHORT": 10,
    "FRAME_SKIP_LONG": 25,
    "ACTION_SKIP": 1,
}

FrankaObsConfig = {
    "scale": {
        "ee_position": 1.0,
        "ee_velocity": 1.0,
        "object_position": 1.0,
        "object_rotation": 1.0,
        "object_velp": 1.0,
        "object_velr": 1.0,
    },
    "noise": {
        "noise_level": 0.0,
    },
}

CurriculumConfig = {
    "ground_contact_body_names": [],
}

RewardConfig = {
    "reach": {
        "sparse": False,
        "distance_threshold": 0.05,
    },
    "pick_and_place": {
        "sparse": False,
        "distance_threshold": 0.05,
    },
}

FrankaRobotConfig = {
    "panda": {
        "base_joint_name": "link0",
        "arm_joint_names": [
            "joint1", "joint2", "joint3",
            "joint4", "joint5", "joint6", "joint7",
        ],
        "gripper_joint_names": ["finger_joint1", "finger_joint2"],
        "neutral_joint_values": np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00]),

        "actuator_names": [
            "actuator1", "actuator2", "actuator3",
            "actuator4", "actuator5", "actuator6", "actuator7",
            "r_gripper_finger_joint", "l_gripper_finger_joint",
        ],
        "actuator_type": "position",
        "action_scale": np.ones(9),
        "kps": np.ones(9) * 20.0,
        "kds": np.ones(9) * 0.5,

        "ee_site_name": "ee_center_site",
        "obj_site_name": "object_site",
        "obj_joint_name": "object_joint",
        "mocap_name": "panda_mocap",
        "goal_site_name": "goal",
        "mocap_pos_range": np.array([[-20.0, 20.0], [-20.0, 20.0], [0.0, 2.0]]),

        "has_object": False,
        "block_gripper": True,
        "distance_threshold": 0.05,
        "goal_xy_range": 0.5,
        "obj_xy_range": 0.3,
        "goal_x_offset": 0.0,
        "goal_z_range": 0.3,

        "soft_joint_qpos_limit": 1.0,
        "soft_joint_qvel_limit": 1.0,
        "soft_torque_limit": 1.0,
        "joint_qvel_range": np.ones(9),

        "log_agent_names": [],
        "visualize_command_agent_names": [],
        "playable_agent_name": "",
        "command_indicator_name": "goal_mocap",
    },
}

TaskConfig = {
    "reach": {
        "has_object": False,
        "block_gripper": True,
        "distance_threshold": 0.05,
        "goal_xy_range": 0.5,
        "obj_xy_range": 0.3,
        "goal_x_offset": 0.0,
        "goal_z_range": 0.3,
        "max_episode_steps": 250,
        "frame_skip": FrankaEnvConfig["FRAME_SKIP_LONG"],
        "action_size": 3,
    },
    "pick_and_place": {
        "has_object": True,
        "block_gripper": False,
        "distance_threshold": 0.05,
        "goal_xy_range": 0.3,
        "obj_xy_range": 0.3,
        "goal_x_offset": 0.0,
        "goal_z_range": 0.2,
        "max_episode_steps": 500,
        "frame_skip": FrankaEnvConfig["FRAME_SKIP_SHORT"],
        "action_size": 4,
    },
}
