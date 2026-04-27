so101_config = {
    "robot_type": "single_arm",
    "base": {
        "base_body_name": "base",
        "base_site_name": "baseframe",
    },
    "arm": {
        "joint_names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ],
        "neutral_joint_values": [0.0, 0.0, 0.0, 0.0, 0.0],
        "motor_names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ],
        "position_names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ],
        "ee_center_site_name": "gripperframe",
    },
    "gripper": {
        "joint_name": "gripper",
        "actuator_name": "gripper",
        "neutral_value": 0.0,
    },
}
