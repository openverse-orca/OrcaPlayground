d12_robot_config = {
    "robot_type": "dual_arm_waist",
    "base": {
        "base_body_name": "base_link",
        "dummy_joint_name": "dummy_joint",
    },
    "waist": {
        "joint_names": ["waist_yaw_joint"],
        "motor_names": ["M_waist_01"],
        "position_names": ["P_waist_01"],
    },
    "right_arm": {
        "joint_names": [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_pitch_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        "neutral_joint_values": [-0.67, -0.72, 0.87, 0.03, 0.83, 0.0, 0.0],
        "motor_names": [
            "M_arm_r_01", "M_arm_r_02", "M_arm_r_03", "M_arm_r_04",
            "M_arm_r_05", "M_arm_r_06", "M_arm_r_07",
        ],
        "position_names": [
            "P_arm_r_01", "P_arm_r_02", "P_arm_r_03", "P_arm_r_04",
            "P_arm_r_05", "P_arm_r_06", "P_arm_r_07",
        ],
        "positions_init_ctrl": [-1.9, 0.5, 0, 2.0, -1.5708, 0, 0],
        "positions_ranges": [
            (-2.96706, 2.96706), (-1.8326, 1.8326), (-2.96706, 2.96706),
            (0, 2.96706), (-2.96706, 2.96706), (-1.8326, 1.8326), (-1.0472, 1.0472),
        ],
        "ee_center_site_name": "ee_center_site_r",
    },
    "left_arm": {
        "joint_names": [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
            "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ],
        "neutral_joint_values": [-0.67, 0.72, -0.87, -0.03, -0.83, 0.0, 0.0],
        "motor_names": [
            "M_arm_l_01", "M_arm_l_02", "M_arm_l_03", "M_arm_l_04",
            "M_arm_l_05", "M_arm_l_06", "M_arm_l_07",
        ],
        "position_names": [
            "P_arm_l_01", "P_arm_l_02", "P_arm_l_03", "P_arm_l_04",
            "P_arm_l_05", "P_arm_l_06", "P_arm_l_07",
        ],
        "positions_init_ctrl": [1.9, -0.5, 0, 2.0, 1.5708, 0, 0],
        "positions_ranges": [
            (-2.96706, 2.96706), (-1.8326, 1.8326), (-2.96706, 2.96706),
            (0, 2.96706), (-2.96706, 2.96706), (-1.8326, 1.8326), (-1.0472, 1.0472),
        ],
        "ee_center_site_name": "ee_center_site",
    },
    "gripper_l": {
        "joint_names": ["l_left_driver_joint"],
        "actuator_names": ["l_fingers_actuator"],
        "actuator_ranges": [(0, 255)],
        "init_ctrl": [0],
    },
    "gripper_r": {
        "joint_names": ["r_right_driver_joint"],
        "actuator_names": ["r_fingers_actuator"],
        "actuator_ranges": [(0, 255)],
        "init_ctrl": [0],
    },
    "cameras": {
        "head": "camera_head_color",
        "wrist_l": "camera_wrist_l_color",
        "wrist_r": "camera_wrist_r_color",
    },
}
