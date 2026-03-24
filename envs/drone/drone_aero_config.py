from dataclasses import dataclass


@dataclass(frozen=True)
class RotorAeroConfig:
    hover_omega_reference: float = 12.0
    torque_ratio: float = 0.025
    motor_time_constant: float = 0.045
    idle_omega: float = 6.0
    max_omega: float = 26.0


@dataclass(frozen=True)
class BodyDragConfig:
    linear_xy: float = 0.10
    linear_z: float = 0.14
    quadratic_xy: float = 0.05
    quadratic_z: float = 0.08
    angular_xy: float = 0.060
    angular_z: float = 0.050
    # 世界系水平线速度阻尼 (N/(m/s))；不宜过大，高速时与二次阻力叠加会失控
    world_xy_velocity_damping: float = 0.18
    # 计算阻力/阻尼时用的线速度裁剪 (m/s)，避免 v≫10 时 0.05*v² 量级达数百牛
    aero_model_velocity_clip: float = 6.0
    # 角阻尼用力矩分量上限 (N·m)；用真实角速度算 -k*ω，再裁剪，避免 ω≫25 时等效阻尼过弱而翻飞
    angular_drag_torque_axis_max: float = 0.60
    # 姿态杆全在死区内时，额外角阻尼系数与力矩上限（耗散机械臂扰动）
    zero_cmd_angular_hold_k: float = 0.02
    zero_cmd_angular_torque_axis_max: float = 0.08
    # 机体 xfrc 力矩向量范数上限 (N·m)，防止数值尖峰
    max_body_torque_norm: float = 1.20
    # 世界系线合力上限 = 系数 × hover_thrust，防止数值/接触偶发巨大外力
    max_total_linear_force_over_hover: float = 8.0
    # 杆量接近零时：用真实线速度做世界系阻尼（带力上限），耗散机械臂等扰动，避免只靠裁剪速度导致“飘走”
    zero_cmd_hold_deadband: float = 0.05
    zero_cmd_xy_hold_k: float = 0.30  # N/(m/s)，与 world_xy_velocity_damping 叠加
    zero_cmd_xy_hold_force_cap: float = 1.5  # N/轴
    zero_cmd_z_hold_k: float = 0.45  # N/(m/s)
    zero_cmd_z_hold_force_cap: float = 1.2  # N
    # free joint 速度安全钳制（数值保险丝，防止耦合导致发散）
    free_linear_speed_cap: float = 60.0  # m/s
    free_angular_speed_cap: float = 35.0  # rad/s


@dataclass(frozen=True)
class VerticalZOnlyConfig:
    """仅世界 Z 推力 + vz 阻尼的一维竖直实验；与 full 模式互斥。"""

    enabled: bool = False
    vz_linear_k: float = 0.10
    vz_quadratic_k: float = 0.05
    # 每步锁定姿态，并将水平线速度、角速度置零，仅保留 vz
    lock_pose_and_planar_velocity: bool = True
    # True：四元数固定为世界朝上 (wxyz=1,0,0,0)，使世界 +Z 推力对应机体系 +Z；False 则锁回场景初始四元数
    lock_quat_world_up: bool = True
    # True：xfrc 加在 drone_frame（与 free 关节同体）。False 则加在子体 Drone 上——装配角会使世界 +Z 力对 free 关节产生力矩，
    # 与每步清零角速度冲突，易出现地面弹跳/假「方向反了」。
    apply_thrust_at_free_frame: bool = True
    # 推力爬升时每间隔多少秒打一条 INFO（sim_t、T/mg、vz、z、桨平均角速度）
    ramp_progress_log_interval_s: float = 1.0
    # 推力线性爬升：T(t) = hover * lerp(t0_factor, t1_factor, min(1, sim_time/duration))
    thrust_ramp_enabled: bool = False
    thrust_ramp_t0_factor: float = 0.65
    # 默认 >1 较多：地面接触/软穿透下 T≈1.35mg 常仍压在支撑面上，难以持续离地爬升
    thrust_ramp_t1_factor: float = 2.05
    thrust_ramp_duration_s: float = 25.0
    # 固定 T/(mg)（>=0 时启用，且与 thrust_ramp 互斥；用于二分搜索临界推力）
    fixed_thrust_over_hover: float = -1.0
    # 无固定、无 ramp 时：推力 = keyboard_baseline_thrust_over_hover·mg + (R−F 杆量)·scale；用于在二分结果附近手调
    keyboard_baseline_thrust_over_hover: float = 1.0022
    # 首次 vz 过阈（易与弹跳混淆）；False 时仅依赖下方「持续起飞」判据
    takeoff_log_first_vz_spike: bool = False
    takeoff_vz_threshold: float = 0.08
    # 持续起飞：相对 reset 时 drone_frame 高度、vz、持续时间同时满足，作为更稳的临界标定
    takeoff_sustain_vz_threshold: float = 0.04
    takeoff_sustain_dz_m: float = 0.06
    takeoff_sustain_time_s: float = 0.22


@dataclass(frozen=True)
class GroundEffectConfig:
    enabled: bool = True
    rotor_radius: float = 0.045
    min_height: float = 0.05
    active_height: float = 0.32
    gain: float = 0.18
    max_factor: float = 1.22
    # 机体系 +Z 与世界 +Z 夹角超过该值时不做地面效应（大倾角时放大 Z 易数值失稳）
    max_tilt_deg: float = 48.0


@dataclass(frozen=True)
class DroneAeroConfig:
    rotor: RotorAeroConfig = RotorAeroConfig()
    drag: BodyDragConfig = BodyDragConfig()
    ground_effect: GroundEffectConfig = GroundEffectConfig()
    vertical_z_only: VerticalZOnlyConfig = VerticalZOnlyConfig()


DEFAULT_DRONE_AERO_CONFIG = DroneAeroConfig()
