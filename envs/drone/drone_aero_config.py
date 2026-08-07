from dataclasses import dataclass, field, replace


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
    angular_xy: float = 0.016
    angular_z: float = 0.016
    # 世界系水平线速度阻尼 (N/(m/s))；不宜过大，高速时与二次阻力叠加会失控
    world_xy_velocity_damping: float = 0.10
    # 计算阻力/阻尼时用的线速度裁剪 (m/s)，避免 v≫10 时 0.05*v² 量级达数百牛
    aero_model_velocity_clip: float = 6.0
    # 角阻尼用力矩分量上限 (N·m)；用真实角速度算 -k*ω，再裁剪，避免 ω≫25 时等效阻尼过弱而翻飞
    angular_drag_torque_axis_max: float = 0.18
    # 姿态杆全在死区内时，额外角阻尼系数与力矩上限（耗散机械臂扰动）
    zero_cmd_angular_hold_k: float = 0.01
    zero_cmd_angular_torque_axis_max: float = 0.035
    # 机体 xfrc 力矩向量范数上限 (N·m)，防止数值尖峰
    max_body_torque_norm: float = 0.38
    # 世界系线合力上限 = 系数 × hover_thrust，防止数值/接触偶发巨大外力
    max_total_linear_force_over_hover: float = 8.0
    # 全量四旋翼：以倾转为主，叠加少量世界系水平助推，避免视觉有倾斜但平移过肉
    quad_world_xy_stick_force_factor: float = 0.012
    # 全量模式推力目标一阶低通时间常数 (s)，略放慢以避免 R/F 大动作时瞬间拉爆姿态
    full_mode_thrust_lpf_tau_s: float = 0.16
    # 杆量接近零时：用真实线速度做世界系阻尼（带力上限），耗散机械臂等扰动，避免只靠裁剪速度导致“飘走”
    zero_cmd_hold_deadband: float = 0.05
    zero_cmd_xy_hold_k: float = 0.12  # N/(m/s)；过大易与倾转 PD 耦合导致开局水平晃
    zero_cmd_xy_hold_force_cap: float = 0.85  # N/轴
    zero_cmd_z_hold_k: float = 0.45  # N/(m/s)
    zero_cmd_z_hold_force_cap: float = 1.2  # N
    # free joint 速度安全钳制（数值保险丝，防止耦合导致发散）
    free_linear_speed_cap: float = 60.0  # m/s
    free_angular_speed_cap: float = 18.0  # rad/s


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
    # 世界系水平探索：f_x=lateral_cmd·factor·mg, f_y=forward_cmd·factor·mg（与键盘 W/S A/D 一致）；0 则保持纯 Z 并每步清零 vx,vy
    keyboard_world_xy_force_factor: float = 0.055
    # 世界系水平速度阻尼 (N/(m/s))；过小时接触/积分泄漏的水平速度易累积导致「横飞」
    keyboard_world_xy_vel_damping: float = 0.38
    # k_xy>0 时每步对 free 线速度水平分量裁剪 (m/s)；0 表示不裁剪
    keyboard_world_xy_max_speed: float = 0.75
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


@dataclass(frozen=True)
class FullModeControlConfig:
    thrust_cmd_scale_over_hover: float = 0.38
    tau_yaw_over_hover: float = 0.012
    thrust_max_over_hover: float = 2.2
    max_tilt_deg: float = 18.0
    planar_forward_axis_body: tuple[float, float, float] = (0.0, 1.0, 0.0)
    planar_right_axis_body: tuple[float, float, float] = (1.0, 0.0, 0.0)
    attitude_kp_scale: float = 1.0
    attitude_kd_scale: float = 1.0
    attitude_rate_cap_scale: float = 1.0
    attitude_torque_limit_scale: float = 1.0
    idle_attitude_kp_scale: float = 1.0
    idle_attitude_torque_limit_scale: float = 1.0
    hover_rotor_speed: float = 42.0
    rotor_speed_delta: float = 24.0
    rotor_ramp_rate: float = 80.0
    demo_rotor_bias: tuple[float, float, float, float] = (60.0, 60.0, 60.0, 60.0)
    reset_height_offset_m: float = 0.25
    fullmode_reset_thrust_ramp_s: float = 0.8
    fullmode_reset_thrust_start_factor: float = 0.2
    fullmode_reset_minimal_stab_s: float = 0.35


@dataclass(frozen=True)
class DroneModelProfile:
    key: str
    display_name: str
    aero: DroneAeroConfig = field(default_factory=lambda: DEFAULT_DRONE_AERO_CONFIG)
    full_mode: FullModeControlConfig = field(default_factory=FullModeControlConfig)
    vertical_keyboard_baseline_tmg: float = 1.0022
    vertical_xy_force_factor: float = 0.055


DEFAULT_DRONE_AERO_CONFIG = DroneAeroConfig()

DEFAULT_DRONE_MODEL = "Drone_ver_1.0"

_X2_DRAG_CONFIG = replace(
    DEFAULT_DRONE_AERO_CONFIG.drag,
    linear_xy=0.22,
    linear_z=0.22,
    quadratic_xy=0.10,
    quadratic_z=0.12,
    angular_xy=0.040,
    angular_z=0.030,
    world_xy_velocity_damping=0.28,
    angular_drag_torque_axis_max=0.30,
    zero_cmd_angular_hold_k=0.022,
    zero_cmd_angular_torque_axis_max=0.070,
    max_body_torque_norm=0.56,
    quad_world_xy_stick_force_factor=0.003,
    full_mode_thrust_lpf_tau_s=0.28,
    zero_cmd_xy_hold_k=0.34,
    zero_cmd_xy_hold_force_cap=1.35,
    zero_cmd_z_hold_k=0.80,
    zero_cmd_z_hold_force_cap=1.70,
)
_X2_GROUND_EFFECT_CONFIG = replace(
    DEFAULT_DRONE_AERO_CONFIG.ground_effect,
    rotor_radius=0.13,
    active_height=0.42,
    gain=0.14,
    max_factor=1.16,
)
_X2_AERO_CONFIG = replace(
    DEFAULT_DRONE_AERO_CONFIG,
    drag=_X2_DRAG_CONFIG,
    ground_effect=_X2_GROUND_EFFECT_CONFIG,
)

DRONE_MODEL_PROFILES: dict[str, DroneModelProfile] = {
    "Drone_ver_1.0": DroneModelProfile(
        key="Drone_ver_1.0",
        display_name="Drone_ver_1.0",
    ),
    "x2": DroneModelProfile(
        key="x2",
        display_name="Skydio X2",
        aero=_X2_AERO_CONFIG,
        full_mode=FullModeControlConfig(
            thrust_cmd_scale_over_hover=0.18,
            tau_yaw_over_hover=0.006,
            thrust_max_over_hover=1.6,
            max_tilt_deg=12.0,
            planar_forward_axis_body=(-1.0, 0.0, 0.0),
            planar_right_axis_body=(0.0, -1.0, 0.0),
            attitude_kp_scale=1.35,
            attitude_kd_scale=1.20,
            attitude_rate_cap_scale=1.35,
            attitude_torque_limit_scale=1.50,
            idle_attitude_kp_scale=1.55,
            idle_attitude_torque_limit_scale=2.00,
            hover_rotor_speed=38.0,
            rotor_speed_delta=14.0,
            rotor_ramp_rate=50.0,
            demo_rotor_bias=(36.0, 36.0, 36.0, 36.0),
            reset_height_offset_m=0.25,
            fullmode_reset_thrust_ramp_s=1.40,
            fullmode_reset_thrust_start_factor=0.14,
            fullmode_reset_minimal_stab_s=0.70,
        ),
        vertical_keyboard_baseline_tmg=1.0022,
        vertical_xy_force_factor=0.04,
    ),
}

DRONE_MODEL_ALIASES: dict[str, str] = {
    "drone_v1": "Drone_ver_1.0",
    "drone-v1": "Drone_ver_1.0",
    "drone_ver_1.0": "Drone_ver_1.0",
    "Drone_ver_1.0": "Drone_ver_1.0",
    "x2": "x2",
    "skydio_x2": "x2",
    "skydio-x2": "x2",
}


def get_drone_model_profile(name: str | None) -> DroneModelProfile:
    requested = DEFAULT_DRONE_MODEL if name is None else str(name).strip()
    canonical = DRONE_MODEL_ALIASES.get(requested, requested)
    if canonical not in DRONE_MODEL_PROFILES:
        choices = ", ".join(sorted(DRONE_MODEL_PROFILES))
        raise KeyError(f"Unknown drone model profile '{name}'. Available: {choices}")
    return DRONE_MODEL_PROFILES[canonical]
