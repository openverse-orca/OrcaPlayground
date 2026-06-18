# DJI LHCG 无人机模型说明

## 模型概述

DJI LHCG 是一款大型六旋翼无人机（10x 缩放），质量 15 kg，惯性矩约 6.25 kg·m²。模型包含 4 组共轴双桨（8 个桨叶），通过 MuJoCo 自由关节 `drone_free` 驱动刚体动力学。

## 物理参数

| 参数 | 值 |
|------|-----|
| 机体质量 | 15.0 kg |
| 惯性矩 (Ixx/Iyy/Izz) | 6.25 / 3.72 / 2.85 kg·m² |
| 桨臂长度 | ~0.875 m |
| 悬停推力 | ~147.2 N |
| 碰撞体 | 圆柱 r=1.5m, h=0.4m |

## 模型结构

- `drone_frame`：根体，含自由关节 `drone_free`
- `Drone`：主机体，质量与惯量集中于此
- `FL/FR/BL/BR_upper_blade` + `FL/FR/BL/BR_lower_blade`：8 个桨叶（4 组共轴双桨）
- 旋翼关节：`FL_joint`, `FR_joint`, `BL_joint`, `BR_joint` 及对应 `2_joint`
- 传感器：陀螺仪、加速度计、四元数（IMU site）

## 运行

```bash
python examples/drone_driver/run_drone_orca.py --drone-model dji_lhcg
```

使用 Xbox 手柄控制：

```bash
python examples/drone_driver/run_drone_orca.py --drone-model dji_lhcg --ctrl-device xbox
```

## 控制方式

### 键盘

| 按键 | 功能 |
|------|------|
| W/S | 前后平移 |
| A/D | 左右平移 |
| R/F | 升降 |
| Q/E | 偏航 |
| Space | 重置 |

### Xbox 手柄

| 输入 | 功能 |
|------|------|
| 左摇杆 | 前后/左右平移 |
| RT/LT | 升降 |
| 右摇杆 X | 偏航 |
| A 键 | 重置 |

## 控制参数设计

由于该机型质量是 Drone_ver_1.0（0.154 kg）的约 100 倍，控制参数以 Drone_ver_1.0 为基准按物理等比放大：

- 力类参数 ×100（阻尼系数、力上限等）
- 力矩类参数按惯量比放大（受 `max_body_torque_norm` 安全钳制）
- 无量纲系数保持不变（`thrust_cmd_scale_over_hover`、`tau_yaw_over_hover` 等）
- 时间常数不变（`full_mode_thrust_lpf_tau_s`）

关键配置差异：

| 参数 | Drone_ver_1.0 | dji_lhcg | 说明 |
|------|---------------|----------|------|
| 质量 | 0.154 kg | 15.0 kg | ~100x |
| `linear_xy` | 0.10 | 10.0 | 阻尼 ×100 |
| `max_body_torque_norm` | 0.38 N·m | 38.0 N·m | 力矩 ×100 |
| `zero_cmd_z_hold_k` | 0.45 | 15.0 | Z 阻尼 ×33（略低以保持升降感） |
| `attitude_kp_scale` | 1.0 | 100.0 | 姿态 P 增益 ×100 |
| `idle_attitude_kp_scale` | 1.0 | 5.0 | 松杆 P 增益降低，防止回正过冲 |
| `attitude_rate_cap_scale` | 1.0 | 3.0 | 限制角速度，防止高速旋转 |
| `reset_height_offset_m` | 0.25 | 1.80 | 远离地面效应区 |
| `demo_rotor_bias` | (60,...) | (0,...) | 无自动轨迹偏置 |

## 文件清单

- `model/dji_lhcg/dji_lhcg.xml`：MuJoCo 模型定义
- `model/dji_lhcg/meshes/`：STL 网格与纹理
- `envs/drone/drone_aero_config.py`：`dji_lhcg` profile 定义
