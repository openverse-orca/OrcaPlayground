# Drone Driver - 无人机推力驱动仿真

基于 MuJoCo 的四旋翼无人机推力驱动仿真示例。支持键盘/手柄实时操控、多机型 profile 切换，并提供标准 Gym 接口以接入 DRL 强化学习训练。

## ⚠️ 资产准备

> **📦 相关资产**：https://simassets.orca3d.cn/ **Drone_ver, dji_lhcg 或 skydio_x2_nofloor 资产包**
>
> **🔧 是否需要手动拖动到布局中**：**是**
>
> **📝 对应布局建议**：`Industry_DataCenter_Plugging_01` 或空布局（`OrcaLab_day`）
>
> **🧭 UI 资产详情**：请在资产面板中搜索 `Drone_ver` 或 `skydio_x2_nofloor`，将对应 actor 拖入布局
>
> **运行方式**：脚本会自动扫描场景中的无人机实例并绑定

## 🔧 手动拖入资产进行调试

1. 在资产面板中搜索 `Drone_ver` 或 `dji_lhcg` 或 `skydio_x2_nofloor`
2. 将 actor 拖入布局，调整到期望的初始位置和朝向
3. 场景中需要且只能有 **1 台**完整匹配的无人机实例
4. 实例名不需要固定，脚本会根据关节、执行器、body、site 后缀自动绑定
5. 运行前可切换 camera 视角来获得更优体验

手动拖动资产的通用流程可参考项目根目录 [README - 手动拖动资产](../../README.md#-手动拖动资产运行前必做)。

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

本示例为基础依赖即可运行，无需额外安装。

### 默认运行（full 四旋翼模式，请在每次运行前指定机型）

```bash
python examples/drone_driver/run_drone_orca.py --drone-model drone_v1
```

### 切换机型

```bash
#drone_v1
python examples/drone_driver/run_drone_orca.py --drone-model drone_v1

# Skydio X2
python examples/drone_driver/run_drone_orca.py --drone-model x2

# DJI LHCG（大型无人机，10x 缩放）
python examples/drone_driver/run_drone_orca.py --drone-model dji_lhcg
```

### 启用自动巡航

```bash
python examples/drone_driver/run_drone_orca.py --drone-model drone_v1 --autoplay
```

### 竖直 Z-only 调试模式

```bash
python examples/drone_driver/run_drone_orca.py --drone-model drone_v1 --vertical-z-only
```

### 使用 Xbox 手柄控制

```bash
python examples/drone_driver/run_drone_orca.py --drone-model drone_v1 --ctrl-device xbox
```

## 🎮 控制说明

### 键盘控制（默认）

| 按键 | 功能 |
|------|------|
| `w/s` | 前后平移（机身伴随轻微前后倾） |
| `a/d` | 左右平移（机身伴随轻微左右倾） |
| `r/f` | 集体推力增减（相对悬停推力） |
| `q/e` | 偏航控制 |
| `z/x` | 抓取机构收拢/扩张（需机型支持） |
| `space` | 重置到初始姿态 |

### Xbox 手柄控制（`--ctrl-device xbox`）

| 输入 | 功能 |
|------|------|
| 左摇杆 Y/X | 前后/左右平移 |
| RT/LT | 推力增减 |
| 右摇杆 X | 偏航控制 |
| LB/RB | 抓取机构收拢/扩张（需机型支持） |
| A 键 | 重置到初始姿态 |

### 竖直 Z-only 模式控制

| 按键 | 功能 |
|------|------|
| `r/f` | 世界 `+Z` 推力控制 |
| `w/s`、`a/d` | 可选世界系水平力（取决于 `--vertical-xy-force-factor`，`--vertical-pure-z` 可彻底关闭） |
| `q/e` | 仅动画，不参与姿态控制链 |

启用 `--autoplay` 后，脚本会持续生成正弦扰动，`space` / `A键` 仍可用于重置。

## 📁 目录结构

```
examples/drone_driver/
├── README.md                          # 本文件
├── run_drone_orca.py                  # 入口脚本（扫描场景、注册环境、驱动仿真循环）
└── model/
    ├── Drone_ver_1.0/
    │   ├── drone-v1.xml               # 默认无人机 MuJoCo 模型
    │   └── meshes/                    # 网格文件
    ├── skydio_x2_nofloor/
    │   ├── x2.xml                     # Skydio X2 模型（框架重构后）
    │   └── Meshes/                    # 网格与贴图
    ├── dji_lhcg/
    │   ├── dji_lhcg.xml               # DJI LHCG 大型无人机模型
    │   └── meshes/                    # 网格文件
    └── firefly/
        └── firefly.xml                # Firefly 模型

envs/drone/
├── __init__.py
├── drone_orca_env.py                  # 推力物理环境（Gym 接口、键盘/手柄输入、xfrc 施力）
└── drone_aero_config.py               # 气动阻尼、full 模式控制参数与机型 profile
```

## 🧭 架构概览

### 物理驱动方式

本示例提供「**推力驱动的刚体飞行**」原型：

- 根节点使用**自由关节** `drone_free`，由 MuJoCo 积分刚体动力学
- 默认 full 模式：键盘/手柄指令映射为作用在 `drone_frame` 刚体上的**集体推力 + CTBR 姿态/偏航控制**（`xfrc_applied`）
- 附带 `drone_aero_config` 中的**线性/二次阻尼**与简易**地面效应**（按高度放大局部升力）
- 四个旋翼关节主要承担**转速动画**（每步写关节相位），模型中不再为桨挂 position 执行器

其中"**集体推力**"指四旋翼总升力的集总控制量，主要由 `R/F` 调节；当前默认 full 模式**不包含位置闭环**，也不是"给定目标高度/目标位置后自动跟踪"的位控。

### 两种控制模式

| 模式 | 推力方向 | 姿态 | 水平运动 | 适用场景 |
|------|----------|------|----------|----------|
| **Full 四旋翼**（默认） | 机体系 +Z | CTBR PD 控制倾转 | 倾转产生水平分力 | 日常飞行、DRL 训练 |
| **Vertical Z-only** | 世界 +Z | 每步锁定朝上 | 可选世界系水平力 | 起飞标定、悬停推力调试 |

### 控制流水线

```
键盘/手柄输入
    │
    ▼
_read_keyboard_command / _read_joystick_command
    │  command = [forward, lateral, vertical, yaw] ∈ [-1, 1]⁴
    ▼
_apply_thrust_and_drag
    ├── Full 模式:
    │   ├── 集体推力: T = hover + vertical_cmd × scale（一阶低通滤波）
    │   ├── CTBR 控制器: command → 目标体角速率 → P+FB 力矩
    │   ├── 姿态回正: 倾角误差叠加回正角速率
    │   ├── 倾角硬限: 超过 max_tilt 时强力回正
    │   ├── 气动阻尼: 线性 + 二次速度阻尼
    │   ├── 地面效应: 近地升力放大
    │   └── 写入 xfrc_applied[Drone]（世界系力 + 力矩）
    │
    └── Vertical Z-only:
        ├── 世界 +Z 推力 + vz 阻尼
        ├── 可选 WASD 世界系水平力
        └── 写入 xfrc_applied[frame/Drone]
    │
    ▼
_update_rotors → 旋翼相位动画
_update_gripper → 抓取机构（如存在）
    │
    ▼
mj_step
```

### CTBR 控制器

全量模式使用 **CTBR（Collective-Thrust-Body-Rate）** 控制器，这是无人机领域常用的内环控制架构：

1. **集体推力**：`T = hover_thrust + vertical_cmd × thrust_cmd_scale`，经一阶低通滤波平滑
2. **目标体角速率**：WASD 杆量通过机体前/右参考轴映射到 roll/pitch 目标角速率
3. **姿态回正**：叠加基于当前倾角误差的回正角速率（松杆时增益更大）
4. **倾角硬限**：超过 `max_tilt_deg` 时额外叠加强力回正
5. **P+FB 力矩**：`τ = J(Kp(ω_des - ω) + ω×Jω)`，范数裁剪到 `max_body_torque_norm`

该架构与主流 DRL 无人机控制（如 Crazyflie、OmniDrones）的内环一致，便于将 RL 策略的输出直接映射为 `[thrust, roll_rate, pitch_rate, yaw_rate]` 四维指令。

## 🤖 DRL 强化学习控制

### Gym 环境接口

`DroneOrcaEnv` 继承自 `OrcaGymLocalEnv`，实现了标准 Gymnasium 接口，可直接接入 SB3、RLlib 等 RL 框架：

```python
import gymnasium as gym

env = gym.make("DroneOrca-OrcaGym-localhost-50051-000")
obs, info = env.reset()

for _ in range(1000):
    action = policy(obs)          # RL 策略推理
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### 动作空间

| 维度 | 范围 | 含义 | 对应键盘 |
|------|------|------|----------|
| 0 | [-1, 1] | 前后平移指令 | W/S |
| 1 | [-1, 1] | 左右平移指令 | A/D |
| 2 | [-1, 1] | 集体推力指令 | R/F |
| 3 | [-1, 1] | 偏航指令 | Q/E |

> **注意**：当前 `step(action)` 中 `action` 参数未直接使用，实际控制指令来自键盘/手柄。若需纯 RL 训练，需将 `action` 替换为控制源（见下方接入指南）。

### 观测空间

| 维度 | 内容 | 说明 |
|------|------|------|
| 0-2 | `position` | 无人机世界系位置 (x, y, z) |
| 3-5 | `euler` | 机体欧拉角 (roll, pitch, yaw) |
| 6-8 | `linear_velocity` | 世界系线速度 (vx, vy, vz) |
| 9-11 | `angular_velocity` | 世界系角速度 (ωx, ωy, ωz) |
| 12-15/19 | `rotor_speeds` | 各旋翼角速度（4 或 8 个） |
| 末 4 维 | `last_command` | 上一帧控制指令 |

观测中线速度/角速度来自 MuJoCo 刚体 `cvel`（世界系）。

### 接入 DRL 训练的推荐方案

当前环境以键盘/手柄实时操控为主，若需接入 DRL 训练，推荐以下改造路径：

#### 方案 1：替换控制源（最简改动）

在 `DroneOrcaEnv.step()` 中，将 `_read_keyboard_command()` 替换为直接使用 `action`：

```python
def step(self, action):
    command = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
    self._last_command[:] = command
    reset_requested = False
    # ... 后续物理步进不变
```

此方案保留现有 CTBR 控制器作为内环，RL 策略只需输出四维高层指令，与 Crazyflie、OmniDrones 等无人机 RL 基准的动作空间对齐。

#### 方案 2：添加奖励函数

在 `step()` 返回值中实现任务相关的奖励，参考仓库中其他 RL 示例的模式：

```python
def _compute_reward(self) -> float:
    # 示例：悬停任务
    pos = self.gym._mjData.xpos[self._frame_body_id]
    target = np.array([0.0, 0.0, 1.0])
    distance = np.linalg.norm(pos - target)
    return -distance
```

#### 方案 3：多环境并行训练

参考仓库中 `franka_rl`（SB3 + HER）和 `ant_rl`（RLlib APPO）的并行训练架构：

```
RL 算法 (SB3 PPO / RLlib APPO)
    │
    ▼
OrcaGymAsyncSubprocVecEnv          # 多进程向量化环境
    ├── env_0: DroneOrcaEnv        # 独立 MuJoCo 实例
    ├── env_1: DroneOrcaEnv
    └── ...
```

关键配置项：
- `orcagym_addr`：每个子环境连接独立的 OrcaGym 地址
- `frame_skip`：控制频率与仿真频率的比值
- `time_step`：物理步长（默认 1/120s）

#### 推荐训练框架

| 框架 | 适用场景 | 参考示例 |
|------|----------|----------|
| **SB3 PPO** | 单机训练、快速迭代 | `examples/legged_gym/` |
| **SB3 SAC/TQC + HER** | 目标条件任务（到达、抓取） | `examples/franka_rl/` |
| **RLlib APPO** | 多机分布式、大规模并行 | `examples/ant_rl/` |

#### 典型任务设计

| 任务 | 观测 | 奖励 | 终止条件 |
|------|------|------|----------|
| **悬停** | 位置 + 速度 + 姿态 | `-‖pos - target‖` | 距离 > 阈值 或 超时 |
| **航点跟踪** | 位置 + 速度 + 目标相对位置 | `-‖pos - waypoint‖` + 速度惩罚 | 到达 / 碰撞 / 超时 |
| **穿障飞行** | 位置 + 速度 + 障碍距离 | 前进奖励 + 碰撞惩罚 | 碰撞 / 通过 / 超时 |
| **抓取投放** | 位置 + 速度 + 物体状态 | 抓取成功 + 投放精度 | 物体到达目标 / 超时 |

### 与仓库中其他 RL 示例的对比

| 特性 | Drone Driver | Franka RL | Ant RL | Legged Gym |
|------|-------------|-----------|--------|------------|
| 控制方式 | xfrc 推力 + CTBR | mocap + PD | 关节力矩 | 关节 PD |
| 动作空间 | 4D（推力+角速率） | 3-4D（位置增量） | 8D（关节力矩） | 12+D（关节位置） |
| 内环控制器 | CTBR | PD position | 无 | PD position |
| RL 就绪 | 需替换控制源 | ✅ 原生支持 | ✅ 原生支持 | ✅ 原生支持 |
| 训练框架 | 待接入 | SB3 + HER | RLlib APPO | SB3 PPO / RLlib APPO |

## 📋 参数说明

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--orcagym_addr` | `localhost:50051` | OrcaGym 服务地址 |
| `--time_step` | `1/120` | 物理步长 (s) |
| `--frame_skip` | `1` | 每次控制对应的仿真步数 |
| `--autoplay` | `False` | 启用循环自动输入 |
| `--drone-model` | `Drone_ver_1.0` | 无人机 profile（支持 `x2`、`dji_lhcg`） |
| `--ctrl-device` | `keyboard` | 控制设备：`keyboard` 或 `xbox` |

### 诊断参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--diag-logs` / `--no-diag-logs` | 启用 | 开关环境侧诊断 WARNING |
| `--diag-every-env-steps` | `0` | 每 N 个 env step 输出一次 periodic 长日志；`0` 为关闭 |

### Reset 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--reset-height-offset` | 取 profile 值 | reset 时给初始 z 额外抬高 (m) |
| `--fullmode-reset-thrust-ramp` | 取 profile 值 | full 模式 reset 后推力渐入时间 (s) |
| `--fullmode-reset-thrust-start-factor` | 取 profile 值 | 推力渐入起点占 hover 比例 |
| `--fullmode-reset-minimal-stab` | 取 profile 值 | reset 后最小稳定窗口 (s) |

### 竖直模式参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vertical-z-only` | `False` | 切回竖直 Z-only 模式 |
| `--vertical-thrust-ramp` | `False` | 线性爬升 T/(mg) |
| `--vertical-ramp-t0` | `0.65` | 爬升起始 T/mg 系数 |
| `--vertical-ramp-t1` | `2.05` | 爬升结束 T/mg 系数 |
| `--vertical-ramp-duration` | `25.0` | 爬升持续时间 (s) |
| `--vertical-fixed-tmg` | `-1.0` | 固定 T/(mg)（≥0 时启用，与 ramp 互斥） |
| `--vertical-keyboard-base-tmg` | 取 profile 值 | 杆量零时 T/(mg) |
| `--vertical-xy-force-factor` | 取 profile 值 | WASD 世界系水平力系数 |
| `--vertical-pure-z` | `False` | 关闭水平力（k_xy=0） |
| `--vertical-takeoff-bisect` | `False` | 二分搜索起飞临界 T/(mg) |
| `--bisect-lo` | `0.98` | 二分下界 T/mg |
| `--bisect-hi` | `1.12` | 二分上界 T/mg |
| `--bisect-iters` | `14` | 二分迭代次数 |
| `--bisect-hold-s` | `3.0` | 每档试验持有时间 (s) |
| `--bisect-dz` | `0.06` | 判定离地的 Δz (m) |

## 🛩️ 支持的机型

### Drone_ver_1.0（默认）

标准四旋翼，阻尼与控制参数适中，适合日常飞行和 DRL 训练基线。

### Skydio X2（`--drone-model x2`）

轻量级四旋翼，阻尼更大、最大倾角更小（12°），姿态增益更高。适配过程中解决了质心偏移、机体系朝向不一致、执行器冲突等问题。

### DJI LHCG（`--drone-model dji_lhcg`）

大型无人机（10x 缩放），阻尼参数按比例放大，reset 抬高量 1.8m，无推力渐入。含双层旋翼（8 个桨关节）和抓取机构。

### 机型别名

| 规范名 | 别名 |
|--------|------|
| `Drone_ver_1.0` | `drone_v1`, `drone-v1`, `drone_ver_1.0` |
| `x2` | `skydio_x2`, `skydio-x2` |
| `dji_lhcg` | `dji-lhcg`, `lhcg` |

## 🔌 适配新无人机

建议把"适配新机型"分成 3 层来看：

### 1. 模型结构层（XML）

让资产满足当前环境的最小约定：

- 有根 body `drone_frame`
- 根上有自由关节 `drone_free`
- 有主机体 body `Drone`
- 有 `FL_joint` / `FR_joint` / `BL_joint` / `BR_joint`
- 有 `rotor_fl_site` / `rotor_fr_site` / `rotor_bl_site` / `rotor_br_site`
- 有 `imu` 与 `drone_body_center_site`

### 2. 动力学近似层（质量/碰撞/朝向）

这是最容易让新机型"看起来能加载，但一飞就乱"的部分：

- `drone_free` 尽量靠近整机近似质心
- 主质量集中在 `Drone` 主体，不要把大量质量挂在旋翼子体上
- 旋翼子体若仅用于动画，只保留极小惯量
- 网格朝向不同时，优先通过 profile 中的 `planar_forward_axis_body` / `planar_right_axis_body` 修正

### 3. 机型参数层（profile）

在 [drone_aero_config.py](../../envs/drone/drone_aero_config.py) 里新增 `DroneModelProfile`：

- `drag`：线性/二次阻尼、角阻尼、零杆量保持参数
- `ground_effect`：旋翼半径、地效高度和增益
- `full_mode`：最大倾角、姿态增益、力矩限制、reset 抬高量、桨速动画参数
- 必要时配置 `planar_forward_axis_body` / `planar_right_axis_body`

### 推荐接入流程

1. 先把 XML 整理成与 `drone-v1` 同类的结构
2. 先跑 `--vertical-z-only`，确认悬停推力、起飞临界高度、接触稳定性大体正常
3. 再切 full 模式，只测 `R/F`，确认纯升降没有异常姿态耦合
4. 然后单独测 `W/S`、`A/D`，倾转方向不对时优先检查机体系朝向配置
5. 若 full 模式"轻微倾转后直接飞走"，第一优先检查 `frame_com_offset`、根体位置和质量分布
6. 最后用 `--diag-every-env-steps` 打开 periodic 日志，针对具体机型微调 profile

### 适配完成判断标准

- reset 后静止时，`frame_com_offset` 无明显大偏移
- 纯 `R/F` 升降时，不会无输入自发滚转/俯仰
- `W/S`、`A/D` 对应到预期的倾转方向
- 松杆后姿态能回收，而非继续沿同一方向累积
- 旋翼动画存在，但不对主动力学产生明显副作用

## 🔍 调试技巧

- 环境启动时会打印 `scene DOF` 观察点 `v[8]` 的 `joint/body/actor` 归属，便于和 `scene_max|qacc|=...@8(...)` 日志对照
- 复杂场景一 reset 就炸时，先加 `--reset-height-offset 0.2~0.4` 验证是否为出生点接触问题
- full 模式定位乱飘时，用 `--diag-every-env-steps 20` 恢复 periodic 长日志
- 竖直模式下可用 `--vertical-takeoff-bisect` 二分搜索起飞临界推力

## ⚠️ 当前限制

- 推力为**集总标量 + 简化姿态/偏航控制**，不是四电机螺旋桨气动力逐个求解
- 旋翼动画与动力学推力**未刚性耦合**（便于调试与展示）
- 轮地差速驱动**未实现**，需另建轮式模型与控制链
- `step(action)` 中 action 未直接作为控制源，接入 DRL 训练需替换控制源

若需更接近真机的分配律与姿态内环，可在本版稳定后再接电机模型与控制器。
