# D12 双臂机器人脚本轨迹演示

本示例源项目地址：`(https://github.com/openverse-orca/OrcaManipulation)`，将 d12 双臂机器人的脚本轨迹演示（Motion Planning）移植到 OrcaPlaygroundMotion。

> **当前范围**：本示例仅包含脚本轨迹执行（YAML 定义轨迹 → OSC 控制器 → 仿真步进），不包含 OpenPI 推理、HDF5 数据采集存储等功能。OpenPI 相关内容作为**后续扩展方向**在教学路线第四课中介绍。

## ⚠️ 资产准备

- **资产**：请订阅 **d12_openpi_pick** 和 **d12_waist** 资产包
- **是否需要手动拖动到布局中**：**否**，建议启动 **orcalab_day** 后打开当前目录**d12.json**布局
- **场景 JSON**：`d12.json`，包含桌子 + d12 机器人 + 3 个任务物体（C12C、Barcode_01、Cardboardbox_01）
- 脚本启动时会扫描场景中的 actuator 后缀，自动检测 agent 前缀

## 文件组成

```
examples/d12/demo/                  # 运行入口
├── run_d12_demo.py                 # 主入口脚本
├── pose_scripted.yaml              # 基座系轨迹（固定场景专用）
├── pose_mp.yaml                    # 物体坐标系轨迹（支持随机化）
├── rand_low.yaml                   # 低随机化配置（±0.1m / ±15°）
├── rand.yaml                       # 中等随机化配置（±0.15m / ±15°）
├── d12.json                        # 场景布局 JSON
└── README.md                       # 本文件

envs/d12/                           # 库代码
├── d12_env.py                      # D12Env 环境子类
├── configs/d12_robot_config.py     # D12 机器人配置（关节、执行器、末端 site、相机）
└── scripts/
    ├── pose_resolver.py            # 物体坐标系 → 基座系解析
    └── object_randomizer.py        # 物体位置/旋转随机化
```

## 运行前准备

- 场景中需要且只能有 1 台完整匹配的 d12 机器人
- 机器人实例名不需要固定，脚本会自动检测 agent 前缀
- `agent_name` 默认为 `d12_waist_usda_1`，一般无需修改

## 运行

```bash
# 固定场景 + 基座系轨迹（最简单，推荐首次运行）
python examples/d12/demo/run_d12_demo.py --mode fixed

# 物体坐标系追踪（物体位置固定但轨迹自动适配）
python examples/d12/demo/run_d12_demo.py --mode mp

# 带低随机化的追踪（物体位置随机偏移，轨迹自动追踪）
python examples/d12/demo/run_d12_demo.py --mode rand-low

# 中等随机化
python examples/d12/demo/run_d12_demo.py --mode rand

# 调试：打印物体在基座系下的位姿
python examples/d12/demo/run_d12_demo.py --mode dump --dump_keywords C12C Barcode Cardboardbox

# 自定义参数
python examples/d12/demo/run_d12_demo.py --pose_file pose_mp.yaml --rand_file rand_low.yaml --episodes 10
```

### 命令行参数说明

- `--mode`：运行模式，可选 `fixed` / `mp` / `rand-low` / `rand` / `dump`
- `--pose_file`：轨迹 YAML 文件路径（与 `--mode` 二选一）
- `--rand_file`：随机化 YAML 文件路径
- `--episodes`：运行轮数
- `--orcagym_addr`：OrcaGym 服务地址，默认 `localhost:50051`
- `--agent_name`：机器人实例名，默认 `d12_waist_usda_1`
- `--dump_keywords`：`dump` 模式下要查询的物体名称关键词

## 场景布局

`d12.json` 中各物体相对于桌子 (Table_green_03) 的位姿：

| 物体 | X (m) | Y (m) | Z (m) | Yaw° |
|------|-------|-------|-------|------|
| Robot (d12) | 0.73 | -0.17 | ≈0 | 180 |
| Cardboardbox_01 | 0.24 | -0.17 | 1.08 | 90 |
| Barcode_01 | 0.25 | 0.19 | 1.10 | 90 |
| C12C | 0.21 | -0.56 | 1.08 | 180 |

机器人面朝桌子（Yaw=180°），三个物体均在桌面上方。

## 仿真参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `time_step` | 0.001 | 物理仿真步长 1kHz |
| `frame_skip` | 20 | 每控制步跳 20 帧物理步 |
| `real_time_step` | 0.02s | 控制频率 50Hz |
| `policy_freq` | 1.0 / env.dt | OSC 控制器频率，与控制频率对齐 |
| 主循环 | 默认 sleep 到实时 | 保证仿真速度与真实时间一致 |

## 教学路线

### 第一课：理解脚本轨迹

脚本轨迹 = 预定义的末端执行器路径，机器人按"剧本"行动。

`pose_scripted.yaml` 中每一段定义了：
- `steps`：这段走多少步
- `l_target_b / r_target_b`：左/右臂末端在基座系下的目标位置 [x, y, z]（米）
- `l_quat_b / r_quat_b`：末端姿态四元数 [x, y, z, w]
- `gripper_l / gripper_r`：夹爪指令（`open` / `close` / `hold`）

运行流程：
```
加载 YAML → 逐段插值（位置线性 + 姿态 SLERP）→ 逐帧写入 OSC 控制器 → 仿真步进
```

局限：所有坐标是写死的绝对值，物体移走就抓不到。

### 第二课：物体坐标系追踪

不写死"手去哪里"，而是写"手去物体上方 0.4m 处"。

`pose_mp.yaml` 中的关键字段：
- `l_object_frame`：要追踪的物体名称（模糊匹配）
- `l_target_o`：相对物体的目标位置（物体坐标系）
- `l_frame_offset_euler_deg`：物体坐标系的旋转补偿

解析过程（`pose_resolver.py`）：
```
1. 查询物体在基座系下的实际位置和姿态
2. 将物体坐标系下的目标转换为基座系坐标
3. 用转换后的基座系坐标构建轨迹
```

为什么需要 `frame_offset_euler_deg`：有些物体在 MuJoCo 中的坐标系和直觉不一致（比如 Z 轴朝下），需要旋转补偿才能让"上方"对应正确的方向。

### 第三课：物体随机化

在每个 episode 开始时，随机偏移物体的位置和旋转。

`rand_low.yaml` 中的字段：
- `position_range`：`[[xmin, xmax], [ymin, ymax], [zmin, zmax]]` 相对偏移范围（米）
- `rotation_range_deg`：旋转偏移范围（度，xyz Euler 序）
- `seed`：随机种子（可复现）

随机化 + 物体坐标系追踪 = 自适应抓取：
```
物体随机偏移 → 物体坐标系解析自动适配 → 轨迹追踪新位置 → 大概率抓到
```

### 第四课：扩展方向 — OpenPI 推理

> **本课内容为后续扩展方向**，当前示例不包含 OpenPI 功能。源项目 `OrcaManipulation` 中有完整的 `eval.py` 和 HDF5 数据采集实现，可按需移植。

[OpenPI](https://github.com/OpenRobotLab/OpenPI) 是一个视觉-语言-动作（VLA）模型，通过相机图像 + 本体感知 + 自然语言指令，实时输出双臂末端动作。本示例的脚本轨迹架构与 OpenPI 存在以下复用关系：

| 本示例已有 | OpenPI 可复用 |
|-----------|-------------|
| OSC 控制器 + B 系→全局坐标转换 | 推理时的动作执行链路 |
| 16 维动作格式 | OpenPI 输出动作空间 |
| 物体随机化 + 追踪 | 生成多样化训练数据 |
| 场景布局 + 机器人资产 | 训练和推理共享场景 |

#### 脚本轨迹 vs OpenPI 推理

| 维度 | 脚本轨迹（本示例） | OpenPI 推理（待移植） |
|------|-------------------|---------------------|
| 输入 | 预定义 YAML | 相机图像 + 本体感知 + 语言指令 |
| 决策 | 固定路径 | 神经网络实时推理 |
| 自适应 | 仅物体坐标系追踪 | 视觉闭环，完全自适应 |
| 需要训练 | 否 | 是 |
| 泛化能力 | 仅限预定义场景 | 可泛化到新物体、新布局 |

#### OpenPI 端到端流程（参考源项目）

```
1. 数据采集 → 脚本轨迹 / 遥操作 → HDF5
2. 格式转换 → HDF5 → LeRobot 格式
3. 模型训练 → OpenPI Server 微调 VLA 模型
4. 部署推理 → eval.py 连接 OpenPI Server
              → 每步发送：3 路相机图像 + 16 维本体感知 + 语言指令
              → 接收 16 维动作 → OSC 控制器 → 仿真步进
```

#### 16 维动作格式

```
[l_pos_x, l_pos_y, l_pos_z,                    # 左臂末端位置（基座系）
 l_quat_x, l_quat_y, l_quat_z, l_quat_w,       # 左臂末端姿态
 l_grip,                                         # 左夹爪
 r_pos_x, r_pos_y, r_pos_z,                    # 右臂末端位置
 r_quat_x, r_quat_y, r_quat_z, r_quat_w,       # 右臂末端姿态
 r_grip]                                         # 右夹爪
```

#### 从脚本轨迹到 OpenPI 的推荐路径

1. **先跑通脚本轨迹**：用 `--mode fixed` 确认场景和控制链路正常
2. **验证物体坐标系追踪**：用 `--mode mp` 确认轨迹能自动适配物体位置
3. **移植数据采集层**：从源项目移植 `DataCollectionManager` + HDF5 存储
4. **批量采集数据**：用 `--mode rand-low --episodes 500` 生成多样化数据
5. **转换格式并训练**：HDF5 → LeRobot → OpenPI 微调
6. **移植推理入口**：从源项目移植 `eval.py`，复用本示例的 OSC 控制器和坐标转换

## 代码架构

```
run_d12_demo.py                       ← 运行入口
    ├── D12Env(OrcaGymLocalEnv)       ← 环境子类，实现 reset_model()
    ├── d12_robot_config.py           ← 机器人配置
    ├── pose_resolver.py              # 物体坐标系解析
    │       └── scipy.spatial.transform.R  ← SLERP / 坐标转换
    ├── object_randomizer.py          # 物体随机化
    │       └── env.set_joint_qpos() ← MuJoCo 接口
    ├── orca_gym OSC 控制器           ← 末端位姿 → 关节扭矩
    └── orca_gym 坐标转换工具         ← quat2axisangle 等
```

不依赖：`DualArmEnv`、`RobomimicEnv`、`DualArmRobot`、`PickPlaceTask`（这些有 import 问题或不需要）。

## 常见问题

### Q: 运行脚本报 "Body xxx not found"？
A: 场景中的物体名称可能与配置不一致。用 `python run_d12_demo.py --mode dump` 查看实际 body 名称，然后修改 YAML 中的 `l_object_frame`。

### Q: 机器人不动或动作异常？
A: 检查 OrcaLab 仿真是否已启动，场景是否已加载 d12 机器人。

### Q: 随机化后抓不到物体？
A: 降低随机化范围（用 `rand_low.yaml`），或调整 `pose_mp.yaml` 中的 `l_target_o` 偏移量。

### Q: 如何自己写一段轨迹？
A: 1) 用 `--mode dump` 查看物体位姿  2) 在 YAML 中添加 segments  3) 先用 `pose_scripted.yaml` 格式验证，再转成 `pose_mp.yaml` 格式

### Q: 本示例支持 OpenPI 推理吗？
A: 当前不支持。本示例仅包含脚本轨迹执行（Motion Planning）。OpenPI 推理（`eval.py`）和 HDF5 数据采集存储需要从源项目 `OrcaManipulation` 额外移植，但本示例的 OSC 控制器、坐标转换、动作格式等底层链路可直接复用。

### Q: 如何开始 OpenPI 训练？
A: 参考 https://github.com/OpenRobotLab/OpenPI ，需要 GPU + LeRobot 格式数据集。建议先跑通脚本轨迹，再从源项目移植数据采集和推理模块。
