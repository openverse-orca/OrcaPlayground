# OrcaGym Euler 示例课程

本课程通过一系列由浅入深的示例，演示如何使用 `OrcaGymEulerEnv` 开发自己的仿真。
每个示例既是 OrcaGymEuler 开发阶段的验证载体，也是面向用户的教学用例。

> 课程设计参考 [OrcaFlow examples/TUTORIAL.md](../../../OrcaFlow/examples/TUTORIAL.md)，
> 每个 example 对应一个开发阶段（P3 → P5），引入 1-2 个新概念。

## 快速开始

```bash
# 进入 OrcaPlayground 根目录
cd /path/to/OrcaPlayground
conda activate orca

# 第 1 课：Hello Euler（离线，不需要 OrcaStudio/OrcaLab）
python examples/euler/01_hello_euler/hello_euler.py

# 第 2 课：在线渲染（需要先启动 OrcaStudio/OrcaLab）
python examples/euler/02_online_render/online_render.py

# 第 3 课：SB3 PPO 训练（离线，约 2-3 分钟）
python examples/euler/03_rl_ppo/train_ppo.py --total-timesteps 100000

# 第 4-9 课：在线端到端验证（需 OrcaStudio/OrcaLab + G1 关卡）
python examples/euler/04_query_api/query_api.py
```

> 详细教程见各课程目录下的 `0X_*.md`。

---

## 目录结构

```
examples/euler/
├── TUTORIAL.md                     ← 你正在阅读的总纲
├── requirements.txt                # 课程依赖
├── .gitignore                      # 忽略训练产物
├── 00_setup.md                     # 阶段四（Lesson 4-9）环境搭建
├── 01_hello_euler/                 # 第 1 课（P3 离线联调）
├── 02_online_render/               # 第 2 课（P3A 在线渲染）
├── 03_rl_ppo/                      # 第 3 课（P3B SB3 PPO）
├── 04_query_api/                   # 第 4 课（P4 状态查询 API）
├── 05_force_apply/                 # 第 5 课（P4 外力应用与约束）
├── 06_jacobian/                    # 第 6 课（P4 雅可比 IK 与 mocap）
├── 07_locomotion/                  # 第 7 课（P4 G1 行走控制链路）
├── 08_video_capture/               # 第 8 课（P4 Studio 视频录制与截帧）
└── 09_body_manipulation/           # 第 9 课（P4 体操作与 equality）

assets/                             # 全局资产（迁移后集中管理）
├── g1/                             # G1 人形机器人资源
│   ├── g1_29dof_camera.xml
│   ├── config/g1_29dof_hist.yaml
│   ├── meshes/                     # G1 网格
│   └── models/
│       ├── dec_loco/model_6600.onnx   # 行走策略
│       └── mimic/                     # mimic 数据
└── scenes/
    └── simple_pendulum.xml         # 第 1-3 课场景
```

> **自包含原则**：每个 example 目录包含全部所需 `.py` 文件，import 使用同目录引用，
> 资产路径通过 `__file__` 上溯 4 层定位 `assets/`。零 `envs/` 共享目录依赖，
> 零 Local 体系（`OrcaGymLocalEnv`）引用。

---

## 课程总览

### 阶段三：离线/在线联调（Lesson 1-3）

| 课 | 阶段 | 主题 | 新概念 | 状态 | OrcaStudio/OrcaLab | 教程 |
|----|------|------|--------|------|-----------|------|
| 1 | P3 | Hello Euler | `OrcaGymEulerEnv` 基本用法、离线模式、模型加载、步进、状态访问 | ✅ | 否 | [01_hello_euler.md](01_hello_euler/01_hello_euler.md) |
| 2 | P3A | 在线渲染与交互 | gRPC 在线模式、render 循环、sync/异步渲染、RTF、override_ctrls | ✅ | 是 | [02_online_render.md](02_online_render/02_online_render.md) |
| 3 | P3B | SB3 PPO 强化学习 | Gymnasium API 契约、SB3 PPO、奖励函数设计、Box 观测、episode 截断 | ✅ | 否（离线训练） | [03_rl_ppo.md](03_rl_ppo/03_rl_ppo.md) |

### 阶段四：在线端到端验证（Lesson 4-9，需 OrcaStudio/OrcaLab + G1 关卡）

阶段四使用 G1 人形机器人在 OrcaStudio/OrcaLab 中进行在线验证。每个课程遵循 **5 步手工验证流程**：

1. **（人工）** 启动 OrcaStudio/OrcaLab，加载含 1 个 G1 机器人的关卡，点击运行
2. **（人工）** 运行 `examples/euler/0X_*/` 下对应的课程脚本
3. **（自动）** 脚本驱动 `G1BaseEnv` 子类完成功能
4. **（人工）** 根据教程指导，观察 Studio 视口，确认画面符合预期
5. **（自动）** 脚本通过 `OnlineVerifier` 检查输出，输出判定报告

| 课 | 阶段 | 主题 | 新概念 | 状态 | 教程 |
|----|------|------|--------|------|------|
| 4 | P4 | 状态查询 API | `query_joint_*`、`query_site_*`、`query_sensor_data`、`query_contact_*` | ✅ | [04_query_api.md](04_query_api/04_query_api.md) |
| 5 | P4 | 外力应用与约束 | `apply_body_force`、`clear_*_force`、`update_equality_constraints`、`set_*` | ✅ | [05_force_apply.md](05_force_apply/05_force_apply.md) |
| 6 | P4 | 雅可比 IK 与 mocap | `mj_jacBody`、`mj_jacSite`、`set_mocap_pos_and_quat`、`anchor_actor` | ✅ | [06_jacobian.md](06_jacobian/06_jacobian.md) |
| 7 | P4 | G1 行走控制链路 | ONNX 推理、PD 控制器、行走稳定性、力矩触限 | ✅ | [07_locomotion.md](07_locomotion/07_locomotion.md) |
| 8 | P4 | Studio 视频录制与截帧 | `get_current_frame`、`get_next_frame`、`get_frame_png`、视频录制 | ✅ | [08_video_capture.md](08_video_capture/08_video_capture.md) |
| 9 | P4 | 体操作与 equality | `equality_find_slot_by_body`、`equality_constraint`、`equality_update`、mocap 拖拽 + 行走 | ✅ | [09_body_manipulation.md](09_body_manipulation/09_body_manipulation.md) |

> 阶段四环境搭建详见 [00_setup.md](00_setup.md)。

---

## 课程简介

### 第 1 课：Hello Euler — 第一个 OrcaGymEulerEnv 程序

用随机动作驱动单铰链倒立摆，验证 `OrcaGymEulerEnv` 体系的端到端 API 契约（模型加载、
状态访问、求解器配置、reset、步进）。聚焦**离线模式**，不需要 OrcaStudio/OrcaLab。

- **运行**：`python examples/euler/01_hello_euler/hello_euler.py`
- **需要 OrcaStudio/OrcaLab**：否
- **预计时长**：< 10 秒

📖 详细教程见 [01_hello_euler/01_hello_euler.md](01_hello_euler/01_hello_euler.md)

---

### 第 2 课：在线渲染与交互 — 连接 OrcaStudio/OrcaLab

在第 1 课基础上启用 gRPC 在线模式，连接 OrcaStudio/OrcaLab 实时渲染。聚焦在线模式特有能力：
渲染循环、同步/异步渲染、RTF 实时同步、Studio UI 交互（override_ctrls / 拖拽物体）。

- **运行**：`python examples/euler/02_online_render/online_render.py`
- **需要 OrcaStudio/OrcaLab**：是（加载 pendulum 场景）
- **预计时长**：持续运行直至 Ctrl+C

📖 详细教程见 [02_online_render/02_online_render.md](02_online_render/02_online_render.md)

---

### 第 3 课：SB3 PPO 强化学习 — 训练倒立摆保持直立

基于 Stable Baselines3 PPO 算法训练 `SimpleEulerEnv`，验证 `OrcaGymEulerEnv` 与
Gymnasium API 契约的兼容性。训练默认离线无头（最高效），评估默认 online（可视化），
Studio 未启动时自动退化离线。

- **运行**：`python examples/euler/03_rl_ppo/train_ppo.py --total-timesteps 100000`
- **需要 OrcaStudio/OrcaLab**：否（训练离线，评估可选 online）
- **预计时长**：100k 步约 2-3 分钟（GPU）

📖 详细教程见 [03_rl_ppo/03_rl_ppo.md](03_rl_ppo/03_rl_ppo.md)

---

### 第 4 课：状态查询 API — G1 全套查询 API 在线验证

使用 G1 人形机器人在线验证 OrcaGymEuler 的全套状态查询 API：关节状态、Body 位姿、
传感器、执行器力矩、接触、质量。通过 `OnlineVerifier` 自动判定数值合理性。

- **运行**：`python examples/euler/04_query_api/query_api.py`
- **需要 OrcaStudio/OrcaLab**：是（加载 G1 关卡）
- **预计时长**：< 1 分钟

📖 详细教程见 [04_query_api/04_query_api.md](04_query_api/04_query_api.md)

---

### 第 5 课：外力应用与约束 — 施力、摩擦、状态写入

验证外力应用 API（`apply_body_force` / `clear_*_force`）、摩擦设置（`set_geom_friction`）、
状态写入（`set_joint_qpos` / `set_joint_qvel`）。通过对照实验验证外力效果。

- **运行**：`python examples/euler/05_force_apply/force_apply.py`
- **需要 OrcaStudio/OrcaLab**：是（加载 G1 关卡）
- **预计时长**：< 1 分钟

📖 详细教程见 [05_force_apply/05_force_apply.md](05_force_apply/05_force_apply.md)

---

### 第 6 课：雅可比 IK 与 mocap — 逆运动学控制

验证雅可比矩阵计算（`mj_jacBody` / `mj_jacSite`）、Site 查询、Mocap 控制
（`set_mocap_pos_and_quat`）、锚点操作（`anchor_actor`）。通过 IK 求解驱动 G1 手部
追踪目标轨迹。

- **运行**：`python examples/euler/06_jacobian/jacobian_ik.py`
- **需要 OrcaStudio/OrcaLab**：是（加载 G1 关卡）
- **预计时长**：< 1 分钟

📖 详细教程见 [06_jacobian/06_jacobian.md](06_jacobian/06_jacobian.md)

---

### 第 7 课：G1 行走控制链路 — ONNX 推理 + PD 控制器 + 稳定性验证

验证 G1 行走控制链路（ONNX 推理 → PD 控制器 → motor 力矩 → 行走稳定性），
聚焦行走控制本身，不耦合视频采集（视频采集见 Lesson 8）。

- **运行**：`python examples/euler/07_locomotion/locomotion.py`
- **需要 OrcaStudio/OrcaLab**：是（加载 G1 关卡）
- **预计时长**：< 1 分钟

📖 详细教程见 [07_locomotion/07_locomotion.md](07_locomotion/07_locomotion.md)

---

### 第 8 课：Studio 视频录制与截帧 — G1 行走录制在线验证

验证 Studio 视频/帧/时间戳采集 API（`get_current_frame` / `get_next_frame` /
`get_frame_png`），复用 Lesson 7 的行走控制链路驱动 G1 行走并录制。

- **运行**：`python examples/euler/08_video_capture/video_capture.py`
- **需要 OrcaStudio/OrcaLab**：是（加载 G1 关卡）
- **预计时长**：约 10 秒（录制 5 秒视频 + 截图）

📖 详细教程见 [08_video_capture/08_video_capture.md](08_video_capture/08_video_capture.md)

---

### 第 9 课：体操作与交互式驱动 — 锚定/mocap 驱动/equality

交互式菜单驱动模式验证 Studio 体操作 API 与 MuJoCo equality 约束。使用公共无状态
原语（`equality_find_slot_by_body` / `equality_constraint` / `equality_update`）自管
编排绑定/释放，通过 mocap 拖拽 weld 约束驱动 G1 pelvis，并结合 ONNX 行走策略。

- **运行**：`python examples/euler/09_body_manipulation/body_manipulation.py`
- **需要 OrcaStudio/OrcaLab**：是（加载 G1 关卡）
- **预计时长**：交互式，取决于用户输入

📖 详细教程见 [09_body_manipulation/09_body_manipulation.md](09_body_manipulation/09_body_manipulation.md)

---

## 环境准备

### 1. 安装依赖

```bash
cd /path/to/OrcaPlayground
pip install -r examples/euler/requirements.txt
```

### 2. 前置条件

| 依赖 | 说明 |
|------|------|
| OrcaGym | 已安装（`orca_gym` 包可在 Python 中 import） |
| MuJoCo | OrcaGym 自带的 `mujoco` 包 |
| OrcaStudio/OrcaLab | **第 2 课、第 4-9 课需要**，需启动 gRPC 服务（默认 `localhost:50051`） |
| stable-baselines3 | **仅第 3 课需要**，`requirements.txt` 已包含 |

### 3. 验证安装

```bash
conda activate orca
python -c "from orca_gym.environment.euler import OrcaGymEulerEnv; print('OK')"
python -c "import stable_baselines3; print('sb3:', stable_baselines3.__version__)"
```

---

## 常见问题

### Q1：第 1 课随机采样，为何奖励看似"上升"？

**原因**：这不是学习效果，是奖励函数目标定义的巧合。

- **修复前**：奖励为 `cos(theta)`，目标是摆杆水平。随机扰动后摆杆恰好经过水平位置，
  导致 reward 接近 1，呈现"上升"假象。
- **修复后**：奖励改为 Pendulum-v1 标准 cost `-(theta² + ...)`，目标是直立。随机动作下
  reward 为大负值，**不再呈现上升趋势**。

### Q2：第 3 课训练时出现 `UserWarning: You are trying to run PPO on the GPU`

**原因**：SB3 检测到 GPU 可用，默认用 GPU，但对 MLP 策略推荐 CPU
（[issue #1245](https://github.com/DLR-RM/stable-baselines3/issues/1245)）。

**说明**：项目统一采用 GPU 训练（`--device cuda`，默认值），实测 CPU 训练 MLP 较慢。
此警告可忽略，不影响训练正确性。

### Q3：训练时出现 `Error 304: OS call failed`

**原因**：在 TRAE sandbox 内运行，CUDA 驱动因 capabilities 限制无法初始化。

**解决**：GPU 训练需 TRAE 命令白名单旁路 sandbox（详见 AGENTS.md 规则 3），或使用
`--device cpu` 退化到 CPU 训练。

### Q4：第 2 课在线模式连接 OrcaStudio/OrcaLab 失败

**排查**：
1. 确认 OrcaStudio/OrcaLab 已启动并监听 `localhost:50051`
2. 确认场景中已加载 pendulum 模型
3. 确认 `--addr` 参数正确
4. 第 2 课默认在线模式（`skip_grpc_load=False`），无需额外参数

### Q5：`ModuleNotFoundError: No module named 'envs'`

**原因**：旧文档遗留。迁移后所有 example 自包含，无 `envs/` 共享目录。

**解决**：本目录已自包含，运行时 Python 自动将脚本所在目录加入 `sys.path[0]`，
同目录 `from simple_env import ...`（第 1-3 课）或 `from g1_base_env import ...`
（第 4-9 课）可直接生效，无需额外 `PYTHONPATH`。若仍报此错，请确认使用的是迁移后
的脚本（位于 `examples/euler/0X_*/`），而非旧的 `envs/euler/` 路径。
