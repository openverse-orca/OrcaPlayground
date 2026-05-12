# OrcaGym Environments

本目录包含各种机器人环境的实现，这些是**参考实现**，供用户学习和定制。

## 📦 重要说明

⚠️ **这些环境不包含在 `orca-gym` PyPI 包中**

原因：
- 这些是**示例环境**，不是通用库
- 用户通常需要根据自己的任务**定制环境**
- 环境与特定的机器人模型和任务绑定

## 🎯 如何使用这些环境

### 方式 1：克隆仓库 + 开发模式安装（推荐）

```bash
# 1. 克隆完整仓库
git clone https://github.com/openverse-orca/OrcaGym.git
cd OrcaGym

# 2. 以开发模式安装
pip install -e .

# 3. 直接使用
python examples/legged_gym/run_legged_sim.py
```

### 方式 2：复制到自己的项目

```bash
# 复制需要的环境
cp -r envs/manipulation my_project/envs/

# 修改导入路径
# 从: from envs.manipulation import SingleArmEnv
# 到: from my_project.envs.manipulation import SingleArmEnv
```

## 📁 环境目录结构

```
envs/
├── README.md                    # 本文件
├── __init__.py
├── common/                      # 公共工具
│   └── model_scanner.py         # 场景模型扫描（机器人发现）
│
├── legged_gym/                  # 🦿 足式机器人（RL 训练 + 交互仿真）
│   ├── legged_gym_env.py        #   Gym 训练环境（OrcaGymAsyncEnv）
│   ├── legged_sim_env.py        #   交互仿真环境（OrcaGymLocalEnv）
│   ├── legged_robot.py          #   机器人 Agent（观测/奖励/PD控制）
│   ├── legged_config.py         #   全局配置
│   ├── legged_utils.py          #   工具函数
│   ├── robot_locator.py         #   动态机器人发现
│   ├── adapters/rllib/          #   RLlib 适配层
│   │   ├── legged_vector_env.py #     动态多机器人向量环境
│   │   ├── legged_env_runner.py #     自定义 EnvRunner
│   │   ├── appo_catalog.py      #     Dict 观测 APPO Catalog
│   │   └── metrics_callback.py  #     训练指标回调
│   ├── robot_config/            #   机器人型号配置
│   │   ├── Lite3_config.py      #     Lite3 四足
│   │   ├── go2_config.py        #     Go2 四足
│   │   ├── g1_config.py         #     G1 双足
│   │   ├── A01B_config.py       #     A01B
│   │   └── AzureLoong_config.py #     AzureLoong
│   ├── scripts/                 #   训练/转换脚本
│   │   └── scene_util.py        #     场景管理
│   ├── utils/                   #   ONNX 推理工具
│   │   ├── onnx_policy.py
│   │   └── lite3_obs_helper.py
│   ├── RLLIB_README.md          #   RLlib 训练详细文档
│   └── MIGRATION_README.md      #   迁移说明
│
├── manipulation/                # 🦾 机械臂操作
│   ├── single_arm_env.py        #   单臂环境
│   ├── dual_arm_env.py          #   双臂环境
│   ├── dual_arm_robot.py        #   双臂机器人
│   └── robots/                  #   机器人模型
│       ├── openloong_gripper_fix_base.py
│       ├── openloong_gripper_mobile_base.py
│       ├── openloong_hand_fix_base.py
│       └── configs/             #     机器人配置
│
├── aloha/                       # 🤖 ALOHA 双臂机器人
│   ├── aloha_env.py
│   ├── aloha_dm_env.py
│   └── aloha_orcagym_task.py
│
├── g1/                          # 🏃 G1 人形机器人
│   ├── g1_env.py
│   ├── rl_policy/               #   RL 策略
│   └── utils/                   #   工具
│
├── so101/                       # 🦾 SO101 机械臂
│   ├── so101_env.py
│   ├── so101_robot.py
│   ├── openpi_client/           #   OpenPI 推理客户端
│   └── configs/
│
├── drone/                       # 🚁 无人机
│   ├── drone_orca_env.py
│   └── drone_aero_config.py
│
├── fluid/                       # 🌊 流体仿真
│   ├── sim_env.py
│   ├── coupling_modes/          #   耦合模式
│   ├── launch/                  #   启动脚本
│   ├── modules/                 #   功能模块
│   └── utils/                   #   工具
│
├── fluid_stats/                 # 📊 流体统计
│
├── character/                   # 👤 人形角色
│   ├── character.py
│   ├── character_env.py
│   └── character_config/
│
├── hand_detection/              # ✋ 手部检测
│   └── hand_detection_env.py
│
├── realman/                     # 🎮 Realman 机器人
│   ├── rm65b_joystick_env.py
│   ├── rm75bv_joystick_env.py
│   ├── rm75bv_vr_env.py
│   └── realman_rm65b/           #   底层驱动
│
├── wheeled_chassis/             # 🚗 轮式底盘
│   ├── wheeled_chassis_env.py
│   └── ackerman_env.py
│
├── xbot_gym/                    # 🤖 XBot 机器人
│   └── xbot_simple_env.py
│
├── zq_sa01/                     # 🏃 ZQ SA01 人形
│   └── zq_sa01_env.py
│
└── mujoco/                      # 🏋️ MuJoCo 示例
    └── ant_orcagym.py
```

## 🦿 足式机器人 (legged_gym)

用于四足/双足机器人的 RL 训练环境，支持 SB3 和 RLlib 两种训练框架。

**包含**:
- Lite3, Go2, G1, A01B, AzureLoong 机器人配置
- 地形生成与 Curriculum learning
- 动态机器人发现（`robot_locator`）
- RLlib APPO 分布式训练
- ONNX 模型转换与推理

**SB3 训练**:
```bash
python examples/legged_gym/run_legged_rl.py \
    --config examples/legged_gym/configs/sb3_ppo_config.yaml --train
```

**RLlib 训练**:
```bash
python examples/legged_gym/run_legged_rl.py \
    --config examples/legged_gym/configs/rllib_appo_config.yaml --train
```

**交互仿真**:
```bash
python examples/legged_gym/run_legged_sim.py \
    --config examples/legged_gym/configs/lite3_sim_config.yaml
```

**详细文档**: [examples/legged_gym/README.md](../examples/legged_gym/README.md)

## 🦾 机械臂操作 (manipulation)

单臂和双臂机械臂操作环境。

**包含**:
- 单臂环境 (OpenLoong + Gripper/Hand)
- 双臂环境 (OpenLoong)
- 固定底座 / 移动底座支持

**使用示例**:
```python
from envs.manipulation.single_arm_env import SingleArmEnv, RunMode

env = SingleArmEnv(
    orcagym_addr="localhost:50051",
    robot_name="franka",
    run_mode=RunMode.SIM
)
```

## 🤖 ALOHA 机器人 (aloha)

ALOHA 双臂移动操作平台。

**使用示例**:
```python
from envs.aloha.aloha_env import AlohaEnv

env = AlohaEnv(orcagym_addr="localhost:50051")
```

## 🚗 轮式底盘 (wheeled_chassis)

差速驱动和阿克曼转向底盘。

**使用示例**:
```python
from envs.wheeled_chassis.wheeled_chassis_env import WheeledChassisEnv

env = WheeledChassisEnv(orcagym_addr="localhost:50051")
```

## 👤 人形角色 (character)

人形角色控制和动画。

**相关示例**: `examples/character/`

## 🎮 Realman 机器人 (realman)

Realman RM65B/RM75BV 机器人接口（摇杆/VR 控制）。

## 🚁 无人机 (drone)

无人机仿真环境。

**相关示例**: `examples/drone_driver/`

## 🌊 流体仿真 (fluid)

SPH 流体与 MuJoCo 耦合仿真。

**相关示例**: `examples/fluid/`

## 🔧 定制自己的环境

### 1. 继承基类

所有环境都继承自 `orca_gym.environment.OrcaGymBaseEnv`:

```python
from orca_gym.environment import OrcaGymRemoteEnv
import gymnasium as gym

class MyCustomEnv(OrcaGymRemoteEnv):
    def __init__(self, **kwargs):
        super().__init__(
            frame_skip=5,
            orcagym_addr="localhost:50051",
            agent_names=["my_robot"],
            time_step=0.002,
            **kwargs
        )
        
    def _get_obs(self):
        pass
        
    def compute_reward(self, achieved_goal, desired_goal, info):
        pass
```

### 2. 定义观察空间

```python
def _get_obs(self):
    obs = {
        'observation': np.concatenate([
            self.data.qpos,
            self.data.qvel,
        ]),
        'achieved_goal': self.get_end_effector_pos(),
        'desired_goal': self.goal_pos,
    }
    return obs
```

### 3. 定义动作空间

```python
self.action_space = gym.spaces.Box(
    low=-1.0, high=1.0, shape=(7,), dtype=np.float32
)
```

### 4. 实现奖励函数

```python
def compute_reward(self, achieved_goal, desired_goal, info):
    distance = np.linalg.norm(achieved_goal - desired_goal)
    return -distance
```

## 📖 相关文档

- [RLlib 多机器人训练详细文档](legged_gym/RLLIB_README.md)
- [Lite3 迁移说明](legged_gym/MIGRATION_README.md)
- [核心库 API](../orca_gym/README.md)
- [示例代码](../examples/README.md)
- [Gymnasium 文档](https://gymnasium.farama.org/)

## 🆘 常见问题

### Q: 为什么这些环境不在 PyPI 包中？

A: 因为这些是**示例和参考实现**，用户通常需要根据自己的任务定制。将其作为独立文件更灵活。

### Q: 如何在我的项目中使用这些环境？

A: 有两种方式：
1. 克隆仓库，以开发模式安装
2. 复制需要的环境到你的项目，修改导入路径

### Q: RLlib 和 SB3 训练有什么区别？

A: SB3 适合单机训练，RLlib 适合多机分布式训练。两者共享相同的环境代码（`legged_gym_env.py`），只是训练框架不同。

### Q: 如何贡献新环境？

A: 
1. Fork 仓库
2. 在 `envs/` 下添加你的环境
3. 在 `examples/` 下添加使用示例
4. 提交 Pull Request

## 📞 获取帮助

- 查看示例代码: `examples/`
- 查看核心库文档: `orca_gym/`
- 提交 Issue: https://github.com/openverse-orca/OrcaGym/issues
