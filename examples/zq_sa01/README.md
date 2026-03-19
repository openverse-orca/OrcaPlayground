# ZQ SA01 使用指南

本示例原项目地址：`git@github.com:engineai-robotics/engineai_legged_gym.git`，主要将由 Isaac Gym 训练的机器人 PPO 模型移植到 OrcaGym 环境中。

## 运行前准备

- 资产位于 **OrcaPlaygroundAssets**，导入后预制体为 `zq_sa01_usda`
- 是否需要手动拖动到布局中：**是**
- 需要先在场景中手动摆好 ZQ SA01 机器人
- 脚本会在启动前扫描场景中的关节和驱动器后缀，自动识别实际机器人实例名
- 如果关节或驱动器没有完整匹配，脚本会直接报错退出

## 🔧 手动拖入资产进行调试

为了增添多场景物理交互，请先把 ZQ SA01 actor 拖到布局中，再和地面、障碍物或其他对象一起摆位。通用拖入方式见**项目根目录 [README - 手动拖动资产（运行前必做）](../../README.md#-手动拖动资产运行前必做)**。

## 启动示例

```bash
python examples/zq_sa01/run_zqsa01.py
```

## 说明

- 当前入口按 `leg_l1_joint ... leg_r6_joint` 这一组后缀模板进行识别
- 不需要手动传入场景中的机器人实例名
