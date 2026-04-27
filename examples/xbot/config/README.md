# XBot配置文件

## 🔧 布局中的 actor 准备

在使用本目录策略文件前，建议先在 OrcaStudio / OrcaLab 中把 XBot 对应 actor 手动拖到布局中。

- **是否需要手动拖动到布局中**：**是**
- **UI 操作建议**：在资产面板中搜索 `XBot-L_usda` 或 `Xbot_usda`，拖入布局后点击“资产详情”确认实际路径
- **路径说明**：不同资产包版本的显示路径可能不同，请以 UI 中“资产详情”的实际结果为准
- **目的**：便于和地形、障碍物或其他 actor 组合，增强多场景物理交互调试效果

## 📦 策略文件

### policy_example.pt
- **来源**: [humanoid-gym](http://github.com/roboterax/humanoid-gym.git)
- **大小**: 2.02 MB
- **格式**: TorchScript JIT模型
- **用途**: XBot机器人的预训练行走策略

**特点**:
- ✅ 已集成在项目内，无需外部依赖
- ✅ 支持稳定的双足行走
- ✅ 可以在OrcaGym环境中直接加载使用

## 📋 训练配置

### xbot_train_config.yaml
- **用途**: XBot训练的环境配置
- **包含**: 物理参数、奖励设置、训练超参数等

## 🔄 更新策略文件

如果需要使用新训练的策略：

1. 从humanoid-gym训练完成后，导出策略：
```bash
cd /path/to/humanoid-gym
python humanoid/scripts/export_policy.py --task XBotL_free
```

2. 复制到config目录：
```bash
cp humanoid-gym/logs/XBot_ppo/exported/policies/policy_*.pt \
   OrcaGym/examples/xbot/config/policy_example.pt
```

3. 直接运行即可生效：
```bash
python examples/xbot/run_xbot_orca.py
```

