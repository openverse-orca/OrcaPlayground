本示例原项目地址：git@github.com:engineai-robotics/engineai_legged_gym.git
主要将由 isaacgym 训练的机器人 PPO 模型移植到 orcagym 环境中。

## ⚠️ 资产准备

- **资产**：位于 **OrcaPlaygroundAssets**，导入后预制体为 `zq_sa01_usda`（源文件在 `robots/zq_sa01/`）。
- **是否需要手动拖动到布局中**：**否**（脚本会通过 spawn/replicator 自动创建场景）。

## 🔧 手动拖入资产进行调试

手动拖动资产的操作方式与命名建议见**项目根目录 [README - 手动拖动资产（调试时）](../../README.md#-手动拖动资产调试时)**。

**本示例修改前样例代码（手动拖入时，不调用 spawn）**：

```python
# 不调用 publish_zqsa01_scene(...)，依赖场景中已存在对应名称的 actor
agent_name = "zq_sa01"
env_id, kwargs = register_env(orcagym_addr, env_name, 0, agent_name, sys.maxsize)
env = gym.make(env_id)
```