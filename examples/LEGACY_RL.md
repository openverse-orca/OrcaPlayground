# 强化学习样例迁移说明

本目录原有的三个强化学习（RL）样例已从主分支移除：

| 样例 | 说明 | 旧基类 | Euler 兼容性 |
|------|------|--------|--------------|
| `ant_rl/` | Ant 机器人 RL（Ray RLlib APPO 多环境并行训练） | `OrcaGymEulerEnv`（env 已迁移，训练入口仍依赖 legged_gym） | 部分 |
| `franka_rl/` | Franka 多机械臂 RL（SB3 + HER） | `OrcaGymAsyncEnv` | 未迁移 |
| `legged_gym/` | 足式机器人 RL 训练 + 交互仿真（SB3 PPO + RLlib APPO） | `OrcaGymAsyncEnv` / `OrcaGymLocalEnv` | 未迁移 |

## 为什么移除

1. **Euler 架构合规**：主分支样例必须基于 `OrcaGymEulerEnv`（见 `AGENTS.md` 规则 2）。`franka_rl` 与 `legged_gym` 仍依赖 `OrcaGymAsyncEnv` / `OrcaGymLocalEnv`，与 Euler 异步路径未对齐。
2. **循环依赖**：`ant_rl` 与 `legged_gym` 存在循环 import（详见原 `examples/CROSS_REFERENCES.md`），单独保留无法独立运行。
3. **训练管线重**：三个样例都包含完整的 RL 训练管线（Ray RLlib / SB3 + HER + checkpoint），不属于 PRD v2 第一部分（渐进式基础）或第二部分（场景构建）的教学范畴。

## 获取旧版本

这些样例的可用版本保留在 `release/26.7.1` 分支，可通过以下方式获取：

```bash
# 方式 1：切换到 release 分支查看
git checkout release/26.7.1
ls examples/ant_rl examples/franka_rl examples/legged_gym

# 方式 2：仅检出三个样例目录到当前工作区（不影响当前分支其他文件）
git checkout release/26.7.1 -- examples/ant_rl examples/franka_rl examples/legged_gym

# 方式 3：在新克隆中查看
git clone -b release/26.7.1 <repo-url> orcaplayground-rl
```

> **注意**：`release/26.7.1` 分支的样例不会随 Euler API 演进更新，仅作为历史参考。运行前请确认 OrcaGym 版本与该分支兼容。

## 新 RL 样例开发计划

新的 Euler 兼容 RL 样例正在规划中，将作为 PRD v2 第三部分（具身场景）的子模块发布，特性包括：

- 完全基于 `OrcaGymEulerEnv`（或 Euler 异步路径就绪后的 `OrcaGymEulerAsyncEnv`）
- 消除跨样例循环依赖，每个样例自包含训练入口
- 提供从零到一的 RL 训练教程（而非完整的工程化训练管线）

跟踪进度请关注 `examples/euler/` 目录扩展与 PRD v2 第三部分规划文档。

## 旧样例引用清单（供迁移参考）

下表为 `release/26.7.1` 分支中旧样例的关键文件，供新样例迁移时参考：

| 旧样例 | 关键文件 | 迁移要点 |
|--------|----------|----------|
| `ant_rl/` | `ant_orcagym.py`（`AntOrcaGymEnv(OrcaGymEulerEnv)`） | env 层已迁移；训练入口需脱离 legged_gym 的 `rllib_appo_rl.py` |
| `franka_rl/` | `franka_gym_env.py`、`franka_agent.py` | env/agent 需从 `OrcaGymAsyncEnv` / `OrcaGymAsyncAgent` 迁移到 Euler 异步路径 |
| `legged_gym/` | `legged_gym_env.py`、`legged_robot.py`、`legged_sim_env.py` | env/agent 需迁移；`LeggedSimEnv(OrcaGymLocalEnv)` 需重写为 Euler 交互分支 |
