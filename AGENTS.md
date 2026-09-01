# OrcaPlayground AI 开发指南

本文件为 AI 代理（如 Trae、Cursor 等）在本仓库工作时提供强制规则。AI 代理必须严格遵守。

## 规则 1：测试与调试环境

AI 代理执行测试、调试、运行示例脚本时，**必须使用 `orca` conda 环境**。

```bash
# 正确
conda activate orca
python examples/character/run_character.py

# 错误 — 不要使用 base 或其他环境
conda activate base
python examples/character/run_character.py
```

`orca` 是 README 推荐的环境名称，已安装本项目所有依赖。使用其他环境会导致依赖缺失或版本不一致。

## 规则 2：新 Example 开发使用 Euler 体系

新开发的 example（位于 `examples/` 和 `envs/` 下）**必须使用 Euler 体系**，即基于 `OrcaGymEulerEnv` 而非 `OrcaGymLocalEnv`。

Euler 体系的架构约束参考 OrcaGym 仓库的架构文档：

`../OrcaGym/docs/design/architecture/orca_gym_euler_architecture.md`

该文档定义了：

- `OrcaGymEulerEnv` 的公共 API 契约（状态读取 / 写入 / 仿真步进 / 求解器配置 / 名称空间）
- 封装隔离机制（禁止直接访问 `_mjModel` / `_mjData`）
- 用户代码的正确使用模式

### 新 Example 的开发规范

> Euler 教程示例（01_hello_euler ~ 11_scene_config）已迁移至独立仓库 `OrcaEulerExamples`，本仓不再保留。

1. **Env 子类**继承 `OrcaGymEulerEnv`，放在对应示例目录下
2. **入口脚本**放在对应示例目录下，提供命令行启动
3. **禁止**在 Env 子类中访问 `env._gym._sim._mjData` 或 `env._gym._mjModel`
4. **状态读取**使用 `env.data.*`（`OrcaGymDataView`）或 `env.query_*()`
5. **状态写入**使用 `env.set_*()` 或 `env.apply_body_force()`
6. **求解器配置**使用 `env.sim_config.*`

### 冲突处理

若开发过程中发现 Euler 体系缺少所需功能，或架构约束与 example 需求存在冲突，**请联系 OrcaGym 开发者寻求协助**，不要在 example 代码中绕过封装隔离机制。

配套的开发阶段分解见：`../OrcaGym/docs/design/development/orca_gym_euler_development.md`

## 规则 3：GPU 加速与 Sandbox 旁路

Euler 体系使用 GPU 加速时（MuJoCoFlow / Flow 在 GPU 上求解），**无法在 TRAE sandbox 内正确运行**。sandbox 剥离了所有进程能力，导致 `cuInit` 返回 `CUDA_ERROR_304`。AI agent 必须使用 **TRAE 命令白名单** 旁路 sandbox，才能调用 GPU。

> 用户侧的白名单配置教程见 `DEVELOPER_GUIDE.md`。本规则约束 AI agent 的命令格式。

### 核心规则

1. **GPU 命令必须以白名单解释器路径开头**。直接使用 `<conda-base>/envs/orca/bin/python` 作为命令首 token（可通过 `conda info --base` 解析 `<conda-base>`）。

2. **禁止使用 shell 管道 `|`**。管道会触发 IDE 用 `trae-sandbox` 包裹命令，重新引入能力限制，导致 `CUDA_ERROR_304`。包括 `| tail`、`| grep`、`2>&1 | ...` 等所有管道构造。

3. **输出捕获用重定向，不用管道**。如需捕获输出，将日志重定向到文件，再单独读取：
   ```bash
   # 正确 — 重定向到文件（通常安全）
   <conda-base>/envs/orca/bin/python examples/character/run_character.py > /tmp/out.log 2>&1

   # 错误 — 管道触发 sandbox 包裹
   <conda-base>/envs/orca/bin/python examples/character/run_character.py 2>&1 | tail -30
   ```

4. **若需切换目录，用 `cd` 链接**。`cd` 已在白名单中，`cd <repo-root> && <conda-base>/envs/orca/bin/python script.py` 整条链在宿主执行。

### 命令格式示例

```bash
# ✅ 正确 — 白名单解释器直接调用，无管道
<conda-base>/envs/orca/bin/python examples/character/run_character.py

# ✅ 正确 — cd 链接 + 白名单解释器
cd <repo-root> && <conda-base>/envs/orca/bin/python examples/character/run_character.py

# ✅ 正确 — 重定向到文件捕获输出
<conda-base>/envs/orca/bin/python examples/character/run_character.py > /tmp/out.log 2>&1

# ❌ 错误 — 管道触发 sandbox 包裹，GPU 不可用
<conda-base>/envs/orca/bin/python examples/character/run_character.py 2>&1 | tail -30

# ❌ 错误 — 非白名单首 token
bash -c "<conda-base>/envs/orca/bin/python examples/character/run_character.py"
```

### 识别 sandbox 包裹

若命令日志中出现 `trae-sandbox '...'` 前缀，说明命令被包裹（白名单未匹配或使用了管道）。此时 GPU 不可用，需简化命令：以白名单解释器路径开头，移除管道。

### CPU 测试无需旁路

仅使用 CPU 或纯 NumPy 的测试可在 sandbox 内直接运行，无需白名单旁路。应将 GPU 依赖的测试与 CPU 测试分离，GPU 测试标记为仅在 sandbox 外运行。

## 规则 4：API 隔离强制

本仓库采用 `_` 前缀社区约定 + ruff SLF001 静态检查，引导 AI 和用户走公共 API（OrcaGym 架构 §7）。

### 禁止穿墙访问

不得访问以下 `_` 前缀内部属性（类内部合法的 `self._xxx` 委托除外）：

- `env._gym` / `env._stub` / `env._channel` / `env._studio_bridge`
- `env._gym._sim` / `env._gym._sim._mjData` / `env._gym._sim._mjModel`
- 任何自研类（含本仓库 env 子类）的 `_` 前缀属性

### 必须使用公共 API

| 操作 | 正确 | 禁止 |
|------|------|------|
| 读取状态 | `env.data.qpos` / `env.data.body_xpos(name)` / `env.query_*()` | `env._gym._sim._mjData.qpos` |
| 写入状态 | `env.set_joint_qpos()` / `env.apply_body_force()` | `env._gym._sim._mjData.xfrc_applied[...]` |
| 步进 | `env.do_simulation(ctrl, n_frames)` / `env.step()` | `env._gym._sim._mjData.step()` |
| 求解器配置 | `env.sim_config.timestep = 0.002` | `env._gym._sim._mjModel.opt.timestep = 0.002` |

### 必须执行 ruff

提交代码前必须执行，零报警方可提交：

    <conda-base>/envs/orca/bin/python -m ruff check --select SLF001 envs/ examples/

### 缺失功能时扩展公共方法

若公共 API 不满足需求，**暂停并提交用户决策**，不要穿墙访问内部属性。扩展途径：
- 在 OrcaGym 侧添加公共方法（联系 OrcaGym 开发者）
- 在本仓库 env 子类中添加公共访问器
