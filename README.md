# OrcaPlayground（新手示例库重构分支）

本分支 `feat/playground-beginner-redesign` 从接近空的状态重建 OrcaPlayground 为**新手优先的示例库**。

- 目标用户：会运行少量 Python，但无机器人 / 物理仿真 / 强化学习经验的用户
- 规划：48 个基础课 + 8 个选修专题（详见 [docs/beginner/REQUIREMENTS.md](docs/beginner/REQUIREMENTS.md)）
- 首版交付范围：01–18 课（场景与资产 → 场景编辑 → 时间与物理）

> 当前阶段：**需求分析已定稿（v0.2，含决策记录），课程代码尚未开始编写。**
> 本分支最终将**整体替换 `dev`**：旧 Euler 课程（01–11）与 embodied 高级样例在合入主线时移除，
> 替换前在 `dev` 上打 `legacy/pre-redesign` 存档标签供历史访问。

## 开发规则

所有 AI 代理与贡献者必须遵守 [AGENTS.md](AGENTS.md)：
- 测试与运行示例必须使用 `orca` conda 环境
- 新示例一律基于 `OrcaGymEulerEnv`（Euler 体系），禁止穿墙访问 `_` 前缀内部属性
- 提交前 ruff `--select SLF001` 零报警
