随着样例逐渐增多，以及 OrcaEuler 体系的完善，为提高专业性和规范化，决定对 OrcaPlayground 进行以下重构。

> **术语约定**：本文档中"Euler 体系"指基于 `OrcaGymEulerEnv` 的开发范式，与 AGENTS.md 规则 2 中定义一致；"OrcaEuler 体系"为其对外品牌名。

## 一、重构目标

1. **样例规范化**：所有样例必须符合 OrcaEuler 体系规范（见 AGENTS.md 规则 2），禁止使用 `OrcaGymLocalEnv` 旧范式。
2. **样例分类**：仿照 Euler 样例课题的方式，按从易到难的顺序组织样例，包括但不限于以下几类：
   - **基础样例**：OrcaLab/OrcaStudio 的特性演示，例如动画（character）、spawnable 资产（replicator）、流体基础样例等。
   - **行走样例**：g1、wheeled_chassis、xbot 等。
   - **训练样例**：legged_gym、franka_rl、ant_rl 等。

   每个样例必须包含以下文件：
   - `run_xxx.py`：入口脚本，位于 `examples/euler/` 下
   - `xxx_env.py`：Env 子类，位于 `envs/euler/` 下，继承 `OrcaGymEulerEnv`
   - `TUTORIAL.md`：从易到难的课题式讲解文档
3. **跨平台自检**：在代码提交前进行检查，确保样例在不同操作系统（Windows、Linux）上都能正常运行；对不支持跨平台的样例，及时提醒用户。
4. **接口规范**：严格遵守 AGENTS.md 规则 4「API 隔离强制」，严禁绕过 OrcaGym 接口直接调用 MuJoCo 底层接口。若公共 API 不满足需求，按规则 4 的「缺失功能时扩展公共方法」流程处理，不在样例代码中穿墙访问。
5. **文档刷新**：提交代码前必须刷新 AGENTS.md 及相关文档，确保文档与代码一致。
6. **项目结构与代码提交**：`dev` 分支加保护，所有代码变更必须通过 Pull Request 审核；提交信息遵循 Conventional Commits（`feat:`/`fix:`/`docs:` 等类型），分支命名遵循 `<type>/<scope>-<short-description>` 格式。
7. **公共 API 清洁度自检**：代码提交前必须自检是否存在废弃、实验性或未被任何样例引用的公共 API（含本仓库 env 子类暴露的方法、OrcaGym 侧临时扩展但已无样例使用的接口）。对确认无引用的 API 及时清理删除，避免公共接口膨胀与认知负担；对存疑的 API 在 PR 中显式标注并经 reviewer 确认。

   **简述**：
   - **分支命名**：采用 `<type>/<scope>-<short-description>` 格式，type 与 Conventional Commits 一致（feat/fix/docs/refactor/chore/test/perf），如 `feat/euler-09-fluid-surface`、`fix/spawn-prefab-path`
   - **分支创建**：基于最新 `dev` 创建特性分支，不直接在 `dev` 上提交；首次推送用 `git push -u` 建立 upstream
   - **提交规范**：使用 Conventional Commits 格式 `<type>(<scope>): <subject>`，subject 用祈使句小写；选择性 `git add`，禁用 `git add .`/`-A` 防止误提交敏感文件
   - **安全约束**：禁止对 `dev`/`main`/`release/*` 使用 `git push --force`；rebase 后推送用 `--force-with-lease`
   - **PR 流程**：特性分支推送到远程后发起 PR 到 `dev`，至少 1 人 review 通过且 CI（ruff SLF001 / 跨平台 check / 文档一致性）绿色方可合并
   - **分支保护**：`main`/`dev`/`release/*` 设为受保护，禁止直接 push，合并方式优先 `Squash and merge` 或 `Rebase and merge`

   > 完整的 git 命令示例（创建分支、安全提交、PR 流程、分支保护配置）后续写入独立的 check 文档，本计划仅保留原则性约束。

## 二、执行阶段

| 阶段 | 优先级 | 内容 | 依赖 |
|------|--------|------|------|
| P0   | 立即   | 第6条分支保护 + Conventional Commits 规范落地；第4条 ruff SLF001 强制检查接入 CI | 无 |
| P1   | 短期   | 第2条样例分类与目录结构调整；存量样例迁移到 Euler 体系（第1条） | P0 完成 |
| P2   | 中期   | 第3条跨平台自检工具实现；第5条文档刷新自动化（pre-commit hook）；第7条公共 API 清洁度自检脚本与 PR 模板勾选项 | P1 完成 |
| P3   | 持续   | 新增样例按分类规范开发；与 OrcaGym 仓库协同演进 | 持续进行 |

## 三、验收标准（DoD）

| 重构项 | 验收标准 |
|--------|----------|
| 1. 样例规范化 | `examples/` 和 `envs/` 下所有样例基于 `OrcaGymEulerEnv`；`ruff check --select SLF001` 零告警 |
| 2. 样例分类 | 每个样例包含 `run_xxx.py`/`xxx_env.py`/`TUTORIAL.md`；目录结构按基础/行走/训练三类划分 |
| 3. 跨平台自检 | CI 在 Windows 和 Linux 双平台运行通过；不支持跨平台的样例在 README 和 TUTORIAL.md 中明确标注 |
| 4. 接口规范 | 代码审查清单包含"无穿墙访问"项；ruff SLF001 在 CI 中强制执行 |
| 5. 文档刷新 | PR 模板包含"文档已刷新"勾选项；AGENTS.md 与代码同步更新 |
| 6. 项目结构 | `dev`/`main`/`release/*` 分支保护已配置；CI 检查 ruff/跨平台/文档一致性全部绿色方可合并 |
| 7. 公共 API 清洁度 | PR 模板包含"已自检无废弃/未引用 API"勾选项；CI 脚本可列出未被任何样例引用的公共方法；存疑 API 在 PR 描述中显式标注 |

## 四、相关文档

- [AGENTS.md](AGENTS.md)：AI 开发指南，定义 Euler 体系约束、API 隔离规则、GPU sandbox 旁路等
- OrcaGym 架构文档：`../OrcaGym/docs/design/architecture/orca_gym_euler_architecture.md`（定义 `OrcaGymEulerEnv` 公共 API 契约）
- OrcaGym 开发阶段分解：`../OrcaGym/docs/design/development/orca_gym_euler_development.md`
- 独立 check 文档（待编写）：完整的 git 命令示例、CI 配置、pre-commit hook

## 五、与 OrcaGym 仓库的协同

- 若重构过程中发现 OrcaEuler 体系缺少所需功能，或架构约束与样例需求存在冲突，**不在样例代码中绕过封装隔离机制**，按 AGENTS.md 规则 2「冲突处理」流程联系 OrcaGym 开发者
- OrcaGym 侧的公共 API 扩展需同步更新架构文档，本仓库样例随之迁移
