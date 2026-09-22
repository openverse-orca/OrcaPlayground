# 选修 A09：GPU 后端切换

## 本课效果
同一套代码、同一个方块落体，跑两遍：一遍 CPU（MuJoCo），一遍 GPU（Euler 求解器）。
`--compare` 给出触地时间差、轨迹偏差和每帧耗时——第一次亲眼验证"物理引擎搬上 GPU"。

## 准备
- 同第 01 课（OrcaPlaygroundAssets 订阅）
- GPU 模式额外要求：NVIDIA GPU + CUDA 环境，且 `orca` 环境已装 `orca.flow` / `orca.euler` / `orca.mujoco_flow`（详见 DEVELOPER_GUIDE.md）
- 注意：GPU 命令需白名单解释器直跑（sandbox 内 `cuInit` 会失败，见 DEVELOPER_GUIDE.md）

## 运行
```bash
# CPU 单跑（基线，任何机器可跑）
python -m examples.euler.beginner.electives.A09_gpu_backend.run --default-scene

# GPU 单跑
python -m examples.euler.beginner.electives.A09_gpu_backend.run --default-scene --device cuda:0

# 双后端对比（推荐）
python -m examples.euler.beginner.electives.A09_gpu_backend.run --default-scene --compare

# 实时观看下落过程（节拍推进；与 --compare 互斥——节拍的 sleep 会污染吞吐测量）
python -m examples.euler.beginner.electives.A09_gpu_backend.run --default-scene --realtime
```

## 关键代码
后端切换只有一个词——构造时的 `device` 参数：
```python
env = sim_link.connect_simulation_env(addr)                  # CPU：1-18 课的路径
env = sim_link.connect_simulation_env(addr, device="cuda:0") # GPU：Euler 求解器
```
之后 `env.data.qpos`、`env.do_simulation(ctrl, n)`、`env.render()` 一切照旧——
公共 API 屏蔽了后端差异，这就是"封装隔离"的价值。

**GPU 后端两条新规矩**（与 CPU 不同）：
1. `timestep` / `gravity` 在构造期固化，初始化后 setter 只读——
   16 课"运行中 `sim_config.gravity = ...`"的做法在 GPU 下会抛 `RuntimeError`
2. 数据读写走 host 缓冲 + 惰性同步（H2D/D2H），对用户透明，但频繁单步读写有同步开销

## 动手改
配方区 `DROP_HEIGHT = 3.0` → `5.0`：落体更长，轨迹对比段更充分；
`SIM_DURATION = 2.0` → `4.0`：触地后静置观察更久。

## 解释结果
- **触地时间差 ≤0.05s、触地前轨迹 max|Δz| 毫米级**——两个后端解同一组物理方程，
  数值路径不同（CPU 逐步标量 vs GPU 并行 kernel），结果应在浮点误差内一致
- **小场景 GPU 不一定快，甚至更慢**——单帧 GPU 有 H2D/D2H 同步与 kernel 启动开销，
  一个方块的物理量喂不饱 GPU。GPU 的价值在**规模**：上千个物体并行求解、
  多世界批量仿真（后续课程）、非刚体（布料/流体）——这些才是 GPU 的主场
- **首次 GPU 运行明显偏慢**——kernel 需要编译，之后有缓存（`~/.cache/orcaflow/`）

## 小挑战
先预测再验证（predict）：
1. 笔算：3 米高的方块在默认重力下触地时间 t=√(2Δz/9.81)（Δz = 3 − 静置高度约 0.5）
2. 预测：`--compare` 模式下，你这个场景 CPU 快还是 GPU 快？差几倍？写下预测
3. 运行 `--compare` 对照——加速比和你想的一样吗？为什么小场景 GPU 未必占优？

**验收标准**
- [ ] CPU 模式：触地时间与理论值同量级
- [ ] GPU 模式：连接成功，日志显示 device=cuda:0
- [ ] `--compare`：触地时间差 ≤0.05s，触地前轨迹 max|Δz| 毫米级以内
- [ ] 能口头回答：为什么切换后端不需要改任何用户代码？

## 常见问题
| 现象 | 修复 |
|---|---|
| GPU 连接失败：cuInit 报 CUDA_ERROR_304 | 在 sandbox 内运行导致——用白名单解释器直跑（`<conda-base>/envs/orca/bin/python ...`，勿加管道），见 DEVELOPER_GUIDE.md |
| GPU 连接失败：No flow GPU library | flow 原生库未构建——按 OrcaFlow DEVELOPER_GUIDE 补建（需 CUDA Toolkit + nvcc） |
| RuntimeError: Euler 后端下 timestep/gravity 不可修改 | 预期行为——构造期固化；本地物理参数在构造前通过参数/overrides 下发 |
| 首次 GPU 运行等了十几秒 | kernel 编译，仅首次；之后走 `~/.cache/orcaflow/` 缓存 |
| GPU 模式视口方块不动/表现异常 | 已知架构限制：视口按 Studio 独立物理渲染，本地 GPU 物理不回推视口状态；以终端日志数值为准（与 16/18 课 FAQ 同源） |
| 视口过程"瞬间结束" | 预期——默认全速推进测吞吐，不做实时节拍；想看过程加 `--realtime`（CPU/GPU 均可）或回 13 课 |

## 延伸
选修课独立于基础课主线，可与任意阶段并行学习。
后续 GPU 系列选修（创建时生效）：多世界并行——一个 GPU 同时算 N 个场景；非刚体求解器——布料/流体/沙雪。
