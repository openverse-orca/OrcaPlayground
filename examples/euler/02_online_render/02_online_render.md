# 第 2 课：在线渲染与交互 — 连接 OrcaStudio

> 阶段三（P3A）在线联调课。在第 1 课离线联调基础上，启用 gRPC 在线模式，
> 连接 OrcaStudio 实时渲染。本课聚焦在线模式特有能力：渲染循环、同步/异步渲染、
> RTF 实时同步、Studio UI 交互。

---

## 1. 课程目标

验证在线模式 8 项能力：

| # | 验证点 | API | 期望 |
|---|--------|-----|------|
| 1 | gRPC 连接 | `SimpleEulerEnv(skip_grpc_load=False)` | 连接成功，nq/nv/nu 正确 |
| 2 | render 同步 | `env.render()` | Studio 视口显示摆杆 |
| 3 | sync_render=True | `sync_render=True` | 每物理步渲染（帧率最高） |
| 4 | sync_render=False | `sync_render=False`（默认） | 按 fps 节流（CPU 占用低） |
| 5 | RTF=1.0 | `--rtf 1.0`（默认） | 仿真时间 ≈ 真实时间 |
| 6 | RTF=0 快进 | `--rtf 0` | 不 sleep，rtf=inf |
| 7 | override_ctrls | Studio UI 滑块 | 程序动作被 UI 覆盖 |
| 8 | do_body_manipulation | Studio UI 拖拽 | 物体可被鼠标拖动（占位接口） |

---

## 2. 前置条件

- ✅ conda `orca` 环境可用
- ✅ OrcaStudio 已启动，并加载 pendulum 场景（`simple_pendulum.xml`）
- ✅ Studio 中已点击「运行」启动仿真，gRPC 监听 `localhost:50051`

---

## 3. 目录结构（自包含）

```
examples/euler/02_online_render/
├── 02_online_render.md   ← 本教程
├── online_render.py      ← 脚本入口
└── simple_env.py         ← SimpleEulerEnv（资产路径指向 assets/scenes/）
```

**资产路径**：`simple_env.py` 顶部通过 `__file__` 上溯 4 层定位项目根，
指向 `assets/scenes/simple_pendulum.xml`。

**依赖**：仅 `orca_gym.environment.euler.*` + `orca_gym.log`，零 Local 体系引用。

---

## 4. 运行步骤

### 步骤 1（人工）：启动 OrcaStudio 并加载 pendulum 场景

1. 打开 OrcaStudio
2. 资产搜索 `pendulum` 或导入 `assets/scenes/simple_pendulum.xml`
3. 点击「运行」启动仿真

### 步骤 2（人工）：运行课程脚本

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 默认在线模式，RTF=1.0 实时同步
python examples/euler/02_online_render/online_render.py

# 同步渲染（每个物理步都渲染，帧率最高但可能卡顿）
python examples/euler/02_online_render/online_render.py --sync-render

# 快进模式（不 sleep，仿真尽量快，适合压测）
python examples/euler/02_online_render/online_render.py --rtf 0

# 慢动作（RTF=0.5，仿真比真实时间慢一半，便于观察细节）
python examples/euler/02_online_render/online_render.py --rtf 0.5

# 指定 Studio 地址
python examples/euler/02_online_render/online_render.py --addr 192.168.1.100:50051
```

### 步骤 3（人工）：观察 Studio 视口 + UI 交互

- 视口应显示摆杆实时运动
- 可在 Studio UI 手动拖动执行器滑块（`override_ctrls` 覆盖程序动作）
- 可用鼠标拖拽物体（`do_body_manipulation`）

---

## 5. 预期输出

```
============================================================
第 2 课：在线渲染与交互 — 连接 OrcaStudio
  模式: 在线 gRPC（addr=localhost:50051）
  步数: 50000
  sync_render: False（异步：按 fps 节流）
  RTF: 1.0（按真实时间同步）
============================================================
[1/4] gRPC 连接成功: nq=1, nv=1, nu=1
[2/4] reset 成功: obs.shape=(3,)
      → 此时 Studio 视口应显示摆杆初始状态（竖直向上）
[3/4] 开始步进渲染循环（可在 Studio UI 手动控制执行器 / 拖拽物体）
  step 100/50000: reward=-0.2442, time=1.0000, rtf=1.002
  step 200/50000: reward=-0.1833, time=2.0000, rtf=1.001
  ...
[3/4] 步进完成: 总奖励=-1234.5678
[4/4] 环境关闭，gRPC 连接断开
============================================================
第 2 课验证通过
============================================================
```

**通过条件**：
- ✅ `[1/4]` gRPC 连接成功
- ✅ 视口显示摆杆实时运动
- ✅ RTF 实测接近设定值（1.0 模式下 rtf≈1.0，快进模式下 rtf=inf）
- ✅ 退出码 0

---

## 6. 本课概念

| 概念 | 说明 |
|------|------|
| 在线模式 | `skip_grpc_load=False`，通过 gRPC 连接 OrcaStudio，同步渲染 |
| `render_mode="human"` | 渲染到 Studio 视口（`"none"` 不渲染） |
| `sync_render` | `True`：每个物理步都渲染（帧率最高）；`False`：按 fps 节流（默认，CPU 占用低） |
| `override_ctrls` | Studio UI 返回的手动控制值，在 `set_ctrl` 中覆盖程序动作 |
| `do_body_manipulation` | Studio UI 拖拽物体的占位接口 |
| RTF | 实时因子，1.0=实时，0=快进不 sleep，0.5=慢动作 |

### 代码解析

```python
env = SimpleEulerEnv(
    orcagym_addr=args.addr,
    skip_grpc_load=False,       # 在线模式：连接 OrcaStudio
    render_mode="human",        # 渲染到 Studio 视口
    sync_render=args.sync_render,
)
```

```python
for step in range(args.steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()   # 将物理状态同步到 Studio 视口
    # RTF 同步：让仿真时间 ≈ 真实时间
    if rtf_mode:
        expected_wall = (step + 1) * step_dt / args.rtf
        elapsed = time.perf_counter() - wall_start
        if elapsed < expected_wall:
            time.sleep(expected_wall - elapsed)
```

- `env.render()` 在在线模式下通过 gRPC 把当前物理状态推送到 Studio。
- `sync_render=True` 时 `step()` 内部已渲染，`render()` 立即返回。
- `sync_render=False` 时 `render()` 按 fps 节流，可能跳过部分帧。

---

## 7. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `localhost:50051` | OrcaStudio gRPC 地址 |
| `--steps` | `50000` | 仿真步数 |
| `--time-step` | `0.002` | 物理时间步长（秒） |
| `--frame-skip` | `5` | 每个动作执行的物理步数 |
| `--sync-render` | `False` | 同步渲染（每个物理步都渲染） |
| `--rtf` | `1.0` | 实时因子（1.0=实时，0 或负值=快进不 sleep） |

---

## 8. 故障排查

### Q1：gRPC 连接失败 / 超时

**原因**：OrcaStudio 未启动或地址不对。

**解决**：
1. 确认 OrcaStudio 已启动并监听 `localhost:50051`
2. 确认场景中已加载 pendulum 模型
3. 确认 `--addr` 参数正确

### Q2：`ModuleNotFoundError: No module named 'simple_env'`

**原因**：未激活 `orca` 环境，或脚本目录不在 `sys.path`。

**解决**：

```bash
cd /path/to/OrcaPlayground
conda activate orca
python examples/euler/02_online_render/online_render.py
```

> 本目录自包含，运行时 Python 自动将脚本所在目录加入 `sys.path[0]`，
> 同目录 `from simple_env import ...` 可直接生效，无需额外 `PYTHONPATH`。

### Q3：视口无画面 / 摆杆不动

**原因**：Studio 未点击「运行」，或 `render_mode` 不是 `human`。

**解决**：
1. 确认 Studio 已点击「运行」启动仿真
2. 确认脚本未传 `--render-mode none`（本课默认 `human`）
