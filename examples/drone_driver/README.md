# Drone Driver 使用说明

本示例当前提供的是一版「**推力驱动的刚体飞行**」原型：

- 根节点使用 **自由关节** `drone_free`，由 MuJoCo 积分刚体动力学
- 默认键盘指令映射为作用在 **drone_frame 刚体** 上的 **集体推力 + 姿态/偏航控制**（`xfrc_applied`），表现为一版可调的四旋翼 full 模式
- 附带 `drone_aero_config` 中的 **线性/二次阻尼** 与简易 **地面效应**（按高度放大局部升力），便于稳定与近地表现
- 四个旋翼关节仍主要承担 **转速动画**（每步写关节相位），**模型中不再为桨挂 position 执行器**，避免与脚本驱动冲突导致桨自由度上出现数千量级的 `qacc`
- 运行链路仍然走 `OrcaStudio` / gRPC

其中“**集体推力**”指四旋翼总升力的集总控制量，主要由 `R/F` 调节；当前默认 full 模式**不包含位置闭环**，也不是“给定目标高度/目标位置后自动跟踪”的位控。

**轮地接触驱动**：当前资产为四旋翼，未包含车轮与轮速执行器；若要做真正的轮式推进，需要在 XML 中增加轮体、接触几何与 `velocity`/`motor` 等执行器，并在环境里用 `ctrl` 驱动轮子而非本节的推力逻辑。

## 文件组成

- `run_drone_orca.py`：入口脚本，负责扫描场景、注册环境并驱动仿真循环
- `model/Drone_ver_1.0/drone-v1.xml`：无人机 MuJoCo 模型（根为 `drone_free`）
- `envs/drone/drone_orca_env.py`：推力物理环境，读取 `OrcaStudio` 键盘并施加外力/力矩
- `envs/drone/drone_aero_config.py`：气动阻尼与地面效应参数（可被环境读取）

## 运行前准备

- 先将 **更新后的** `drone-v1.xml` 导入 `OrcaStudio`（若仍使用旧的 `Tx…Rz` 根关节，场景扫描会失败）
- 将对应 actor 手动拖入当前场景
- 场景中需要且只能有 1 台完整匹配的无人机实例
- 实例名不需要固定，脚本会根据关节、执行器、body、site 后缀自动绑定
- 键盘输入来自 `OrcaStudio`，不依赖终端焦点

手动拖入资产的通用流程可参考项目根目录 `README.md`。

## 运行

默认运行（推荐，默认已是 full 四旋翼模式）：

```bash
python examples/drone_driver/run_drone_orca.py
```

显式切回竖直 Z-only 调试模式：

```bash
python examples/drone_driver/run_drone_orca.py --vertical-z-only
```

启用自动巡航：

```bash
python examples/drone_driver/run_drone_orca.py \
  --orcagym_addr localhost:50051 \
  --time_step 0.008333333 \
  --frame_skip 1 \
  --autoplay
```

参数说明：

- `--orcagym_addr`：OrcaGym 服务地址，默认 `localhost:50051`
- `--time_step`：物理步长，默认约 `1/120s`
- `--frame_skip`：每次控制对应的仿真步数
- `--autoplay`：启用循环自动输入，适合在 gRPC 常驻时反复调试
- `--vertical-z-only`：切回竖直模式，只保留世界 `+Z` 推力与 `vz` 阻尼
- `--vertical-thrust-ramp` / `--vertical-fixed-tmg` / `--vertical-takeoff-bisect`：竖直模式下的起飞标定辅助参数

## 键盘控制

| 按键 | 功能 |
|------|------|
| `w/s` | 前后平移控制，机身会伴随轻微前后倾 |
| `a/d` | 左右平移控制，机身会伴随轻微左右倾 |
| `r/f` | 集体推力增减（相对悬停推力） |
| `q/e` | 偏航控制 |
| `space` | 重置到初始姿态 |

启用 `--autoplay` 后，脚本会持续生成扰动，`space` 仍可用于重置。

若使用 `--vertical-z-only`：

- `r/f`：控制世界 `+Z` 推力
- `w/s`、`a/d`：可选世界系水平力探索（取决于 `--vertical-xy-force-factor`，`--vertical-pure-z` 可彻底关闭）
- `q/e`：仅动画，不参与 full 四旋翼姿态链

## 当前模型说明

- 根关节：`drone_free`（7 维位姿 + 6 维速度）
- 旋翼：`FL/FR/BL/BR_joint`（环境中直接写相位动画；当前模型不再依赖桨 position 执行器）
- 观测中的线速度/角速度来自 MuJoCo 刚体 `cvel`（世界系线速度 + 角速度）

## 实现概览

1. 入口脚本扫描场景，确认存在唯一一台完整匹配的无人机
2. 注册 `DroneOrcaEnv`，主循环 `step()` / `render()`
3. 每个物理子步：`mj_forward` → 按键盘指令写 `drone_frame` 的 `xfrc_applied`
   - 默认 full 模式：集体推力沿 `drone_frame +Z`，通过小角度倾转产生水平分力，并叠加阻尼/偏航稳定
   - `vertical_z_only`：改为世界 `+Z` 推力与 `vz` 阻尼，可选世界系水平力
4. 更新旋翼关节动画 → `ctrl=0` → `mj_step`

## 当前限制

- 推力为 **集总标量 + 简化姿态/偏航控制**，不是四电机螺旋桨气动力逐个求解
- 旋翼动画与动力学推力 **未刚性耦合**（便于调试与展示）
- 轮地差速驱动 **未实现**，需另建轮式模型与控制链

若需更接近真机的分配律与姿态内环，可在本版稳定后再接电机模型与控制器。
