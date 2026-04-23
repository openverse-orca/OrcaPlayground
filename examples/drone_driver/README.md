# Drone Driver 使用说明

## ⚠️ 资产准备

- **资产**：请订阅 **Drone_ver或skydio_x2_nofloor对应布局建议Industry_DataCenter_Plugging_01或者空布局（OrcaLab_day）**。
- **是否需要手动拖动到布局中**：**是**，需要加载建议布局，将运行前需要先把Drone_ver或skydio_x2_nofloor摆进场景。
- 运行前可以切换camera视角来获得更优体验


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
- `model/skydio_x2_nofloor/x2.xml`：按当前框架重构后的 `Skydio X2` 示例模型
- `envs/drone/drone_orca_env.py`：推力物理环境，读取 `OrcaStudio` 键盘并施加外力/力矩
- `envs/drone/drone_aero_config.py`：气动阻尼、full 模式控制参数与机型 profile

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

切到 `x2` 机型：

```bash
python examples/drone_driver/run_drone_orca.py --drone-model x2
```

如需恢复定位乱飘时的 periodic 长日志：

```bash
python examples/drone_driver/run_drone_orca.py --drone-model x2 --diag-every-env-steps 20
```

默认启动已内置一组更稳的 reset 隔离参数：

- `reset_height_offset=0.25`
- `fullmode_reset_thrust_ramp=0.8`
- `fullmode_reset_thrust_start_factor=0.2`
- `fullmode_reset_minimal_stab=0.35`

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

若你想显式覆盖默认值，仍可手动传参；例如只改 reset 抬高量：

```bash
python examples/drone_driver/run_drone_orca.py --reset-height-offset 0.25
```

参数说明：

- `--orcagym_addr`：OrcaGym 服务地址，默认 `localhost:50051`
- `--time_step`：物理步长，默认约 `1/120s`
- `--frame_skip`：每次控制对应的仿真步数
- `--autoplay`：启用循环自动输入，适合在 gRPC 常驻时反复调试
- `--drone-model`：选择无人机 profile，默认 `Drone_ver_1.0`，当前也支持 `x2`
- `--diag-logs` / `--no-diag-logs`：开关环境侧诊断 WARNING
- `--diag-every-env-steps`：每 N 个 env step 输出一次 periodic 长日志；`0` 为关闭
- `--reset-height-offset`：reset 时给初始 `z` 额外抬高若干米，用于排查出生点接触/穿插
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

## `x2` 适配中遇到的主要问题

这次 `Skydio X2` 的适配，核心不是“把旋翼名字映射一下”就结束，而是同时踩到了结构、坐标系和控制三类问题：

- **模型根体与真实质心不一致**：最开始导入的 `x2` 结构里，`free joint` 对应的根体并不在整机近似质心附近，日志中 `frame_com_offset` 一度达到米级，导致 `W` 触发轻微倾转时，会因为大杠杆臂放大出不合理的平移/翻转。
- **机体系前向/右向和控制器假设不一致**：当前 full 模式默认假设“机体前向”和“右向”与默认 `drone-v1` 一致；而 `x2` 的网格朝向不同，直接接入会出现 `W/S` 对应到错误倾转轴的问题。
- **原始资产的执行器/动力链不适配当前框架**：现有框架里推力是通过 `xfrc_applied` 直接作用在刚体上，旋翼关节主要做动画；如果保留原资产里那套 actuator / transmission，会和当前控制链冲突。
- **旋翼子体是运动体，不能完全零质量**：为了让桨还能转动做动画，`rotor_*` body 仍带 hinge joint；MuJoCo 要求这种运动体至少有最小质量和惯量，否则会报 `mass and inertia of moving bodies must be larger than mjMINVAL`。

## 最后采用的解决思路

最终没有继续在旧 `x2` 结构上局部修补，而是把它改造成更接近 `drone-v1` 的框架友好结构：

- 保留一层根 body `drone_frame`，并把 `drone_free` 放在这层，作为环境施加 `xfrc_applied` 的主刚体。
- 在子 body `Drone` 上集中整机主要质量与惯量，使主刚体更接近“单刚体飞行器”假设。
- 4 个旋翼 body 只保留为“可转动的可视件”，只给最小惯量，不再承载真实质量分布。
- 把 mesh、碰撞盒、推力 site、IMU site 一起平移到新的近似质心坐标系，减小 `frame_com_offset`。
- 在 `drone_aero_config.py` 中为 `x2` 增加单独的 `DroneModelProfile`，将阻尼、最大倾角、姿态增益、reset 参数等和默认机型解耦。
- 增加 `planar_forward_axis_body` / `planar_right_axis_body`，把“机体朝向差异”从 XML 硬编码问题，变成 profile 可配置问题。

换句话说，当前方案的重点是：

- **XML 负责把资产整理成统一的动力学骨架**
- **profile 负责表达机型差异**
- **环境控制器尽量保持一套，不为单个资产写分叉逻辑**

## 当前代码架构如何适配一个新无人机

建议把“适配新机型”分成 3 层来看：

1. **模型结构层（XML）**
   目标是让资产满足当前环境的最小约定：
   - 有根 body `drone_frame`
   - 根上有自由关节 `drone_free`
   - 有主机体 body `Drone`
   - 有 `FL_joint` / `FR_joint` / `BL_joint` / `BR_joint`
   - 有 `rotor_fl_site` / `rotor_fr_site` / `rotor_bl_site` / `rotor_br_site`
   - 有 `imu` 与 `drone_body_center_site`

2. **动力学近似层（质量/碰撞/朝向）**
   这是最容易让新机型“看起来能加载，但一飞就乱”的部分：
   - 先保证 `drone_free` 尽量靠近整机近似质心
   - 主质量尽量集中在 `Drone` 主体，不要把大量质量挂在旋翼子体上
   - 旋翼子体若仅用于动画，只保留极小惯量
   - 若网格朝向与默认机型不同，优先通过 profile 中的 `planar_forward_axis_body` / `planar_right_axis_body` 修正，而不是把控制逻辑写死到环境里

3. **机型参数层（profile）**
   在 `envs/drone/drone_aero_config.py` 里新增一个 `DroneModelProfile`：
   - `drag`：线性/二次阻尼、角阻尼、零杆量保持参数
   - `ground_effect`：旋翼半径、地效高度和增益
   - `full_mode`：最大倾角、姿态增益、力矩限制、reset 抬高量、桨速动画参数
   - 必要时配置 `planar_forward_axis_body` / `planar_right_axis_body`

## 新无人机接入的推荐流程

推荐顺序如下：

1. 先把 XML 整理成与 `drone-v1` 同类的结构，而不是一开始就尝试兼容原资产的全部层级和 actuator。
2. 先跑 `--vertical-z-only`，确认悬停推力、起飞临界高度、接触稳定性大体正常。
3. 再切 full 模式，只测 `R/F`，确认纯升降没有异常姿态耦合。
4. 然后单独测 `W/S`、`A/D`，如果出现“按前进却在错误轴上倾转”，优先检查机体系朝向配置。
5. 若 full 模式出现“轻微倾转后直接飞走”，第一优先检查 `frame_com_offset`、根体位置和质量分布，而不是先猛调控制增益。
6. 最后再用 `--diag-every-env-steps` 打开 periodic 日志，针对具体机型微调 profile。

## 判断一个新机型是否已经基本适配完成

至少应满足下面这些现象：

- reset 后静止时，`frame_com_offset` 不再出现明显离谱的大偏移
- 纯 `R/F` 升降时，不会无输入自发滚转/俯仰
- `W/S`、`A/D` 会对应到预期的倾转方向
- 松杆后姿态能回收，而不是继续沿同一方向累积
- 旋翼动画存在，但不会对主动力学产生明显副作用

## 实现概览

1. 入口脚本扫描场景，确认存在唯一一台完整匹配的无人机
2. 注册 `DroneOrcaEnv`，主循环 `step()` / `render()`
3. 每个物理子步：`mj_forward` → 按键盘指令写 `drone_frame` 的 `xfrc_applied`
   - 默认 full 模式：集体推力沿 `drone_frame +Z`，通过小角度倾转产生水平分力，并叠加阻尼/偏航稳定
   - `vertical_z_only`：改为世界 `+Z` 推力与 `vz` 阻尼，可选世界系水平力
4. 更新旋翼关节动画 → `ctrl=0` → `mj_step`

调试提示：

- 环境启动时会额外打印 `scene DOF` 观察点 `v[8]` 的 `joint/body/actor` 归属，便于和 `scene_max|qacc|=...@8(...)` 一类日志直接对照
- 若复杂场景一 reset 就炸，可先加 `--reset-height-offset 0.2~0.4` 验证是否为出生点接触问题

## 当前限制

- 推力为 **集总标量 + 简化姿态/偏航控制**，不是四电机螺旋桨气动力逐个求解
- 旋翼动画与动力学推力 **未刚性耦合**（便于调试与展示）
- 轮地差速驱动 **未实现**，需另建轮式模型与控制链

若需更接近真机的分配律与姿态内环，可在本版稳定后再接电机模型与控制器。
