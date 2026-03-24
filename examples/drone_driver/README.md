# Drone Driver 使用说明

本示例当前提供的是一版「**推力驱动的刚体飞行**」原型：

- 根节点使用 **自由关节** `drone_free`，由 MuJoCo 积分刚体动力学
- 键盘指令映射为作用在 **Drone 刚体** 上的 **机体系合力 + 合力矩**（`xfrc_applied`），近似四旋翼总推力与滚转/俯仰/偏航力矩
- 附带 `drone_aero_config` 中的 **线性/二次阻尼** 与简易 **地面效应**（按高度放大局部升力），便于稳定与近地表现
- 四个旋翼关节仍主要承担 **转速动画**（每步写关节相位），**模型中不再为桨挂 position 执行器**，避免与脚本驱动冲突导致桨自由度上出现数千量级的 `qacc`
- 运行链路仍然走 `OrcaStudio` / gRPC

**轮地接触驱动**：当前资产为四旋翼，未包含车轮与轮速执行器；若要做真正的轮式推进，需要在 XML 中增加轮体、接触几何与 `velocity`/`motor` 等执行器，并在环境里用 `ctrl` 驱动轮子而非本节的推力逻辑。

## 文件组成

- `run_drone_orca.py`：入口脚本，负责扫描场景、注册环境并驱动仿真循环
- `model/Drone_ver_1.0/drone-v1.xml`：无人机 MuJoCo 模型（根为 `drone_free`）
- `envs/drone/drone_orca_env.py`：推力物理环境，读取 `OrcaStudio` 键盘并施加外力螺旋
- `envs/drone/drone_aero_config.py`：气动阻尼与地面效应参数（可被环境读取）

## 运行前准备

- 先将 **更新后的** `drone-v1.xml` 导入 `OrcaStudio`（若仍使用旧的 `Tx…Rz` 根关节，场景扫描会失败）
- 将对应 actor 手动拖入当前场景
- 场景中需要且只能有 1 台完整匹配的无人机实例
- 实例名不需要固定，脚本会根据关节、执行器、body、site 后缀自动绑定
- 键盘输入来自 `OrcaStudio`，不依赖终端焦点

手动拖入资产的通用流程可参考项目根目录 `README.md`。

## 运行

默认运行：

```bash
python examples/drone_driver/run_drone_orca.py
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

## 键盘控制

| 按键 | 功能 |
|------|------|
| `w/s` | 俯仰力矩（前后倾，带动加速） |
| `a/d` | 滚转力矩（左右倾） |
| `r/f` | 总推力增减（相对悬停推力） |
| `q/e` | 偏航力矩 |
| `space` | 重置到初始姿态 |

启用 `--autoplay` 后，脚本会持续生成扰动，`space` 仍可用于重置。

## 当前模型说明

- 根关节：`drone_free`（7 维位姿 + 6 维速度）
- 旋翼：`FL/FR/BL/BR_joint` + 对应 position 执行器（环境中 `ctrl` 置零，推力不走执行器）
- 观测中的线速度/角速度来自 MuJoCo 刚体 `cvel`（世界系线速度 + 角速度）

## 实现概览

1. 入口脚本扫描场景，确认存在唯一一台完整匹配的无人机
2. 注册 `DroneOrcaEnv`，主循环 `step()` / `render()`
3. 每个物理子步：`mj_forward` → 按键盘指令写 `Drone` 的 `xfrc_applied`（推力沿机体系 +Z，阻尼在机体系叠加）→ 更新旋翼关节动画 → `ctrl=0` → `mj_step`

## 当前限制

- 推力为 **集总标量 + 简化力矩映射**，不是四电机螺旋桨气动力逐个求解
- 旋翼动画与动力学推力 **未刚性耦合**（便于调试与展示）
- 轮地差速驱动 **未实现**，需另建轮式模型与控制链

若需更接近真机的分配律与姿态内环，可在本版稳定后再接电机模型与控制器。
