# D12 双臂机器人示例

本示例包含 D12 双臂机器人的脚本轨迹演示和 ACT 策略推理两种模式。

## 目录结构

```
examples/d12/
├── demo/       # 脚本轨迹演示（YAML 轨迹 → OSC 控制器 → 仿真）
├── act/        # ACT 策略推理与训练（详见 act/README.md）
└── README.md   # 本文件
```

## 快速开始

### 脚本轨迹回放

```bash
python examples/d12/demo/run_d12_demo.py --mode mp
```

### ACT 闭环推理

```bash
python examples/d12/act/run_d12_act.py \
    --checkpoint envs/d12/checkpoints/act_det_pkg_scan/best_model.pt \
    --max_steps 6300 --frame_skip 5 --exec_mode chunk \
    --ref_trajectory examples/d12/act/ref_trajectory/ref_demo.hdf5 \
    --no_sleep
```
