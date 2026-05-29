# cloth_3d — MjcPBD phase1（推块）

MuJoCo 刚体 → OrcaLink → XPBD 耦合的 Python 侧工程目录（与 `examples/fluid` 隔离）。

## S0 交付（当前）

| 路径 | 说明 |
|------|------|
| `cloth_sim_config.phase1_slide.json` | 主配置（端口、body 映射、XPBD 参数） |
| `assets/phase1_slide/scene.xml` | MuJoCo 场景（5 body） |
| `modules/phase1_trajectory.py` | 夹爪轨迹 `compute_ctrl` |
| `scripts/run_phase1_viewer.py` | 本地 viewer + 轨迹 |
| `scripts/validate_s0.py` | S0 配置与 MJCF 一致性检查 |
| `explain.md` | S0 讲解 |

## 快速验证

```bash
cd /home/hjadmin/OrcaApr24/OrcaPlayground/examples/cloth_3d

# S0：配置 + MJCF 对齐
python3 scripts/validate_s0.py

# 仅 MuJoCo 场景
python3 -m mujoco.viewer --mjcf=assets/phase1_slide/scene.xml

# 场景 + 推块轨迹
python3 scripts/run_phase1_viewer.py --realtime
```

## 后续（S1+）

见 `XPBD/MjcPBD_orcalink/MjcPBD_implement.md`：`body_map.py`、`cloth_orcalink_bridge.py`、`launch/run_cloth_simulation.py`、XPBD `phase1_slide_mjc`。
