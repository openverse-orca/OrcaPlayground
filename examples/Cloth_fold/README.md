# Cloth_fold — XPBD/PBD 布料粒子显示（与 fluid 隔离）

本目录为 **布料折叠 / PBD 粒子 gRPC 回放** 专用，与 `examples/fluid`（SPH/OrcaGym 流体管线）**无代码依赖、勿混用**。

| 项 | 路径 |
|----|------|
| 回放客户端 | `XPBD/Orca/ParticleRender/` |
| Studio 关卡 | `OrcaStudio_2409/Levels/cloth_demo/` |
| 离线数据 | `/home/hjadmin/PBDX/data_particle30Hz/` |

## 快速联调

```bash
# 终端 1：Editor + cloth_demo + Play + 等待 50251
/home/hjadmin/OrcaApr24/OrcaPlayground/examples/Cloth_fold/start_cloth_demo.sh

# 终端 2：30 Hz 回放
/home/hjadmin/OrcaApr24/OrcaPlayground/examples/Cloth_fold/run_pbd_particle_replay.sh
```

停止：`start_cloth_demo.sh --stop`

## 文件

| 文件 | 作用 |
|------|------|
| `auto_start_cloth_demo.py` | OrcaEditor `--runpython`：开关卡、进 Play |
| `start_cloth_demo.sh` | 启动 Editor 并等待 gRPC |
| `run_pbd_particle_replay.sh` | 调用 `pbd_particle_replay` |

## 与 fluid 的区别

- **不**使用 `run_fluid_sim.py`、`sph_sim_config.json`、`auto_start_scene.py`
- **不**占用 OrcaGym 50051；PBD 仅用 ParticleRender **50251**
- 日志前缀：`[CLOTH_FOLD]`（fluid 为 `[AUTO_START]` / `[ONE_CLICK]`）

兼容入口：`scripts/start_pbd_cloth_demo.sh` 会转调本目录的 `start_cloth_demo.sh`。
