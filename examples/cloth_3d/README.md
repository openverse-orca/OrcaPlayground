# cloth_3d — MjcPBD phase1（推块）

MuJoCo 刚体 → OrcaLink → XPBD 耦合的 Python 侧工程目录（与 `examples/fluid` 隔离）。

## S0 交付（当前）

| 路径 | 说明 |
|------|------|
| `cloth_sim_config.phase1_slide.json` | 主配置（端口、body 映射、XPBD 参数） |
| `assets/phase1_slide/scene.xml` | MuJoCo 场景（5 body） |
| `modules/phase1_trajectory.py` | 夹爪轨迹 `compute_ctrl` |
| `modules/sim_frames.py` | 宏帧计数（Mjc 20 子步 / XPBD 40 子步 → +1，0.02s） |
| `scripts/run_phase1_viewer.py` | 本地 viewer + 轨迹 |
| `scripts/validate_s0.py` | S0 配置与 MJCF 一致性检查 |
| `explain.md` | S0 讲解 |

## 环境

需要 **MuJoCo Python 包**（本仓库常用 conda 环境 **`orcapc`**，已含 `mujoco` 3.x）。

```bash
conda activate orcapc
pip install -r requirements.txt   # 若 import mujoco 失败再装
which python                      # 应为 .../envs/orcapc/bin/python
python -c "import mujoco; print(mujoco.__version__)"
```

**注意**：部分机器上提示符已是 `(orcapc)`，但 `python3` 仍指向 `/usr/bin/python3`（系统 Python **无** mujoco）。请用环境内的 **`python`**（无 `3`），或：

```bash
./scripts/run_phase1_viewer.sh --realtime
```

## 快速验证

```bash
cd /home/gvorca/OrcaSys/OrcaApr24/OrcaPlayground/examples/cloth_3d

# S0：配置 + MJCF 对齐
python scripts/validate_s0.py

# 宏帧计数（无 GUI）
python scripts/verify_macro_frames.py

# 仅 MuJoCo 场景（含正四面体锚点 SITE + 外接球 + 棱边）
python -m mujoco.viewer --mjcf=assets/phase1_slide/scene.xml
# Viewer 中勾选 Rendering → Site；半透明绿球为外接球，彩点为 4 锚点

# 场景 + 推块轨迹
python scripts/run_phase1_viewer.py --realtime
# 或: ./scripts/run_phase1_viewer.sh --realtime

# 打印 MuJoCo macro_frame（每 20 个 mj_step）
python scripts/run_phase1_viewer.py --realtime --print-frames
```

### 宏帧（0.02 s）

| 侧 | 子步 | 条件 | `macro_frame` |
|----|------|------|----------------|
| MuJoCo | `mj_step`，0.001 s | 满 **20** 次 | +1 → 发 OrcaLink |
| XPBD | `phys_world_step`，0.0005 s | 满 **40** 次 | +1 → 与上一宏步对齐 |

实现：`modules/sim_frames.py`（S1 的 `run_cloth_simulation.py` / `phase1_slide_mjc` 应复用同一逻辑）。




run_cloth_simulation.py 宏步循环
    └── ClothOrcaLinkBridge.publish_anchor_macro_frame()   ← ③ 宏步边界入口
            │
            ├── ① collect_anchor_frame()     modules/anchor_frame.py
            │      · mj_forward
            │      · site_xpos, jacp@qvel（锚点 pos/vel）
            │      · xquat, cvel[:3]（刚体 quat/ω）
            │
            ├── ② frame_to_units()           modules/anchor_publish.py
            │      · 组装与 PublishFrame 相同的 DataUnit
            │
            ├── AnchorDebugCsvWriter.write_macro_frame()  modules/anchor_debug_export.py
            │      · 写上述两个 CSV
            │
            └── publish_anchor_frame()       orcalink_client（gRPC 发出）



## S1 Gate-0：MuJoCo → OrcaLink 锚点发布

### 构建与依赖

```bash
# OrcaLink Server（含 ORCALINK_DEBUG_ANCHOR 日志）
cd OrcaLink/build && cmake .. && cmake --build . -j
# 可执行文件: OrcaLink/bin/orcalink

# Python 客户端（orcapc 环境）
pip install -e OrcaLink
```

### 运行（每 macro_frame 打印发送/接收）

终端 1 — Server：

```bash
export ORCALINK_DEBUG_ANCHOR=1
# 端口仅来自 cloth_sim_config.phase1_slide.json → orcalink.port
PORT=$(python -c "import json; print(json.load(open('cloth_sim_config.phase1_slide.json'))['orcalink']['port'])")
OrcaLink/bin/orcalink --port "$PORT"
```

终端 2 — MuJoCo 发送端：

```bash
cd OrcaPlayground/examples/cloth_3d
export ORCALINK_DEBUG_ANCHOR=1 CLOTH_DEBUG_ANCHOR=1
python launch/run_cloth_simulation.py --max-seconds 0.5
```

- Python 端：`[MUJOCO SEND] macro_frame=…`（每刚体 4 锚点 pos/vel + quat/ω）
- Server 端：`[OrcaLink RECV_PUBLISH] macro_frame=… units=72`（6 刚体 × 12 unit/宏步，含 body_p/body_v）

配置里 `debug.publish_only=true` 时 `expected_clients=1`，无需 XPBD 即可联调。  
`anchor_discovery.auto_from_model=true` 时扫描 MJCF 中带 `_anchor_` 的 SITE，与 `rigid_body_map` 合并；**增删刚体请改 scene.xml 的 body + 四面体 SITE**，无需改 Python 硬编码。

规格：`XPBD_orcalink/MjcPBD_orcalink/anchor_transport_module.md`

### CSV 手动校验（推荐）

每次运行 `launch/run_cloth_simulation.py`（或 `verify_anchor_orcalink.py`）会在配置目录写入：

`XPBD_orcalink/MjcPBD_orcalink/debug_log/`

| 文件 | 内容 |
|------|------|
| `mujoco_orcalink_units.csv` | **线上 OrcaLink DataUnit**（与 Server `[unit] id=...` 逐行对照） |
| `mujoco_anchor_samples.csv` | **MuJoCo 原始采样**（site 位置/速度 + body 四元数/角速度，Z-up） |
| `run_meta.txt` | 本次运行 UTC 时间戳 |

配置项：`debug.export_csv`、`debug.debug_log_dir`（见 `cloth_sim_config.phase1_slide.json`）。

**数据采集代码位置：**

1. `modules/anchor_frame.py` → `collect_anchor_frame()` — MuJoCo 采样  
2. `modules/anchor_publish.py` → `frame_to_units()` — 组包  
3. `modules/cloth_orcalink_bridge.py` → `publish_anchor_macro_frame()` — 宏步边界调用  
4. `modules/anchor_debug_export.py` → `AnchorDebugCsvWriter.write_macro_frame()` — 写 CSV  

### 自动校验（发送 vs 接收逐 unit 对比）

```bash
python scripts/verify_anchor_orcalink.py --macro-frames 30
```

脚本会：启动 Server → 发布端跑 MuJoCo（每宏步 `publish_anchor_frame`）→ 订阅端 `subscribe_anchor_frames` 拉取 DataFrame，按 `object_id` 对比 pos/quat/vel/omega（容差 `1e-4`）。全部 `PASS` 即数据正确。

端到端通讯流程（三角色、JoinSession、宏步对齐）：见 [`XPBD_orcalink/MjcPBD_orcalink/OrcaLink_connect_flow.md`](../../XPBD_orcalink/MjcPBD_orcalink/OrcaLink_connect_flow.md)。

### XPBD 阶段 A/B/D（无图形，先于双窗 demo）

```bash
cd XPBD_orcalink && ./build.sh mjc_pbd_bridge_recv
cd .. && python XPBD_orcalink/MjcPBD_orcalink/checkpoints/scripts/verify_xpbd_orcalink_abd.py --macro-frames 10
```

- 订阅端：`build/mjc_pbd_bridge_recv`（无 X11，仅 JoinSession + SubscribeFrame + `xpbd_orcalink_raw_zup.csv`）
- 图形 demo `phase1_slide_mjc` 与 MuJoCo/XPBD 双窗同步在 A/B/D 通过后再做

## 后续（S1+）

XPBD 订阅端：`CoordinateTransform`、`AnchorFrameDecoder`、`MjcPbdOrcaLinkBridge`（见 `MjcPBD_implement.md`）。
