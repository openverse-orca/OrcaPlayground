# phase1 双窗图形联调

观察 XPBD 是否随 MuJoCo 宏步更新刚体线框（锚点 J2c 子步跟踪 + **同框对比**）。

XPBD 窗口配色：**青色**=当前 XPBD 盒体，**灰色**=MuJoCo 快照（`MJC_PBD_OVERLAY_MJC=1`），**黑球**=SITE 锚点。

## 前提

```bash
conda activate orca-apr24   # 或 orcapc
cd ~/OrcaApr24/XPBD && ./build.sh phase1_slide_mjc
export PYTHONPATH=~/OrcaApr24/OrcaLink/Client/Python
```

配置使用 **`cloth_sim_config.phase1_slide.dual_graphic.json`**（`control_mode: async`，避免 sync 卡死）。

## 一键启动（60 宏步，约 3s 墙钟）

```bash
cd ~/OrcaApr24
export PYTHONPATH=~/OrcaApr24/OrcaLink/Client/Python
bash XPBD/MjcPBD_orcalink/checkpoints/scripts/start_phase1_dual_60.sh 60 0.05
```

## 三终端（推荐顺序）

### 终端 1 — OrcaLink Server

```bash
cd ~/OrcaApr24
OrcaLink/bin/orcalink --port 50361
```

### 终端 2 — XPBD 窗口（先启动，JoinSession）

```bash
cd ~/OrcaApr24/XPBD/build
export MJC_PBD_CONFIG=~/OrcaApr24/OrcaPlayground/examples/embodied/cloth/cloth_sim_config.phase1_slide.dual_graphic.json
export MJC_PBD_OVERLAY_MJC=1
export MJC_PBD_DEBUG_ORCALINK=1
./phase1_slide_mjc
```

应看到：`scene loaded bodies=5`、`CONNECT_OK`、窗口内青/灰双色盒体线框与黑球锚点。

### 终端 3 — MuJoCo viewer + 发包

```bash
cd ~/OrcaApr24/OrcaPlayground/examples/embodied/cloth
export PYTHONPATH=~/OrcaApr24/OrcaLink/Client/Python
python scripts/run_phase1_viewer_orcalink.py \
  --config cloth_sim_config.phase1_slide.dual_graphic.json \
  --realtime --macro-frames 60 --macro-delay 0.05
```

## 预期

| 窗口 | 内容 |
|------|------|
| MuJoCo | Z-up 场景：底座、立方体、夹爪推块 |
| XPBD | Y-up：5 个盒体线框随宏步与 MuJoCo **大致同步**运动 |

终端 2 应周期性打印 `[MjcPbdOrcaLinkBridge] RECV macro_frame=...`。

## 若 XPBD 不动

1. 确认终端 2 先于终端 3 启动且 Session ready（2/2 clients）
2. 不要用生产配置里的 `control_mode: sync` 做首次双窗
3. 检查端口 50361 未被旧进程占用：`pkill -x orcalink; pkill -f phase1_slide_mjc`

## 初态盒体 8 顶点（MuJoCo vs XPBD，无图形）

```bash
cd ~/OrcaApr24
conda activate orca-apr24
python XPBD/MjcPBD_orcalink/checkpoints/scripts/verify_initial_box_vertices_mjc_xpbd.py --write-report
```

通过表示 `xpbd_scene_from_mjcf.json` 与 MuJoCo `mj_reset` 后碰撞盒 8 角点（Y-up）一致。

## 仅 MuJoCo 目视（无 OrcaLink）

```bash
python scripts/run_phase1_viewer.py --realtime
```
