# 操作 → 录制 → 回放

本文说明 `run_fluid_sim.py` 的 **`--mode`**（`live` / `record` / `playback`）、人类操作轨迹、粒子 HDF5 及相关参数。命令均在 **`examples/fluid`** 目录下执行（已 `cd` 到该目录）。

## 总览

| 阶段 | `--mode` | 作用 |
|------|-----------|------|
| 实时操作 | `live`（默认） | MuJoCo 主循环；粒子经 gRPC 发往 Orca（与 `sph_sim_config.json` 中 particle 渲染一致）。 |
| 录制 | `record` | 将粒子帧写入 HDF5；默认不向 OrcaStudio 推粒子流；会话内并行录制 MuJoCo 全 `qpos`，结束时合并进同一粒子 HDF5。 |
| 回放 | `playback` | 不启动 MuJoCo / 耦合仿真；将已有 HDF5 经 `orca-sph` 客户端发往 OrcaStudio 粒子渲染。 |

**共同前提**：`live` / `record` 需在 OrcaLab（或 OrcaStudio）中已加载含 SPH 的流体场景，并按 README 启动仿真入口。`playback` 仅需 Orca 侧能接收对应 gRPC（通常仍开 OrcaStudio / 同场景）。

---

## 1. 操作（live）

```bash
python run_fluid_sim.py
# 等价
python run_fluid_sim.py --mode live
```

### 人类操作轨迹（可选）

与粒子 HDF5 独立：记录人在 Studio 侧对 **ctrl / mocap / equality** 等操作（不含 SPH 专用 mocap 体），便于与粒子录制按时间戳配对。当前轨迹 HDF5 为 **`schema_version=3`**（无 actuator 时 **`nu==0`** 且不写入 `ctrl` 数据集；帧数见 `mocap_pos` 时间维）。设计说明：[`envs/fluid/Docs/DESIGN_mujoco_human_trajectory_hdf5.md`](../../envs/fluid/Docs/DESIGN_mujoco_human_trajectory_hdf5.md)。

| 参数 | 说明 |
|------|------|
| `--trajectory-record` | 开启写入；默认路径：`trajectory_records/<前缀>_<时间戳>.h5`。 |
| `--trajectory-record-output PATH` | 指定输出 HDF5 完整路径。 |
| `--trajectory-record-prefix NAME` | 默认文件名前缀（仅字母、数字、`_`、`-`）；默认 `trajectory_record`。 |

示例：

```bash
python run_fluid_sim.py --trajectory-record
python run_fluid_sim.py --trajectory-record --trajectory-record-prefix my_run
```

---

## 2. 录制（record）

```bash
python run_fluid_sim.py --mode record
```

默认输出：`particle_records/<前缀>_<时间戳>.h5`。自动拉起 OrcaSPH 时，其标准输出可写入 `~/.orcagym/tmp/orcasph_<时间戳>.log`，其中含 `[PARTICLE_RECORD_STATS]` 行供统计窗口解析。

### 输出与帧率

| 参数 | 说明 |
|------|------|
| `--record-output PATH` | 指定粒子 HDF5 路径；父目录不存在会自动创建。 |
| `--record-prefix NAME` | 默认前缀（同上字符限制）；默认 `particle_record`。 |
| `--record-fps HZ` | 覆盖配置中的 `recording.record_fps`，并与 gRPC 更新率对齐。 |

### 是否在 Orca 侧预览粒子流

| 参数 | 说明 |
|------|------|
| `--render-particle` | 录制同时向 OrcaStudio 发送粒子 gRPC（默认关闭，仅写文件）。 |

### 录制统计子窗口（matplotlib）

默认会启动子进程：大字显示当前已录 **仿真时间 `sim_time`**（秒），并绘制基于日志的 FPS 等曲线（跳过开头若干条、滑动窗口等见下表）。无显示器或不想弹窗：

| 参数 | 说明 |
|------|------|
| `--no-record-stats-plot` | 不启动统计子进程。 |
| `--record-stats-interval SEC` | 刷新间隔，默认 `5`。 |
| `--record-stats-window SEC` | FPS 滑动曲线时间窗，默认 `5`。 |
| `--record-stats-skip-head N` | 跳过开头 N 条统计行再画曲线，默认 `5`。 |
| `--record-stats-rolling N` | 每条曲线最多保留最近 N 个点，默认 `50`。 |
| `--orcasph-log PATH` | **手动**启动 OrcaSPH 时指定日志路径，供统计窗 tail。 |

无 Tk / 无显示时可设 `MPLBACKEND=Agg`；子进程失败时终端会有提示，主仿真通常仍可继续。依赖问题见本目录 `requirements.txt`（`matplotlib`、`packaging`）。

单独查看已有日志（需在仓库根目录且 `PYTHONPATH=.` 或已 `pip install -e .`）：

```bash
PYTHONPATH=. python envs/fluid/utils/particle_record_stats_plot_viewer.py \
  --log ~/.orcagym/tmp/orcasph_xxx.log --interval 5 --skip-head 5 --rolling 50
```

### 在 record 时叠加已录好的人类轨迹

先用 `live` + `--trajectory-record` 得到轨迹 HDF5，再在 `record` 时回放：

| 参数 | 说明 |
|------|------|
| `--trajectory-playback PATH` | 从该 HDF5 在 `bridge.step` 之后叠加 mocap / eq / ctrl。 |

```bash
python run_fluid_sim.py --mode record --trajectory-playback trajectory_records/trajectory_record_YYYYMMDD_HHMMSS.h5
```

---

## 3. 回放（playback）

需已安装 **`orca-sph`**（提供 `orcasph_client.particle_replay`）。

```bash
python run_fluid_sim.py --mode playback --h5 particle_records/foo_20260101_120000.h5
# 与末尾位置参数等价
python run_fluid_sim.py --mode playback particle_records/foo.h5
```

| 参数 | 说明 |
|------|------|
| `--h5 PATH` 或末尾 `H5_FILE` | 录制得到的 HDF5。 |
| `--playback-target HOST:PORT` | ParticleRender gRPC 地址；省略则从 `sph_sim_config` 模板读取。 |
| `--playback-fps FPS` | 墙钟帧率；`0` 表示使用文件内 `record_fps` 属性。 |

---

## 4. 通用与其它参数

| 参数 | 说明 |
|------|------|
| `--config FILE` | 主配置 JSON，默认 `fluid_sim_config.json`（相对 **本脚本目录**）。 |
| `--gui` | 为自动拉起的 OrcaSPH 追加 `--gui`。 |
| `--manual-mode` | 不自动启动 OrcaLink / OrcaSPH；需自行先启动服务再运行脚本。 |
| `--use-all-cpu` | 关闭为 OrcaStudio 预留核、将 OrcaSPH 绑到高编号 CPU 的默认亲和策略。 |

**手动分步示例**（与 `--manual-mode` 配合）：

```bash
# 终端 1
orcalink --port 50351

# 终端 2（scene 以运行时 ~/.orcagym/tmp/sph_scene_*.json 为准）
orcasph --scene ~/.orcagym/tmp/sph_scene_xxx.json --gui

# 终端 3
python run_fluid_sim.py --manual-mode
```

`fluid_sim_config.json` 中与链路相关的字段示例：`orcalink.port`、`orcasph.enabled`、`orcasph.config_template`（如 `sph_sim_config.json`）等。

---

## 5. 相关文档与排障

- 粒子与 MuJoCo `qpos` 耦合录制 / 合并设计：[`envs/fluid/Docs/DESIGN_particle_record_mujoco_qpos_coupled_playback.md`](../../envs/fluid/Docs/DESIGN_particle_record_mujoco_qpos_coupled_playback.md)
- 流体环境模块说明：[`envs/fluid/README.md`](../../envs/fluid/README.md)
- **`packaging` 安装异常**（conda/pip 混用、`pip list` 显示 `None`、`Cannot uninstall packaging` 等）：可尝试 `pip install --ignore-installed "packaging>=21.0"`，或用 `python -c "import packaging; print(packaging.__version__)"` 确认能否导入；仍失败时按 `site-packages` 下实际路径清理损坏的 `packaging` 目录与 dist-info 后重装。
