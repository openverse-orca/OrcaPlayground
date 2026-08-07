# run_simulation.py 组织架构

> 源文件：`OrcaPlayground/envs/fluid/launch/run_simulation.py`（约 972 行；下文行号随源文件维护）

## 一、顶层入口

```
run_simulation_with_config(config, session_timestamp, cpu_affinity, max_steps)
```

唯一公开函数，由 `run_fluid_sim.py` 调用。执行流程：

```
_preflight_session()              ← 预检：时间戳、日志、端口冲突检查
_init_atexit_state_for_session()  ← 初始化全局 atexit 状态
注册 SIGTERM handler              ← 保证启动阶段也能响应 OrcaLab stop
try:
    步骤1~5（见下文）+ 可选 stats 图进程
    主循环
except KeyboardInterrupt / Exception:
    ...
finally:
    _finalize_simulation_session()  ← 全量清理
```

---

## 二、启动步骤（步骤 1→5，严格有序）

| 步骤 | 函数 | 行号（约） | 依赖 | 作用 |
|------|------|------------|------|------|
| 0 | `_preflight_session` | 150 | 无 | 时间戳、`setup_python_logging`、临时目录、OrcaLink 端口冲突检查 |
| 0' | `_init_atexit_state_for_session` | 186 | 无 | 设置 `_fluid_atexit_state` 全局字典 |
| 1 | `_create_and_reset_gym_env` | 194 | OrcaStudio / OrcaGym 已就绪（运行期依赖） | `gym.register` + `gym.make` + `env.reset()` → 一般 `mjData.time=0` |
| 1' | `_maybe_launch_mujoco_viewer` | 251 | 步骤1 | 可选：`mujoco.viewer.launch_passive()` |
| 2 | `_maybe_generate_sph_scene` | 345 | 步骤1 | `SceneGenerator` 生成 scene.json + particle_render 覆盖配置 |
| 3 | `_start_orcalink_if_configured` | 397 | 无 | 子进程启动 `orcalink`（端口见配置，默认 50351），`startup_delay` 秒 |
| 4 | `_start_orcasph_if_configured` | 432 | 步骤2+3 | 子进程启动 `orcasph --config ... --scene ...`，固定再 `sleep(2)` |
| 4' | `_try_start_record_stats_plot_viewer` | （见 `fluid_session`） | 配置启用时 | 可选：独立进程拉性能图（与主循环无硬依赖） |
| 5 | `_connect_sph_bridge_if_enabled` | 488 | 步骤3+4 | `OrcaLinkBridge(env, config)` → `connect()`（内联等待 OrcaLink **session_ready** / 超时） |

**依赖关系图**：

```
OrcaStudio (外部)
    │
    ▼
步骤1: _create_and_reset_gym_env ──→ 步骤1': _maybe_launch_mujoco_viewer
    │
    ├──→ 步骤2: _maybe_generate_sph_scene
    │         │
    │         ▼
步骤3: _start_orcalink ──→ 步骤4: _start_orcasph ──→ (4') stats 图可选 ──→ 步骤5: _connect_sph_bridge
```

---

## 三、核心数据结构

### FluidSimulationContext（dataclass）

贯穿整个会话的可变状态，所有函数通过 `ctx` 参数共享：

| 字段 | 类型 | 用途 |
|------|------|------|
| `config` | Dict | 完整配置（运行时可能被修改，如 `orcasph.enabled=False`） |
| `session_timestamp` | str | 会话时间戳，统一日志文件名 |
| `cpu_affinity` | Optional[str] | CPU 亲和性，如 `"4-15"` |
| `orcagym_tmp_dir` | Path | `~/.orcagym/tmp/`，存放 scene.json、orcasph_config.json、日志 |
| `process_manager` | ProcessManager | 管理子进程（orcalink、orcasph） |
| `shutdown_event` | threading.Event | SIGTERM/SIGHUP 置位，主循环协作退出 |
| `max_steps` | int | >0 时限制主循环步数 |
| `env` | Any | gymnasium 环境（SimEnv） |
| `sph_wrapper` | Any | OrcaLinkBridge 实例 |
| `mujoco_viewer` | Any | MuJoCo 被动查看器 Handle |
| `traj_rec` | Any | TrajectoryRecorder（人类轨迹录制） |
| `traj_player` | Any | TrajectoryPlayer（人类轨迹回放） |
| `mujoco_qpos_sidecar` | Any | MuJoCo qpos HDF5 录制器 |
| `scene_output_path` | Optional[Path] | 生成的 scene.json 路径 |
| `particle_render_override` | Any | 由 scene 生成阶段写入的 particle_render 覆盖（可为 None） |
| `sphscale` | float | SPH 缩放因子 |
| `prev_sigterm_handler` | Any | 前一个 SIGTERM handler（清理时恢复） |

---

## 四、主循环

### 入口

```
_setup_main_loop_recorders(ctx)   ← 注册 SIGHUP、初始化录制器
_run_cooperative_main_loop(ctx)   ← 主循环本体
```

### 单圈逻辑（`_run_cooperative_main_loop`，约 L598–788）

```
while not shutdown_event.is_set():
    │
    ├── 1. 轨迹回放耗尽检查
    │      if traj_player.exhausted → break
    │
    ├── 2. SPH 同步（决定 should_step）
    │      should_step = True
    │      if orcasph.enabled and sph_wrapper:
    │          should_step = sph_wrapper.step()
    │          │
    │          └── OrcaLinkBridge.step()
    │              ├── is_connected()? → False 则返回 False
    │              ├── is_session_ready_status()? → False 则返回 False
    │              └── current_mode.step()  ← 策略模式（multi_point_force 等）
    │                  以 multi_point_force 为例（见 ``coupling_modes/multi_point_force_mode.py``）：
    │                  ├── subscribe_forces_and_positions()（或等价路径）
    │                  │     → 在 ``orcalink_client`` 内按 sync 规则调整 ``current_sync_window``
    │                  ├── should_pause_this_cycle() → 窗口≤0 则本周期返回 False
    │                  └── publish_site_positions() → 发布 SITE，窗口再递减（细节见客户端）
    │
    ├── 3. MuJoCo 动力学步进
    │      if should_step:
    │          if traj_player: push_pending + env.step(None) + advance_cursor
    │          else:           env.step(None)
    │          mujoco_qpos_sidecar.append_row()
    │          traj_rec.append_frame()
    │          [MONITOR] 写 MuJoCo body 状态 CSV
    │          env.render()
    │      else:
    │          env.render()   ← should_step=False 时仍渲染（仅不 env.step）
    │
    ├── 4. MuJoCo 查看器同步
    │      if mujoco_viewer:
    │          update_viewer_sph_anchor_markers()
    │          viewer.sync()
    │
    ├── 5. 时间同步 CSV
    │      写 mj_wall_time, step_count, sim_time, sph_sim_time, orcalink_cycle
    │
    ├── 6. 实时步进节拍
    │      elapsed = now - start_time
    │      if elapsed < REALTIME_STEP(0.02s):
    │          shutdown_event.wait(remaining)  ← 可被 SIGTERM 唤醒
    │
    └── 7. 步数递增与退出检查
           step_count += 1
           if max_steps and step_count >= max_steps → break
```

### 仿真开始时间点

第一次 `env.step(None)` 执行的墙钟时刻取决于：

| 场景 | should_step 何时首次为 True |
|------|---------------------------|
| orcasph.enabled=True，connect 成功 | `sph_wrapper.step()` **首次**返回 True：需 **会话已就绪** 且耦合模式未因流控返回 False；**不保证**主循环第一圈即为 True（例如首圈仍 `session_ready=False` 或 sync 暂停） |
| orcasph.enabled=True，connect 失败 | connect 失败后 `orcasph.enabled=False`，should_step 保持默认 True，几乎立即 env.step |
| orcasph.enabled=False（配置禁用） | should_step 始终为默认 True，主循环首圈即 env.step |

---

## 五、信号处理

| 信号 | 注册位置 | 行为 |
|------|---------|------|
| SIGTERM | `run_simulation_with_config` 入口 | `_make_sigterm_cleanup_handler`：同步执行全量清理后 `os._exit(0)`，不依赖 finally |
| SIGHUP | `_setup_main_loop_recorders` | `ctx.shutdown_event.set()`，主循环协作退出，由 finally 清理 |
| KeyboardInterrupt | try-except | 主循环自然退出，由 finally 清理 |

**SIGTERM handler 实际顺序**（见 `_make_sigterm_cleanup_handler`，约 L88–145；且仅在 `viewport_reset_done` 为假时执行）：

1. `shutdown_event.set()` — 通知主循环协作退出  
2. `_terminate_stats_plot_proc()` — 若有独立 stats 图进程则收尾  
3. `ctx.sph_wrapper.close()` — 断开 OrcaLink Bridge（若存在）  
4. 若 `owns_shared_services`：`_fluid_send_end_simulation_from_config()` — 告知 ParticleRender 结束  
5. `ctx.process_manager.cleanup_all()` — 终止 orcasph / orcalink 子进程  
6. 若 `owns`：`time.sleep(0.2)` — 等待在途粒子帧被丢弃  
7. 若 `env_ref` 且 `owns`：`_fluid_sync_initial_viewport_to_engine` + `env.close()`；若仅有 `env_ref` 则尝试 `close()`  
8. 更新 `_fluid_atexit_state`，`os._exit(0)`

---

## 六、清理（`_finalize_simulation_session`，约 L791–870）

finally 块中执行，顺序：

```
1. 恢复 SIGTERM handler
2. _terminate_stats_plot_proc()
3. 关闭 mujoco_viewer
4. 关闭 traj_rec / mujoco_qpos_sidecar / traj_player / traj_stats_log_f
5. sph_wrapper.close()
6. _fluid_send_end_simulation_from_config()  ← 仅 owns_shared_services
7. process_manager.cleanup_all()
8. time.sleep(0.2)                           ← 仅 owns_shared_services
9. merge_particle_mujoco_sidecar_into_particle_h5()  ← record 模式合并
10. 若 `ctx.env` 非空：`owns` 为真时 `_fluid_sync_initial_viewport_to_engine`；随后 **总是**尝试 `env.close()`（与 `owns` 无关）
11. 更新 _fluid_atexit_state
```

---

## 七、辅助函数

| 函数 | 行号 | 作用 |
|------|------|------|
| `_resolve_cli_binary` | 71 | 在 Python bin 目录或 PATH 中查找可执行文件 |
| `_make_sigterm_cleanup_handler` | 88 | 生成 SIGTERM handler 闭包 |
| `_preflight_session` | 150 | 时间戳、日志、端口冲突检查 |
| `_init_atexit_state_for_session` | 186 | 初始化全局 atexit 状态字典 |
| `_create_and_reset_gym_env` | 194 | 注册并创建 gymnasium 环境 |
| `_maybe_launch_mujoco_viewer` | 251 | 可选启动 MuJoCo 被动查看器 |
| `_maybe_generate_sph_scene` | 345 | 生成 SPH scene.json |
| `_start_orcalink_if_configured` | 397 | 子进程启动 OrcaLink Server |
| `_start_orcasph_if_configured` | 432 | 子进程启动 OrcaSPH |
| `_connect_sph_bridge_if_enabled` | 488 | 创建 OrcaLinkBridge 并连接 |
| `_setup_main_loop_recorders` | 536 | 初始化录制器（轨迹/qpos/统计） |
| `_run_cooperative_main_loop` | 598 | 主循环本体 |
| `_finalize_simulation_session` | 791 | 全量清理 |
| `run_simulation_with_config` | 873 | 顶层入口 |

---

## 八、外部依赖关系

```
run_simulation.py
    │
    ├── orcalink_bridge.OrcaLinkBridge          ← SPH-MuJoCo 通信适配器
    │       ├── orcalink_client (pip)           ← Python gRPC 客户端
    │       └── coupling_modes/                 ← 策略模式（spring_constraint / multi_point_force）
    │
    ├── trajectory.TrajectoryRecorder/Player    ← 人类轨迹 HDF5 录制/回放
    │
    ├── utils.scene_generator.SceneGenerator    ← MuJoCo → SPH scene.json 生成
    │
    ├── utils.merge_particle_mujoco_h5          ← 录制结束后合并粒子+qpos HDF5
    │
    ├── utils.mujoco_qpos_sidecar_recorder      ← MuJoCo qpos 并行录制
    │
    ├── modules.mujoco_anchor_viewer_overlay    ← MuJoCo 查看器锚点球叠加
    │
    ├── launch.fluid_session                    ← ParticleRender/视口/EndSimulation
    │
    ├── launch.process_utils.ProcessManager     ← 子进程管理（start/kill/cleanup）
    │
    └── launch.sph_config                       ← 生成 orcasph_config.json
```

---

## 九、关键数值（区分「源文件常量 / gym kwargs / 配置文件」）

| 名称 | 典型值 | 定义位置 | 含义 |
|------|--------|-----------|------|
| `REALTIME_STEP` | `0.02` | `run_simulation.py` 模块常量 | 主循环墙钟节拍（秒）；**不等于** OrcaLink `update_rate_hz` 的倒数（二者常恰好同为 0.02，但职责不同） |
| `frame_skip` | `20` | `_create_and_reset_gym_env` 传入 `gym.register` 的 `kwargs` | SimEnv 每步内部子步次数（与下面 `time_step` 相乘得到每步仿真秒数） |
| `time_step` | `0.001` | 同上 | SimEnv 子步长（秒） |
| `ready_timeout_sec` | `30.0`（可改） | `fluid_sim_config.json` / `sph_sim_config.json` 的 `orcalink.client.session` | OrcaLink 客户端等待 **session_ready** 的超时（秒） |
| `sync_window_size` | `2`（可改） | 同上 `session.sync_params` | 同步模式本地窗口大小 |
| `update_rate_hz` | `50`（可改） | 同上 `orcalink` 配置 | OrcaLink 数据交换标称频率（Hz） |
