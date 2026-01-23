# Fluid-MuJoCo 耦合仿真示例

SPH 流体与 MuJoCo 刚体耦合仿真，使用 OrcaLink 进行通信。

## 📋 运行前提

### 1. 启动 OrcaStudio 或 OrcaLab

**重要**：在运行仿真脚本之前，必须先启动 OrcaStudio 或 OrcaLab 并加载对应的流体仿真场景。

**场景要求**：
- 场景中包含带 SPH 标记的刚体
- 刚体需要有 `SPH_MESH_GEOM`、`SPH_SITE`、`SPH_MOCAP_SITE` 等标记

**推荐使用 OrcaLab**：
```bash
# 启动 OrcaLab 并加载 Fluid 示例场景
orcalab --scene fluid_example
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `orcalink-client>=0.2.0` - OrcaLink 客户端
- `orca-gym>=25.10.0` - OrcaGym 核心库

## 🚀 快速开始

### 自动模式（推荐）

脚本会自动启动 `orcalink` 和 `orcasph`：

```bash
python run_fluid_sim.py
```

**启动流程**：
1. 读取 `fluid_config.json` 配置
2. 创建 MuJoCo 环境
3. 自动生成 SPH scene.json
4. 启动 OrcaLink Server（端口从配置读取，等待 2 秒）
5. 动态生成 orcasph 配置文件（端口自动同步）
6. 启动 OrcaSPH（使用生成的配置和 scene.json）
7. 连接并开始仿真

### 手动模式

如果您想手动控制 OrcaLink 和 OrcaSPH 的启动（用于调试）：

#### 步骤 1：启动 OrcaLink Server

```bash
orcalink --port 50052
```

#### 步骤 2：启动 OrcaSPH

```bash
# 使用自动生成的 scene.json（位于 ~/.orcagym/tmp/）
orcasph --scene ~/.orcagym/tmp/sph_scene_xxx.json --gui

# 或使用自定义 scene.json
orcasph --scene my_scene.json --gui
```

#### 步骤 3：运行仿真脚本（手动模式）

```bash
python run_fluid_sim.py --manual-mode
```

## ⚙️ 配置文件

### fluid_config.json - 主配置文件

包含所有仿真配置，统一管理 OrcaGym、OrcaLink、OrcaSPH 的参数：

```json
{
  "orcagym": {
    "address": "localhost:50051",
    "agent_name": "NoRobot",
    "env_name": "SimulationLoop"
  },
  "orcalink": {
    "enabled": true,
    "host": "localhost",
    "port": 50351,
    "auto_start": true,
    "startup_delay": 2,
    "command": "orcalink",
    "args": [],
    "client": {
      "session_id": 1,
      "client_name": "mujoco_client",
      "update_rate_hz": 50,
      "session": {
        "control_mode": "sync",
        "expected_clients": 2
      }
    },
    "bridge": {
      "coupling_mode": "multi_point_force"
    }
  },
  "orcasph": {
    "enabled": true,
    "auto_start": true,
    "command": "orcasph",
    "args": ["--gui"],
    "scene_auto_generate": true,
    "config": {
      "orcalink_client": { ... },
      "orcalink_bridge": { ... },
      "physics": { ... },
      "debug": { ... }
    }
  },
  "sph": {
    "scene_config": "scene_config.json",
    "include_fluid_blocks": true,
    "include_wall": true
  }
}
```

**关键配置说明**：

- `orcalink.port`: OrcaLink 服务器端口，**自动应用到所有地方**（启动命令、客户端连接、orcasph 配置）
- `orcalink.startup_delay`: OrcaLink 启动后的等待时间（秒），默认 2 秒
- `orcasph.scene_auto_generate`: 自动生成 scene.json（启动 orcasph 前完成）
- `orcasph.config`: orcasph 的完整配置，`server_address` 会自动从 `orcalink.port` 填充

### scene_config.json - SPH 场景配置

定义 SPH 物理属性、流体块、墙体等。

### 端口自动同步

**重要**：端口号从 `orcalink.port` 自动同步到：

1. **启动 OrcaLink 服务器**: `orcalink --port 50351`
2. **OrcaLinkBridge 连接**: `server_address = localhost:50351`（从 `orcalink.client` 自动构建）
3. **OrcaSPH 连接**: 动态生成的配置文件中 `server_address = localhost:50351`

**无需手动配置多处，只需修改 `orcalink.port` 即可！**

### 使用自定义配置

```bash
python run_fluid_sim.py --config my_config.json
```

## 📖 使用场景

### 场景 1：快速测试（自动模式）

```bash
# 一键启动，所有服务自动管理
python run_fluid_sim.py
```

### 场景 2：调试模式（手动启动服务）

```bash
# 终端 1：启动 OrcaLink（可查看日志）
orcalink --port 50052

# 终端 2：启动 OrcaSPH（可查看日志）
orcasph --scene scene.json --gui

# 终端 3：运行仿真
python run_fluid_sim.py --manual-mode
```

### 场景 3：自定义配置

```bash
# 创建自定义配置
cp fluid_config.json my_config.json
# 编辑 my_config.json...

# 使用自定义配置运行
python run_fluid_sim.py --config my_config.json
```

## 🛠️ 高级用法

### 资源文件路径

配置文件中的几何文件路径支持三种格式：

1. **包资源路径**（推荐）：
   ```json
   {
     "geometryFile": "package://orcasph/data/models/UnitBox.obj"
   }
   ```
   从 `orcasph_client` 包中加载，兼容所有安装方式（包括 `pip install -e .`）。

2. **绝对路径**：
   ```json
   {
     "geometryFile": "/absolute/path/to/UnitBox.obj"
   }
   ```

3. **相对路径**（自动 fallback）：
   ```json
   {
     "geometryFile": "../../../data/models/UnitBox.obj"
   }
   ```
   相对于 `scene_generator.py` 解析，如果不存在则自动从 `orcasph_client` 包中查找。

**注意**：当使用相对路径且文件不存在时，系统会自动尝试从 `orcasph_client` 包中查找对应的资源文件（如 `data/models/UnitBox.obj`），确保在普通安装和可编辑安装（`pip install -e .`）模式下都能正常工作。

### 生成 SPH 场景文件

如果需要单独生成 scene.json（不运行仿真）：

```bash
python -m envs.fluid.tools.generate_scene_cli \\
    /path/to/model.xml \\
    output_scene.json \\
    --config scene_config.json
```

### 禁用 SPH 集成

在配置文件中设置：

```json
{
  "orcasph": {
    "enabled": false
  }
}
```

或仅运行 MuJoCo 仿真（不启动 SPH）。

## 🏗️ 架构说明

本示例使用 `envs.fluid` 模块，核心组件：

- **FluidSimEnv** - Gymnasium 环境封装
- **OrcaLinkBridge** - SPH-MuJoCo 通信桥接
- **SceneGenerator** - 从 MuJoCo 模型生成 SPH 场景
- **ConfigGenerator** - 动态生成 OrcaLink 配置

详细 API 文档：`envs/fluid/README.md`

## 🐛 常见问题

### Q1: 提示 "无法连接到 OrcaLink"

**原因**：OrcaLink Server 未启动或端口不匹配

**解决**：
1. 检查 OrcaLink 是否运行：`ps aux | grep orcalink`
2. 检查端口配置：确保 `fluid_config.json` 中的端口与 OrcaLink 启动端口一致
3. 检查 startup_delay：可能需要增加等待时间（如改为 10 秒）
4. 尝试手动启动：`orcalink --port 50052`

### Q2: OrcaSPH 窗口未显示

**原因**：可能是图形界面问题或 scene.json 路径错误

**解决**：
1. 检查 scene.json 是否生成：`ls ~/.orcagym/tmp/sph_scene_*.json`
2. 手动启动查看错误：`orcasph --scene <path> --gui`
3. 检查日志：`~/.orcagym/tmp/orcasph_*.log`

### Q3: 场景中没有流体

**原因**：MuJoCo 场景中缺少 SPH 标记的刚体

**解决**：
1. 确保场景中包含带 `SPH_MESH_GEOM` 的刚体
2. 检查 scene.json 中的 `FluidBlocks` 配置
3. 参考 `scene_config.json` 调整流体块位置

### Q4: 提示 "配置文件不存在"

**解决**：
1. 确保 `fluid_config.json` 位于 `examples/fluid/` 目录
2. 或使用 `--config` 参数指定配置文件路径

## 📞 获取帮助

- 查看核心模块文档：`envs/fluid/README.md`
- 提交 Issue：https://github.com/openverse-orca/OrcaGym/issues
- 联系：huangwei@orca3d.cn
