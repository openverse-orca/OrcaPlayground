# 阶段四环境搭建教程

本教程指导你搭建 OrcaGym Euler 阶段四在线验证所需的运行环境。
阶段四（Lesson 4–8）使用 G1 人形机器人在 OrcaStudio/OrcaLab 中进行在线端到端验证。

> 阶段四总体设计见 [orca_gym_euler_phase4_online_validation_development.md](../../../OrcaGym/docs/design/development/orca_gym_euler_phase4_online_validation_development.md)。

---

## 1. 前置条件确认

| 依赖 | 说明 | 验证命令 |
|------|------|---------|
| conda `orca` 环境 | OrcaGym 推荐环境，已安装全部依赖 | `conda activate orca && python -c "import orca_gym"` |
| OrcaStudio/OrcaLab | 阶段四在线验证的仿真前端，提供 gRPC 服务 + 视口渲染 | 启动后监听 `127.0.0.1:50051` |
| G1 机器人资产 | OrcaStudio/OrcaLab 中可搜索到的 G1 人形机器人 | 资产搜索框输入 `g1` |
| `onnxruntime` | Lesson 7/8 的 ONNX 行走策略推理依赖 | `python -c "import onnxruntime"` |

> **注意**：阶段四使用 conda `orca` 环境（非 OrcaFlow_Flow）。所有脚本均在 `orca` 环境下运行。

---

## 2. 资源路径说明

阶段四使用的 G1 模型资源位于 `envs/euler/robots/`：

```
envs/euler/robots/
├── g1_29dof_camera.xml              # G1 模型（含摄像头传感器、mocap body、测试 box、weld 约束）
├── config/
│   └── g1_29dof_hist.yaml           # G1 配置（关节顺序、默认角度、history 长度、观测缩放）
└── models/
    └── dec_loco/
        └── model_6600.onnx          # ONNX 行走策略（deepmimic_dec_loco_height）
```

### 2.1 G1 模型关键元素

`g1_29dof_camera.xml` 在原 G1 模型基础上加装了阶段四所需的测试元素：

| 元素 | 名称 | 用途 |
|------|------|------|
| 摄像头传感器 | `camera_head`（`user="7070 7071"`） | Lesson 7 视频录制（color port 7070, depth port 7071） |
| Mocap body | `TestMocapAnchor` | Lesson 6/8 mocap 拖拽锚点 |
| 测试 box | `manipulation_box` | Lesson 8 体操作目标物体 |
| Weld 约束 | `anchor_box_weld` | Lesson 8 锚点-box 焊接约束 |

### 2.2 验证资源就位

```bash
cd /path/to/OrcaPlayground

# 检查文件存在
ls envs/euler/robots/g1_29dof_camera.xml
ls envs/euler/robots/config/g1_29dof_hist.yaml
ls envs/euler/robots/models/dec_loco/model_6600.onnx

# 验证 XML 可加载（CPU 测试，可在 sandbox 内运行）
python -c "import mujoco; m = mujoco.MjModel.from_xml_path('envs/euler/robots/g1_29dof_camera.xml'); print(f'nq={m.nq}, nv={m.nv}, nu={m.nu}')"
```

预期输出：`nq=36, nv=35, nu=29`（7 free joint + 29 hinge joints）。

---

## 3. OrcaStudio/OrcaLab 启动与关卡加载

### 3.1 启动 OrcaStudio/OrcaLab

1. 打开 OrcaStudio/OrcaLab 应用
2. 确认 gRPC 服务已启动（默认监听 `127.0.0.1:50051`）

### 3.2 加载 G1 关卡

1. 在资产搜索框中输入 `g1`，搜索 G1 人形机器人
2. 将 G1 拖入布局场景
3. 确认场景中**恰好 1 台** G1（阶段四要求单机器人场景）
4. 点击「运行」按钮，启动仿真

> **Lesson 6/8 额外要求**：场景需包含 mocap body `TestMocapAnchor`、测试 box `manipulation_box`、weld 约束 `anchor_box_weld`。这些元素已内置在 `g1_29dof_camera.xml` 中，若使用 Studio 自带的 G1 资产，需确认这些元素已加载。

---

## 4. 首次连通性测试

确认 OrcaStudio/OrcaLab 已启动并加载 G1 关卡后，运行以下测试验证环境就绪：

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 连通性测试：扫描场景中的 G1 agent_name
python -c "
from examples.euler.g1_base_env import build_g1_template, resolve_g1_agent_name
agent_name = resolve_g1_agent_name('127.0.0.1:50051')
print(f'G1 agent_name: {agent_name}')
print('环境就绪！')
"
```

**预期输出**：

```
G1 agent_name: g1
环境就绪！
```

若输出 `ValueError: 找不到对应的机器人型号：G1`，说明场景中未加载 G1 或关节不匹配，请回到第 3 步检查关卡加载。

---

## 5. 框架代码验证

阶段四框架代码（步骤 0 产物）的可离线验证部分：

```bash
cd /path/to/OrcaPlayground
conda activate orca

# 5.1 验证 OnlineVerifier
python -c "
from examples.euler.online_verifier import OnlineVerifier
v = OnlineVerifier('setup_test')
v.check('test_pass', 1 == 1, 1, 1)
v.check_range('height', 0.8, 0.7, 0.9)
v.observe('standing', 'G1 应站立')
report = v.report()
assert report['summary']['all_passed']
print('OnlineVerifier 验证通过')
"

# 5.2 验证 G1BaseEnv 可离线实例化
python -c "
from examples.euler.g1_base_env import G1BaseEnv
env = G1BaseEnv(skip_grpc_load=True)
print(f'G1BaseEnv 离线实例化成功: nq={env.model.nq}, nu={env.model.nu}')
env.close()
"

# 5.3 验证 G1Locomotion 可加载 ONNX
python -c "
from examples.euler.g1_locomotion import G1Locomotion
loco = G1Locomotion(agent_name='g1')
print(f'G1Locomotion 初始化成功: onnx_path={loco.onnx_path}')
print('G1Locomotion 验证通过')
"
```

---

## 6. 常见问题排查

### Q1：连通性测试报 `grpc.RpcError: failed to connect to all addresses`

**原因**：OrcaStudio/OrcaLab 未启动或 gRPC 地址不对。

**解决**：
1. 确认 OrcaStudio/OrcaLab 已启动
2. 确认监听地址为 `127.0.0.1:50051`（默认）
3. 若端口不同，在脚本中通过 `orcagym_addr` 参数指定

### Q2：场景扫描报 `ValueError: 找不到对应的机器人型号：G1`

**原因**：场景中未加载 G1，或 G1 资产的关节命名不匹配后缀模板。

**解决**：
1. 确认场景中已拖入 G1 机器人
2. 确认场景中**恰好 1 台** G1（多台会报数量过多）
3. 检查 G1 资产是否为 29 dof 版本

### Q3：`ModuleNotFoundError: No module named 'onnxruntime'`

**原因**：`orca` 环境未安装 onnxruntime（Lesson 7/8 需要）。

**解决**：

```bash
conda activate orca
pip install onnxruntime
```

### Q4：G1Locomotion 报 ONNX 输入维度不匹配

**原因**：`g1_locomotion.py` 的观测组装布局与 `model_6600.onnx` 的实际输入规范不一致。

**解决**：检查 ONNX 模型的实际输入维度（`session.get_inputs()[0].shape`），对照 `g1_locomotion.py` 的 `_build_obs` 方法调整观测拼接顺序或 history 配置。
