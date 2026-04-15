# SO101 仿真场景文件放置说明

运行仿真脚本需要两类文件：
1. **OrcaStudio 场景文件**（Assets + Levels）：放入 OrcaStudio 安装目录
2. **机械臂模型文件**（so101_new_calib.xml + assets/）：已包含在本目录

---

## 一、OrcaStudio 场景文件

### Levels（已随仓库提供）

仓库根目录下的 `Levels/` 直接**覆盖替换** OrcaSim 安装目录下的 `Levels/`：

```
OrcaSim 安装目录/
└── Levels/    ←  用仓库根目录的 Levels/ 覆盖替换
```

替换后在 OrcaStudio 中打开 `Levels/manidp2d` 场景，点击"运行"即可。

### Assets（需单独下载，约 2.5GB）

`Assets/` 体积过大（2.5GB），未包含在仓库中，请从百度云下载：

> 链接：https://pan.baidu.com/s/1nLnQ09DF1zXdJiTif3TFqA  
> 提取码：`gq9y`

下载后将 `Assets/` 直接**覆盖替换** OrcaSim 安装目录下的 `Assets/`：

```
OrcaSim 安装目录/
└── Assets/    ←  用下载的 Assets/ 覆盖替换
```

---

## 二、机械臂模型文件（已包含在本目录）

`so101_new_calib.xml` 和 `assets/`（STL 网格文件）已随仓库提供，无需额外放置：

```
assets/so101/
├── so101_new_calib.xml              ← SO101 机械臂 XML 模型文件
├── so101_new_calib_usda.prefab      ← OrcaStudio Prefab（含相机配置，见下方说明）
└── assets/                          ← 网格 STL 文件目录
    ├── base_so101_v2.stl
    ├── upper_arm_so101_v1.stl
    └── ...
```

### ⚠️ 关于 so101_new_calib_usda.prefab（相机缺失修复）

百度云下载的 `Assets/` 中，`so101_new_calib_usda.prefab` 存在**相机配置缺失**问题，会导致相机画面无法正常推流。

**修复方法**：将仓库 `assets/so101/so101_new_calib_usda.prefab` 替换到 OrcaSim 安装目录对应位置：

```
OrcaSim 安装目录/Assets/Prefabs/so101_new_calib_usda.prefab  ←  替换为此文件
```

---

## 验证

```bash
conda activate so101
cd OrcaGym-SO101
python -c "import os; assert os.path.exists('assets/so101/so101_new_calib.xml'); print('✓ XML 文件路径正确')"
```

---

## 自定义路径

如果需要使用其他位置的 XML 文件，通过 `--xml_path` 参数指定：

```bash
python examples/so101/so101_leader_teleoperation.py \
    --xml_path /custom/path/so101_new_calib.xml
```
