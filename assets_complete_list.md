# 完整资产列表

本文档列出 examples 中所有需要的资产及其在 OrcaGym_Assets 中的源路径。

## 机器人资产

### 1. Go2 机器人

- **源路径**: `robots/unitree/go2/`
- **目标路径**: `assets/prefabs/go2_usda/`
- **说明**: 完整目录复制

### 2. Lite3 机器人

- **源路径**: `robots/deeprobotics/Lite3/`
- **目标路径**: `assets/prefabs/lite3_usda/`
- **说明**: 完整目录复制

### 3. OpenLoong 机器人（带移动底盘）

- **源路径**: `robots/openloong/`
- **目标路径**: `assets/prefabs/openloong_gripper_2f85_mobile_base_usda/`
- **说明**: 完整目录复制
- **使用位置**: `examples/wheeled_chassis/run_wheeled_chassis.py`
- **依赖**:
  - `robots/realman/realman_rm75b/` 目录下的 STL 文件（见下方）
  - `textures/Metal009_2K-PNG/` 纹理目录
  - `textures/Plastic006_2K-PNG/` 纹理目录

### 4. Hummer H2 车辆

- **源路径**: `robots/hummer_h2/`
- **目标路径**: `assets/prefabs/hummer_h2_usda/`
- **说明**: 完整目录复制
- **使用位置**: `examples/wheeled_chassis/run_ackerman.py`

###  Remy 角色

- **源路径**: 需要确认（可能在 `robots/character/Remy/` 或 `scene_assets/character/Remy/`）
- **目标路径**: `assets/prefabs/Remy/`
- **说明**: 完整目录复制
- **使用位置**: `examples/character/run_character.py`, `envs/character/character_config/remy.yaml`

## 地形资产

所有地形文件从 `terrains/legged_gym/` 复制，重命名为 `*_usda` 后缀：

| 源文件                                                  | 目标文件                                            |
| ------------------------------------------------------- | --------------------------------------------------- |
| `terrains/legged_gym/terrain_brics_high.xml`          | `assets/prefabs/terrain_brics_high_usda`          |
| `terrains/legged_gym/terrain_brics_low.xml`           | `assets/prefabs/terrain_brics_low_usda`           |
| `terrains/legged_gym/terrain_ellipsoid_low.xml`       | `assets/prefabs/terrain_ellipsoid_low_usda`       |
| `terrains/legged_gym/terrain_perlin_rough_slope.xml`  | `assets/prefabs/terrain_perlin_rough_slope_usda`  |
| `terrains/legged_gym/terrain_perlin_rough.xml`        | `assets/prefabs/terrain_perlin_rough_usda`        |
| `terrains/legged_gym/terrain_perlin_smooth_slope.xml` | `assets/prefabs/terrain_perlin_smooth_slope_usda` |
| `terrains/legged_gym/terrain_perlin_smooth.xml`       | `assets/prefabs/terrain_perlin_smooth_usda`       |
| `terrains/legged_gym/terrain_slope_10.xml`            | `assets/prefabs/terrain_slope_10_usda`            |
| `terrains/legged_gym/terrain_slope_5.xml`             | `assets/prefabs/terrain_slope_5_usda`             |
| `terrains/legged_gym/terrain_stair_high.xml`          | `assets/prefabs/terrain_stair_high_usda`          |
| `terrains/legged_gym/terrain_stair_low_flat.xml`      | `assets/prefabs/terrain_stair_low_flat_usda`      |
| `terrains/legged_gym/terrain_stair_low.xml`           | `assets/prefabs/terrain_stair_low_usda`           |
| `terrains/legged_gym/terrain_test.xml`                | `assets/prefabs/terrain_test_usda`                |

### 地形资源文件

- `terrains/legged_gym/height_field.png` -> `terrains/legged_gym/height_field.png`
- `terrains/legged_gym/height_field_rough.png` -> `terrains/legged_gym/height_field_rough.png`

## Replicator 场景资产（仅 XML）

| 源文件                                                                      | 目标文件                                                      |
| --------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `scene_assets/usdz/converted_files/Cart_Basket/Cart_Basket.xml`           | `assets/prefabs/cart_basket_usda/Cart_Basket.xml`           |
| `scene_assets/usdz/converted_files/Cup_of_Coffee/Cup_of_Coffee.xml`       | `assets/prefabs/cup_of_coffee_usda/Cup_of_Coffee.xml`       |
| `scene_assets/usdz/converted_files/Office_Desk_7_MB/Office_Desk_7_MB.xml` | `assets/prefabs/office_desk_7_mb_usda/Office_Desk_7_MB.xml` |

## USDZ 源文件

| 源文件                                                       | 目标文件                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `scene_assets/usdz/Cart_Basket.usdz`                       | `scene_assets/usdz/Cart_Basket.usdz`                       |
| `scene_assets/usdz/Cup_of_Coffee.usdz`                     | `scene_assets/usdz/Cup_of_Coffee.usdz`                     |
| `scene_assets/usdz/Fully_textured_tank.usdz`               | `scene_assets/usdz/Fully_textured_tank.usdz`               |
| `scene_assets/usdz/Office_Desk_7_MB.usdz`                  | `scene_assets/usdz/Office_Desk_7_MB.usdz`                  |
| `scene_assets/usdz/Railway_Signal_Box_-_Bytom_Poland.usdz` | `scene_assets/usdz/Railway_Signal_Box_-_Bytom_Poland.usdz` |
| `scene_assets/usdz/Round_Table.usdz`                       | `scene_assets/usdz/Round_Table.usdz`                       |

## OpenLoong 依赖的 Realman 资产

OpenLoong 的 XML 文件引用了以下 Realman 的 STL 文件：

| 源文件                                                     | 目标文件                                                                                           |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `robots/realman/realman_rm75b/base_link_underpan.STL`    | `assets/prefabs/openloong_gripper_2f85_mobile_base_usda/realman_rm75b/base_link_underpan.STL`    |
| `robots/realman/realman_rm75b/link_right_wheel.STL`      | `assets/prefabs/openloong_gripper_2f85_mobile_base_usda/realman_rm75b/link_right_wheel.STL`      |
| `robots/realman/realman_rm75b/link_left_wheel.STL`       | `assets/prefabs/openloong_gripper_2f85_mobile_base_usda/realman_rm75b/link_left_wheel.STL`       |
| `robots/realman/realman_rm75b/link_swivel_wheel_1_1.STL` | `assets/prefabs/openloong_gripper_2f85_mobile_base_usda/realman_rm75b/link_swivel_wheel_1_1.STL` |
| `robots/realman/realman_rm75b/link_swivel_wheel_1_2.STL` | `assets/prefabs/openloong_gripper_2f85_mobile_base_usda/realman_rm75b/link_swivel_wheel_1_2.STL` |
| `robots/realman/realman_rm75b/link_swivel_wheel_2_1.STL` | `assets/prefabs/openloong_gripper_2f85_mobile_base_usda/realman_rm75b/link_swivel_wheel_2_1.STL` |
| `robots/realman/realman_rm75b/link_swivel_wheel_2_2.STL` | `assets/prefabs/openloong_gripper_2f85_mobile_base_usda/realman_rm75b/link_swivel_wheel_2_2.STL` |
| `robots/realman/realman_rm75b/link_swivel_wheel_3_1.STL` | `assets/prefabs/openloong_gripper_2f85_mobile_base_usda/realman_rm75b/link_swivel_wheel_3_1.STL` |
| `robots/realman/realman_rm75b/link_swivel_wheel_3_2.STL` | `assets/prefabs/openloong_gripper_2f85_mobile_base_usda/realman_rm75b/link_swivel_wheel_3_2.STL` |
| `robots/realman/realman_rm75b/link_swivel_wheel_4_1.STL` | `assets/prefabs/openloong_gripper_2f85_mobile_base_usda/realman_rm75b/link_swivel_wheel_4_1.STL` |
| `robots/realman/realman_rm75b/link_swivel_wheel_4_2.STL` | `assets/prefabs/openloong_gripper_2f85_mobile_base_usda/realman_rm75b/link_swivel_wheel_4_2.STL` |
| `robots/realman/realman_rm75b/body_base_link.STL`        | `assets/prefabs/openloong_gripper_2f85_mobile_base_usda/realman_rm75b/body_base_link.STL`        |

**注意**: 需要创建 `assets/prefabs/openloong_gripper_2f85_mobile_base_usda/realman_rm75b/` 目录

## OpenLoong 依赖的纹理资产

| 源目录                          | 目标目录                        |
| ------------------------------- | ------------------------------- |
| `textures/Metal009_2K-PNG/`   | `textures/Metal009_2K-PNG/`   |
| `textures/Plastic006_2K-PNG/` | `textures/Plastic006_2K-PNG/` |

## 相机和灯光资产

这些资产通常在 OrcaStudio 中已内置，但如果在代码中明确引用：

- `assets/prefabs/cameraviewport` - 相机视口
- `assets/prefabs/cameraviewport_mujoco` - MuJoCo 相机视口
- `assets/prefabs/spotlight` - 聚光灯

## 复制说明

1. **机器人资产**: 复制整个目录，保持内部结构
2. **地形资产**: 复制 XML 文件，重命名为 `*_usda`（去掉 `.xml` 扩展名）
3. **Replicator 资产**: 只复制 XML 文件
4. **USDZ 文件**: 保持原路径结构
5. **依赖资产**: OpenLoong 需要的 realman STL 文件和纹理需要单独复制

## 快速复制命令

```bash
SOURCE="/home/guojiatao/OrcaWorkStation/OrcaGym_Assets"
TARGET="/home/guojiatao/OrcaWorkStation/OrcaPlaygroundAssets"

# 机器人
cp -r "$SOURCE/robots/unitree/go2" "$TARGET/assets/prefabs/go2_usda"
cp -r "$SOURCE/robots/deeprobotics/Lite3" "$TARGET/assets/prefabs/lite3_usda"
cp -r "$SOURCE/robots/openloong" "$TARGET/assets/prefabs/openloong_gripper_2f85_mobile_base_usda"
cp -r "$SOURCE/robots/hummer_h2" "$TARGET/assets/prefabs/hummer_h2_usda"

# 地形（12个文件）
# ... 见上方表格

# Replicator XML（3个文件）
# ... 见上方表格

# USDZ（6个文件）
cp -r "$SOURCE/scene_assets/usdz" "$TARGET/scene_assets/"

# OpenLoong 依赖
mkdir -p "$TARGET/assets/prefabs/openloong_gripper_2f85_mobile_base_usda/realman_rm75b"
cp "$SOURCE/robots/realman/realman_rm75b"/*.STL "$TARGET/assets/prefabs/openloong_gripper_2f85_mobile_base_usda/realman_rm75b/"

# 纹理
cp -r "$SOURCE/textures/Metal009_2K-PNG" "$TARGET/textures/"
cp -r "$SOURCE/textures/Plastic006_2K-PNG" "$TARGET/textures/"
```
