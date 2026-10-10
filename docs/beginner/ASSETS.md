# 配套资产包介绍（OrcaPlaygroundAssets）

新手课程（01–30）使用的全部教具与机器人来自同一个配套资产包 **OrcaPlaygroundAssets**。本文介绍包里有什么、课程用了什么、还有哪些资产可供后续课程或自组场景时浏览查阅。

- 资产平台：https://simassets.orca3d.cn/
- 订阅方式：见 [01 课 README](../../examples/euler/beginner/stage1_scene_basics/lesson_01_hello_world/README.md)（订阅后需重启 OrcaLab 才生效）
- 课程侧引用：[assets.py](../../examples/euler/beginner/_common/assets.py) 单一事实源——当前为本地导入包 `345a60e1cced` 的过渡路径，正式上架后只需切换 `_PREFIX` 一处即可全课换源

## 包内容总览

| 分类 | 数量 | 资源类型 | 与已发布课程的关系 |
|---|---|---|---|
| 教学教具 | 16 个 MJCF | 物理（刚体/关节/执行器） | 01–30 课在用 |
| 机器人模型 | 8 个机型族 | 物理（MJCF/URDF + 网格） | ⚠ 超纲储备（见下） |
| 场景物件 | 3 件 | 美术（usdc/obj 网格）+ 物理（xml） | 储备，当前未引用 |
| 地形 | legged_gym 系列 | 物理（MJCF） | 储备，面向足式主题 |
| 纹理 | 8 套 2K PBR | 美术 | 教具共用贴图源 |

> 约定（REQUIREMENTS FR-A06）：只有视觉网格、没有刚体/碰撞配置的资产不得用于物理课程——选资产做物理实验前先确认它带物理定义。

## 教学教具（01–30 课在用）

| 教具 | 说明 | 使用课程 |
|---|---|---|
| `floor` | 5×5 m 地面 | 01–30 全部课程 |
| `cube` | 1 m 方块 | 01/04/07/09/10/11/12、14–18 |
| `cube_small` | 小方块 | 13/23/24 |
| `cuboid` | 长方体（带方向标记，旋转可见） | 08 |
| `sphere` | 球（半径 0.15 m） | 02/04/05/06、19–22/24 |
| `table` | 桌子（桌面顶面 0.75 m，长边沿 x） | 02/04/05/06、19–21 |
| `metal_shelf` | 金属货架 | 02 |
| `box_red` / `box_blue` | 红蓝方块对（spawn API 无运行时改色，以双色对替代） | 备用，主线未引用 |
| `block_arm` | 积木臂（二连杆转台臂） | 19/20/21 |
| `pendulum_passive` | 被动摆（无执行器） | 25/26 |
| `pendulum_position` | 位置伺服摆（kp=100 kv=8） | 27 |
| `wheel_velocity` | 速度伺服轮（kv=0.1） | 28 |
| `rotor_torque` | 力矩旋臂（±5 N·m） | 29/30 |
| `go2` | Unitree Go2（URDF + 网格） | 03/05/06 **仅加载观看** |
| `h1` | H1 人形机器人（源资产） | 03 **仅加载观看** |

> 教具物理参数与课程理论数字逐位对应（例：`pendulum_position` 的 kp/kv 正是 27 课静差公式参数）。修改教具参数前请先阅读对应课程 README，避免破坏"实测命中理论"的教学验证链。

## 机器人模型（⚠ 超出已发布课程范围）

**01–30 课不涉及下列模型的控制。** 课程实际用到的机器人只有 go2 与 h1，且仅在阶段 1 作为场景摆件"加载观看"（不驱动关节）。完整的机器人控制属于规划中的**阶段 6（31–36「控制完整机器人」）**，下列模型为其储备，现阶段标注为超纲内容：

| 机型 | 类型 | 备注 |
|---|---|---|
| Unitree Go2 | 四足机器狗 | 已在 03/05/06 课亮相 |
| G1（29 DoF） | 人形机器人 | MJCF + config |
| DeepRobotics Lite3 | 四足机器狗 | URDF + MJCF + STL |
| OpenLoong | 人形机器人 | 含 Robotiq 2F-85 夹爪 + 移动底盘（带操作能力） |
| RealMan RM65B / RM75B | 六/七轴协作机械臂 | |
| XBot-L | 人形机器人 | 54 body MJCF |
| zqsa01 | 腿式平台 | SolidWorks 导出 URDF |
| Hummer H2 | 车辆模型 | body/wheel usdc + MJCF |

## 场景物件与地形（美术 + 物理储备）

以下资产当前课程主线未引用，供后续课程、用户自组场景（层 3 倡导路径）或浏览查阅：

- **场景物件**：Cart_Basket（手推车篮）、Cup_of_Coffee（咖啡杯）、Office_Desk_7_MB（办公桌）——各含 usdc（美术呈现）+ xml（物理定义）+ obj（网格）三套格式
- **地形**：legged_gym 系列地形 MJCF（10° 坡面、低/高台阶、brics 等）+ height_map_helper——面向足式机器人运动主题（阶段 6+ / 选修）
- **纹理**：Metal009 / Metal044A / Metal044B / Plastic006 / Plastic010 / Plastic013A / oak_veneer_01 / wood_table_worn 共 8 套 2K PBR 纹理（Color / Normal / Roughness / Metalness）+ transparent_red.png——教具与机器人模型的共用贴图源

## 维护提醒

- 教具 XML 源在 OrcaPlayground 仓库 `examples/euler/beginner/assets/PlayGroundAssets/`（课程运行时单一事实源）；包制作源在本机 `OrcaPlaygroundAssets/` 目录——修改教具后需双侧同步并重新导入/上架
- robots/ 各模型的"超纲"标注是快照：阶段 6 开课后需更新本文使用状态
