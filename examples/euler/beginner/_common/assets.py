"""教学资产路径 — 初学者课程（beginner）共用的 spawnable 资产清单。

全部课程从本模块引用资产路径（单一事实源）：资产正式上传资产库后，
只需在此切换为云端正式包地址，无需逐课修改。

TODO(asset-lib): 当前为本地导入包 345a60e1cced 的过渡路径；正式格式为
assets/<hash>/default_project/prefabs/<name>（hash 为上传者 ID，
资产库见 https://simassets.orca3d.cn/）。
"""

from __future__ import annotations

_PREFIX = "assets/345a60e1cced/prefabs"

# 地面与基础物体
FLOOR = f"{_PREFIX}/floor_usda"  # 5×5 m 地面
CUBE = f"{_PREFIX}/cube_usda"  # 1 m 方块（body 中心静置高约 0.5 m）
CUBE_SMALL = f"{_PREFIX}/cube_small_usda"  # 小方块（13 课观察用）
CUBOID = f"{_PREFIX}/cuboid_usda"  # 长方体（08 课旋转演示用）
BALL = f"{_PREFIX}/sphere_usda"  # 球（半径 0.15 m）
TABLE = f"{_PREFIX}/table_usda"  # 桌子（桌面顶面 0.75 m，长边沿 x）
SHELF = f"{_PREFIX}/metal_shelf_usda"  # 金属货架

# 教具（阶段 4 桌前操作）
BLOCK_ARM = f"{_PREFIX}/block_arm_usda"  # 积木臂（二连杆转台臂，横梁中心高于底盘底面 0.15 m）

# 教具（阶段 5 关节控制；源 XML 在 assets/PlayGroundAssets/，
# 已导入本地资产包 345a60e1cced）
PENDULUM = f"{_PREFIX}/pendulum_passive_usda"    # 被动摆（25/26 课，无执行器）
PENDULUM_POS = f"{_PREFIX}/pendulum_position_usda"  # 位置伺服摆（27 课，kp=100 kv=8）
WHEEL_VEL = f"{_PREFIX}/wheel_velocity_usda"     # 速度伺服轮（28 课，kv=0.1）
ROTOR_TORQUE = f"{_PREFIX}/rotor_torque_usda"    # 力矩旋臂（29/30 课，gear=1）

# 机器人
GO2 = f"{_PREFIX}/go2_usda"  # go2 机器狗
H1 = f"{_PREFIX}/h1_usda"  # h1 人形机器人
