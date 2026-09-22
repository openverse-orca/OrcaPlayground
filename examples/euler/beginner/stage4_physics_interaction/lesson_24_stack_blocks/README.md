# 第 24 课：搭积木与推倒 — 多体接触与碰撞事件

## 课程目标

前几课都是"一个物体 + 地面"；这一课让物体**互相叠起来**，看两个面：

- **塔为什么能站着**：幕 1 用 `query_contact_simple` / `query_contact_force`
  打印三级接触力链——顶↔中 ≈ mg、中↔底 ≈ 2mg、底↔地 ≈ 3mg，
  接触力像金字塔一级级往下压
- **塔为什么会倒**：幕 2 撞球（复用 23 课的写初速度技能）滚过去撞塔，
  倒塌全程扫描接触对——**新接触对出现 = 一次碰撞事件**（球撞塔、
  块撞地、块撞块），逐条打印（本课融原"detect_collision 碰到了什么"
  主题：接触对过滤与碰撞事件检测）

## 前置条件

- ✅ 已在 OrcaLab 资产库中订阅 **OrcaPlaygroundAssets** 资产包
- ✅ 完成 23 课（知道 set_joint_qvel 怎么写初速度）
- ✅ OrcaLab 已启动（gRPC `localhost:50051`）
- 运行期间手离开视口（拖拽会物理干扰演示）

## 目录结构

| 文件 | 用途 |
|---|---|
| `run.py` | 课程入口（`python -m` 启动，配方区在文件顶部） |
| `example.yaml` | 课程元数据（导航/挑战/可调参数） |
| `README.md` | 本文档 |

## 资产说明

| 资产 | 用途 | 是否需要手动拖动 |
|---|---|---|
| `floor_usda` | 5×5m 地面 | 否，`--default-scene` 自动摆 |
| `cube_small_usda` × 3 | 三层塔（球心高度对齐中间块质心，正撞塔腰） | 否，`--default-scene` 自动摆 |
| `sphere_usda` | 撞球（1.4kg，从 1.5m 外滚来） | 否，`--default-scene` 自动摆 |

## 运行步骤

```bash
conda activate orca
python -m examples.euler.beginner.stage4_physics_interaction.lesson_24_stack_blocks.run --default-scene
```

也可以先在 OrcaLab 里自己摆三块塔 + 一个球 + 地面（不指定
`--default-scene`），脚本按 body 名检索（块按高度排底→顶）。

## 本课概念

- **静力链（牛顿第三定律的传递）**：塔静止时每级接触的法向力 =
  这一级上方压着的全部重量——顶块只扛自己（mg），底块扛整座塔
  （3mg）。力沿着接触链一级级往下传
- **接触对与碰撞事件**：`query_contact_simple` 列出当前全部接触
  （geom 对 + 接触点位置）；持续扫描，**新接触对出现 = 一次撞击**。
  过滤指定碰撞对（只看塔相关的）就是碰撞检测的最小实现
- **接触力读法**：`query_contact_force` 按 contact frame 返回 6D 力
  （前 3 分量是力，法向排第一）；面接触拆成 4 个角点，求和才是整面
- **为什么倒**：撞击动量让块获得速度，塔的重心一旦越过底块边缘，
  重力从「扶正力」变成「翻倒力」——稳定性 = 重心投影落在支撑面内

## 关键代码

```python
# 归约接触：body 对 → 法向力之和（4 个角点求和才是整面）
pairs = {}
for i, contact in enumerate(env.query_contact_simple()):
    key = tuple(sorted((body1, body2)))
    pairs[key] = pairs.get(key, 0.0) + forces[i][0]  # 法向分量

# 碰撞事件检测：扫描间隔内出现的新接触对
for key in set(cur_pairs) - prev_pairs:
    _logger.info(f"[碰撞] {key[0]} ↔ {key[1]}")

# 撞球发车（23 课技能复用）
sim_link.kick_body(env, striker, np.array([STRIKER_SPEED, 0.0, 0.0]))
```

## 预期结果与解释

- 幕 1：静力链 0.49 / 0.98 / 1.47 N，与 mg / 2mg / 3mg 理论值
  **精确一致**（单块 0.05kg）
- 幕 2：撞球滚约 0.9s 撞塔腰（首起碰撞事件法向力 ~13 N——撞击力
  远大于重量，这就是"冲击"），塔倒，全程 ~15 起碰撞事件连环打印
- 直观记忆：叠三层积木，最下面那块最"累"；保龄球撞瓶阵，
  瓶子四散时的每一次碰撞都能被仿真"看见"

## 常见问题

| 现象 | 原因与处置 |
|---|---|
| "未找到三块/球/地面" | 自摆场景缺件：拖三个小方块堆塔 + 一个球 + 地面，或加 `--default-scene` |
| 静力链有一级是 0 | spawn 后块未完全对齐（悬空）——塔没搭稳，`--default-scene` 的配方保证对齐 |
| 碰撞事件一条都没有 | 撞球速度太低没撞到塔（检查 STRIKER_SPEED）或球被挪走 |
| 塔倒后事件刷屏 | 正常——倒塌本来就是连环碰撞，扫描间隔 0.1s 已做了节流 |

## 动手改

配方区 `STRIKER_SPEED = 2.0` → `4.0`：撞球快一倍，冲击力更大、
块飞得更散；给 `block_bottom` 加 `scale=1.5`（改 `build_default_recipe`）：
底块变宽变重（22 课的"质量 ∝ 尺寸³"），支撑面变大——更难推倒。

## 小挑战（predict）

先写预测再运行：撞球速度加倍后，首起碰撞的法向力大概变几倍？
底块放大 1.5 倍后塔会怎么倒（推倒？滑走？纹丝不动？）——
体会动量与支撑面各自的角色。

## 参见

- 上一课：`lesson_23_friction_slide`（d=v²/(2μg)：摩擦管刹车）
- 阶段收束：本课是阶段 4（物理交互）最后一课，下一阶段进入关节控制
- `_common/sim_link.py`（连接/步进/kick_body）、`_common/assets.py`（资产路径）
