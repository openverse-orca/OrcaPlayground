"""
力施加模块 —— 将 SPH 侧计算得到的流体力施加到 MuJoCo 刚体上

在多点力（multi_point_force）耦合模式下，SPH 将流体对刚体的总力/总力矩
按四面体分解到 4 个锚点，每个锚点力通过 OrcaLink 传输到 MuJoCo 侧，
本模块负责接收这些力并写入 MuJoCo 的 xfrc_applied 数组。

核心机制：
  - 脉冲力方案：每帧先清零上一帧的外力，再施加新力，避免累积误差
  - 力的安全裁剪：NaN/Inf 丢弃，L2 范数超限则等比缩放
  - 首帧更严裁剪：防止耦合启动阶段数值垃圾把刚体打飞

力的施加路径：
  SPH (ForceDecompositionModule) → OrcaLink → 本模块 → mj_apply_force_at_site()
  → xfrc_applied[body_id] → MuJoCo mj_step() 求解

  mj_apply_force_at_site() 的物理原理：
    力 F 作用在 SITE 点，等效到 body 中心时：
    - 力不变：F_body = F
    - 产生附加力矩：τ_induced = r × F，其中 r = site_pos - body_pos
    - 总力矩：τ_total = r × F + τ_user
    最终写入 data.xfrc_applied[body_id, :3]（力）和 [body_id, 3:]（力矩）
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# SPH→MJ：首包非空力列表使用更严的 L2 范数上限，避免首周期数值垃圾把刚体打飞；之后放宽。
_DEFAULT_CAP_FIRST = 10_000.0
_DEFAULT_CAP_REST = 1_000_000.0
_DEFAULT_TAU_CAP_FIRST = 1_000.0
_DEFAULT_TAU_CAP_REST = 100_000.0


class ForceApplicationModule:
    """力施加模块 —— 将 SPH 计算的力施加到 MuJoCo 刚体或 SITE 点

    支持两种施力方式：
      1. 刚体级施力（subscribe_and_apply_forces）：
         用于 ForcePositionMode，直接对 body 施加力/力矩

      2. SITE 点施力（subscribe_and_apply_site_forces / apply_site_forces_from_list）：
         用于 MultiPointForceMode，对 SITE 点施加力，底层自动计算力臂产生的附加力矩

    脉冲力方案：
      MuJoCo 的 xfrc_applied 是持续生效的外力数组，不会自动清零。
      如果每帧只写入新力而不清零旧力，力会不断累加导致刚体飞出。
      因此采用"先清零、再施加"的脉冲力方案：
        - 每帧开始时清零上一帧施加过力的 body 的 xfrc_applied
        - 然后施加本帧从 SPH 收到的新力
        - 如果本帧没有收到新力，则保持上一帧的力不变（SPH 侧未发送更新）
    """

    def __init__(self, env, orcalink_client, loop):
        """
        Args:
            env: OrcaGym 环境实例，提供 mj_apply_force_at_site / mj_clear_xfrc_applied_for_site 等接口
            orcalink_client: OrcaLinkClient 实例，用于订阅 SPH 发来的力数据
            loop: asyncio 事件循环，用于运行异步订阅操作
        """
        import sys
        print("[PRINT-DEBUG] ForceApplicationModule.__init__() - START", file=sys.stderr, flush=True)
        logger.debug("[DEBUG] ForceApplicationModule.__init__() - Start")
        self.env = env
        self.client = orcalink_client
        self.loop = loop

        # 脉冲力方案：记录上一帧施加过力的 site 名称集合
        # 下一帧开始时，对这些 site 所属的 body 执行 xfrc_applied 清零
        self._previous_site_names = set()

        # 本周期实际施加到 SITE 的力快照（MuJoCo 世界系）
        # 格式：[(site_name, force_vec), ...]
        # 供外部（如 steering 力矩 CSV）读取；无新力包时保持上一值
        self.last_applied_site_forces: List[Tuple[str, np.ndarray]] = []

        # 首帧标志：OrcaLink 第一次收到非空力包时使用更严格的范数裁剪
        # 处理完首包后置 False，后续使用宽松上限
        self._strict_sph_force_clip = True

        print("[PRINT-DEBUG] ForceApplicationModule.__init__() - END", file=sys.stderr, flush=True)
        logger.debug("[DEBUG] ForceApplicationModule.__init__() - Completed")

    def _force_norm_cap(self) -> float:
        """获取当前力向量的 L2 范数上限

        首帧使用更严上限（默认 10000 N），防止耦合启动阶段数值异常；
        后续帧使用宽松上限（默认 1000000 N），允许正常大力通过。
        可通过环境变量 ORCA_SITE_FORCE_NORM_CAP_FIRST / ORCA_SITE_FORCE_NORM_CAP 覆盖。
        """
        if self._strict_sph_force_clip:
            return float(os.environ.get("ORCA_SITE_FORCE_NORM_CAP_FIRST", str(_DEFAULT_CAP_FIRST)))
        return float(os.environ.get("ORCA_SITE_FORCE_NORM_CAP", str(_DEFAULT_CAP_REST)))

    def _torque_norm_cap(self) -> float:
        """获取当前力矩向量的 L2 范数上限

        首帧使用更严上限（默认 1000 N·m），后续帧使用宽松上限（默认 100000 N·m）。
        可通过环境变量 ORCA_BODY_TORQUE_NORM_CAP_FIRST / ORCA_BODY_TORQUE_NORM_CAP 覆盖。
        """
        if self._strict_sph_force_clip:
            return float(os.environ.get("ORCA_BODY_TORQUE_NORM_CAP_FIRST", str(_DEFAULT_TAU_CAP_FIRST)))
        return float(os.environ.get("ORCA_BODY_TORQUE_NORM_CAP", str(_DEFAULT_TAU_CAP_REST)))

    def _sanitize_force3(self, vec: np.ndarray, label: str) -> Optional[np.ndarray]:
        """清洗三维力向量：NaN/Inf 丢弃，L2 范数超限则等比缩放

        Args:
            vec: 原始力向量
            label: 日志标签（如 "site[cup_SPH_SITE_000]"）

        Returns:
            清洗后的力向量，或 None（表示应丢弃）
        """
        v = np.asarray(vec, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(v)):
            logger.warning("%s: non-finite force, skip apply", label)
            return None
        cap = self._force_norm_cap()
        n = float(np.linalg.norm(v))
        if n > cap:
            logger.warning("%s: clamp ||F|| %.4g -> %.4g (SPH→MJ)", label, n, cap)
            v = v * (cap / n)
        return v

    def _sanitize_torque3(self, vec: np.ndarray, label: str) -> Optional[np.ndarray]:
        """清洗三维力矩向量：NaN/Inf 丢弃，L2 范数超限则等比缩放

        Args:
            vec: 原始力矩向量
            label: 日志标签

        Returns:
            清洗后的力矩向量，或 None（表示应丢弃）
        """
        t = np.asarray(vec, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(t)):
            logger.warning("%s: non-finite torque, skip apply", label)
            return None
        cap = self._torque_norm_cap()
        n = float(np.linalg.norm(t))
        if n > cap:
            logger.warning("%s: clamp ||τ|| %.4g -> %.4g (SPH→MJ)", label, n, cap)
            t = t * (cap / n)
        return t

    def _mark_sph_force_batch_consumed(self, forces_nonempty: bool) -> None:
        """标记首帧力包已消费，后续帧切换为宽松裁剪上限

        Args:
            forces_nonempty: 本帧是否收到了非空的力数据
        """
        if forces_nonempty and self._strict_sph_force_clip:
            self._strict_sph_force_clip = False

    def subscribe_and_apply_forces(self):
        """订阅刚体级力并施加（用于 ForcePositionMode）

        从 OrcaLink 订阅 SPH 发来的刚体整体力/力矩，
        通过 env.apply_force_to_body() 施加到对应刚体。
        """
        if not self.client or not self.loop:
            return

        try:
            forces = self.loop.run_until_complete(
                self.client.subscribe_forces()
            )

            if not forces:
                return

            for force_data in forces:
                self._apply_force_to_body(force_data)
            self._mark_sph_force_batch_consumed(True)
        except Exception as e:
            logger.error(f"Error applying forces: {e}", exc_info=True)

    def subscribe_and_apply_site_forces(self):
        """订阅多点力并施加到 SITE 点（用于 MultiPointForceMode）

        脉冲力方案（避免 xfrc_applied 累积误差）：
        - Step 1: 从 OrcaLink 订阅新力
        - Step 2: 如果没有新力，保持上一帧的力不变（SPH 侧未更新）
        - Step 3: 如果有新力，清零旧 sites 的 xfrc_applied + 施加新力

        优势：
        - 无累积误差：每次更新时从零开始设置力
        - 逻辑简单：不需要存储和计算旧力的反向力
        - 保持现状：未收到更新时，维持上一帧的力
        """
        if not self.client or not self.loop:
            logger.debug("[DEBUG] subscribe_and_apply_site_forces - client or loop not available")
            return

        try:
            # Step 1: 从 OrcaLink 订阅 SPH 发来的锚点力
            logger.debug("[DEBUG] subscribe_and_apply_site_forces - About to call subscribe_forces()...")
            forces = self.loop.run_until_complete(
                self.client.subscribe_forces()
            )

            # Step 2: 如果没有新数据，保持上一帧的力不变（SPH 侧未发送更新）
            if not forces:
                logger.debug("[DEBUG] subscribe_and_apply_site_forces - No forces received, keeping previous forces")
                return

            # Step 3: 有新数据，统计并输出日志
            site_names = [f.object_id for f in forces]
            logger.debug(f"[DEBUG] subscribe_and_apply_site_forces - Received {len(forces)} SITE forces: {site_names}")

            # Step 4: 清零上一帧施加过力的 site 对应的 body 的 xfrc_applied
            if self._previous_site_names:
                if hasattr(self.env, 'mj_clear_xfrc_applied_for_site'):
                    for site_name in self._previous_site_names:
                        self.env.mj_clear_xfrc_applied_for_site(site_name)
                    logger.debug(f"[DEBUG] Cleared xfrc_applied for {len(self._previous_site_names)} sites")
                else:
                    logger.warning("Environment does not support mj_clear_xfrc_applied_for_site")

            # Step 5: 清空旧记录，准备记录本帧的 site
            self._previous_site_names.clear()

            # Step 6: 应用所有新力（在已清零的基础上累加，等价于直接设置）
            snap: List[Tuple[str, np.ndarray]] = []
            torque_zero = np.zeros(3, dtype=np.float64)
            for force_data in forces:
                site_name = force_data.object_id
                fin = self._sanitize_force3(
                    np.array(force_data.force, dtype=np.float64), f"site[{site_name}]"
                )
                if fin is None or float(np.linalg.norm(fin)) < 1e-9:
                    continue
                self._previous_site_names.add(site_name)
                if hasattr(self.env, "mj_apply_force_at_site"):
                    self.env.mj_apply_force_at_site(site_name, fin, torque_zero)
                else:
                    logger.warning("Environment does not support mj_apply_force_at_site")
                snap.append((site_name, fin.copy()))

            self.last_applied_site_forces = snap
            self._mark_sph_force_batch_consumed(True)

            logger.debug("[DEBUG] Applied %d impulse forces to sites (after sanitize)", len(snap))

        except Exception as e:
            logger.error(f"Error applying site forces: {e}", exc_info=True)

    def apply_site_forces_from_list(self, forces: list):
        """将已获取的 SITE 力施加到 MuJoCo（用于 subscribe_forces_and_positions 合并订阅路径）

        与 subscribe_and_apply_site_forces 的区别：
        - 后者自己调用 OrcaLink 订阅力
        - 本方法接收外部已订阅好的力列表，避免二次订阅吃掉 POSITION 数据

        脉冲力方案同上：先清零旧力，再施加新力。

        Args:
            forces: RigidBodyForce 对象列表，已从 OrcaLink 获取
        """
        try:
            if not forces:
                return

            site_names = [f.object_id for f in forces]
            logger.debug(f"[DEBUG] apply_site_forces_from_list - Applying {len(forces)} SITE forces: {site_names}")

            # 清零上一帧施加过力的 body 的 xfrc_applied
            if self._previous_site_names:
                if hasattr(self.env, 'mj_clear_xfrc_applied_for_site'):
                    for site_name in self._previous_site_names:
                        self.env.mj_clear_xfrc_applied_for_site(site_name)
                    logger.debug(f"[DEBUG] Cleared xfrc_applied for {len(self._previous_site_names)} sites")

            self._previous_site_names.clear()

            # 施加新力
            snap: List[Tuple[str, np.ndarray]] = []
            torque_zero = np.zeros(3, dtype=np.float64)
            for force_data in forces:
                site_name = force_data.object_id
                fin = self._sanitize_force3(
                    np.array(force_data.force, dtype=np.float64), f"site[{site_name}]"
                )
                if fin is None or float(np.linalg.norm(fin)) < 1e-9:
                    continue
                self._previous_site_names.add(site_name)
                if hasattr(self.env, "mj_apply_force_at_site"):
                    self.env.mj_apply_force_at_site(site_name, fin, torque_zero)
                else:
                    logger.warning("Environment does not support mj_apply_force_at_site")
                snap.append((site_name, fin.copy()))

            self.last_applied_site_forces = snap
            self._mark_sph_force_batch_consumed(True)

            logger.debug("[DEBUG] Applied %d forces from list (after sanitize)", len(snap))

        except Exception as e:
            logger.error(f"Error applying site forces from list: {e}", exc_info=True)

    def _apply_force_to_body(self, force_data):
        """将力/力矩施加到刚体质心（用于 ForcePositionMode 的刚体级施力）

        通过 env.apply_force_to_body() 施加，底层操作 xfrc_applied。

        Args:
            force_data: RigidBodyForce 对象，包含 object_id、force、torque
        """
        try:
            body_name = force_data.object_id
            force_s = self._sanitize_force3(
                np.array(force_data.force, dtype=np.float64), f"body[{body_name}] F"
            )
            if force_s is None:
                return
            if hasattr(force_data, "torque") and force_data.torque is not None:
                raw_t = np.array(force_data.torque, dtype=np.float64)
                torque_s = self._sanitize_torque3(raw_t, f"body[{body_name}] τ")
                if torque_s is None:
                    torque_s = np.zeros(3, dtype=np.float64)
            else:
                torque_s = np.zeros(3, dtype=np.float64)

            if not hasattr(self.env, 'apply_force_to_body'):
                raise AttributeError(
                    f"Environment does not provide 'apply_force_to_body' method. "
                    f"Cannot apply forces to rigid bodies. "
                    f"Environment type: {type(self.env).__name__}"
                )

            self.env.apply_force_to_body(body_name, force_s, torque_s)
            logger.debug("Applied force to body '%s': F=%s, τ=%s", body_name, force_s, torque_s)

        except Exception as e:
            logger.error(f"Error applying force to body '{body_name}': {e}", exc_info=True)
            raise

    @classmethod
    def close_force_csv(cls):
        pass
