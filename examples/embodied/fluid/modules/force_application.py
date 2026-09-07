"""
Module for applying forces to MuJoCo
"""

import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ForceApplicationModule:
    """Module for applying forces to MuJoCo rigid bodies or SITE points"""
    
    def __init__(self, env, orcalink_client, loop):
        """
        Args:
            env: OrcaGym environment instance
            orcalink_client: OrcaLinkClient instance
            loop: Event loop for async operations
        """
        import sys
        print("[PRINT-DEBUG] ForceApplicationModule.__init__() - START", file=sys.stderr, flush=True)
        logger.debug("[DEBUG] ForceApplicationModule.__init__() - Start")
        self.env = env
        self.client = orcalink_client
        self.loop = loop
        
        # 脉冲力方案：只需记录上一帧施加过力的 site 名称
        # 用于在下一帧开始时清零这些 site 对应的 body 的外力
        self._previous_site_names = set()
        self._warned_no_body_force_api = False
        self._body_name_map = None
        
        print("[PRINT-DEBUG] ForceApplicationModule.__init__() - END", file=sys.stderr, flush=True)
        logger.debug("[DEBUG] ForceApplicationModule.__init__() - Completed")
    
    def subscribe_and_apply_forces(self, macro_step: int = 0):
        """Subscribe to rigid body-level forces and apply (ForcePositionMode)"""
        if not self.client or not self.loop:
            return
        
        try:
            forces = self.loop.run_until_complete(
                self.client.subscribe_forces()
            )
            
            if not forces:
                return

            force_seq = int(getattr(self.client, "subscribe_sequence", macro_step))
            applied_flags = {}
            for force_data in forces:
                applied_flags[force_data.object_id] = self._apply_force_to_body(force_data)

            try:
                from ..debug.force_position_debug_trace import get_active_trace

                trace = get_active_trace()
                if trace is not None and trace.should_log_cp5(force_seq):
                    trace.log_cp5_after_force_apply(
                        force_seq, macro_step, forces, applied_flags
                    )
            except ImportError:
                pass
        except Exception as e:
            logger.error(f"Error applying forces: {e}", exc_info=True)
    
    def subscribe_and_apply_site_forces(self):
        """Subscribe to multi-point forces and apply to SITE (MultiPointForceMode)
        
        脉冲力方案（避免累积误差）：
        - Step 1: 订阅新力
        - Step 2: 如果没有新力，保持上一帧的力不变（SPH 侧未更新）
        - Step 3: 如果有新力，清零旧 sites + 应用新 sites 的力
        
        优势：
        - 无累积误差：每次更新时从零开始设置力
        - 逻辑简单：不需要存储和计算旧力的反向力
        - 保持现状：未收到更新时，维持上一帧的力
        """
        if not self.client or not self.loop:
            logger.debug("[DEBUG] subscribe_and_apply_site_forces - client or loop not available")
            return
        
        try:
            # Step 1: 订阅新力
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

            # Step 4: 清零上一帧施加过力的 site 对应的 body
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
            for force_data in forces:
                site_name = force_data.object_id  # SITE point ID
                
                # 直接使用接收到的力（已经是MuJoCo Z-up坐标系）
                # 坐标转换已经在C++的GrpcDataMapper中完成
                force_mujoco = np.array(force_data.force, dtype=np.float64)
                torque_mujoco = np.zeros(3, dtype=np.float64)

                # 防御：SPH 可能回传 inf/nan 力（已知根因：OrcaSPH AccelerationForceInference
                # 的 AccumulatedData 累加器未初始化，对某些 body 持续吐 nan）。这种力会让 MuJoCo
                # 第一步就 QACC NaN、触发 mj 自动重置，导致仿真时间被反复打回（data->time 封顶）。
                # 跳过非有限力，避免单个坏帧炸掉整个仿真。每个坏 site 只告警一次，防止刷屏。
                if not np.all(np.isfinite(force_mujoco)):
                    if not hasattr(self, "_warned_nonfinite_sites"):
                        self._warned_nonfinite_sites = set()
                    if site_name not in self._warned_nonfinite_sites:
                        self._warned_nonfinite_sites.add(site_name)
                        logger.warning(
                            f"SPH returned non-finite force {force_data.force} for site "
                            f"'{site_name}'; skipping its forces (further occurrences silenced)"
                        )
                    continue

                # 记录 site 名称（下次更新时需要清零）
                self._previous_site_names.add(site_name)
                
                # 性能优化：如果是0值力，跳过 mj_applyFT 调用（已经清零了）
                force_norm = np.linalg.norm(force_mujoco)
                torque_norm = np.linalg.norm(torque_mujoco)
                if force_norm < 1e-9 and torque_norm < 1e-9:
                    continue  # 0 值力，已经清零，不需要应用
                
                if hasattr(self.env, 'mj_apply_force_at_site'):
                    self.env.mj_apply_force_at_site(site_name, force_mujoco, torque_mujoco)
                else:
                    logger.warning(f"Environment does not support mj_apply_force_at_site")
            
            logger.debug(f"[DEBUG] Applied {len(forces)} impulse forces to sites")
            
        except Exception as e:
            logger.error(f"Error applying site forces: {e}", exc_info=True)
    
    def _gym_model(self):
        """取当前流体环境里的 OrcaGym 模型。可能包了一层 unwrapped。"""
        env = self.env
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        return getattr(env, "model", None)

    def _apply_force_to_body(self, force_data) -> bool:
        """Apply force to a rigid body using OrcaGym API. Returns True if written to sim memory."""
        object_id = force_data.object_id
        body_name = object_id
        try:
            model = self._gym_model()
            if model is not None:
                if self._body_name_map is None:
                    from ..utils.body_name_map import FluidBodyNameMap
                    self._body_name_map = FluidBodyNameMap(model)
                body_name = self._body_name_map.mujoco_body_name(object_id)
            force = np.array(force_data.force, dtype=np.float64)
            torque = (
                np.array(force_data.torque, dtype=np.float64)
                if hasattr(force_data, "torque")
                else np.zeros(3)
            )

            if not hasattr(self.env, "apply_force_to_body"):
                if not self._warned_no_body_force_api:
                    logger.warning(
                        "Environment has no apply_force_to_body (%s); "
                        "skipping OrcaLink rigid-body force application.",
                        type(self.env).__name__,
                    )
                    self._warned_no_body_force_api = True
                return False

            self.env.apply_force_to_body(body_name, force, torque)
            logger.debug(
                f"Applied force to body '{body_name}' (object_id='{object_id}'): "
                f"F={force}, τ={torque}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error applying force to body '{body_name}' "
                f"(object_id='{object_id}'): {e}",
                exc_info=True,
            )
            return False

