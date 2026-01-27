"""
Module for applying forces to MuJoCo
"""

import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)
# 配置 logger
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('[ForceApplicationModule] %(levelname)s: %(message)s'))
    logger.addHandler(handler)


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
        
        # 增量式力管理：记录我们上一次对每个 site 施加的力
        # 格式：{site_name: (force_array, torque_array)}
        # 用于在下一帧中取消旧力，避免累积
        self._previous_site_forces = {}
        
        print("[PRINT-DEBUG] ForceApplicationModule.__init__() - END", file=sys.stderr, flush=True)
        logger.debug("[DEBUG] ForceApplicationModule.__init__() - Completed")
    
    def subscribe_and_apply_forces(self):
        """Subscribe to rigid body-level forces and apply (ForcePositionMode)"""
        if not self.client or not self.loop:
            return
        
        try:
            forces = self.loop.run_until_complete(
                self.client.subscribe_forces()
            )
            
            if not forces:
                return
            
            for force_data in forces:
                # Apply force to rigid body
                self._apply_force_to_body(force_data)
        except Exception as e:
            logger.error(f"Error applying forces: {e}", exc_info=True)
    
    def subscribe_and_apply_site_forces(self):
        """Subscribe to multi-point forces and apply to SITE (MultiPointForceMode)
        
        增量式力管理（不依赖系统状态）：
        1. 取消上一帧我们对每个site的力（用负值应用）
        2. 应用新力
        3. 记录新力供下一帧使用
        
        由于 mj_applyFT 是累加操作，通过 -F_old + F_new 实现力的替换。
        """
        if not self.client or not self.loop:
            logger.debug("[DEBUG] subscribe_and_apply_site_forces - client or loop not available")
            return
        
        try:
            # Step 1: 取消上一帧我们施加的力
            if self._previous_site_forces:
                for site_name, (prev_force, prev_torque) in self._previous_site_forces.items():
                    # 用负值取消旧力
                    cancel_force = -prev_force
                    cancel_torque = -prev_torque
                    
                    if hasattr(self.env, 'mj_apply_force_at_site'):
                        self.env.mj_apply_force_at_site(site_name, cancel_force, cancel_torque)
                        logger.debug(f"[DEBUG] Cancelled previous force for site '{site_name}'")
                
                logger.debug(f"[DEBUG] Cancelled {len(self._previous_site_forces)} previous site forces")
            
            # Step 2: 订阅新力
            logger.debug("[DEBUG] subscribe_and_apply_site_forces - About to call subscribe_forces()...")
            forces = self.loop.run_until_complete(
                self.client.subscribe_forces()
            )
            logger.debug(f"[DEBUG] subscribe_and_apply_site_forces - subscribe_forces() returned {len(forces) if forces else 0} forces")
            
            # Step 3: 清空旧记录
            self._previous_site_forces.clear()
            
            if not forces:
                logger.debug("[DEBUG] subscribe_and_apply_site_forces - No forces received")
                return
            
            # Step 4: 应用所有新力并记录
            for force_data in forces:
                site_name = force_data.object_id  # SITE point ID
                
                # 直接使用接收到的力（已经是MuJoCo Z-up坐标系）
                # 坐标转换已经在C++的GrpcDataMapper中完成
                force_mujoco = np.array(force_data.force, dtype=np.float64)
                torque_mujoco = np.zeros(3, dtype=np.float64)
                
                # Apply force at SITE point (累加到 qfrc_applied)
                if hasattr(self.env, 'mj_apply_force_at_site'):
                    self.env.mj_apply_force_at_site(site_name, force_mujoco, torque_mujoco)
                    
                    # 记录这次施加的力，供下一帧取消
                    self._previous_site_forces[site_name] = (force_mujoco.copy(), torque_mujoco.copy())
                else:
                    logger.warning(f"Environment does not support mj_apply_force_at_site")
            
            logger.debug(f"[DEBUG] Applied {len(forces)} new forces to sites")
            
        except Exception as e:
            logger.error(f"Error applying site forces: {e}", exc_info=True)
    
    def _apply_force_to_body(self, force_data):
        """Apply force to a rigid body using OrcaGym API"""
        try:
            body_name = force_data.object_id
            force = np.array(force_data.force, dtype=np.float64)
            torque = np.array(force_data.torque, dtype=np.float64) if hasattr(force_data, 'torque') else np.zeros(3)
            
            # Apply force using OrcaGym API (required)
            if not hasattr(self.env, 'apply_force_to_body'):
                raise AttributeError(
                    f"Environment does not provide 'apply_force_to_body' method. "
                    f"Cannot apply forces to rigid bodies. "
                    f"Environment type: {type(self.env).__name__}"
                )
            
            self.env.apply_force_to_body(body_name, force, torque)
            logger.debug(f"Applied force to body '{body_name}': F={force}, τ={torque}")
            
        except Exception as e:
            logger.error(f"Error applying force to body '{body_name}': {e}", exc_info=True)
            raise

