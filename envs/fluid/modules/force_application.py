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
        """Subscribe to multi-point forces and apply to SITE (MultiPointForceMode)"""
        if not self.client or not self.loop:
            logger.debug("[DEBUG] subscribe_and_apply_site_forces - client or loop not available")
            return
        
        try:
            logger.debug("[DEBUG] subscribe_and_apply_site_forces - About to call subscribe_forces()...")
            forces = self.loop.run_until_complete(
                self.client.subscribe_forces()
            )
            logger.debug(f"[DEBUG] subscribe_and_apply_site_forces - subscribe_forces() returned {len(forces) if forces else 0} forces")
            
            if not forces:
                logger.debug("[DEBUG] subscribe_and_apply_site_forces - No forces received")
                return
            
            for force_data in forces:
                # Apply force to SITE point
                site_name = force_data.object_id  # SITE point ID
                
                # Convert force from SPH Y-up to MuJoCo Z-up if needed
                force_mujoco = np.array([
                    force_data.force[0],   # fx
                    force_data.force[2],   # fy
                    -force_data.force[1]   # fz
                ], dtype=np.float64)
                
                # Zero torque (handled by MuJoCo)
                torque_mujoco = np.zeros(3, dtype=np.float64)
                
                # Apply force at SITE point
                if hasattr(self.env, 'mj_apply_force_at_site'):
                    self.env.mj_apply_force_at_site(site_name, force_mujoco, torque_mujoco)
                else:
                    logger.warning(f"Environment does not support mj_apply_force_at_site")
            
            logger.debug(f"[DEBUG] subscribe_and_apply_site_forces - Applied {len(forces)} forces successfully")
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

