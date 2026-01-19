"""
Spring Constraint coupling mode implementation
"""

import numpy as np
from typing import Optional, Dict, Any
from .base import ICouplingMode
from ..modules.position_publish import PositionPublishModule


class SpringConstraintMode(ICouplingMode):
    """Spring Constraint coupling mode
    
    This mode implements bidirectional position coupling with spring constraints:
    - Both SPH and MuJoCo send positions to each other
    - MuJoCo receives target positions from SPH and applies them to mocap bodies
    """
    
    def __init__(self):
        self.position_publish_module: Optional[PositionPublishModule] = None
        self.env = None
        self.orcalink_client = None
        self.loop = None
        self.config = {}
    
    def initialize(self, config: Dict[str, Any], env, orcalink_client) -> bool:
        """Initialize the mode"""
        self.env = env
        self.orcalink_client = orcalink_client
        self.loop = orcalink_client.loop if hasattr(orcalink_client, 'loop') else None
        self.config = config
        
        # Initialize position publish module
        self.position_publish_module = PositionPublishModule(
            env, orcalink_client, self.loop, config.get('rigid_bodies', []))
        
        return True
    
    def register_channels(self):
        """Register channels: bidirectional POSITION"""
        # Channel registration is handled by OrcaLinkClient during JoinSession
        pass
    
    def step(self) -> bool:
        """Execute one step"""
        # 1. Receive remote positions (SPH sends target positions)
        self._receive_and_apply_target_positions()
        
        # 2. Check flow control
        if self.orcalink_client and hasattr(self.orcalink_client, 'should_pause_this_cycle'):
            if self.orcalink_client.should_pause_this_cycle():
                return False  # Pause MuJoCo step
        
        # 3. Publish local positions
        if self.position_publish_module:
            self.position_publish_module.publish_positions()
        
        return True
    
    def shutdown(self):
        """Shutdown the mode"""
        self.position_publish_module = None
    
    def _receive_and_apply_target_positions(self):
        """Receive target positions from SPH and apply to mocap bodies"""
        if not self.orcalink_client or not self.loop:
            return
        
        try:
            import mujoco
            import logging
            logger = logging.getLogger(__name__)
            
            # Subscribe to positions (target positions from SPH)
            positions = self.loop.run_until_complete(
                self.orcalink_client.subscribe_positions()
            )
            
            if not positions:
                return
            
            # Check if environment has MuJoCo model/data
            if not hasattr(self.env, 'mj_model') or not hasattr(self.env, 'mj_data'):
                # Fallback: try environment-specific method
                if hasattr(self.env, 'set_mocap_pos'):
                    for pos_data in positions:
                        self.env.set_mocap_pos(pos_data.object_id, pos_data.position)
                        if hasattr(self.env, 'set_mocap_quat'):
                            self.env.set_mocap_quat(pos_data.object_id, pos_data.rotation)
                    return
                else:
                    logger.warning("Environment does not have mj_model/mj_data or set_mocap_pos method")
                    return
            
            # Apply target positions to mocap bodies using MuJoCo API
            for pos_data in positions:
                body_name = pos_data.object_id
                
                # Find body ID
                body_id = mujoco.mj_name2id(self.env.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id < 0:
                    logger.debug(f"Body '{body_name}' not found in MuJoCo model")
                    continue
                
                # Check if this body is a mocap body
                mocap_id = self.env.mj_model.body_mocapid[body_id]
                if mocap_id < 0:
                    logger.warning(f"Body '{body_name}' (id={body_id}) is not a mocap body")
                    continue
                
                # Set mocap position and quaternion
                # Convert position from SPH Y-up to MuJoCo Z-up if needed
                # SPH: [x, y, z] where y is up
                # MuJoCo: [x, y, z] where z is up
                # Conversion: MuJoCo_y = SPH_x, MuJoCo_z = SPH_y, MuJoCo_x = -SPH_z
                # Actually, positions might already be in MuJoCo format from OrcaLink
                pos = np.array(pos_data.position, dtype=np.float64)
                quat = np.array(pos_data.rotation, dtype=np.float64) if hasattr(pos_data, 'rotation') else np.array([1, 0, 0, 0], dtype=np.float64)
                
                # Ensure quaternion is in [w, x, y, z] format (MuJoCo format)
                if len(quat) == 4:
                    self.env.mj_data.mocap_pos[mocap_id] = pos
                    self.env.mj_data.mocap_quat[mocap_id] = quat
                    
                    logger.debug(f"Set mocap target for '{body_name}' (mocap_id={mocap_id}): pos={pos}, quat={quat}")
                else:
                    logger.warning(f"Invalid quaternion format for body '{body_name}': {quat}")
        except ImportError:
            import logging
            logger = logging.getLogger(__name__)
            logger.error("mujoco module not available, cannot apply target positions")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error receiving target positions: {e}", exc_info=True)

