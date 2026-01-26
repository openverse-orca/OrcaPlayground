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
    
    def initialize(self, config: Dict[str, Any], env, orcalink_client, loop) -> bool:
        """Initialize the mode"""
        self.env = env
        self.orcalink_client = orcalink_client
        self.loop = loop  # 直接使用传入的 loop，不再从 orcalink_client 获取
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
        """Subscribe to target positions and apply to mocap bodies using OrcaGym API"""
        if not self.orcalink_client or not self.loop:
            return
        
        try:
            import logging
            logger = logging.getLogger(__name__)
            
            # Subscribe to positions (target positions from SPH)
            positions = self.loop.run_until_complete(
                self.orcalink_client.subscribe_positions()
            )
            
            if not positions:
                logger.debug("No target positions received")
                return
            
            # Check if environment supports mocap API
            if not hasattr(self.env, 'set_mocap_pos_and_quat'):
                raise AttributeError(
                    f"Environment does not provide 'set_mocap_pos_and_quat' method. "
                    f"Cannot set mocap targets for spring constraint mode. "
                    f"Environment type: {type(self.env).__name__}"
                )
            
            # Build mocap position/quaternion dictionary for batch update
            mocap_pos_and_quat_dict = {}
            
            for pos_data in positions:
                body_name = pos_data.object_id
                pos = np.array(pos_data.position, dtype=np.float64)
                quat = np.array(pos_data.rotation, dtype=np.float64) if hasattr(pos_data, 'rotation') else np.array([1, 0, 0, 0], dtype=np.float64)
                
                # Ensure quaternion is in [w, x, y, z] format (MuJoCo format)
                if len(quat) != 4:
                    logger.warning(f"Invalid quaternion format for body '{body_name}': {quat}, skipping")
                    continue
                
                mocap_pos_and_quat_dict[body_name] = {
                    "pos": pos,
                    "quat": quat
                }
                
                logger.debug(f"Prepared mocap target for '{body_name}': pos={pos}, quat={quat}")
            
            # Apply all mocap targets in batch
            if mocap_pos_and_quat_dict:
                self.env.set_mocap_pos_and_quat(mocap_pos_and_quat_dict)
                logger.debug(f"Applied {len(mocap_pos_and_quat_dict)} mocap targets")
        
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error applying mocap targets: {e}", exc_info=True)
            raise

