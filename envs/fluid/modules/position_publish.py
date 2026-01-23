"""
Module for publishing positions to SPH
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)
# 配置 logger
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('[PositionPublishModule] %(levelname)s: %(message)s'))
    logger.addHandler(handler)


class PositionPublishModule:
    """Module for publishing rigid body or SITE positions to SPH"""
    
    def __init__(self, env, orcalink_client, loop, rigid_bodies_config: List[Dict[str, Any]]):
        """
        Args:
            env: OrcaGym environment instance
            orcalink_client: OrcaLinkClient instance
            loop: Event loop for async operations
            rigid_bodies_config: List of rigid body configurations
        """
        import sys
        print(f"[PRINT-DEBUG] PositionPublishModule.__init__() - START", file=sys.stderr, flush=True)
        logger.debug("[DEBUG] PositionPublishModule.__init__() - Start")
        self.env = env
        self.client = orcalink_client
        self.loop = loop
        self.rigid_bodies = rigid_bodies_config
        print(f"[PRINT-DEBUG] PositionPublishModule.__init__() - END (rigid_bodies count: {len(rigid_bodies_config)})", file=sys.stderr, flush=True)
        logger.debug(f"[DEBUG] PositionPublishModule.__init__() - Initialized with {len(rigid_bodies_config)} rigid bodies")
    
    def publish_positions(self):
        """Publish rigid body positions (ForcePositionMode, SpringConstraintMode)"""
        if not self.client or not self.loop:
            return
        
        try:
            positions = self._collect_body_positions()
            if positions:
                self.loop.run_until_complete(
                    self.client.publish_positions(positions)
                )
        except Exception as e:
            logger.error(f"Error publishing positions: {e}", exc_info=True)
    
    def publish_site_positions(self):
        """Publish SITE positions (MultiPointForceMode)"""
        if not self.client or not self.loop:
            return
        
        try:
            positions = self._collect_site_positions()
            if positions:
                self.loop.run_until_complete(
                    self.client.publish_positions(positions)
                )
        except Exception as e:
            logger.error(f"Error publishing site positions: {e}", exc_info=True)
    
    def _collect_body_positions(self) -> List:
        """Collect rigid body positions from MuJoCo"""
        positions = []
        
        if not hasattr(self.env, 'mj_data') or not hasattr(self.env, 'mj_model'):
            return positions
        
        mj_data = self.env.mj_data
        mj_model = self.env.mj_model
        
        # Iterate through rigid bodies
        for body_config in self.rigid_bodies:
            body_name = body_config.get('mujoco_body', '')
            if not body_name:
                continue
            
            # Find body ID
            body_id = None
            for i in range(mj_model.nbody):
                if mj_model.body(i).name == body_name:
                    body_id = i
                    break
            
            if body_id is None:
                continue
            
            # Get position and rotation
            pos = mj_data.xpos[body_id].copy()
            quat = mj_data.xquat[body_id].copy()
            
            # Convert to position data structure
            try:
                from data_structures import RigidBodyPosition
                position_data = RigidBodyPosition()
                position_data.object_id = body_name
                position_data.position = pos
                position_data.rotation = quat
            except ImportError:
                # Fallback: create dict-like object
                position_data = type('RigidBodyPosition', (), {
                    'object_id': body_name,
                    'position': pos,
                    'rotation': quat
                })()
            
            positions.append(position_data)
        
        return positions
    
    def _collect_site_positions(self) -> List:
        """Collect SITE positions from MuJoCo"""
        positions = []
        
        if not hasattr(self.env, 'mj_data') or not hasattr(self.env, 'mj_model'):
            return positions
        
        mj_data = self.env.mj_data
        mj_model = self.env.mj_model
        
        # Iterate through rigid bodies and their connection points
        for body_config in self.rigid_bodies:
            connection_points = body_config.get('connection_points', [])
            
            for cp in connection_points:
                site_name = cp.get('site_name', '')
                if not site_name:
                    continue
                
                # Find site ID
                site_id = None
                for i in range(mj_model.nsite):
                    if mj_model.site(i).name == site_name:
                        site_id = i
                        break
                
                if site_id is None:
                    continue
                
                # Get site position
                pos = mj_data.site_xpos[site_id].copy()
                
                # Use identity quaternion for sites (or get from body)
                quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
                
                # Convert to position data structure
                try:
                    from data_structures import RigidBodyPosition
                    position_data = RigidBodyPosition()
                    position_data.object_id = site_name  # Use site name as object ID
                    position_data.position = pos
                    position_data.rotation = quat
                except ImportError:
                    # Fallback: create dict-like object
                    position_data = type('RigidBodyPosition', (), {
                        'object_id': site_name,
                        'position': pos,
                        'rotation': quat
                    })()
                
                positions.append(position_data)
        
        return positions

