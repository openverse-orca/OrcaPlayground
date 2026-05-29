"""
Multi-Point Force coupling mode implementation
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List
from .base import ICouplingMode
from ..modules.force_application import ForceApplicationModule
from ..modules.position_publish import PositionPublishModule

logger = logging.getLogger(__name__)


class MultiPointForceMode(ICouplingMode):
    """Multi-Point Force coupling mode
    
    This mode implements multi-point force coupling:
    - MuJoCo sends SITE point positions to SPH
    - SPH decomposes fluid forces to tetrahedron anchor points
    - SPH sends decomposed forces to MuJoCo SITE points
    """
    
    def __init__(self):
        self.force_application_module: Optional[ForceApplicationModule] = None
        self.position_publish_module: Optional[PositionPublishModule] = None
        self.env = None
        self.orcalink_client = None
        self.loop = None
        self.config = {}
    
    def initialize(self, config: Dict[str, Any], env, orcalink_client, loop) -> bool:
        """Initialize the mode"""
        self.env = env
        self.orcalink_client = orcalink_client
        self.loop = loop
        self.config = config
        
        self.force_application_module = ForceApplicationModule(env, orcalink_client, self.loop)
        self.position_publish_module = PositionPublishModule(
            env, orcalink_client, self.loop, config.get('rigid_bodies', []))
        
        return True
    
    def register_channels(self):
        """Register channels: subscribe FORCE, publish POSITION (SITE)"""
        pass
    
    def step(self) -> bool:
        """Execute one step"""
        # 1. 订阅 SPH 力数据
        if self.force_application_module and self.orcalink_client and self.loop:
            try:
                self.force_application_module.subscribe_and_apply_site_forces()
            except Exception as e:
                logger.debug(f"subscribe_and_apply_site_forces failed: {e}", exc_info=True)
        
        # 2. 流控
        if self.orcalink_client and hasattr(self.orcalink_client, 'should_pause_this_cycle'):
            if self.orcalink_client.should_pause_this_cycle():
                return False
        
        # 3. 发布 SITE 位置到 SPH
        if self.position_publish_module:
            self.position_publish_module.publish_site_positions()
        
        return True
    
    def shutdown(self):
        """Shutdown the mode"""
        self.force_application_module = None
        self.position_publish_module = None

