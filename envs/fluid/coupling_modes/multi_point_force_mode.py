"""
Multi-Point Force coupling mode implementation
"""

from typing import Optional, Dict, Any
from .base import ICouplingMode
from ..modules.force_application import ForceApplicationModule
from ..modules.position_publish import PositionPublishModule


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
    
    def initialize(self, config: Dict[str, Any], env, orcalink_client) -> bool:
        """Initialize the mode"""
        self.env = env
        self.orcalink_client = orcalink_client
        self.loop = orcalink_client.loop if hasattr(orcalink_client, 'loop') else None
        self.config = config
        
        # Initialize modules
        self.force_application_module = ForceApplicationModule(env, orcalink_client, self.loop)
        self.position_publish_module = PositionPublishModule(
            env, orcalink_client, self.loop, config.get('rigid_bodies', []))
        
        return True
    
    def register_channels(self):
        """Register channels: subscribe FORCE (multi-point), publish POSITION (SITE)"""
        # Channel registration is handled by OrcaLinkClient during JoinSession
        pass
    
    def step(self) -> bool:
        """Execute one step"""
        # 1. Subscribe to multi-point forces and apply to SITE points
        if self.force_application_module:
            self.force_application_module.subscribe_and_apply_site_forces()
        
        # 2. Check flow control
        if self.orcalink_client and hasattr(self.orcalink_client, 'should_pause_this_cycle'):
            if self.orcalink_client.should_pause_this_cycle():
                return False  # Pause MuJoCo step
        
        # 3. Publish SITE positions
        if self.position_publish_module:
            self.position_publish_module.publish_site_positions()
        
        return True
    
    def shutdown(self):
        """Shutdown the mode"""
        self.force_application_module = None
        self.position_publish_module = None

