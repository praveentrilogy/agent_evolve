"""
Tracker Configuration

Flexible configuration system for the generic tracker.
"""

from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass, field as dataclass_field
from .extractors import create_extractor, PathExtractor


@dataclass
class RoleConfig:
    """Configuration for role detection."""
    field: str = "role"  # Path to role field
    mapping: Dict[str, List[str]] = dataclass_field(default_factory=lambda: {
        "human": ["user", "human", "customer"],
        "assistant": ["ai", "assistant", "bot", "agent", "system"]
    })
    default: str = "unknown"


@dataclass
class TrackerConfig:
    """Main configuration for the tracker."""
    
    # Storage configuration
    storage: str = "memory"  # memory, sqlite://path, postgresql://...
    
    # Extraction configuration - can be strings, lists, or dicts
    extractors: Dict[str, Union[str, List[str], dict]] = dataclass_field(default_factory=dict)
    
    # What to track
    track_operations: bool = True
    track_messages: bool = True
    track_custom_fields: List[str] = dataclass_field(default_factory=list)
    
    # Role configuration
    role_config: Optional[RoleConfig] = None
    
    # Advanced options
    async_mode: bool = True
    batch_size: int = 100
    
    def __post_init__(self):
        """Initialize extractors and validate configuration."""
        # Set default extractors if not provided
        if not self.extractors:
            self.extractors = {
                "thread_id": [
                    "config.thread_id",
                    "config.configurable.thread_id",
                    "state.thread_id",
                    "thread_id",
                    "kwargs.thread_id",
                    "generate_from:0"  # Generate from first arg
                ],
                "user_id": [
                    "state.user_id",
                    "user_id",
                    "kwargs.user_id",
                    "state.user.id"
                ],
                "message": [
                    "message",
                    "state.messages[-1]",
                    "kwargs.message",
                    "0"  # First argument
                ],
                "role": "role"
            }
        
        # Initialize role config if not provided
        if self.role_config is None:
            self.role_config = RoleConfig()
        
        # Create PathExtractor instances
        self._extractors = {}
        for key, config in self.extractors.items():
            self._extractors[key] = create_extractor(config)
    
    def get_extractor(self, key: str) -> Optional[PathExtractor]:
        """Get a configured extractor by key."""
        return self._extractors.get(key)
    
    def extract(self, key: str, *args, **kwargs) -> Any:
        """Extract a value using the configured extractor."""
        extractor = self.get_extractor(key)
        if extractor:
            return extractor.extract(*args, **kwargs)
        return None
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrackerConfig':
        """Create configuration from dictionary."""
        # Handle role_config separately
        role_config_dict = config_dict.pop('role_config', None)
        if role_config_dict:
            role_config = RoleConfig(**role_config_dict)
        else:
            role_config = None
        
        return cls(role_config=role_config, **config_dict)
    
    @classmethod
    def from_simple(cls, 
                   thread_id: Union[str, List[str]] = None,
                   user_id: Union[str, List[str]] = None,
                   message: Union[str, List[str]] = None,
                   role: Union[str, List[str]] = None,
                   **kwargs) -> 'TrackerConfig':
        """
        Create configuration from simple parameters.
        
        Example:
            config = TrackerConfig.from_simple(
                thread_id="request.session.id",
                user_id="request.user.id",
                message="request.json.message"
            )
        """
        extractors = {}
        if thread_id:
            extractors['thread_id'] = thread_id
        if user_id:
            extractors['user_id'] = user_id
        if message:
            extractors['message'] = message
        if role:
            extractors['role'] = role
            
        return cls(extractors=extractors, **kwargs)