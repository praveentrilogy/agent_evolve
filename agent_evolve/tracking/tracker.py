"""
Core Tracker Implementation

Main tracker class that coordinates extraction, storage, and tracking logic.
"""

import uuid
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import logging
from queue import Queue
from threading import Thread, Event

from .config import TrackerConfig
from .models import TrackedMessage, TrackedOperation, OperationStatus
from .storage import MemoryStorage, SyncMemoryStorage
from .sqlite_storage import SQLiteStorage

logger = logging.getLogger(__name__)


class Tracker:
    """Main tracker class that handles all tracking operations."""
    
    def __init__(self, config: TrackerConfig = None):
        """
        Initialize tracker with configuration.
        
        Args:
            config: TrackerConfig instance or None for defaults
        """
        self.config = config or TrackerConfig()
        self._storage = self._create_storage()
        self._initialized = False
        
        # For async mode, we use a queue and background thread
        if self.config.async_mode:
            self._queue = Queue()
            self._stop_event = Event()
            self._worker_thread = Thread(target=self._process_queue, daemon=True)
            self._worker_thread.start()
    
    def _create_storage(self):
        """Create storage instance based on configuration."""
        storage_type = self.config.storage.lower()
        
        if storage_type == "memory":
            return SyncMemoryStorage() if not self.config.async_mode else MemoryStorage()
        elif storage_type.startswith("sqlite://"):
            db_path = storage_type.replace("sqlite://", "")
            return SQLiteStorage(db_path)
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")
    
    def initialize(self):
        """Initialize the tracker and storage."""
        if not self._initialized:
            if hasattr(self._storage, 'initialize'):
                if hasattr(self._storage.initialize, '__call__'):
                    try:
                        import asyncio
                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(self._storage.initialize())
                        loop.close()
                    except:
                        self._storage.initialize()
            self._initialized = True
    
    def close(self):
        """Close the tracker and cleanup resources."""
        if self.config.async_mode and hasattr(self, '_stop_event'):
            self._stop_event.set()
            if hasattr(self, '_worker_thread'):
                self._worker_thread.join(timeout=5)
        
        if hasattr(self._storage, 'close'):
            if hasattr(self._storage.close, '__call__'):
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self._storage.close())
                    loop.close()
                except:
                    self._storage.close()
    
    def _process_queue(self):
        """Background thread to process queued items."""
        while not self._stop_event.is_set():
            try:
                items = []
                deadline = datetime.utcnow().timestamp() + 1  # 1 second batch window
                
                while len(items) < self.config.batch_size:
                    remaining = deadline - datetime.utcnow().timestamp()
                    if remaining <= 0:
                        break
                    
                    try:
                        item = self._queue.get(timeout=remaining)
                        items.append(item)
                    except:
                        break
                
                if items:
                    self._process_batch(items)
                    
            except Exception as e:
                logger.error(f"Error processing queue: {e}")
    
    def _process_batch(self, items: List[Dict[str, Any]]):
        """Process a batch of items."""
        messages = []
        operations = []
        
        for item in items:
            if item['type'] == 'message':
                messages.append(item['data'])
            elif item['type'] == 'operation':
                operations.append(item['data'])
        
        # Store in batches
        for msg in messages:
            self._storage.store_message(msg)
        for op in operations:
            self._storage.store_operation(op)
    
    def track_message(self, 
                     content: Any,
                     role: str,
                     thread_id: str,
                     user_id: Optional[str] = None,
                     parent_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Track a message.
        
        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        message = TrackedMessage(
            id=message_id,
            thread_id=thread_id,
            role=role,
            content=content,
            user_id=user_id,
            parent_id=parent_id,
            metadata=metadata or {}
        )
        
        if self.config.async_mode:
            self._queue.put({'type': 'message', 'data': message})
        else:
            self._storage.store_message(message)
        
        return message_id
    
    def track_operation(self,
                       operation_name: str,
                       thread_id: str,
                       status: OperationStatus = OperationStatus.STARTED,
                       input_data: Optional[Dict[str, Any]] = None,
                       output_data: Optional[Dict[str, Any]] = None,
                       error: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Track an operation.
        
        Returns:
            Operation ID
        """
        operation_id = str(uuid.uuid4())
        operation = TrackedOperation(
            id=operation_id,
            thread_id=thread_id,
            operation_name=operation_name,
            status=status,
            started_at=datetime.utcnow(),
            input_data=input_data,
            output_data=output_data,
            error=error,
            metadata=metadata or {}
        )
        
        if self.config.async_mode:
            self._queue.put({'type': 'operation', 'data': operation})
        else:
            self._storage.store_operation(operation)
        
        return operation_id
    
    def extract_and_track(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Extract values using configured extractors and return tracking context.
        
        This is used by the decorator to extract values from function arguments.
        """
        context = {}
        
        # Extract configured values
        for key in ['thread_id', 'user_id', 'message', 'role']:
            value = self.config.extract(key, *args, **kwargs)
            if value is not None:
                context[key] = value
        
        # Extract custom fields
        for field_path in self.config.track_custom_fields:
            extractor = self.config._extractors.get(field_path)
            if not extractor:
                from .extractors import PathExtractor
                extractor = PathExtractor(field_path)
            value = extractor.extract(*args, **kwargs)
            if value is not None:
                context[f'custom_{field_path}'] = value
        
        return context
    
    def detect_role(self, value: Any, context: Dict[str, Any]) -> str:
        """Detect role from value using role configuration."""
        role_config = self.config.role_config
        
        # Try to extract role field
        if isinstance(value, dict):
            from .extractors import NestedExtractor
            role_value = NestedExtractor.extract(value, role_config.field)
            
            if role_value:
                role_str = str(role_value).lower()
                for role, patterns in role_config.mapping.items():
                    if role_str in [p.lower() for p in patterns]:
                        return role
        
        return role_config.default


# Global tracker instance
_global_tracker: Optional[Tracker] = None


def get_tracker(config: Optional[TrackerConfig] = None) -> Tracker:
    """Get or create global tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = Tracker(config)
        _global_tracker.initialize()
    return _global_tracker


def set_tracker(tracker: Tracker):
    """Set global tracker instance."""
    global _global_tracker
    _global_tracker = tracker