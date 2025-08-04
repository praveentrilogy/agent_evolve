"""
Storage Implementation for Generic Tracker

Simple in-memory storage that keeps everything in memory.
"""

from typing import List, Optional, Dict
from collections import defaultdict
import asyncio
from .models import TrackedMessage, TrackedOperation


class MemoryStorage:
    """Async in-memory storage implementation."""
    
    def __init__(self):
        self.messages: List[TrackedMessage] = []
        self.operations: List[TrackedOperation] = []
        self.messages_by_thread: Dict[str, List[TrackedMessage]] = defaultdict(list)
        self.operations_by_thread: Dict[str, List[TrackedOperation]] = defaultdict(list)
        self._lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None
    
    async def initialize(self) -> None:
        """No initialization needed for memory storage."""
        pass
    
    async def close(self) -> None:
        """No cleanup needed for memory storage."""
        pass
    
    async def store_message(self, message: TrackedMessage) -> None:
        """Store a message in memory."""
        if self._lock:
            async with self._lock:
                await self._store_message_sync(message)
        else:
            await self._store_message_sync(message)
    
    async def _store_message_sync(self, message: TrackedMessage) -> None:
        """Internal sync method for storing message."""
        self.messages.append(message)
        self.messages_by_thread[message.thread_id].append(message)
    
    async def store_operation(self, operation: TrackedOperation) -> None:
        """Store an operation in memory."""
        if self._lock:
            async with self._lock:
                await self._store_operation_sync(operation)
        else:
            await self._store_operation_sync(operation)
    
    async def _store_operation_sync(self, operation: TrackedOperation) -> None:
        """Internal sync method for storing operation."""
        self.operations.append(operation)
        self.operations_by_thread[operation.thread_id].append(operation)
    
    async def get_messages(self, 
                          thread_id: Optional[str] = None,
                          user_id: Optional[str] = None,
                          limit: int = 100) -> List[TrackedMessage]:
        """Retrieve messages with filters."""
        if self._lock:
            async with self._lock:
                return await self._get_messages_sync(thread_id, user_id, limit)
        else:
            return await self._get_messages_sync(thread_id, user_id, limit)
    
    async def _get_messages_sync(self, thread_id, user_id, limit) -> List[TrackedMessage]:
        """Internal sync method for getting messages."""
        if thread_id:
            messages = self.messages_by_thread.get(thread_id, [])
        else:
            messages = self.messages
        
        if user_id:
            messages = [m for m in messages if m.user_id == user_id]
        
        return messages[-limit:]
    
    async def get_operations(self,
                            thread_id: Optional[str] = None,
                            operation_name: Optional[str] = None,
                            limit: int = 100) -> List[TrackedOperation]:
        """Retrieve operations with filters."""
        if self._lock:
            async with self._lock:
                return await self._get_operations_sync(thread_id, operation_name, limit)
        else:
            return await self._get_operations_sync(thread_id, operation_name, limit)
    
    async def _get_operations_sync(self, thread_id, operation_name, limit) -> List[TrackedOperation]:
        """Internal sync method for getting operations."""
        if thread_id:
            operations = self.operations_by_thread.get(thread_id, [])
        else:
            operations = self.operations
        
        if operation_name:
            operations = [o for o in operations if o.operation_name == operation_name]
        
        return operations[-limit:]
    
    async def batch_store_messages(self, messages: List[TrackedMessage]) -> None:
        """Store multiple messages."""
        if self._lock:
            async with self._lock:
                for message in messages:
                    await self._store_message_sync(message)
        else:
            for message in messages:
                await self._store_message_sync(message)
    
    async def batch_store_operations(self, operations: List[TrackedOperation]) -> None:
        """Store multiple operations."""
        if self._lock:
            async with self._lock:
                for operation in operations:
                    await self._store_operation_sync(operation)
        else:
            for operation in operations:
                await self._store_operation_sync(operation)


class SyncMemoryStorage:
    """Synchronous in-memory storage implementation."""
    
    def __init__(self):
        self.messages: List[TrackedMessage] = []
        self.operations: List[TrackedOperation] = []
        self.messages_by_thread: Dict[str, List[TrackedMessage]] = defaultdict(list)
        self.operations_by_thread: Dict[str, List[TrackedOperation]] = defaultdict(list)
    
    def initialize(self) -> None:
        """No initialization needed for memory storage."""
        pass
    
    def close(self) -> None:
        """No cleanup needed for memory storage."""
        pass
    
    def store_message(self, message: TrackedMessage) -> None:
        """Store a message in memory."""
        self.messages.append(message)
        self.messages_by_thread[message.thread_id].append(message)
    
    def store_operation(self, operation: TrackedOperation) -> None:
        """Store an operation in memory."""
        self.operations.append(operation)
        self.operations_by_thread[operation.thread_id].append(operation)
    
    def get_messages(self, 
                    thread_id: Optional[str] = None,
                    user_id: Optional[str] = None,
                    limit: int = 100) -> List[TrackedMessage]:
        """Retrieve messages with filters."""
        if thread_id:
            messages = self.messages_by_thread.get(thread_id, [])
        else:
            messages = self.messages
        
        if user_id:
            messages = [m for m in messages if m.user_id == user_id]
        
        return messages[-limit:]
    
    def get_operations(self,
                      thread_id: Optional[str] = None,
                      operation_name: Optional[str] = None,
                      limit: int = 100) -> List[TrackedOperation]:
        """Retrieve operations with filters."""
        if thread_id:
            operations = self.operations_by_thread.get(thread_id, [])
        else:
            operations = self.operations
        
        if operation_name:
            operations = [o for o in operations if o.operation_name == operation_name]
        
        return operations[-limit:]