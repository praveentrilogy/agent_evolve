"""
Data Models for Generic Tracker

Defines the data structures used for tracking operations and messages.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field as dataclass_field
from enum import Enum


class OperationStatus(Enum):
    """Status of a tracked operation."""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class MessageRole(Enum):
    """Role of a message sender."""
    HUMAN = "human"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class TrackedMessage:
    """Represents a tracked message."""
    id: str
    thread_id: str
    role: str
    content: Any  # Can be string, dict, etc.
    timestamp: datetime = dataclass_field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)
    parent_id: Optional[str] = None  # For conversation threading
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "metadata": self.metadata,
            "parent_id": self.parent_id
        }


@dataclass
class TrackedOperation:
    """Represents a tracked operation/function execution."""
    id: str
    thread_id: str
    operation_name: str
    status: OperationStatus
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "operation_name": self.operation_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_ms": self.duration_ms,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class TrackingContext:
    """Context for a tracking session."""
    thread_id: str
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)
    messages: List[TrackedMessage] = dataclass_field(default_factory=list)
    operations: List[TrackedOperation] = dataclass_field(default_factory=list)