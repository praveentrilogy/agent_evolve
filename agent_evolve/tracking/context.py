"""
Execution Context for Tracking Operation Sequences

Provides context management for tracking nested operations and call chains.
"""

from contextvars import ContextVar
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class OperationContext:
    """Context for a single operation in the call chain."""
    operation_id: str
    operation_name: str
    thread_id: str
    started_at: datetime
    parent_id: Optional[str] = None
    depth: int = 0


# Context variable to track current operation stack
_operation_stack: ContextVar[List[OperationContext]] = ContextVar('operation_stack', default=[])


def get_current_operation() -> Optional[OperationContext]:
    """Get the current operation context."""
    stack = _operation_stack.get()
    return stack[-1] if stack else None


def get_operation_stack() -> List[OperationContext]:
    """Get the full operation stack."""
    return _operation_stack.get().copy()


def push_operation(context: OperationContext):
    """Push a new operation onto the stack."""
    stack = _operation_stack.get().copy()
    stack.append(context)
    _operation_stack.set(stack)


def pop_operation() -> Optional[OperationContext]:
    """Pop the current operation from the stack."""
    stack = _operation_stack.get().copy()
    if stack:
        context = stack.pop()
        _operation_stack.set(stack)
        return context
    return None


def get_operation_depth() -> int:
    """Get the current operation depth (nesting level)."""
    return len(_operation_stack.get())


def get_operation_chain() -> List[str]:
    """Get the chain of operation names from root to current."""
    return [op.operation_name for op in _operation_stack.get()]


# Additional context variables for cross-operation data sharing
_context_data: ContextVar[dict] = ContextVar('context_data', default={})


def set_context_value(key: str, value):
    """Set a value in the operation context."""
    data = _context_data.get().copy()
    data[key] = value
    _context_data.set(data)


def get_context_value(key: str, default=None):
    """Get a value from the operation context."""
    return _context_data.get().get(key, default)