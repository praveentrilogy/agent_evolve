"""
Generic Track Decorator

Main decorator for tracking function executions with flexible configuration.
"""

import functools
import inspect
import time
import uuid
from typing import Callable, Optional, Union, Dict, Any
from datetime import datetime
import logging

from .config import TrackerConfig
from .tracker import get_tracker, Tracker
from .models import OperationStatus, MessageRole
from .context import (
    OperationContext, get_current_operation, push_operation, 
    pop_operation, get_operation_depth, get_operation_chain
)

logger = logging.getLogger(__name__)


def track(
    config: Optional[Union[TrackerConfig, Dict[str, Any]]] = None,
    tracker: Optional[Tracker] = None,
    # Shorthand parameters for simple configuration
    thread_id: Optional[Union[str, list]] = None,
    user_id: Optional[Union[str, list]] = None,
    message: Optional[Union[str, list]] = None,
    role: Optional[Union[str, list]] = None,
    operation_name: Optional[str] = None,
    track_messages: bool = True,
    track_operations: bool = True,
):
    """
    Decorator to track function executions.
    
    Args:
        config: TrackerConfig instance or dict to create one
        tracker: Specific Tracker instance to use (otherwise uses global)
        thread_id: Path(s) to extract thread ID
        user_id: Path(s) to extract user ID
        message: Path(s) to extract message content
        role: Path(s) to extract role
        operation_name: Name for the operation (defaults to function name)
        track_messages: Whether to track messages
        track_operations: Whether to track operations
        
    Examples:
        # Using full configuration
        @track(config=TrackerConfig(
            extractors={
                "thread_id": "state.config.thread_id",
                "user_id": "state.user.id"
            }
        ))
        def process(state):
            pass
            
        # Using shorthand
        @track(
            thread_id="request.session.id",
            user_id="request.user.id",
            message="request.json.message"
        )
        def api_endpoint(request):
            pass
            
        # Using existing tracker
        my_tracker = Tracker(my_config)
        @track(tracker=my_tracker)
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        # Create configuration
        if config is not None:
            if isinstance(config, dict):
                actual_config = TrackerConfig.from_dict(config)
            else:
                actual_config = config
        elif any([thread_id, user_id, message, role]):
            # Create from shorthand parameters
            actual_config = TrackerConfig.from_simple(
                thread_id=thread_id,
                user_id=user_id,
                message=message,
                role=role
            )
        else:
            # Use default configuration
            actual_config = TrackerConfig()
        
        # Override tracking options if specified
        if not track_messages:
            actual_config.track_messages = False
        if not track_operations:
            actual_config.track_operations = False
        
        # Get tracker instance
        actual_tracker = tracker or get_tracker(actual_config)
        
        # Get operation name
        op_name = operation_name or func.__name__
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            """Synchronous wrapper."""
            logger.info(f"🔥 TRACK DECORATOR CALLED for {func.__name__}")
            start_time = time.time()
            operation_id = None
            context = {}
            
            try:
                # Extract tracking context manually since LangGraph doesn't pass config to nodes
                state = args[0] if args else {}
                
                # Try to get thread_id from LangGraph context using multiple methods
                thread_id = 'unknown'
                try:
                    # Method 1: Try LangGraph context
                    from langgraph.context import get_langgraph_context
                    lg_context = get_langgraph_context()
                    if isinstance(lg_context, dict):
                        thread_id = lg_context.get('thread_id') or lg_context.get('configurable', {}).get('thread_id', 'unknown')
                except Exception:
                    pass
                
                if thread_id == 'unknown':
                    # Method 2: Generate a stable thread_id based on conversation, not function
                    import hashlib
                    # Use the first message content to generate a consistent thread_id for the entire conversation
                    if isinstance(state, dict) and "messages" in state and state["messages"]:
                        first_msg = state["messages"][0]
                        conversation_seed = getattr(first_msg, 'content', 'no-content')[:50]  # First 50 chars
                        thread_id = hashlib.md5(conversation_seed.encode()).hexdigest()[:8]
                    else:
                        thread_id = "no-thread"
                
                # Extract message and role from state
                message_content = "no-message"
                role = "assistant"
                if isinstance(state, dict) and "messages" in state and state["messages"]:
                    last_msg = state["messages"][-1]
                    message_content = getattr(last_msg, 'content', 'no-content')
                    role = getattr(last_msg, 'type', 'unknown')
                
                # Create context manually
                context = {
                    'thread_id': thread_id,
                    'user_id': None,  # Not available in LangGraph state
                    'message': message_content,
                    'role': role
                }
                
                print(f"🔍 Manual context: {context}")
                
                # Only track messages from specific nodes to avoid duplicates
                # Track input message only for orchestrator_node, track output for function-specific nodes
                should_track_message = False
                message_to_track = message_content
                role_to_track = role
                
                if func.__name__ == "orchestrator_node":
                    # Track the input message (user message) only once
                    should_track_message = True
                    print(f"🔗 Will track user message from {func.__name__}")
                elif func.__name__ == "chatbot_node":
                    # For chatbot, track the assistant response after function completes
                    should_track_message = False  # We'll track after execution
                    print(f"🤖 Will track assistant response after {func.__name__} completes")
                else:
                    print(f"⚠️ No message tracking for {func.__name__}")
                
                # Get parent operation context
                parent_op = get_current_operation()
                parent_id = parent_op.operation_id if parent_op else None
                depth = get_operation_depth()
                
                # Use the manually extracted thread_id
                thread_id = context.get('thread_id', 'unknown')
                
                # Create operation record (will be updated on completion)
                if actual_config.track_operations:
                    logger.info(f"📝 Creating operation record for {op_name}")
                    import uuid
                    operation_id = str(uuid.uuid4())
                    # We'll create the full record after execution
                    
                    # Push operation context
                    from datetime import datetime
                    op_context = OperationContext(
                        operation_id=operation_id,
                        operation_name=op_name,
                        thread_id=thread_id,
                        started_at=datetime.utcnow(),
                        parent_id=parent_id,
                        depth=depth
                    )
                    push_operation(op_context)
                
                # Track incoming message only for orchestrator to avoid duplicates
                last_user_message_id = None
                if should_track_message and actual_config.track_messages:
                    # Store the message ID for linking responses
                    msg_id = actual_tracker.track_message(
                        content=message_to_track,
                        role=role_to_track,
                        thread_id=thread_id,
                        user_id=context.get('user_id'),
                        metadata={"source": op_name}
                    )
                    print(f"🔗 Tracked user message with ID: {msg_id}")
                    # Store the user message ID in context for later use
                    context['last_user_message_id'] = msg_id
                    
                    # Also store in operation context for cross-node access
                    if operation_id:
                        from .context import set_context_value
                        set_context_value('last_user_message_id', msg_id)
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Track assistant output for specific nodes
                print(f"🤖 Checking assistant output: func={func.__name__}, track_messages={actual_config.track_messages}, has_result={bool(result)}")
                if actual_config.track_messages and func.__name__ == "chatbot_node" and result:
                    output_message = None
                    output_role = 'assistant'
                    print(f"🤖 Processing chatbot result: {type(result)}")
                    
                    # For LangGraph nodes, result is typically a dict with messages
                    if isinstance(result, dict):
                        print(f"🤖 Result keys: {list(result.keys())}")
                        # Look for messages in the result
                        if 'messages' in result and result['messages']:
                            # Get the last message which should be the assistant response
                            last_msg = result['messages'][-1]
                            print(f"🤖 Last message type: {type(last_msg)}, content preview: {str(last_msg)[:100]}...")
                            if hasattr(last_msg, 'content'):
                                output_message = last_msg.content
                                print(f"🤖 Extracted content: {output_message[:100]}...")
                        else:
                            # Look for other common response fields
                            for field in ['response', 'output', 'content']:
                                if field in result:
                                    output_message = result[field]
                                    print(f"🤖 Found output in {field}: {output_message[:100]}...")
                                    break
                    elif isinstance(result, str):
                        output_message = result
                        print(f"🤖 String result: {output_message[:100]}...")
                    
                    if output_message:
                        # Find the last user message ID to link the response
                        parent_message_id = context.get('last_user_message_id')
                        
                        # Try operation context if not in local context
                        if not parent_message_id:
                            try:
                                from .context import get_context_value
                                parent_message_id = get_context_value('last_user_message_id')
                            except Exception:
                                pass
                        
                        # If still not found, query the database directly
                        if not parent_message_id:
                            try:
                                # Access storage directly for sync operation
                                import sqlite3
                                db_path = actual_tracker._storage.db_path
                                conn = sqlite3.connect(db_path)
                                try:
                                    cursor = conn.execute(
                                        "SELECT id FROM tracked_messages WHERE thread_id = ? AND role = 'human' ORDER BY timestamp DESC LIMIT 1",
                                        (thread_id,)
                                    )
                                    row = cursor.fetchone()
                                    if row:
                                        parent_message_id = row[0]
                                        print(f"🔍 Found parent message ID from DB: {parent_message_id}")
                                finally:
                                    conn.close()
                            except Exception as e:
                                print(f"❌ Failed to find parent message: {e}")
                                pass
                        
                        print(f"🔗 Linking assistant response to parent: {parent_message_id}")
                        
                        actual_tracker.track_message(
                            content=output_message,
                            role=output_role,
                            thread_id=thread_id,
                            user_id=context.get('user_id'),
                            parent_id=parent_message_id,
                            metadata={"source": op_name}
                        )
                
                # Create complete operation record
                if actual_config.track_operations and operation_id:
                    duration_ms = (time.time() - start_time) * 1000
                    from .models import TrackedOperation
                    from datetime import datetime
                    
                    # Create complete operation record with all data
                    operation = TrackedOperation(
                        id=operation_id,
                        thread_id=thread_id,
                        operation_name=op_name,
                        status=OperationStatus.COMPLETED,
                        started_at=datetime.fromtimestamp(start_time),
                        ended_at=datetime.utcnow(),
                        duration_ms=duration_ms,
                        input_data=_safe_serialize_args(args, kwargs),
                        output_data=_safe_serialize_result(result),
                        metadata={
                            "function": func.__name__,
                            "parent_operation_id": parent_id,
                            "depth": depth,
                            "call_chain": get_operation_chain()
                        }
                    )
                    
                    # Store the complete operation
                    if actual_tracker.config.async_mode:
                        actual_tracker._queue.put({'type': 'operation', 'data': operation})
                    else:
                        actual_tracker._storage.store_operation(operation)
                    
                    # Pop operation context
                    pop_operation()
                
                return result
                
            except Exception as e:
                # Check if it's an interruption (for LangGraph compatibility)
                if "NodeInterrupt" in str(type(e).__name__):
                    status = OperationStatus.INTERRUPTED
                else:
                    status = OperationStatus.FAILED
                
                # Create failed operation record
                if actual_config.track_operations and 'operation_id' in locals() and operation_id:
                    duration_ms = (time.time() - start_time) * 1000
                    from .models import TrackedOperation
                    from datetime import datetime
                    
                    # Create failed operation record
                    operation = TrackedOperation(
                        id=operation_id,
                        thread_id=context.get('thread_id', 'unknown'),
                        operation_name=op_name,
                        status=status,
                        started_at=datetime.fromtimestamp(start_time),
                        ended_at=datetime.utcnow(),
                        duration_ms=duration_ms,
                        input_data=_safe_serialize_args(args, kwargs),
                        error=str(e),
                        metadata={
                            "function": func.__name__,
                            "error_type": type(e).__name__,
                            "parent_operation_id": parent_id if 'parent_id' in locals() else None,
                            "depth": depth if 'depth' in locals() else 0
                        }
                    )
                    
                    # Store the failed operation
                    if actual_tracker.config.async_mode:
                        actual_tracker._queue.put({'type': 'operation', 'data': operation})
                    else:
                        actual_tracker._storage.store_operation(operation)
                    
                    # Pop operation context on error
                    if 'operation_id' in locals() and operation_id:
                        pop_operation()
                        
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Asynchronous wrapper."""
            # TODO: Implement async version similar to sync_wrapper
            return await func(*args, **kwargs)
        
        # Return appropriate wrapper
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def _safe_serialize_args(args, kwargs, max_length=1000):
    """Safely serialize function arguments for storage."""
    try:
        # Simple serialization - can be enhanced
        serialized = {
            "args": str(args)[:max_length] if args else None,
            "kwargs": str(kwargs)[:max_length] if kwargs else None
        }
        return serialized
    except:
        return {"error": "Could not serialize arguments"}


def _safe_serialize_result(result, max_length=1000):
    """Safely serialize function result for storage."""
    try:
        if result is None:
            return None
        elif isinstance(result, (str, int, float, bool)):
            return {"value": result}
        elif isinstance(result, dict):
            # Truncate large dicts
            return {k: str(v)[:100] for k, v in list(result.items())[:10]}
        else:
            return {"type": type(result).__name__, "str": str(result)[:max_length]}
    except:
        return {"error": "Could not serialize result"}


# Convenience function for LangGraph compatibility
def track_node(**kwargs):
    """
    Convenience decorator for LangGraph nodes.
    
    Automatically configures common LangGraph extraction paths.
    """
    print(f"🚀 track_node called with kwargs: {kwargs}")
    
    # Use configured database path
    from ..config import get_db_path
    db_path = get_db_path()
    
    # Create TrackerConfig with SQLite storage
    from .config import TrackerConfig
    
    tracker_config = TrackerConfig(
        storage=f"sqlite://{db_path}",
        extractors={
            "thread_id": [
                "0.config.configurable.thread_id",  # From state.config.configurable.thread_id
                "generate_from:0.messages[0].content"  # Fallback from first message
            ],
            "user_id": [
                "0.user_id",  # From state.user_id
                "0.config.configurable.user_id"  # From state.config.configurable.user_id
            ],
            "message": [
                "0.messages[-1].content",  # Last message content
                "0.messages[-1]"  # Full last message object
            ],
            "role": [
                "0.messages[-1].type",  # Last message type
                "assistant"  # Default fallback
            ]
        }
    )
    
    # Override with any provided kwargs
    if kwargs:
        for key, value in kwargs.items():
            if hasattr(tracker_config, key):
                setattr(tracker_config, key, value)
            elif key in ['thread_id', 'user_id', 'message', 'role']:
                tracker_config.extractors[key] = value
    
    return track(config=tracker_config)