"""
Safe Auto-Tracer - Non-intrusive tracing system

This version uses context managers and explicit tracing calls instead of
automatic module instrumentation to avoid system crashes.
"""

import sqlite3
import json
import uuid
import time
import threading
import os
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from queue import Queue, Empty
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class SafeTracer:
    """Safe tracing system that doesn't instrument modules automatically."""
    
    def __init__(self, database_path: str, include_modules: Optional[List[str]] = None):
        self.database_path = database_path
        self.include_modules = include_modules or []
        self._event_queue = Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._call_stack = threading.local()
        
        # System modules to skip tracing
        self._skip_modules = {
            'functools', 'threading', 'queue', 'logging', 'sqlite3', 'json',
            'uuid', 'time', 'datetime', 'os', 'sys', 'contextlib', 'typing',
            'builtins', 'collections', 'abc', 'weakref', 're',
            'copy', 'pickle', 'traceback', 'warnings', 'operator', 'itertools'
        }
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for trace storage."""
        db_dir = os.path.dirname(self.database_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        conn = sqlite3.connect(self.database_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS trace_events (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                function_name TEXT,
                module_name TEXT,
                thread_id TEXT,
                duration_ms REAL,
                args JSON,
                result JSON,
                error TEXT,
                success BOOLEAN,
                metadata JSON
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON trace_events(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_function ON trace_events(function_name)')
        conn.commit()
        conn.close()
        
        logger.info(f"Safe trace database initialized: {self.database_path}")
    
    def _should_skip_module(self, module_name: str) -> bool:
        """Check if a module should be skipped from tracing."""
        if not module_name or module_name == 'unknown':
            return False
            
        # Skip system modules (exact match or starts with module.)
        for skip_mod in self._skip_modules:
            if module_name == skip_mod or module_name.startswith(skip_mod + '.'):
                return True
                
        # Only include specified modules if include_modules is set
        if self.include_modules:
            return not any(module_name == mod or module_name.startswith(mod + '.') for mod in self.include_modules)
            
        return False
    
    def start(self):
        """Start the background worker for database writes."""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        
        logger.info("Starting safe tracing system...")
        
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._db_worker, daemon=True)
        self._worker_thread.start()
        
        logger.info("Safe tracing enabled")
    
    def stop(self):
        """Stop the tracing system."""
        if self._worker_thread:
            self._stop_event.set()
            self._worker_thread.join(timeout=5)
        
        logger.info("Safe tracing stopped")
    
    def _serialize_safely(self, obj, max_len=1000):
        """Safely serialize an object to JSON-compatible format."""
        try:
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, (list, tuple)):
                return {
                    'type': type(obj).__name__,
                    'length': len(obj),
                    'items': [self._serialize_safely(item, max_len//10) for item in obj[:10]]  # First 10 items
                }
            elif isinstance(obj, dict):
                return {
                    'type': 'dict',
                    'length': len(obj),
                    'keys': list(obj.keys())[:10],  # First 10 keys
                    'items': {k: self._serialize_safely(v, max_len//10) for k, v in list(obj.items())[:10]}
                }
            else:
                obj_str = str(obj)
                return {
                    'type': type(obj).__name__,
                    'string_repr': obj_str[:max_len] + ('...' if len(obj_str) > max_len else '')
                }
        except Exception as e:
            return {
                'error': f'Serialization failed: {str(e)}',
                'type': type(obj).__name__ if obj is not None else 'None'
            }
    
    def trace_function_call(self, function_name: str, module_name: str = "unknown", 
                           duration_ms: float = 0, success: bool = True, 
                           error: str = None, args: Any = None, result: Any = None,
                           metadata: Dict[str, Any] = None):
        """Manually record a function call trace."""
        event = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'function_call',
            'function_name': function_name,
            'module_name': module_name,
            'thread_id': str(threading.current_thread().ident),
            'duration_ms': duration_ms,
            'args': self._serialize_safely(args),
            'result': self._serialize_safely(result),
            'error': error,
            'success': success,
            'metadata': metadata or {}
        }
        logger.info(f"SafeTracer queuing event for {function_name}")
        self._event_queue.put(event)
        logger.info(f"SafeTracer queue size now: {self._event_queue.qsize()}")
    
    @contextmanager
    def trace_context(self, function_name: str, module_name: str = "unknown", args: Any = None):
        """Context manager for tracing function execution."""
        start_time = time.time()
        error = None
        success = True
        result = None
        
        try:
            yield_result = yield
            if yield_result is not None:
                result = yield_result
        except Exception as e:
            error = str(e)
            success = False
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.trace_function_call(
                function_name=function_name,
                module_name=module_name,
                duration_ms=duration_ms,
                success=success,
                error=error,
                args=args,
                result=result
            )
    
    def trace_decorator(self, function_name: str = None, module_name: str = None):
        """Safe decorator for tracing individual functions."""
        def decorator(func: Callable) -> Callable:
            fname = function_name or func.__name__
            mname = module_name or getattr(func, '__module__', 'unknown')
            
            # Skip if this is from system modules we don't want to trace
            if self._should_skip_module(mname):
                return func
            
            def wrapper(*args, **kwargs):
                start_time = time.time()
                error = None
                success = True
                result = None
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    error = str(e)
                    success = False
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    logger.info(f"SafeTracer capturing: {fname} from {mname} (success={success})")
                    self.trace_function_call(
                        function_name=fname,
                        module_name=mname,
                        duration_ms=duration_ms,
                        success=success,
                        error=error,
                        args={'args': args, 'kwargs': kwargs},
                        result=result
                    )
            
            return wrapper
        return decorator
    
    def _db_worker(self):
        """Background worker that writes events to database."""
        batch = []
        batch_size = 50
        
        while not self._stop_event.is_set():
            try:
                # Collect events
                try:
                    event = self._event_queue.get(timeout=1.0)
                    batch.append(event)
                except Empty:
                    event = None
                
                # Write batch when full or timeout
                if len(batch) >= batch_size or (batch and not event):
                    self._write_batch(batch)
                    batch = []
                    
            except Exception as e:
                logger.error(f"Database worker error: {e}")
        
        # Write remaining events
        if batch:
            self._write_batch(batch)
    
    def _write_batch(self, events: List[Dict[str, Any]]):
        """Write a batch of events to database."""
        if not events:
            return
        
        try:
            conn = sqlite3.connect(self.database_path)
            conn.executemany('''
                INSERT INTO trace_events 
                (id, timestamp, event_type, function_name, module_name, 
                 thread_id, duration_ms, args, result, error, success, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [(
                event['id'], 
                event['timestamp'], 
                event['event_type'], 
                event['function_name'],
                event['module_name'], 
                event['thread_id'], 
                event['duration_ms'],
                json.dumps(event['args']) if event['args'] is not None else None,
                json.dumps(event['result']) if event['result'] is not None else None,
                event['error'],
                event['success'],
                json.dumps(event['metadata'])
            ) for event in events])
            conn.commit()
            conn.close()
            
            logger.info(f"Wrote {len(events)} safe trace events to database")
            
        except Exception as e:
            logger.error(f"Failed to write trace events: {e}")
    
    def get_recent_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trace events from database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.execute(
                'SELECT * FROM trace_events ORDER BY timestamp DESC LIMIT ?', 
                (limit,)
            )
            
            columns = [description[0] for description in cursor.description]
            traces = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            
            return traces
        except Exception as e:
            logger.error(f"Failed to get traces: {e}")
            return []


# Global safe tracer instance
_safe_tracer: Optional[SafeTracer] = None


def get_safe_tracer(database_path: str = "safe_traces.db") -> SafeTracer:
    """Get or create the global safe tracer."""
    global _safe_tracer
    if _safe_tracer is None:
        _safe_tracer = SafeTracer(database_path)
        _safe_tracer.start()
    return _safe_tracer


def trace_call(function_name: str, module_name: str = "unknown", **kwargs):
    """Quick function to manually trace a call."""
    tracer = get_safe_tracer()
    tracer.trace_function_call(function_name, module_name, **kwargs)


def safe_trace_decorator(function_name: str = None, module_name: str = None):
    """Safe tracing decorator."""
    tracer = get_safe_tracer()
    return tracer.trace_decorator(function_name, module_name)


# Context manager for easy use
def trace_operation(operation_name: str, module_name: str = "unknown", args: Any = None):
    """Context manager for tracing operations."""
    tracer = get_safe_tracer()
    return tracer.trace_context(operation_name, module_name, args)