"""
Auto-Tracing System - Bootstrap function for automatic operation tracing

Provides the enable_auto_tracing() function that can be called once during
application startup to automatically instrument all operations and ship
traces to a database.
"""

import sys
import inspect
import functools
import threading
import time
import sqlite3
import json
import uuid
import os
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
import logging

logger = logging.getLogger(__name__)


class AutoTracer:
    """Automatic tracing system that instruments all function calls."""
    
    def __init__(self, database_path: str, include_modules: Optional[List[str]] = None, 
                 exclude_modules: Optional[List[str]] = None):
        self.database_path = database_path
        self.include_modules = include_modules or []
        self.exclude_modules = exclude_modules or [
            'sys', 'os', 'logging', 'threading', 'queue', 're', 'json',
            'datetime', 'uuid', 'pathlib', 'importlib', 'traceback',
            '__builtin__', 'builtins', '_frozen_importlib', 'agent_evolve'
        ]
        
        self._original_import = __builtins__.__import__
        self._instrumented_modules = set()
        self._event_queue = Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._call_stack = threading.local()
        
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
                filename TEXT,
                line_number INTEGER,
                thread_id TEXT,
                parent_trace_id TEXT,
                duration_ms REAL,
                args TEXT,
                result TEXT,
                error TEXT,
                metadata TEXT
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON trace_events(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_function ON trace_events(function_name)')
        conn.commit()
        conn.close()
        
        logger.info(f"Trace database initialized: {self.database_path}")
    
    def start(self):
        """Start the auto-tracing system."""
        logger.info("Starting auto-tracing system...")
        
        # Install import hook
        __builtins__.__import__ = self._traced_import
        
        # Start background worker for database writes
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._db_worker, daemon=True)
        self._worker_thread.start()
        
        # Instrument already loaded modules
        self._instrument_existing_modules()
        
        logger.info("Auto-tracing enabled")
    
    def stop(self):
        """Stop the auto-tracing system."""
        logger.info("Stopping auto-tracing...")
        
        # Restore original import
        __builtins__.__import__ = self._original_import
        
        # Stop worker
        if self._worker_thread:
            self._stop_event.set()
            self._worker_thread.join(timeout=5)
        
        logger.info("Auto-tracing stopped")
    
    def _traced_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Traced import function that auto-instruments imported modules."""
        module = self._original_import(name, globals, locals, fromlist, level)
        
        if name not in self._instrumented_modules and self._should_instrument_module(name):
            self._instrument_module(module, name)
            self._instrumented_modules.add(name)
        
        return module
    
    def _should_instrument_module(self, module_name: str) -> bool:
        """Check if a module should be instrumented."""
        # Skip excluded modules
        for exclude in self.exclude_modules:
            if exclude in module_name:
                return False
        
        # If include list is specified, only include those
        if self.include_modules:
            return any(include in module_name for include in self.include_modules)
        
        return True
    
    def _instrument_existing_modules(self):
        """Instrument already loaded modules."""
        for module_name, module in list(sys.modules.items()):
            if module and self._should_instrument_module(module_name):
                try:
                    self._instrument_module(module, module_name)
                    self._instrumented_modules.add(module_name)
                except Exception as e:
                    logger.debug(f"Could not instrument {module_name}: {e}")
    
    def _instrument_module(self, module, module_name: str):
        """Instrument all functions in a module."""
        if not hasattr(module, '__dict__'):
            return
        
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
            
            try:
                attr = getattr(module, attr_name)
                if (callable(attr) and 
                    hasattr(attr, '__module__') and 
                    attr.__module__ == module_name and
                    not getattr(attr, '_traced', False)):
                    
                    traced_func = self._create_traced_function(attr, module_name)
                    setattr(module, attr_name, traced_func)
            except Exception as e:
                logger.debug(f"Could not instrument {module_name}.{attr_name}: {e}")
    
    def _create_traced_function(self, func: Callable, module_name: str) -> Callable:
        """Create a traced version of a function."""
        @functools.wraps(func)
        def traced_wrapper(*args, **kwargs):
            # Get call stack
            if not hasattr(self._call_stack, 'stack'):
                self._call_stack.stack = []
            
            trace_id = str(uuid.uuid4())
            parent_trace_id = self._call_stack.stack[-1] if self._call_stack.stack else None
            self._call_stack.stack.append(trace_id)
            
            # Get function info
            frame = inspect.currentframe()
            filename = frame.f_code.co_filename if frame else 'unknown'
            line_number = frame.f_lineno if frame else 0
            
            start_time = time.time()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Record successful execution
                duration = (time.time() - start_time) * 1000
                self._record_event({
                    'id': trace_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'event_type': 'function_call',
                    'function_name': func.__name__,
                    'module_name': module_name,
                    'filename': filename,
                    'line_number': line_number,
                    'thread_id': threading.current_thread().ident,
                    'parent_trace_id': parent_trace_id,
                    'duration_ms': duration,
                    'args': self._serialize_args(args, kwargs),
                    'result': self._serialize_result(result),
                    'error': None,
                    'metadata': json.dumps({'success': True})
                })
                
                return result
                
            except Exception as e:
                # Record error
                duration = (time.time() - start_time) * 1000
                self._record_event({
                    'id': trace_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'event_type': 'function_error',
                    'function_name': func.__name__,
                    'module_name': module_name,
                    'filename': filename,
                    'line_number': line_number,
                    'thread_id': threading.current_thread().ident,
                    'parent_trace_id': parent_trace_id,
                    'duration_ms': duration,
                    'args': self._serialize_args(args, kwargs),
                    'result': None,
                    'error': str(e),
                    'metadata': json.dumps({'success': False, 'error_type': type(e).__name__})
                })
                raise
            
            finally:
                # Pop from call stack
                if self._call_stack.stack and self._call_stack.stack[-1] == trace_id:
                    self._call_stack.stack.pop()
        
        traced_wrapper._traced = True
        return traced_wrapper
    
    def _serialize_args(self, args, kwargs, max_length: int = 500) -> str:
        """Serialize function arguments safely."""
        try:
            data = {
                'args': [str(arg)[:max_length] for arg in args],
                'kwargs': {k: str(v)[:max_length] for k, v in kwargs.items()}
            }
            return json.dumps(data)
        except Exception:
            return '{"error": "Could not serialize arguments"}'
    
    def _serialize_result(self, result, max_length: int = 500) -> str:
        """Serialize function result safely."""
        try:
            return str(result)[:max_length]
        except Exception:
            return "Could not serialize result"
    
    def _record_event(self, event: Dict[str, Any]):
        """Queue an event for database storage."""
        self._event_queue.put(event)
    
    def _db_worker(self):
        """Background worker that writes events to database."""
        batch = []
        batch_size = 100
        
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
                (id, timestamp, event_type, function_name, module_name, filename, 
                 line_number, thread_id, parent_trace_id, duration_ms, args, result, error, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [(
                event['id'], event['timestamp'], event['event_type'], event['function_name'],
                event['module_name'], event['filename'], event['line_number'], 
                event['thread_id'], event['parent_trace_id'], event['duration_ms'],
                event['args'], event['result'], event['error'], event['metadata']
            ) for event in events])
            conn.commit()
            conn.close()
            
            logger.debug(f"Wrote {len(events)} trace events to database")
            
        except Exception as e:
            logger.error(f"Failed to write trace events: {e}")


# Global tracer instance
_global_tracer: Optional[AutoTracer] = None


def enable_auto_tracing(
    database_path: str = "trace_events.db",
    include_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None
) -> AutoTracer:
    """
    Enable automatic tracing for the entire application.
    
    Call this ONCE during application bootstrap to instrument all functions
    and operations automatically.
    
    Args:
        database_path: Path to SQLite database file for storing traces
        include_modules: Module patterns to trace (if None, traces all except excluded)
        exclude_modules: Module patterns to exclude from tracing
        
    Returns:
        AutoTracer instance for advanced control
        
    Example:
        from agent_evolve import enable_auto_tracing
        
        # Enable tracing for your application
        tracer = enable_auto_tracing(
            database_path="my_traces.db",
            include_modules=["my_app", "my_package"]
        )
        
        # Your application code runs normally - everything is traced automatically
        def my_function():
            return "Hello World"  # This call will be traced
    """
    global _global_tracer
    
    if _global_tracer:
        logger.warning("Auto-tracing is already enabled")
        return _global_tracer
    
    _global_tracer = AutoTracer(database_path, include_modules, exclude_modules)
    _global_tracer.start()
    
    return _global_tracer


def disable_auto_tracing():
    """Disable automatic tracing."""
    global _global_tracer
    if _global_tracer:
        _global_tracer.stop()
        _global_tracer = None