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
        
        # Handle both dict and module forms of __builtins__
        import builtins
        self._original_import = builtins.__import__
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
        
        # Start background worker for database writes
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._db_worker, daemon=True)
        self._worker_thread.start()
        
        # Instrument already loaded modules (but skip import hooking for now)
        self._instrument_existing_modules()
        
        logger.info("Auto-tracing enabled")
    
    def stop(self):
        """Stop the auto-tracing system."""
        logger.info("Stopping auto-tracing...")
        
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
        logger.info("Instrumenting existing modules...")
        
        for module_name, module in list(sys.modules.items()):
            if (module is not None and 
                module_name not in self._instrumented_modules and 
                self._should_instrument_module(module_name)):
                try:
                    self._instrument_module(module, module_name)
                    self._instrumented_modules.add(module_name)
                    logger.debug(f"Instrumented module: {module_name}")
                except Exception as e:
                    logger.debug(f"Could not instrument module {module_name}: {e}")
        
        logger.info(f"Instrumented {len(self._instrumented_modules)} modules")
    
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
        # Skip tracing functions that are part of the tracer itself or critical system functions
        if module_name in ['agent_evolve.auto_tracing', 'uuid', 'time', 'threading', 'sqlite3', 'json', 'logging']:
            return func
            
        @functools.wraps(func)
        def traced_wrapper(*args, **kwargs):
            # Simple recursion check using thread-local storage
            if not hasattr(self._call_stack, 'tracing'):
                self._call_stack.tracing = False
            
            if self._call_stack.tracing:
                return func(*args, **kwargs)
            
            self._call_stack.tracing = True
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                
                # Simple event recording (without recursion risk)
                event = {
                    'id': str(uuid.uuid4()),
                    'timestamp': datetime.utcnow().isoformat(),
                    'event_type': 'function_call',
                    'function_name': func.__name__,
                    'module_name': module_name,
                    'duration_ms': duration,
                    'thread_id': str(threading.current_thread().ident),
                    'success': True
                }
                self._event_queue.put(event)
                return result
                
            except Exception as e:
                duration = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
                event = {
                    'id': str(uuid.uuid4()),
                    'timestamp': datetime.utcnow().isoformat(),
                    'event_type': 'function_error',
                    'function_name': func.__name__,
                    'module_name': module_name,
                    'duration_ms': duration,
                    'thread_id': str(threading.current_thread().ident),
                    'error': str(e),
                    'success': False
                }
                self._event_queue.put(event)
                raise
            finally:
                self._call_stack.tracing = False
        
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
                (id, timestamp, event_type, function_name, module_name, thread_id, duration_ms, error, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [(
                event['id'], 
                event['timestamp'], 
                event['event_type'], 
                event['function_name'],
                event['module_name'], 
                event['thread_id'], 
                event.get('duration_ms', 0),
                event.get('error'),
                json.dumps({'success': event.get('success', True)})
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
):
    """
    Enable automatic tracing for the application using safe approach.
    
    Args:
        database_path: Path to SQLite database file for storing traces
        include_modules: Module patterns to trace 
        exclude_modules: Module patterns to exclude from tracing
        
    Returns:
        SafeTracer with automatic function patching
    """
    from .safe_tracer import SafeTracer
    
    logger.info(f"Enabling safe auto-tracing with database: {database_path}")
    
    # Create tracer with module filtering
    tracer = SafeTracer(database_path, include_modules)
    tracer.start()
    
    # Auto-patch functions in target modules 
    _auto_patch_functions(tracer, include_modules)
    
    logger.info("Safe auto-tracing enabled successfully")
    return tracer


def _auto_patch_functions(tracer, include_modules: Optional[List[str]] = None):
    """Automatically patch functions in local codebase modules."""
    import sys
    import inspect
    import os
    
    # Get the calling module (where enable_auto_tracing was called)
    frame = inspect.currentframe()
    try:
        caller_frame = frame.f_back.f_back  # Go up two frames
        caller_module = inspect.getmodule(caller_frame)
        if caller_module and hasattr(caller_module, '__file__'):
            caller_file = os.path.abspath(caller_module.__file__)
            project_root = os.path.dirname(caller_file)
            
            # Find project root by looking for common markers
            while project_root != '/' and not any(os.path.exists(os.path.join(project_root, marker)) 
                                                 for marker in ['pyproject.toml', 'setup.py', '.git', 'src']):
                project_root = os.path.dirname(project_root)
            
            logger.info(f"Auto-discovering modules from project root: {project_root}")
        else:
            logger.warning("Could not determine caller module, using current directory")
            project_root = os.getcwd()
    finally:
        del frame
    
    # Find all local codebase modules
    local_modules = []
    for module_name, module in list(sys.modules.items()):
        if module and hasattr(module, '__file__') and module.__file__:
            try:
                module_path = os.path.abspath(module.__file__)
                # Check if module is in project directory and not in site-packages
                if (module_path.startswith(project_root) and 
                    'site-packages' not in module_path and
                    '.pyenv' not in module_path and
                    not module_name.startswith('_') and
                    module_name not in tracer._skip_modules):
                    local_modules.append(module_name)
                    logger.debug(f"Found local module: {module_name} at {module_path}")
            except:
                continue
    
    logger.info(f"Found {len(local_modules)} local modules to patch: {local_modules}")
    
    # Debug: specifically check for the calling module
    if caller_module:
        caller_name = caller_module.__name__
        caller_file = getattr(caller_module, '__file__', 'no-file')
        logger.info(f"Calling module: {caller_name} at {caller_file}")
        logger.info(f"Is calling module in discovered list: {caller_name in local_modules}")
    patched_count = 0
    
    # First pass: patch all discovered modules
    for module_name in local_modules:
        module = sys.modules[module_name]
        if module and hasattr(module, '__dict__'):
            count = _patch_module_functions(module, module_name, tracer)
            if count > 0:
                patched_count += count
                logger.info(f"Patched {count} functions in {module_name}")
    
    # Second pass: patch the calling module (which might not be fully loaded yet)
    if caller_module and caller_module.__name__ not in local_modules:
        caller_name = caller_module.__name__
        logger.info(f"Attempting to patch calling module: {caller_name}")
        count = _patch_module_functions(caller_module, caller_name, tracer)
        patched_count += count  # Add count regardless
        if count > 0:
            logger.info(f"Patched {count} functions in calling module {caller_name}")
        else:
            logger.warning(f"No functions found to patch in calling module {caller_name}")
        local_modules.append(caller_name)
    
    # Also patch any imported references in other modules
    _patch_imported_references(local_modules, tracer, project_root)
    
    # Special handling for LangGraph: update already-stored function references in builder
    _patch_langgraph_builder_nodes(caller_module, tracer)
    
    logger.info(f"Total functions patched: {patched_count}")
    if patched_count == 0:
        logger.warning("No functions were patched! Auto-tracing will not work.")


def _patch_module_functions(module, module_name: str, tracer):
    """Patch all functions in a module with tracing."""
    patched_count = 0
    
    # Debug: List all functions in the module
    all_functions = []
    for attr_name in dir(module):
        if not attr_name.startswith('_'):
            try:
                attr = getattr(module, attr_name)
                if callable(attr):
                    attr_module = getattr(attr, '__module__', 'no-module')
                    all_functions.append(f"{attr_name}({attr_module})")
            except:
                pass
    
    logger.info(f"All callable functions in {module_name}: {all_functions[:10]}")
    
    for attr_name in dir(module):
        if not attr_name.startswith('_'):
            try:
                attr = getattr(module, attr_name)
                if (callable(attr) and 
                    hasattr(attr, '__module__') and 
                    attr.__module__ == module_name and
                    not getattr(attr, '_traced', False)):
                    
                    # Apply tracing decorator
                    traced_func = tracer.trace_decorator(attr_name, module_name)(attr)
                    traced_func._traced = True
                    setattr(module, attr_name, traced_func)
                    
                    # Verify the patching worked
                    new_attr = getattr(module, attr_name)
                    if hasattr(new_attr, '_traced'):
                        logger.info(f"✓ Auto-traced function: {module_name}.{attr_name}")
                        patched_count += 1
                    else:
                        logger.error(f"✗ Failed to patch: {module_name}.{attr_name}")
                        
                    # Debug: Check if it's actually the traced version
                    logger.info(f"Function {attr_name}: traced={hasattr(new_attr, '_traced')}, wrapper={new_attr.__name__ if hasattr(new_attr, '__name__') else 'unknown'}")
                    
            except Exception as e:
                logger.warning(f"Could not trace {module_name}.{attr_name}: {e}")
    
    return patched_count


def _patch_imported_references(local_modules, tracer, project_root):
    """Patch imported references to traced functions in other modules."""
    import sys
    
    logger.info("Patching imported references to traced functions...")
    
    for module_name in local_modules:
        if module_name not in sys.modules:
            continue
            
        source_module = sys.modules[module_name]
        if not source_module or not hasattr(source_module, '__dict__'):
            continue
        
        # Find modules that might have imported from this module  
        for other_module_name, other_module in sys.modules.items():
            if (other_module and 
                other_module != source_module and 
                hasattr(other_module, '__dict__') and
                hasattr(other_module, '__file__') and
                other_module.__file__ and
                other_module.__file__.startswith(project_root)):
                
                _update_imported_refs(source_module, module_name, other_module, other_module_name)


def _update_imported_refs(source_module, source_module_name, target_module, target_module_name):
    """Update imported function references in target module."""
    updated_count = 0
    
    for attr_name in dir(target_module):
        if attr_name.startswith('_'):
            continue
            
        try:
            target_attr = getattr(target_module, attr_name)
            if hasattr(source_module, attr_name):
                source_attr = getattr(source_module, attr_name)
                
                # If source has a traced version and target has the old version
                if (hasattr(source_attr, '_traced') and 
                    not hasattr(target_attr, '_traced') and
                    callable(target_attr)):
                    
                    setattr(target_module, attr_name, source_attr)
                    logger.info(f"Updated imported reference: {target_module_name}.{attr_name} -> {source_module_name}.{attr_name}")
                    updated_count += 1
                    
        except Exception as e:
            logger.debug(f"Could not update {target_module_name}.{attr_name}: {e}")
    
    if updated_count > 0:
        logger.info(f"Updated {updated_count} imported references in {target_module_name}")


def disable_auto_tracing():
    """Disable automatic tracing."""
    global _global_tracer
    if _global_tracer:
        _global_tracer.stop()
        _global_tracer = None