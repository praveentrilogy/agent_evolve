"""
Trace-based Auto-Tracer - Uses sys.settrace() for complete call tracing

This version uses Python's built-in tracing mechanism to capture ALL function calls
without needing to patch or instrument any modules.
"""

import sys
import sqlite3
import json
import uuid
import time
import threading
import os
import contextvars
import concurrent.futures
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
from queue import Queue, Empty
import logging

logger = logging.getLogger(__name__)


class TraceTracer:
    """Auto-tracer using sys.settrace() to capture all function calls."""
    
    def __init__(self, database_path: str, project_root: str = None, include_modules: Optional[List[str]] = None):
        self.database_path = database_path
        self.project_root = os.path.abspath(project_root) if project_root else os.getcwd()
        self.include_modules = include_modules or []
        self._event_queue = Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._original_trace = None
        self._call_stack = {}  # thread_id -> call stack
        
        # Context variables for trace propagation across async boundaries
        self._trace_id_context = contextvars.ContextVar('trace_id', default=None)
        self._trace_depth_context = contextvars.ContextVar('trace_depth', default=0)
        
        # Store original functions for cleanup
        self._original_thread_start = None
        self._original_executor_submit = None
        self._original_create_task = None
        self._original_run_in_executor = None
        
        # Monkey-patch threading to propagate context across threads
        self._setup_thread_context_propagation()
        
        # System modules/paths to skip
        self._skip_patterns = {
            'site-packages', 'lib/python', 'threading.py', 'queue.py', 
            'logging', 'sqlite3', 'json', 'uuid', 'time', 'datetime',
            'agent_evolve', '_bootstrap', 'importlib', 'traceback'
        }
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for trace storage."""
        db_dir = os.path.dirname(self.database_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        conn = sqlite3.connect(self.database_path)
        
        # Check if tables exist and have correct schema
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trace_events'")
        trace_table_exists = cursor.fetchone() is not None
        
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prompts'")
        prompts_table_exists = cursor.fetchone() is not None
        
        if trace_table_exists:
            # Check if required columns exist
            cursor = conn.execute("PRAGMA table_info(trace_events)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'filename' not in columns or 'trace_id' not in columns:
                # Drop old table and recreate with new schema
                conn.execute('DROP TABLE IF EXISTS trace_events')
                logger.info("Dropped old trace_events table with incompatible schema")
        
        # Create trace_events table (simplified without prompts column)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS trace_events (
                id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                function_name TEXT,
                module_name TEXT,
                filename TEXT,
                line_number INTEGER,
                thread_id TEXT,
                duration_ms REAL,
                args JSON,
                result JSON,
                error TEXT,
                success BOOLEAN,
                used_prompt_id TEXT,
                metadata JSON,
                FOREIGN KEY (used_prompt_id) REFERENCES prompts(id)
            )
        ''')
        
        # Create prompts table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prompts (
                id TEXT PRIMARY KEY,
                prompt_name TEXT NOT NULL,
                prompt_type TEXT NOT NULL,
                definition_location TEXT NOT NULL,
                full_code TEXT NOT NULL,
                content TEXT NOT NULL,
                variables JSON,
                function_signature TEXT,
                enum_values JSON,
                created_at TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0,
                UNIQUE(prompt_name, definition_location)
            )
        ''')
        
        # Create prompt_usages table to link traces to prompts
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prompt_usages (
                id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                variable_values JSON,
                rendered_content TEXT,
                FOREIGN KEY (trace_id) REFERENCES trace_events(trace_id),
                FOREIGN KEY (prompt_id) REFERENCES prompts(id)
            )
        ''')
        
        # Create indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON trace_events(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_function ON trace_events(function_name)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_trace_id ON trace_events(trace_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_prompt_name ON prompts(prompt_name)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_prompt_usage_trace ON prompt_usages(trace_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_prompt_usage_prompt ON prompt_usages(prompt_id)')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Trace database initialized with prompts tables: {self.database_path}")
    
    def _setup_thread_context_propagation(self):
        """Setup monkey-patching to propagate context across thread boundaries."""
        # Store original functions to avoid double-patching
        if hasattr(threading.Thread.start, '_opik_patched'):
            return
        
        # 1. Install trace function on all future threads
        threading.settrace(self._trace_function)
        
        # 2. Monkey-patch Thread.start to carry contextvars
        orig_thread_start = threading.Thread.start
        self._original_thread_start = orig_thread_start
        
        def patched_thread_start(thread_self):
            ctx = contextvars.copy_context()
            # Wrap the target so ctx (and trace_id) flows in
            if thread_self._target:
                target = thread_self._target
                args = thread_self._args or ()
                kwargs = thread_self._kwargs or {}
                
                def wrapped_target():
                    return ctx.run(target, *args, **kwargs)
                
                # Replace the target with wrapped version
                thread_self._target = wrapped_target
                thread_self._args = ()
                thread_self._kwargs = {}
            
            return orig_thread_start(thread_self)
        
        patched_thread_start._opik_patched = True
        threading.Thread.start = patched_thread_start
        
        # 3. Monkey-patch ThreadPoolExecutor.submit
        orig_submit = concurrent.futures.ThreadPoolExecutor.submit
        self._original_executor_submit = orig_submit
        
        def patched_submit(executor_self, fn, *args, **kwargs):
            ctx = contextvars.copy_context()
            return orig_submit(executor_self, lambda: ctx.run(fn, *args, **kwargs))
        
        patched_submit._opik_patched = True
        concurrent.futures.ThreadPoolExecutor.submit = patched_submit
        
        # 4. Monkey-patch asyncio.create_task 
        orig_create_task = asyncio.create_task
        self._original_create_task = orig_create_task
        
        def patched_create_task(coro, *, name=None, context=None):
            # If no context provided, copy current context
            if context is None:
                context = contextvars.copy_context()
            return orig_create_task(coro, name=name, context=context)
        
        patched_create_task._opik_patched = True
        asyncio.create_task = patched_create_task
        
        # 5. Monkey-patch loop.create_task (instance method)
        # This is trickier since we need to patch all event loop instances
        try:
            # Patch the default event loop if it exists
            loop = asyncio.get_running_loop()
            self._patch_event_loop(loop)
        except RuntimeError:
            # No running loop, will patch when loop is created
            pass
        
        # 6. Monkey-patch asyncio.run_in_executor
        try:
            loop = asyncio.get_running_loop()
            orig_run_in_executor = loop.run_in_executor
            self._original_run_in_executor = orig_run_in_executor
            
            def patched_run_in_executor(loop_self, executor, func, *args):
                ctx = contextvars.copy_context()
                return orig_run_in_executor(loop_self, executor, lambda: ctx.run(func, *args))
            
            loop.run_in_executor = patched_run_in_executor.__get__(loop, type(loop))
        except RuntimeError:
            # No running loop
            pass
        
        logger.info("Thread and asyncio context propagation patches installed")
    
    def _patch_event_loop(self, loop):
        """Patch an event loop's create_task method."""
        if hasattr(loop.create_task, '_opik_patched'):
            return
            
        orig_loop_create_task = loop.create_task
        
        def patched_loop_create_task(coro, *, name=None, context=None):
            if context is None:
                context = contextvars.copy_context()
            return orig_loop_create_task(coro, name=name, context=context)
        
        patched_loop_create_task._opik_patched = True
        loop.create_task = patched_loop_create_task.__get__(loop, type(loop))
    
    def _should_trace_file(self, filename: str) -> bool:
        """Check if a file should be traced based on path."""
        if not filename:
            return False
        
        # Skip frozen modules and system internals
        if filename.startswith('<frozen') or filename.startswith('<string'):
            return False
            
        # Skip system modules and libraries
        for skip_pattern in self._skip_patterns:
            if skip_pattern in filename:
                return False
        
        # If include_modules is specified, only include those
        if self.include_modules:
            return any(include_mod in filename for include_mod in self.include_modules)
        
        # Only trace files within the project root
        try:
            abs_filename = os.path.abspath(filename)
            return abs_filename.startswith(self.project_root)
        except:
            return False
    
    def _get_module_name(self, filename: str) -> str:
        """Extract module name from filename."""
        try:
            # Convert file path to module name relative to project root
            abs_filename = os.path.abspath(filename)
            if abs_filename.startswith(self.project_root):
                rel_path = os.path.relpath(abs_filename, self.project_root)
                module_name = rel_path.replace('.py', '').replace(os.sep, '.')
                return module_name
            else:
                return os.path.basename(filename).replace('.py', '')
        except:
            return os.path.basename(filename).replace('.py', '')
    
    def _serialize_safely(self, obj, max_len=None):
        """Safely serialize an object for storage."""
        try:
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, (list, tuple)):
                return {
                    'type': type(obj).__name__,
                    'length': len(obj),
                    'items': [self._serialize_safely(item) for item in obj]
                }
            elif isinstance(obj, dict):
                return {
                    'type': 'dict',
                    'length': len(obj),
                    'items': {k: self._serialize_safely(v) for k, v in obj.items()}
                }
            else:
                return {
                    'type': type(obj).__name__,
                    'string_repr': str(obj)
                }
        except Exception as e:
            return {'error': f'serialization_failed: {str(e)}', 'type': str(type(obj))}
    
    def _detect_and_store_prompts(self, frame) -> str:
        """Detect prompt definitions in current scope and store them in prompts table. Returns prompt_id if found."""
        try:
            import inspect
            import ast
            import re
            from datetime import datetime
            
            local_vars = frame.f_locals
            global_vars = frame.f_globals
            filename = frame.f_code.co_filename
            
            # Look for prompt-related variable names in the current scope
            prompt_var_patterns = [
                'prompt', 'template', 'instruction', 'message', 'system', 'user',
                'query', 'completion', 'generation', 'chat'
            ]
            
            all_vars = {**global_vars, **local_vars}
            used_prompt_id = None
            
            for var_name, var_value in all_vars.items():
                try:
                    # Skip non-string values and internal variables
                    if not isinstance(var_value, str) or var_name.startswith('_'):
                        continue
                        
                    var_name_lower = var_name.lower()
                    
                    # Check if variable name suggests it's a prompt
                    is_prompt_var = any(pattern in var_name_lower for pattern in prompt_var_patterns)
                    
                    # Check content for prompt-like patterns
                    if len(var_value) > 20:
                        prompt_indicators = [
                            'you are', 'you will', 'your role', 'your task', 'act as',
                            'please', 'instruction:', 'system:', 'user:', 'assistant:',
                            'given the following', 'analyze', 'generate', 'create'
                        ]
                        
                        value_lower = var_value.lower()
                        has_prompt_content = any(ind in value_lower for ind in prompt_indicators)
                        
                        if is_prompt_var or has_prompt_content:
                            # Extract the full code definition
                            full_code = self._extract_prompt_definition(var_name, filename)
                            variables = self._extract_template_variables(var_value)
                            
                            # Generate unique ID for this prompt
                            prompt_id = str(uuid.uuid4())
                            definition_location = f"{filename}:{var_name}"
                            
                            # Determine prompt type
                            prompt_type = self._determine_prompt_type(var_name, var_value, all_vars)
                            
                            # Store in prompts table
                            self._store_prompt_definition(
                                prompt_id=prompt_id,
                                prompt_name=var_name,
                                prompt_type=prompt_type,
                                definition_location=definition_location,
                                full_code=full_code,
                                content=var_value,
                                variables=variables,
                                function_signature=None,  # Will be set if it's a function-generated prompt
                                enum_values=None         # Will be set if it's an enum prompt
                            )
                            
                            # Return the first prompt ID found (for linking to trace)
                            if used_prompt_id is None:
                                used_prompt_id = prompt_id
                                
                except Exception:
                    continue
                    
            return used_prompt_id
            
        except Exception as e:
            # Don't let prompt detection break tracing
            return None
    
    def _extract_prompt_definition(self, var_name: str, filename: str) -> str:
        """Extract the full code definition of a prompt variable from source file."""
        try:
            import re
            with open(filename, 'r', encoding='utf-8') as f:
                source = f.read()
            
            lines = source.split('\n')
            
            # Find the line where the variable is defined
            for i, line in enumerate(lines):
                # Look for variable assignment with flexible whitespace
                if re.match(rf'^\s*{re.escape(var_name)}\s*=', line):
                    # Start capturing from this line
                    assignment_lines = [line.rstrip()]
                    
                    # Check if this is a multi-line string (triple quotes)
                    line_content = line.strip()
                    has_triple_double = '"""' in line_content
                    has_triple_single = "'''" in line_content
                    
                    if has_triple_double or has_triple_single:
                        # Count quotes to see if string is closed on same line
                        quote_type = '"""' if has_triple_double else "'''"
                        quote_count = line_content.count(quote_type)
                        
                        # If odd number of quotes, string continues to next lines
                        if quote_count % 2 == 1:
                            j = i + 1
                            while j < len(lines):
                                assignment_lines.append(lines[j].rstrip())
                                if quote_type in lines[j]:
                                    # Found closing quotes
                                    break
                                j += 1
                    else:
                        # Check for line continuation or multi-line parentheses/brackets
                        j = i + 1
                        open_parens = line.count('(') - line.count(')')
                        open_brackets = line.count('[') - line.count(']')
                        open_braces = line.count('{') - line.count('}')
                        
                        # Continue if line ends with continuation or has unmatched brackets
                        while (j < len(lines) and 
                               (line.rstrip().endswith('\\') or 
                                open_parens > 0 or open_brackets > 0 or open_braces > 0 or
                                (j < len(lines) and lines[j].strip().startswith((' ', '\t'))))):
                            
                            next_line = lines[j]
                            assignment_lines.append(next_line.rstrip())
                            
                            # Update bracket counts
                            open_parens += next_line.count('(') - next_line.count(')')
                            open_brackets += next_line.count('[') - next_line.count(']')
                            open_braces += next_line.count('{') - next_line.count('}')
                            
                            # If next line doesn't start with whitespace, we're done
                            if j + 1 >= len(lines) or not lines[j + 1].strip().startswith((' ', '\t')):
                                if open_parens <= 0 and open_brackets <= 0 and open_braces <= 0:
                                    break
                            
                            j += 1
                    
                    return '\n'.join(assignment_lines)
            
            return f"{var_name} = <definition not found in {filename}>"
            
        except Exception as e:
            return f"{var_name} = <error reading source: {str(e)}>"
    
    def _extract_template_variables(self, content: str) -> dict:
        """Extract template variables and their types from prompt content."""
        import re
        
        variables = {}
        
        # Find {variable_name} patterns
        var_pattern = r'\{([^}]+)\}'
        matches = re.findall(var_pattern, content)
        
        for match in matches:
            var_name = match.strip()
            # Try to infer type from variable name
            var_type = "str"  # default
            if any(word in var_name.lower() for word in ['count', 'num', 'number', 'id']):
                var_type = "int"
            elif any(word in var_name.lower() for word in ['price', 'rate', 'percent', 'score']):
                var_type = "float"
            elif any(word in var_name.lower() for word in ['is_', 'has_', 'can_', 'should_']):
                var_type = "bool"
            elif any(word in var_name.lower() for word in ['list', 'items', 'data']):
                var_type = "list"
                
            variables[var_name] = var_type
            
        return variables
    
    def _determine_prompt_type(self, var_name: str, content: str, all_vars: dict) -> str:
        """Determine the type of prompt (variable, function, enum)."""
        # Check if it's from an enum
        if any(hasattr(v, '__members__') and var_name in str(v.__members__) for v in all_vars.values()):
            return "enum"
        
        # Check if it's function-generated (has template variables)
        if '{' in content and '}' in content:
            return "template"
        
        # Check naming patterns
        if var_name.isupper():
            return "constant"
        
        return "variable"
    
    def _store_prompt_definition(self, prompt_id: str, prompt_name: str, prompt_type: str,
                                definition_location: str, full_code: str, content: str,
                                variables: dict, function_signature: str, enum_values: dict):
        """Store prompt definition in the prompts table."""
        try:
            from datetime import datetime
            
            conn = sqlite3.connect(self.database_path)
            now = datetime.utcnow().isoformat()
            
            # Check if prompt already exists
            cursor = conn.execute(
                'SELECT id FROM prompts WHERE prompt_name = ? AND definition_location = ?',
                (prompt_name, definition_location)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update last_seen and usage_count
                conn.execute('''
                    UPDATE prompts 
                    SET last_seen = ?, usage_count = usage_count + 1
                    WHERE id = ?
                ''', (now, existing[0]))
                prompt_id = existing[0]
            else:
                # Insert new prompt
                conn.execute('''
                    INSERT INTO prompts 
                    (id, prompt_name, prompt_type, definition_location, full_code, content, 
                     variables, function_signature, enum_values, created_at, last_seen, usage_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ''', (
                    prompt_id, prompt_name, prompt_type, definition_location, full_code, content,
                    json.dumps(variables) if variables else None,
                    function_signature,
                    json.dumps(enum_values) if enum_values else None,
                    now, now
                ))
            
            conn.commit()
            conn.close()
            return prompt_id
            
        except Exception as e:
            logger.error(f"Failed to store prompt definition: {e}")
            return None
    
    def _trace_function(self, frame, event, arg):
        """The actual trace function that gets called by Python."""
        try:
            code = frame.f_code
            filename = code.co_filename
            
            # Skip if not a file we want to trace
            if not self._should_trace_file(filename):
                return self._trace_function
            
            func_name = code.co_name
            module_name = self._get_module_name(filename)
            thread_id = str(threading.current_thread().ident)
            
            if event == "call":
                # Get or create trace_id using context vars
                current_trace_id = self._trace_id_context.get()
                current_depth = self._trace_depth_context.get()
                
                # Create new trace_id if we don't have one
                if current_trace_id is None:
                    current_trace_id = str(uuid.uuid4())
                    self._trace_id_context.set(current_trace_id)
                    current_depth = 0
                    self._trace_depth_context.set(0)
                
                # Get or create call stack for this thread
                if thread_id not in self._call_stack:
                    self._call_stack[thread_id] = []
                
                # Update depth
                self._trace_depth_context.set(current_depth + 1)
                
                # Get function arguments
                args_info = self._get_function_args(frame)
                
                # Detect and store prompts in current scope
                used_prompt_id = self._detect_and_store_prompts(frame)
                
                # Record function entry
                call_info = {
                    'id': str(uuid.uuid4()),
                    'trace_id': current_trace_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'event_type': 'function_call',
                    'function_name': func_name,
                    'module_name': module_name,
                    'filename': filename,
                    'line_number': frame.f_lineno,
                    'thread_id': thread_id,
                    'start_time': time.time(),
                    'args': args_info,
                    'used_prompt_id': used_prompt_id,
                    'depth': current_depth,
                }
                
                # Store in call stack for return handling
                self._call_stack[thread_id].append(call_info)
                
            elif event == "return":
                # Record function exit
                if (thread_id in self._call_stack and 
                    self._call_stack[thread_id] and
                    self._call_stack[thread_id][-1]['function_name'] == func_name):
                    
                    call_info = self._call_stack[thread_id].pop()
                    duration_ms = (time.time() - call_info['start_time']) * 1000
                    
                    # Create final event
                    event_data = {
                        'id': call_info['id'],
                        'trace_id': call_info['trace_id'],
                        'timestamp': call_info['timestamp'],
                        'event_type': 'function_call',
                        'function_name': func_name,
                        'module_name': module_name,
                        'filename': filename,
                        'line_number': call_info['line_number'],
                        'thread_id': thread_id,
                        'duration_ms': duration_ms,
                        'args': call_info['args'],
                        'result': self._serialize_safely(arg),
                        'success': True,
                        'used_prompt_id': call_info.get('used_prompt_id'),
                        'metadata': {'depth': call_info.get('depth', 0)}
                    }
                    
                    self._event_queue.put(event_data)
                    
                    # Store prompt usage if a prompt was used
                    if call_info.get('used_prompt_id'):
                        self._store_prompt_usage(
                            trace_id=call_info['trace_id'],
                            prompt_id=call_info['used_prompt_id'],
                            variable_values=self._extract_variable_values(call_info['args']),
                            rendered_content=None  # Could be enhanced to show final rendered prompt
                        )
                    
                    # Decrease depth
                    current_depth = self._trace_depth_context.get()
                    if current_depth > 0:
                        self._trace_depth_context.set(current_depth - 1)
                    
            elif event == "exception":
                # Record function exception
                if (thread_id in self._call_stack and 
                    self._call_stack[thread_id] and
                    self._call_stack[thread_id][-1]['function_name'] == func_name):
                    
                    call_info = self._call_stack[thread_id].pop()
                    duration_ms = (time.time() - call_info['start_time']) * 1000
                    
                    event_data = {
                        'id': call_info['id'],
                        'trace_id': call_info['trace_id'],
                        'timestamp': call_info['timestamp'],
                        'event_type': 'function_error',
                        'function_name': func_name,
                        'module_name': module_name,
                        'filename': filename,
                        'line_number': call_info['line_number'],
                        'thread_id': thread_id,
                        'duration_ms': duration_ms,
                        'args': call_info['args'],
                        'error': str(arg),
                        'success': False,
                        'used_prompt_id': call_info.get('used_prompt_id'),
                        'metadata': {'depth': call_info.get('depth', 0)}
                    }
                    
                    self._event_queue.put(event_data)
                    
                    # Decrease depth on exception
                    current_depth = self._trace_depth_context.get()
                    if current_depth > 0:
                        self._trace_depth_context.set(current_depth - 1)
        except:
            # Don't let tracing errors break the application
            pass
        
        return self._trace_function
    
    def _store_prompt_usage(self, trace_id: str, prompt_id: str, variable_values: dict, rendered_content: str = None):
        """Store a prompt usage record in the prompt_usages table."""
        try:
            from datetime import datetime
            
            usage_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            
            conn = sqlite3.connect(self.database_path)
            conn.execute('''
                INSERT INTO prompt_usages 
                (id, trace_id, prompt_id, timestamp, variable_values, rendered_content)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                usage_id,
                trace_id,
                prompt_id,
                timestamp,
                json.dumps(variable_values) if variable_values else None,
                rendered_content
            ))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store prompt usage: {e}")
    
    def _extract_variable_values(self, args: dict) -> dict:
        """Extract variable values that might be used in prompt templates."""
        variable_values = {}
        
        try:
            for arg_name, arg_value in args.items():
                # Only capture string values that look like they might be template variables
                if isinstance(arg_value, str) and len(arg_value) > 0:
                    # Common template variable names
                    template_var_names = [
                        'brand_guidelines', 'task', 'plan', 'draft', 'critique', 'research',
                        'content', 'message', 'text', 'input', 'data', 'context',
                        'current_date', 'brand_input', 'query', 'instruction'
                    ]
                    
                    if arg_name.lower() in template_var_names:
                        variable_values[arg_name] = arg_value[:500]  # Limit length for storage
                elif isinstance(arg_value, (int, float, bool)):
                    # Always capture simple types
                    variable_values[arg_name] = arg_value
                elif isinstance(arg_value, (list, dict)):
                    # For complex types, store a summary
                    if isinstance(arg_value, list):
                        variable_values[arg_name] = f"<list with {len(arg_value)} items>"
                    else:
                        variable_values[arg_name] = f"<dict with {len(arg_value)} keys>"
                        
        except Exception as e:
            logger.error(f"Failed to extract variable values: {e}")
            
        return variable_values
    
    def _get_raw_string_args(self, frame):
        """Extract raw string arguments from frame for prompt detection."""
        try:
            raw_args = {}
            code = frame.f_code
            
            # Get argument names
            arg_names = code.co_varnames[:code.co_argcount]
            
            # Get raw argument values (only strings)
            for arg_name in arg_names:
                if arg_name in frame.f_locals:
                    value = frame.f_locals[arg_name]
                    if isinstance(value, str):
                        raw_args[arg_name] = value
            
            return raw_args
        except:
            return {}
    
    def _get_function_args(self, frame):
        """Extract function arguments from frame."""
        try:
            args_info = {}
            code = frame.f_code
            
            # Get argument names
            arg_names = code.co_varnames[:code.co_argcount]
            
            # Get argument values
            for i, arg_name in enumerate(arg_names):
                if arg_name in frame.f_locals:
                    args_info[arg_name] = self._serialize_safely(frame.f_locals[arg_name])
            
            return args_info
        except:
            return {}
    
    def start(self):
        """Start the tracing system."""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        
        logger.info("Starting trace-based auto-tracing...")
        
        # Store original trace function
        self._original_trace = sys.gettrace()
        
        # Set our trace function
        sys.settrace(self._trace_function)
        
        # Start database worker
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._db_worker, daemon=True)
        self._worker_thread.start()
        
        logger.info("Trace-based auto-tracing enabled")
    
    def stop(self):
        """Stop the tracing system."""
        logger.info("Stopping trace-based auto-tracing...")
        
        # Restore original trace function
        sys.settrace(self._original_trace)
        
        # Restore original threading and asyncio functions
        if self._original_thread_start:
            threading.Thread.start = self._original_thread_start
        if self._original_executor_submit:
            concurrent.futures.ThreadPoolExecutor.submit = self._original_executor_submit
        if self._original_create_task:
            asyncio.create_task = self._original_create_task
        
        # Stop worker
        if self._worker_thread:
            self._stop_event.set()
            self._worker_thread.join(timeout=5)
        
        logger.info("Trace-based auto-tracing stopped")
    
    def reset_trace_context(self):
        """Reset the trace context for a new request/execution."""
        self._trace_id_context.set(None)
        self._trace_depth_context.set(0)
        # Clear call stacks for all threads
        self._call_stack.clear()
    
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
                (id, trace_id, timestamp, event_type, function_name, module_name, 
                 filename, line_number, thread_id, duration_ms, args, 
                 result, error, success, used_prompt_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [
                (
                    event['id'],
                    event['trace_id'],
                    event['timestamp'], 
                    event['event_type'], 
                    event['function_name'],
                    event['module_name'],
                    event['filename'],
                    event['line_number'],
                    event['thread_id'], 
                    event['duration_ms'],
                    json.dumps(event['args']) if event['args'] else None,
                    json.dumps(event['result']) if event.get('result') is not None else None,
                    event.get('error'),
                    event['success'],
                    event.get('used_prompt_id'),
                    json.dumps(event['metadata'])
                ) for event in events
            ])
            conn.commit()
            conn.close()
            
            logger.info(f"Wrote {len(events)} trace events to database")
            
        except Exception as e:
            logger.error(f"Failed to write trace events: {e}")


def analyze_prompts(database_path: str = "trace_events.db"):
    """
    Analyze prompt definitions and usage from the database.
    
    Args:
        database_path: Path to SQLite database file
        
    Returns:
        Dictionary with prompt definitions and usage statistics
    """
    if not os.path.exists(database_path):
        logger.error(f"Database not found: {database_path}")
        return {}
    
    try:
        conn = sqlite3.connect(database_path)
        
        # Get all prompt definitions
        cursor = conn.execute('''
            SELECT id, prompt_name, prompt_type, definition_location, full_code, 
                   content, variables, function_signature, enum_values, 
                   created_at, last_seen, usage_count
            FROM prompts 
            ORDER BY usage_count DESC
        ''')
        
        prompts = []
        for row in cursor.fetchall():
            prompt_id, name, ptype, location, full_code, content, variables_json, \
            func_sig, enum_json, created_at, last_seen, usage_count = row
            
            variables = json.loads(variables_json) if variables_json else {}
            enum_values = json.loads(enum_json) if enum_json else {}
            
            prompts.append({
                'id': prompt_id,
                'name': name,
                'type': ptype,
                'definition_location': location,
                'full_code': full_code,
                'content': content,
                'variables': variables,
                'function_signature': func_sig,
                'enum_values': enum_values,
                'created_at': created_at,
                'last_seen': last_seen,
                'usage_count': usage_count
            })
        
        # Get usage patterns from prompt_usages table
        cursor = conn.execute('''
            SELECT p.prompt_name, COUNT(pu.id) as usage_count,
                   GROUP_CONCAT(DISTINCT t.function_name) as used_in_functions
            FROM prompts p
            LEFT JOIN prompt_usages pu ON p.id = pu.prompt_id
            LEFT JOIN trace_events t ON pu.trace_id = t.trace_id
            GROUP BY p.prompt_name, p.id
            ORDER BY usage_count DESC
        ''')
        
        usage_patterns = []
        for row in cursor.fetchall():
            prompt_name, count, functions = row
            usage_patterns.append({
                'prompt_name': prompt_name,
                'usage_count': count or 0,
                'used_in_functions': functions.split(',') if functions else []
            })
        
        # Get detailed usage records
        cursor = conn.execute('''
            SELECT pu.id, p.prompt_name, t.function_name, pu.timestamp, 
                   pu.variable_values, pu.rendered_content
            FROM prompt_usages pu
            JOIN prompts p ON pu.prompt_id = p.id
            JOIN trace_events t ON pu.trace_id = t.trace_id
            ORDER BY pu.timestamp DESC
            LIMIT 50
        ''')
        
        recent_usages = []
        for row in cursor.fetchall():
            usage_id, prompt_name, function_name, timestamp, var_values_json, rendered = row
            var_values = json.loads(var_values_json) if var_values_json else {}
            recent_usages.append({
                'usage_id': usage_id,
                'prompt_name': prompt_name,
                'function_name': function_name,
                'timestamp': timestamp,
                'variable_values': var_values,
                'rendered_content': rendered
            })
        
        conn.close()
        
        result = {
            'total_prompts': len(prompts),
            'prompts': prompts,
            'usage_patterns': usage_patterns,
            'recent_usages': recent_usages
        }
        
        logger.info(f"Found {len(prompts)} prompt definitions with usage patterns")
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing prompts: {e}")
        return {}


def enable_trace_tracing(
    database_path: str = "trace_events.db",
    project_root: str = None,
    include_modules: Optional[List[str]] = None
):
    """
    Enable trace-based auto-tracing using sys.settrace().
    
    Args:
        database_path: Path to SQLite database file
        project_root: Root directory of project to trace (default: current working directory)
        include_modules: Module patterns to include (if None, traces all files in project_root)
        
    Returns:
        TraceTracer instance
    """
    logger.info(f"Enabling trace-based auto-tracing with database: {database_path}")
    
    tracer = TraceTracer(database_path, project_root, include_modules)
    tracer.start()
    
    logger.info("Trace-based auto-tracing enabled successfully")
    return tracer