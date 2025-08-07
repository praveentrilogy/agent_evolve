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
        
        # Variable capture cache for better timing
        self._variable_cache = {}  # trace_id -> {var_name -> value}
        
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
        
        # Create functions table to catalog all unique functions
        conn.execute('''
            CREATE TABLE IF NOT EXISTS functions (
                id TEXT PRIMARY KEY,
                function_name TEXT NOT NULL,
                full_function_name TEXT NOT NULL,
                class_name TEXT,
                filename TEXT NOT NULL,
                module_name TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                call_count INTEGER DEFAULT 0,
                signature TEXT,
                docstring TEXT,
                UNIQUE(full_function_name, filename)
            )
        ''')
        
        # Create indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON trace_events(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_function ON trace_events(function_name)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_trace_id ON trace_events(trace_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_prompt_name ON prompts(prompt_name)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_prompt_usage_trace ON prompt_usages(trace_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_prompt_usage_prompt ON prompt_usages(prompt_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_function_name ON functions(full_function_name)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_function_file ON functions(filename)')
        
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
        """Detect prompt definitions in current scope and store them in prompts table. Return prompt ID if a prompt is being used."""
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
                'query', 'completion', 'generation', 'chat', 'role', 'persona', 
                'behavior', 'directive', 'command', 'task', 'request', 'ask',
                'text', 'content', 'description', 'guideline', 'rule', 'spec'
            ]
            
            all_vars = {**global_vars, **local_vars}
            used_prompt_id = None
            
            # Get function parameter names to avoid treating them as prompts
            import inspect
            try:
                func_sig = inspect.signature(frame.f_globals.get(frame.f_code.co_name, lambda: None))
                param_names = set(func_sig.parameters.keys())
            except:
                # Fallback: get from the code object
                param_names = set(frame.f_code.co_varnames[:frame.f_code.co_argcount])
            
            # Store all prompt definitions found and detect usage
            for var_name, var_value in all_vars.items():
                try:
                    # Skip non-string values, internal variables, and function parameters
                    if (not isinstance(var_value, str) or 
                        var_name.startswith('_') or 
                        var_name in param_names or
                        var_name in ['self', 'cls']):
                        continue
                        
                    var_name_lower = var_name.lower()
                    
                    # Check if variable name suggests it's a prompt
                    is_prompt_var = any(pattern in var_name_lower for pattern in prompt_var_patterns)
                    
                    # Check content for prompt-like patterns (lowered threshold)
                    if len(var_value) > 10:
                        prompt_indicators = [
                            'you are', 'you will', 'your role', 'your task', 'act as',
                            'please', 'instruction:', 'system:', 'user:', 'assistant:',
                            'given the following', 'analyze', 'generate', 'create',
                            'respond to', 'answer', 'help', 'explain', 'describe',
                            'summarize', 'write', 'based on', 'considering',
                            'as a', 'as an', 'i want you', 'can you', 'should you',
                            'task:', 'goal:', 'objective:', 'context:', 'background:'
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
                            
                            # Store the prompt definition and get its ID
                            stored_prompt_id = self._store_prompt_definition(
                                prompt_id=prompt_id,
                                prompt_name=var_name,
                                prompt_type=prompt_type,
                                definition_location=definition_location,
                                full_code=full_code,
                                content=var_value,
                                variables=variables,
                                function_signature=None,
                                enum_values=None
                            )
                            
                            # Check if this prompt is being used in the current function
                            # Be more aggressive - if we found a prompt in this scope, it's likely being used
                            if (is_prompt_var or has_prompt_content) and var_name in local_vars:
                                used_prompt_id = stored_prompt_id
                                logger.info(f"Prompt '{var_name}' found in local scope, marking as used, ID: {used_prompt_id}")
                                
                except Exception:
                    continue
            
            return used_prompt_id
                    
        except Exception as e:
            # Don't let prompt detection break tracing
            return None
    
    def _is_prompt_being_used(self, frame, var_name, var_value):
        """Check if a prompt is being actively used in the current function call."""
        try:
            # Check if the prompt variable is in the function arguments
            local_vars = frame.f_locals
            
            # More aggressive detection - if a prompt exists in local scope, assume it's being used
            # Check if our prompt value appears anywhere in the local variables
            for local_var_name, local_var_value in local_vars.items():
                if local_var_value == var_value:
                    logger.info(f"Prompt '{var_name}' detected in use as '{local_var_name}'")
                    return True
                    
                # Check if it's a formatted/modified version of the prompt
                if isinstance(local_var_value, str) and isinstance(var_value, str) and len(var_value) > 20:
                    # Check various ways the prompt might be used
                    if (var_value in local_var_value or  # Direct substring
                        ('{' in var_value and var_value.split('{')[0] in local_var_value) or  # Template prefix
                        (len(local_var_value) > 50 and len(var_value) > 50 and 
                         local_var_value[:50] == var_value[:50])):  # Same beginning
                        logger.info(f"Prompt '{var_name}' detected in use (formatted)")
                        return True
            
            # Check if it's being passed to common LLM/prompt-related functions
            llm_functions = [
                'generate', 'complete', 'chat', 'respond', 'invoke', 
                'call', 'run', 'execute', 'send', 'format', 'render',
                'process', 'handle', 'create_message', 'send_message',
                'langchain', 'openai', 'anthropic', 'llm', 'prompt',
                'research', 'improve', 'create', 'reflect', 'classify'
            ]
            
            # Check if the current function name suggests LLM usage
            func_name = frame.f_code.co_name.lower()
            if any(pattern in func_name for pattern in llm_functions):
                # If we're in a prompt-related function and the prompt is in scope, it's likely being used
                if var_name in local_vars or var_value in str(local_vars.values()):
                    logger.info(f"Prompt '{var_name}' detected in use in function '{func_name}'")
                    return True
            
            # Also check for specific patterns in your codebase
            # If the prompt is passed as 'messages' parameter (common in chat APIs)
            if 'messages' in local_vars:
                messages_str = str(local_vars['messages'])
                if var_value in messages_str:
                    logger.info(f"Prompt '{var_name}' detected in messages")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking prompt usage: {e}")
            return False
    
    def _find_prompt_in_scope(self, frame):
        """Find any prompt that might be in scope for known prompt-using functions."""
        try:
            # Look up the prompts table for prompts from this module
            filename = frame.f_code.co_filename
            func_name = frame.f_code.co_name
            conn = sqlite3.connect(self.database_path)
            
            # Try to find prompts used by this function through execution context analysis
            print(f"[TRACE] Looking for prompt for function '{func_name}' in module {filename}")
            
            # Look for evidence of prompt usage in the current execution frame
            local_vars = frame.f_locals
            global_vars = frame.f_globals
            all_vars = {**global_vars, **local_vars}
            
            # Skip global constants - focus on local variables that contain formatted prompts
            print(f"[TRACE] Analyzing local execution context for prompt usage...")
            detected_prompt_id = None
            
            # Prioritize message variables that likely contain formatted prompts
            message_vars = [v for v in local_vars.keys() if 'message' in v.lower()]
            other_local_vars = [v for v in local_vars.keys() if 'message' not in v.lower()]
            
            # Check message variables first, then other local variables
            for var_name in message_vars + other_local_vars:
                var_value = local_vars[var_name]
                
                if isinstance(var_value, str) and len(var_value) > 50:
                    # Skip if this looks like a raw prompt constant (all caps variable name)
                    if var_name.isupper() and '_PROMPT' in var_name:
                        continue
                        
                    # Check if this variable content matches any known prompt
                    cursor = conn.execute('''
                        SELECT id, prompt_name, content 
                        FROM prompts 
                        WHERE definition_location LIKE ?
                    ''', (f"{filename}:%",))
                    
                    for prompt_id, prompt_name, prompt_content in cursor.fetchall():
                        # Check for partial content match (in case of formatted prompts)
                        if prompt_content.strip() in var_value or var_value in prompt_content.strip():
                            detected_prompt_id = prompt_id
                            print(f"[TRACE] Found prompt '{prompt_name}' being used in local variable '{var_name}'")
                            logger.info(f"Detected prompt '{prompt_name}' usage in function '{func_name}' via variable analysis")
                            break
                
                # Check message lists that might contain formatted prompts
                elif hasattr(var_value, '__iter__') and not isinstance(var_value, str):
                    print(f"[TRACE] Checking message list in variable '{var_name}'")
                    try:
                        for i, item in enumerate(var_value):
                            if hasattr(item, 'content'):
                                content_str = str(item.content)
                                print(f"[TRACE] Message {i} content length: {len(content_str)}")
                                if len(content_str) > 50:
                                    cursor = conn.execute('''
                                        SELECT id, prompt_name, content 
                                        FROM prompts 
                                        WHERE definition_location LIKE ?
                                    ''', (f"{filename}:%",))
                                    
                                    for prompt_id, prompt_name, prompt_content in cursor.fetchall():
                                        # Check for content overlap - even if template variables are replaced
                                        prompt_words = set(prompt_content.split())
                                        content_words = set(content_str.split())
                                        overlap = len(prompt_words.intersection(content_words))
                                        overlap_ratio = overlap / len(prompt_words) if prompt_words else 0
                                        print(f"[TRACE] Testing '{prompt_name}': {overlap}/{len(prompt_words)} words overlap ({overlap_ratio:.2f})")
                                        
                                        if overlap > min(10, len(prompt_words) * 0.3):  # At least 30% word overlap or 10 words
                                            detected_prompt_id = prompt_id
                                            print(f"[TRACE] MATCH! Found prompt '{prompt_name}' being used in {var_name} message content")
                                            logger.info(f"Detected prompt '{prompt_name}' usage in function '{func_name}' via message analysis")
                                            break
                                    if detected_prompt_id:
                                        break
                    except Exception as ex:
                        print(f"[TRACE] Error checking message list: {ex}")
                        continue
                
                if detected_prompt_id:
                    break
                        
            if detected_prompt_id:
                conn.close()
                return detected_prompt_id
            
            # Fallback: Find prompts from this file
            cursor = conn.execute('''
                SELECT id, prompt_name, content 
                FROM prompts 
                WHERE definition_location LIKE ?
                ORDER BY usage_count DESC
                LIMIT 1
            ''', (f"{filename}:%",))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                prompt_id, prompt_name, content = result
                logger.info(f"Found prompt '{prompt_name}' for function {func_name}")
                return prompt_id
                
        except Exception as e:
            logger.error(f"Error finding prompt in scope: {e}")
            
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
        
        # Enhanced patterns for different template formats
        patterns = [
            r'\{([^}]+)\}',  # Standard {variable}
            r'\{(\d+)\}',    # Positional {0}, {1}
            r'\{\{([^}]+)\}\}',  # Double braces {{variable}}
            r'\{([^}:]+):[^}]*\}',  # With format specifiers {var:format}
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                var_name = match.strip()
                if var_name.isdigit():
                    var_name = f"arg_{var_name}"  # Convert positional to named
                
                # Enhanced type inference
                var_type = "str"  # default
                if any(word in var_name.lower() for word in ['count', 'num', 'number', 'id', 'index']):
                    var_type = "int"
                elif any(word in var_name.lower() for word in ['price', 'rate', 'percent', 'score', 'amount']):
                    var_type = "float"
                elif any(word in var_name.lower() for word in ['is_', 'has_', 'can_', 'should_', 'enable', 'disable']):
                    var_type = "bool"
                elif any(word in var_name.lower() for word in ['list', 'items', 'data', 'array', 'collection']):
                    var_type = "list"
                elif any(word in var_name.lower() for word in ['dict', 'map', 'object', 'config']):
                    var_type = "dict"
                    
                variables[var_name] = var_type
                
        return variables
    
    def _find_variable_value(self, var_name: str, all_vars: dict) -> Any:
        """Enhanced variable finding with multiple strategies."""
        # 1. Direct match
        if var_name in all_vars:
            return all_vars[var_name]
        
        # 2. Case-insensitive match
        for key, value in all_vars.items():
            if key.lower() == var_name.lower():
                return value
        
        # 3. Common suffixes
        for suffix in ['_text', '_str', '_data', '_info', '_content', '_input', '_value', '_message', '_prompt']:
            candidate = var_name + suffix
            if candidate in all_vars:
                return all_vars[candidate]
        
        # 4. Common prefixes removed
        for prefix in ['user_', 'input_', 'template_', 'prompt_', 'system_', 'assistant_']:
            if var_name.startswith(prefix):
                base_name = var_name[len(prefix):]
                if base_name in all_vars:
                    return all_vars[base_name]
        
        # 5. Partial matching (more sophisticated)
        var_clean = var_name.replace('_', '').lower()
        best_match = None
        best_score = 0
        
        for scope_var in all_vars.keys():
            scope_clean = scope_var.replace('_', '').lower()
            
            # Calculate similarity score
            if var_clean in scope_clean or scope_clean in var_clean:
                score = min(len(var_clean), len(scope_clean)) / max(len(var_clean), len(scope_clean))
                if score > best_score and len(scope_clean) >= 3:
                    best_score = score
                    best_match = all_vars[scope_var]
        
        return best_match if best_score > 0.5 else None
    
    def _find_rendered_content(self, prompt_content: str, prompt_variables: dict, variable_values: dict, all_vars: dict) -> str:
        """Enhanced rendered content detection."""
        # 1. Try to render with captured variables
        if variable_values and prompt_variables:
            try:
                rendered = prompt_content.format(**variable_values)
                print(f"[TRACE] Successfully rendered prompt using captured variables")
                return rendered
            except (KeyError, ValueError) as e:
                print(f"[TRACE] Could not render prompt with variables: {e}")
        
        # 2. Look for rendered content in execution context
        prompt_vars = list(prompt_variables.keys()) if prompt_variables else []
        
        for var_name, var_value in all_vars.items():
            if var_name in ['messages', 'prompt', 'system_message', 'user_message', 'generation_messages', 
                          'plan_messages', 'chat_messages', 'improvement_messages', 'reflection_messages',
                          'classification_messages', 'research_messages', 'guidelines_messages', 'content']:
                
                # Check string variables
                if isinstance(var_value, str) and len(var_value) >= len(prompt_content):
                    if self._is_rendered_content(var_value, prompt_content, prompt_vars):
                        print(f"[TRACE] Found rendered content in {var_name}")
                        return var_value
                
                # Check iterable objects (like message lists)
                elif hasattr(var_value, '__iter__') and not isinstance(var_value, str):
                    try:
                        for item in var_value:
                            content_str = str(getattr(item, 'content', item))
                            if len(content_str) >= len(prompt_content) and self._is_rendered_content(content_str, prompt_content, prompt_vars):
                                print(f"[TRACE] Found rendered content in {var_name} message")
                                return content_str
                    except Exception:
                        continue
        
        return prompt_content  # Fallback to original
    
    def _is_rendered_content(self, content: str, template: str, template_vars: list) -> bool:
        """Check if content appears to be a rendered version of the template."""
        if not template_vars:
            # For non-template prompts, check for exact content match
            return template.strip() in content
        
        # For template prompts, check if placeholders are replaced
        for var in template_vars:
            placeholder = "{" + var + "}"
            if placeholder in content:
                return False  # Still has placeholders
        
        # Check if template structure is preserved
        template_words = set(template.split())
        content_words = set(content.split())
        common_words = template_words & content_words
        
        # If more than 50% of template words are in content, it's likely rendered
        return len(common_words) / len(template_words) > 0.5 if template_words else False
    
    def _capture_variables_at_call(self, frame, trace_id: str):
        """Capture variables at function call time for better timing."""
        try:
            local_vars = frame.f_locals
            global_vars = frame.f_globals
            all_vars = {**global_vars, **local_vars}
            
            # Store all string variables that might be template values
            string_vars = {}
            for name, value in all_vars.items():
                if isinstance(value, str) and len(value) > 10:
                    string_vars[name] = value
            
            if string_vars:
                self._variable_cache[trace_id] = string_vars
                print(f"[TRACE] Cached {len(string_vars)} string variables for trace {trace_id}")
                
        except Exception as e:
            print(f"[TRACE] Error capturing variables at call: {e}")
    
    def _get_cached_variables(self, trace_id: str) -> dict:
        """Get cached variables for a trace."""
        return self._variable_cache.get(trace_id, {})
    
    def _store_prompt_usage(self, event_id: str, prompt_id: str, trace_id: str, function_name: str, frame=None):
        """Store prompt usage in the prompt_usages table."""
        try:
            from datetime import datetime
            
            print(f"[TRACE] _store_prompt_usage called with prompt_id={prompt_id}, function={function_name}")
            
            conn = sqlite3.connect(self.database_path)
            usage_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            
            # Check if prompt exists and get its content
            cursor = conn.execute('SELECT id, prompt_name, content, variables FROM prompts WHERE id = ?', (prompt_id,))
            result = cursor.fetchone()
            if not result:
                print(f"[TRACE] ERROR: Prompt {prompt_id} not found in prompts table")
                logger.warning(f"Prompt {prompt_id} not found in prompts table")
                conn.close()
                return
            
            prompt_id_db, prompt_name, prompt_content, prompt_variables_json = result
            print(f"[TRACE] Found prompt: {prompt_name}")
            
            # Extract variable values and rendered content if we have frame context
            variable_values = {}
            rendered_content = prompt_content  # Default to original content
            prompt_variables = {}
            
            if frame:
                try:
                    local_vars = frame.f_locals
                    global_vars = frame.f_globals
                    all_vars = {**global_vars, **local_vars}
                    
                    # Also try cached variables from function call time
                    cached_vars = self._get_cached_variables(trace_id)
                    all_vars.update(cached_vars)
                    
                    print(f"[TRACE] Available variables in scope: {list(local_vars.keys())}")
                    print(f"[TRACE] Cached variables: {list(cached_vars.keys())}")
                    
                    # Parse template variables if they exist
                    if prompt_variables_json:
                        prompt_variables = json.loads(prompt_variables_json)
                        print(f"[TRACE] Template variables to find: {prompt_variables}")
                        
                        # Enhanced variable finding with better matching strategies
                        for var_name, var_type in prompt_variables.items():
                            found_value = self._find_variable_value(var_name, all_vars)
                            if found_value is not None:
                                value_str = str(found_value)
                                variable_values[var_name] = value_str
                                print(f"[TRACE] Found variable {var_name} = {value_str[:100]}... (length: {len(value_str)})")
                            else:
                                print(f"[TRACE] Variable {var_name} not found in scope")
                            

                    else:
                        print(f"[TRACE] No template variables defined for this prompt")
                    
                    # Enhanced rendered content detection
                    rendered_content = self._find_rendered_content(prompt_content, prompt_variables, variable_values, all_vars)
                                
                except Exception as e:
                    print(f"[TRACE] Error extracting variables: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Debug the rendered content before storing
            print(f"[TRACE] Original prompt_content length: {len(prompt_content)}")
            print(f"[TRACE] Rendered content length: {len(rendered_content) if rendered_content else 0}")
            print(f"[TRACE] Are they equal? {rendered_content == prompt_content if rendered_content else False}")
            
            # Store the full rendered content without size limits
            
            # Insert usage record with correct schema
            conn.execute('''
                INSERT INTO prompt_usages 
                (id, trace_id, prompt_id, timestamp, variable_values, rendered_content)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                usage_id, trace_id, prompt_id, now,
                json.dumps(variable_values) if variable_values else json.dumps({}),
                rendered_content  # Always store rendered content, even if same as original
            ))
            
            conn.commit()
            conn.close()
            print(f"[TRACE] Successfully stored prompt usage: {usage_id} for prompt {prompt_id}")
            print(f"[TRACE] Variables captured: {variable_values}")
            logger.info(f"Successfully stored prompt usage: {usage_id} for prompt {prompt_id}")
            
        except Exception as e:
            print(f"[TRACE] ERROR storing prompt usage: {e}")
            logger.error(f"Failed to store prompt usage: {e}")
            import traceback
            traceback.print_exc()
    
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
                conn.commit()
                conn.close()
                return existing[0]  # Return the existing prompt_id
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
            
            # Special handling for module-level code execution
            if func_name == '<module>':
                # Detect and store prompts at module level
                self._detect_and_store_prompts(frame)
            
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
                
                # Capture variables at function entry for better timing
                self._capture_variables_at_call(frame, current_trace_id)
                
                # Get or create call stack for this thread
                if thread_id not in self._call_stack:
                    self._call_stack[thread_id] = []
                
                # Update depth
                self._trace_depth_context.set(current_depth + 1)
                
                # Extract class and function information
                class_name, full_function_name = self._extract_class_and_function_name(frame)
                
                # Store function information
                self._store_function_info(frame, full_function_name, class_name)
                
                # Get function arguments
                args_info = self._get_function_args(frame)
                
                # Detect and store prompts in current scope
                used_prompt_id = self._detect_and_store_prompts(frame)
                
                # Additional check: Look for any prompt usage in function that processes messages
                if not used_prompt_id and func_name in ['chat_response', 'classify_intent', 'improve_draft', 
                                                         'create_plan', 'research_for_plan', 'generate_essay',
                                                         'reflect_on_draft', 'research_for_critique', 
                                                         'research_for_brand', 'generate_brand_guidelines']:
                    # These functions are known to use prompts, link them if we find prompts in scope
                    used_prompt_id = self._find_prompt_in_scope(frame)
                    if used_prompt_id:
                        print(f"[TRACE] Found prompt {used_prompt_id} for function {func_name}")
                
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
                    'frame': frame,  # Store frame for later variable extraction
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
                    
                    # If a prompt was used, record it in prompt_usages table
                    if call_info.get('used_prompt_id'):
                        print(f"[TRACE] Recording prompt usage for function {func_name} with prompt {call_info['used_prompt_id']}")
                        logger.info(f"Recording prompt usage for function {func_name} with prompt {call_info['used_prompt_id']}")
                        self._store_prompt_usage(
                            call_info['id'], 
                            call_info['used_prompt_id'],
                            call_info['trace_id'],
                            func_name,
                            call_info.get('frame')  # Pass frame for variable extraction
                        )
                    
                    self._event_queue.put(event_data)
                    
                    # Decrease depth
                    current_depth = self._trace_depth_context.get()
                    if current_depth > 0:
                        self._trace_depth_context.set(current_depth - 1)
                    
                    # Clean up variable cache for this trace
                    if current_trace_id in self._variable_cache:
                        del self._variable_cache[current_trace_id]
                    
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
                        'used_prompt_id': None,  # Removed since we can't reliably link prompts to usage
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
    
    def _extract_class_and_function_name(self, frame) -> tuple:
        """Extract class name and full function name from frame."""
        try:
            func_name = frame.f_code.co_name
            filename = frame.f_code.co_filename
            
            # Try to get class name from frame locals
            class_name = None
            local_vars = frame.f_locals
            
            # Look for 'self' parameter (instance method)
            if 'self' in local_vars:
                self_obj = local_vars['self']
                class_name = self_obj.__class__.__name__
            # Look for 'cls' parameter (class method)
            elif 'cls' in local_vars:
                cls_obj = local_vars['cls']
                class_name = cls_obj.__name__
            else:
                # Try to infer from qualname if available
                try:
                    # Look in globals for the function object
                    for name, obj in frame.f_globals.items():
                        if (hasattr(obj, '__code__') and 
                            obj.__code__ is frame.f_code and
                            hasattr(obj, '__qualname__') and
                            '.' in obj.__qualname__):
                            # qualname like "ClassName.method_name"
                            parts = obj.__qualname__.split('.')
                            if len(parts) >= 2:
                                class_name = '.'.join(parts[:-1])  # Everything except last part
                            break
                except Exception:
                    pass
            
            # Create full function name
            if class_name:
                full_function_name = f"{class_name}.{func_name}"
            else:
                full_function_name = func_name
                
            return class_name, full_function_name
            
        except Exception as e:
            logger.error(f"Failed to extract class and function name: {e}")
            return None, frame.f_code.co_name
    
    def _store_function_info(self, frame, full_function_name: str, class_name: str = None):
        """Store function information in functions table."""
        try:
            from datetime import datetime
            import inspect
            
            func_name = frame.f_code.co_name
            filename = frame.f_code.co_filename
            
            # Get module name
            module_name = self._get_module_name(filename)
            
            # Try to extract signature and docstring
            signature = None
            docstring = None
            
            try:
                # Look for the actual function object in globals
                for name, obj in frame.f_globals.items():
                    if (hasattr(obj, '__code__') and 
                        obj.__code__ is frame.f_code):
                        if hasattr(inspect, 'signature'):
                            try:
                                signature = str(inspect.signature(obj))
                            except Exception:
                                pass
                        docstring = inspect.getdoc(obj)
                        break
            except Exception:
                pass
            
            # Store in database
            now = datetime.utcnow().isoformat()
            function_id = str(uuid.uuid4())
            
            conn = sqlite3.connect(self.database_path)
            
            # Check if function already exists
            cursor = conn.execute(
                'SELECT id, call_count FROM functions WHERE full_function_name = ? AND filename = ?',
                (full_function_name, filename)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing function
                conn.execute('''
                    UPDATE functions 
                    SET last_seen = ?, call_count = call_count + 1
                    WHERE id = ?
                ''', (now, existing[0]))
            else:
                # Insert new function
                conn.execute('''
                    INSERT INTO functions 
                    (id, function_name, full_function_name, class_name, filename, 
                     module_name, first_seen, last_seen, call_count, signature, docstring)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ''', (
                    function_id, func_name, full_function_name, class_name, filename,
                    module_name, now, now, signature, docstring
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store function info: {e}")
    
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
            
            # Prompt usage tracking removed since we can't reliably link functions to specific prompts
            
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
        
        # Usage patterns now based on usage_count in prompts table (updated when prompts are detected)
        usage_patterns = []
        for prompt in prompts:
            usage_patterns.append({
                'prompt_name': prompt['name'],
                'usage_count': prompt['usage_count'],
                'used_in_functions': []  # Can't determine without proper linking
            })
        
        # No detailed usage records since we can't reliably link functions to specific prompts
        recent_usages = []
        
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