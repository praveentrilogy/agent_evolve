import streamlit as st
import pandas as pd
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

def connect_to_database(db_path: str) -> Optional[sqlite3.Connection]:
    """Connect to the trace database"""
    try:
        if not Path(db_path).exists():
            return None
        return sqlite3.connect(db_path)
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return None

def get_prompts_from_db(conn: sqlite3.Connection) -> List[Dict]:
    """Get all prompts from the database"""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, prompt_name, content, created_at, updated_at
            FROM prompts
            ORDER BY prompt_name
        """)
        
        prompts = []
        for row in cursor.fetchall():
            prompts.append({
                'id': row[0],
                'name': row[1],
                'content': row[2],
                'created_at': row[3],
                'updated_at': row[4]
            })
        return prompts
    except Exception as e:
        st.error(f"Failed to fetch prompts: {e}")
        return []

def get_functions_from_db(conn: sqlite3.Connection) -> List[Dict]:
    """Get all functions from the database"""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, full_function_name, signature, first_seen, last_seen
            FROM functions
            ORDER BY full_function_name
        """)
        
        functions = []
        for row in cursor.fetchall():
            functions.append({
                'id': row[0],
                'name': row[1],
                'signature': row[2],
                'first_seen': row[3],
                'last_seen': row[4]
            })
        return functions
    except Exception as e:
        st.error(f"Failed to fetch functions: {e}")
        return []

def get_trace_events_for_prompt(conn: sqlite3.Connection, prompt_id: str, limit: int = 100) -> List[Dict]:
    """Get trace events that used a specific prompt"""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT te.id, te.trace_id, te.timestamp, te.function_name, te.args, te.result, 
                   te.duration_ms, te.success, te.error, te.metadata
            FROM trace_events te
            WHERE te.used_prompt_id = ?
            ORDER BY te.timestamp DESC
            LIMIT ?
        """, (prompt_id, limit))
        
        events = []
        for row in cursor.fetchall():
            # Parse JSON fields safely
            try:
                args = json.loads(row[4]) if row[4] else {}
            except:
                args = row[4] if row[4] else {}
            
            try:
                result = json.loads(row[5]) if row[5] else None
            except:
                result = row[5] if row[5] else None
            
            try:
                metadata = json.loads(row[9]) if row[9] else {}
            except:
                metadata = row[9] if row[9] else {}
            
            events.append({
                'id': row[0],
                'trace_id': row[1],
                'timestamp': row[2],
                'function_name': row[3],
                'args': args,
                'result': result,
                'duration_ms': row[6],
                'success': row[7],
                'error': row[8],
                'metadata': metadata
            })
        return events
    except Exception as e:
        st.error(f"Failed to fetch trace events for prompt: {e}")
        return []

def get_trace_events_for_function(conn: sqlite3.Connection, function_name: str, limit: int = 100) -> List[Dict]:
    """Get trace events for a specific function"""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT te.id, te.trace_id, te.timestamp, te.args, te.result, 
                   te.duration_ms, te.success, te.error, te.metadata
            FROM trace_events te
            WHERE te.function_name = ?
            ORDER BY te.timestamp DESC
            LIMIT ?
        """, (function_name, limit))
        
        events = []
        for row in cursor.fetchall():
            # Parse JSON fields safely
            try:
                args = json.loads(row[3]) if row[3] else {}
            except:
                args = row[3] if row[3] else {}
            
            try:
                result = json.loads(row[4]) if row[4] else None
            except:
                result = row[4] if row[4] else None
            
            try:
                metadata = json.loads(row[8]) if row[8] else {}
            except:
                metadata = row[8] if row[8] else {}
            
            events.append({
                'id': row[0],
                'trace_id': row[1],
                'timestamp': row[2],
                'args': args,
                'result': result,
                'duration_ms': row[5],
                'success': row[6],
                'error': row[7],
                'metadata': metadata
            })
        return events
    except Exception as e:
        st.error(f"Failed to fetch trace events for function: {e}")
        return []

def render_trace_events_table(events: List[Dict], item_type: str = "prompt"):
    """Render trace events as a table"""
    if not events:
        st.info("No trace events found for this item.")
        return
    
    # Create a flattened view for the table
    flattened_events = []
    for event in events:
        flattened_event = {
            'Timestamp': event['timestamp'],
            'Trace ID': event['trace_id'][:8] + "...",
            'Duration (ms)': event['duration_ms'],
            'Success': "✅" if event['success'] else "❌",
        }
        
        if item_type == "prompt":
            flattened_event['Function'] = event['function_name']
        
        # Add args preview
        if isinstance(event['args'], dict):
            args_preview = str(event['args'])[:100] + "..." if len(str(event['args'])) > 100 else str(event['args'])
        else:
            args_preview = str(event['args'])[:100] + "..." if len(str(event['args'])) > 100 else str(event['args'])
        flattened_event['Args Preview'] = args_preview
        
        # Add result preview
        if event['result'] is not None:
            result_preview = str(event['result'])[:100] + "..." if len(str(event['result'])) > 100 else str(event['result'])
        else:
            result_preview = "None"
        flattened_event['Result Preview'] = result_preview
        
        if event['error']:
            flattened_event['Error'] = event['error'][:100] + "..." if len(event['error']) > 100 else event['error']
        
        flattened_events.append(flattened_event)
    
    # Display as dataframe
    df = pd.DataFrame(flattened_events)
    st.dataframe(df, use_container_width=True)
    
    # Show summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Events", len(events))
    with col2:
        success_count = sum(1 for e in events if e['success'])
        st.metric("Successful", success_count)
    with col3:
        error_count = sum(1 for e in events if not e['success'])
        st.metric("Failed", error_count)
    
    # Show detailed view for selected event
    if events:
        st.subheader("Event Details")
        event_options = [f"{e['timestamp']} - {e['trace_id'][:8]}" for e in events]
        selected_event_idx = st.selectbox(
            "Select an event to view details:",
            range(len(event_options)),
            format_func=lambda x: event_options[x]
        )
        
        if selected_event_idx is not None:
            selected_event = events[selected_event_idx]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Input (Args):**")
                st.json(selected_event['args'])
            
            with col2:
                st.markdown("**Output (Result):**")
                if selected_event['result'] is not None:
                    st.json(selected_event['result'])
                else:
                    st.text("No result")
            
            if selected_event['error']:
                st.markdown("**Error:**")
                st.error(selected_event['error'])
            
            if selected_event['metadata']:
                st.markdown("**Metadata:**")
                st.json(selected_event['metadata'])

def render_real_app_data_tab(tool_data, selected_tool):
    """Render the Real App Data tab content"""
    
    st.subheader("📊 Real Application Data")
    
    # Find the database path dynamically
    tool_path = Path(tool_data['path'])
    current_dir = Path.cwd()
    
    # Look for database in common locations, prioritizing .agent_evolve paths
    possible_db_paths = [
        # Prioritize .agent_evolve locations first
        tool_path.parent.parent / ".agent_evolve" / "data" / "graph.db",
        current_dir / ".agent_evolve" / "data" / "graph.db",
        current_dir.parent / ".agent_evolve" / "data" / "graph.db",
        # Search up the directory tree for .agent_evolve patterns
        *[p / ".agent_evolve" / "data" / "graph.db" for p in current_dir.parents if (p / ".agent_evolve").exists()],
        # Fallback to regular data directories
        tool_path.parent.parent / "data" / "graph.db",
        current_dir / "data" / "graph.db",
        current_dir.parent / "data" / "graph.db",
        *[p / "data" / "graph.db" for p in current_dir.parents if (p / "data").exists()],
    ]
    
    db_path = None
    for path in possible_db_paths:
        if path.exists():
            db_path = path
            break
    
    if not db_path:
        st.warning("🔍 No trace database found. Make sure tracing is enabled and the application has been used.")
        st.info("Expected database locations:")
        for path in possible_db_paths:
            st.code(str(path))
        return
    
    st.success(f"📈 Connected to database: `{db_path}`")
    
    # Connect to database
    conn = connect_to_database(str(db_path))
    if not conn:
        return
    
    try:
        # Get all prompts and functions from database
        prompts = get_prompts_from_db(conn)
        functions = get_functions_from_db(conn)
        
        # Create tabs for prompts and functions
        prompt_tab, function_tab = st.tabs(["🔤 Prompts", "⚙️ Functions"])
        
        with prompt_tab:
            if not prompts:
                st.info("No prompts found in the database.")
            else:
                st.markdown(f"Found **{len(prompts)}** prompts in the database:")
                
                # Let user select a prompt
                prompt_names = [p['name'] for p in prompts]
                selected_prompt_name = st.selectbox(
                    "Select a prompt to view its usage data:",
                    prompt_names,
                    key="real_data_prompt_select"
                )
                
                if selected_prompt_name:
                    # Find selected prompt
                    selected_prompt = next(p for p in prompts if p['name'] == selected_prompt_name)
                    
                    # Show prompt details
                    with st.expander("📝 Prompt Details", expanded=False):
                        st.markdown(f"**Name:** {selected_prompt['name']}")
                        if 'created_at' in selected_prompt and selected_prompt['created_at']:
                            st.markdown(f"**Created:** {selected_prompt['created_at']}")
                        if 'updated_at' in selected_prompt and selected_prompt['updated_at']:
                            st.markdown(f"**Updated:** {selected_prompt['updated_at']}")
                        if 'content' in selected_prompt and selected_prompt['content']:
                            st.markdown("**Content:**")
                            st.code(selected_prompt['content'], language="text")
                        else:
                            st.info("No content available for this prompt")
                    
                    # Control for number of events to show
                    limit = st.slider(
                        "Number of recent events to show:",
                        min_value=10,
                        max_value=500,
                        value=100,
                        key="real_data_prompt_limit"
                    )
                    
                    # Get trace events for this prompt
                    with st.spinner("Loading trace events..."):
                        trace_events = get_trace_events_for_prompt(conn, selected_prompt['id'], limit)
                    
                    if trace_events:
                        st.markdown(f"### 📊 Usage Data for '{selected_prompt_name}'")
                        render_trace_events_table(trace_events, "prompt")
                    else:
                        st.info(f"No trace events found for prompt '{selected_prompt_name}'. The prompt may not have been used yet.")
        
        with function_tab:
            if not functions:
                st.info("No functions found in the database.")
            else:
                st.markdown(f"Found **{len(functions)}** functions in the database:")
                
                # Let user select a function
                function_names = [f['name'] for f in functions]
                selected_function_name = st.selectbox(
                    "Select a function to view its usage data:",
                    function_names,
                    key="real_data_function_select"
                )
                
                if selected_function_name:
                    # Find selected function
                    selected_function = next(f for f in functions if f['name'] == selected_function_name)
                    
                    # Show function details
                    with st.expander("⚙️ Function Details", expanded=False):
                        st.markdown(f"**Name:** {selected_function['name']}")
                        if selected_function.get('first_seen'):
                            st.markdown(f"**First Seen:** {selected_function['first_seen']}")
                        if selected_function.get('last_seen'):
                            st.markdown(f"**Last Seen:** {selected_function['last_seen']}")
                        if selected_function.get('signature'):
                            st.markdown("**Signature:**")
                            st.code(selected_function['signature'], language="python")
                    
                    # Control for number of events to show
                    limit = st.slider(
                        "Number of recent events to show:",
                        min_value=10,
                        max_value=500,
                        value=100,
                        key="real_data_function_limit"
                    )
                    
                    # Get trace events for this function
                    with st.spinner("Loading trace events..."):
                        trace_events = get_trace_events_for_function(conn, selected_function_name, limit)
                    
                    if trace_events:
                        st.markdown(f"### 📊 Usage Data for '{selected_function_name}'")
                        render_trace_events_table(trace_events, "function")
                    else:
                        st.info(f"No trace events found for function '{selected_function_name}'. The function may not have been called yet.")
        
        # Show database statistics
        st.markdown("---")
        st.subheader("📊 Database Statistics")
        
        # Get counts
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trace_events")
        total_events = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM prompts")
        total_prompts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM functions")
        total_functions = cursor.fetchone()[0]
        
        # Show stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Events", total_events)
        with col2:
            st.metric("Total Prompts", total_prompts)
        with col3:
            st.metric("Total Functions", total_functions)
        
    finally:
        conn.close()