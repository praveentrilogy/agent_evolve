"""
Code page for Agent Evolve Dashboard
"""

import streamlit as st
import sqlite3
import pandas as pd
import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from agent_evolve.shared_sidebar import render_sidebar, get_selected_item
from agent_evolve.evaluator_engine import auto_generate_evaluator_on_evolution

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Code - Agent Evolve",
    page_icon="💻",
    layout="wide"
)

# Render shared sidebar and get database path
db_path = render_sidebar("code")

# Get selected item from query parameters
selected_item = get_selected_item()
selected_function_id = selected_item['id'] if selected_item and selected_item['type'] == 'code' else None

def get_function_details(function_id: str):
    """Get detailed information about a specific function"""
    if not os.path.exists(db_path):
        return None
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute('''
            SELECT id, function_name, full_function_name, class_name, filename, 
                   module_name, first_seen, last_seen, call_count, signature, docstring
            FROM functions 
            WHERE id = ?
        ''', (function_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
            
        function_id, name, full_name, class_name, filename, module_name, \
        first_seen, last_seen, call_count, signature, docstring = row
        
        function_details = {
            'id': function_id,
            'name': name,
            'full_name': full_name,
            'class_name': class_name,
            'filename': filename,
            'module_name': module_name,
            'first_seen': first_seen,
            'last_seen': last_seen,
            'call_count': call_count,
            'signature': signature,
            'docstring': docstring
        }
        
        conn.close()
        return function_details
        
    except Exception as e:
        st.error(f"Error loading function details: {e}")
        return None

def get_function_usage(function_id: str) -> list:
    """Get usage records for a specific function"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute('''
            SELECT timestamp, event_type, args, return_value, execution_time
            FROM trace_events 
            WHERE function_name = (SELECT full_function_name FROM functions WHERE id = ?)
            ORDER BY timestamp DESC
            LIMIT 10
        ''', (function_id,))
        
        usage_records = []
        for row in cursor.fetchall():
            timestamp, event_type, args, return_value, execution_time = row
            usage_records.append({
                'timestamp': timestamp,
                'event_type': event_type,
                'args': args,
                'return_value': return_value,
                'execution_time': execution_time
            })
        
        conn.close()
        return usage_records
    except Exception as e:
        st.error(f"Error loading function usage: {e}")
        return []

def queue_function_for_evolution(function_id: str, function_name: str):
    """Add a function to the evolution queue"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        
        # Check if already in queue or running
        cursor.execute("SELECT status FROM code_evolve_queue WHERE function_id = ?", (function_id,))
        existing_queue_item = cursor.fetchone()
        
        if existing_queue_item and existing_queue_item[0] in ['queued', 'running']:
            st.warning(f"Function '{function_name}' is already in the queue or running (Status: {existing_queue_item[0]}).")
        else:
            # Add to queue or update status to queued if completed/failed
            if existing_queue_item:
                cursor.execute('''
                    UPDATE code_evolve_queue SET status = 'queued', updated_at = ? WHERE function_id = ?
                ''', (now, function_id))
            else:
                cursor.execute('''
                    INSERT INTO code_evolve_queue (id, function_id, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (str(uuid.uuid4()), function_id, 'queued', now, now))
            conn.commit()
            
            # Auto-generate evaluator for the function
            auto_generate_evaluator_on_evolution(db_path, function_id, 'code')
            
            st.success(f"✅ Queued '{function_name}' for evolution")
            st.rerun()
        
        conn.close()
    except Exception as e:
        st.error(f"Error queueing function: {e}")

def load_functions_from_db(db_path):
    """Load all functions from the database."""
    try:
        if not os.path.exists(db_path):
            return []
            
        conn = sqlite3.connect(db_path)
        cursor = conn.execute('''
            SELECT id, function_name, full_function_name, class_name, filename, 
                   module_name, first_seen, last_seen, call_count, signature, docstring, marked_for_evolution
            FROM functions 
            ORDER BY filename ASC, function_name ASC
        ''')
        
        functions = []
        for row in cursor.fetchall():
            func_id, name, full_name, class_name, filename, module_name, \
            first_seen, last_seen, call_count, signature, docstring, marked_for_evolution = row
            
            functions.append({
                'id': func_id,
                'name': name,
                'full_name': full_name,
                'class_name': class_name,
                'filename': filename,
                'module_name': module_name,
                'first_seen': first_seen,
                'last_seen': last_seen,
                'call_count': call_count,
                'signature': signature,
                'docstring': docstring,
                'marked_for_evolution': marked_for_evolution
            })
        
        conn.close()
        return functions
        
    except Exception as e:
        st.error(f"Error loading functions: {e}")
        return []




# Main content
if selected_function_id:
    # Show detailed view for selected function
    function_details = get_function_details(selected_function_id)
    if function_details:
        st.header(f"💻 {function_details['full_name']}")
        st.caption(f"Module: {function_details['module_name']} | File: {function_details['filename']}")
        
        # Add queue button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🚀 Queue for Evolution", key="queue_btn"):
                queue_function_for_evolution(selected_function_id, function_details['full_name'])
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Details", "Usage", "Signature", "Source"])
        
        with tab1:
            st.subheader("Function Details")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Function Name:** {function_details['name']}")
                st.write(f"**Full Name:** {function_details['full_name']}")
                if function_details['class_name']:
                    st.write(f"**Class:** {function_details['class_name']}")
                st.write(f"**Call Count:** {function_details['call_count']}")
            
            with col2:
                st.write(f"**Module:** {function_details['module_name']}")
                st.write(f"**File:** {function_details['filename']}")
                st.write(f"**First Seen:** {function_details['first_seen'][:19]}")
                st.write(f"**Last Seen:** {function_details['last_seen'][:19]}")
            
            if function_details['docstring']:
                st.subheader("Documentation")
                st.text(function_details['docstring'])
        
        with tab2:
            st.subheader("Recent Usage")
            usage_data = get_function_usage(selected_function_id)
            if usage_data:
                for i, usage in enumerate(usage_data[:5]):  # Show last 5 usages
                    with st.expander(f"{usage['event_type'].upper()} at {usage['timestamp'][:19]}", expanded=(i==0)):
                        if usage['args']:
                            st.write("**Arguments:**")
                            try:
                                args_data = json.loads(usage['args']) if usage['args'] else {}
                                if args_data:
                                    st.json(args_data)
                                else:
                                    st.write("No arguments")
                            except:
                                st.text(usage['args'])
                        
                        if usage['return_value']:
                            st.write("**Return Value:**")
                            try:
                                return_data = json.loads(usage['return_value'])
                                st.json(return_data)
                            except:
                                st.text(usage['return_value'])
                        
                        if usage['execution_time']:
                            st.write(f"**Execution Time:** {usage['execution_time']}ms")
            else:
                st.write("No usage records found.")
        
        with tab3:
            st.subheader("Function Signature")
            if function_details['signature']:
                st.code(function_details['signature'], language='python')
            else:
                st.write("No signature information available.")
        
        with tab4:
            st.subheader("Source Code")
            st.write("Source code extraction not yet implemented.")
            st.info("💡 This will show the actual source code of the function when implemented.")
    else:
        st.error("Function not found.")
else:
    # Show main code page
    st.header("💻 Code")
    st.write("Select a function from the table or sidebar to view details.")
    
    functions = load_functions_from_db(db_path)
    
    if not functions:
        st.warning("No functions found in the database.")
        st.info("Functions will appear here after running your traced application.")
    else:
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Functions", len(functions))
        with col2:
            total_calls = sum(f['call_count'] for f in functions)
            st.metric("Total Calls", total_calls)
        with col3:
            marked_count = sum(1 for f in functions if f.get('marked_for_evolution', 0) == 1)
            st.metric("Marked for Evolution", marked_count)
        with col4:
            unique_modules = len(set(f['module_name'] for f in functions if f['module_name']))
            st.metric("Unique Modules", unique_modules)
        
        st.divider()
        
        # Create table data
        table_data = []
        for func in functions:
            file_short = os.path.basename(func['filename']) if func['filename'] else "-"
            
            table_data.append({
                "ID": func['id'],
                "Evolve": func.get('marked_for_evolution', 0) == 1,
                "Function": func['full_name'],
                "Class": func['class_name'] or "-",
                "Module": func['module_name'] or "-",
                "File": file_short,
                "Call Count": func['call_count'],
                "First Seen": func['first_seen'][:19],
                "Last Seen": func['last_seen'][:19]
            })
        
        # Display as dataframe with selection
        df = pd.DataFrame(table_data)
        
        # Use st.data_editor for interactive selection
        edited_df = st.data_editor(
            df.drop(columns=['ID']),
            column_config={
                "Evolve": st.column_config.CheckboxColumn(
                    "Evolve",
                    help="Select for evolution",
                    default=False,
                ),
                "Function": st.column_config.TextColumn(
                    "Function",
                    help="Function name",
                    max_chars=60,
                ),
            },
            disabled=["Class", "Module", "File", "Call Count", "First Seen", "Last Seen"],
            hide_index=True,
            use_container_width=True,
            on_change=lambda: st.rerun()
        )
        
        # Detect changes and update database automatically
        for i, (_, row) in enumerate(edited_df.iterrows()):
            function_id = table_data[i]['ID']
            current_state = row['Evolve']
            original_state = functions[i].get('marked_for_evolution', 0) == 1
            
            # If state changed, update database
            if current_state != original_state:
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    now = datetime.utcnow().isoformat()
                    
                    if current_state:
                        # Check if already in queue or running
                        cursor.execute("SELECT status FROM code_evolve_queue WHERE function_id = ?", (function_id,))
                        existing_queue_item = cursor.fetchone()

                        if existing_queue_item and existing_queue_item[0] in ['queued', 'running']:
                            st.warning(f"Function '{functions[i]['full_name']}' is already in the queue or running (Status: {existing_queue_item[0]}).")
                        else:
                            # Add to queue or update status to queued if completed/failed
                            if existing_queue_item:
                                cursor.execute('''
                                    UPDATE code_evolve_queue SET status = 'queued', updated_at = ? WHERE function_id = ?
                                ''', (now, function_id))
                            else:
                                cursor.execute('''
                                    INSERT INTO code_evolve_queue (id, function_id, status, created_at, updated_at)
                                    VALUES (?, ?, ?, ?, ?)
                                ''', (str(uuid.uuid4()), function_id, 'queued', now, now))
                            
                            # Auto-generate evaluator when marked for evolution
                            auto_generate_evaluator_on_evolution(db_path, function_id, 'code')
                    
                    cursor.execute('''
                        UPDATE functions 
                        SET marked_for_evolution = ? 
                        WHERE id = ?
                    ''', [1 if current_state else 0, function_id])
                    conn.commit()
                    conn.close()
                    
                    # Show feedback
                    if current_state:
                        st.success(f"✅ Queued '{functions[i]['full_name']}' for evolution")
                    else:
                        st.info(f"ℹ️ Unmarked '{functions[i]['full_name']}' for evolution")
                        
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error updating function: {e}")
        
        
        st.markdown("💡 Use the checkboxes above to mark items for evolution, or use the sidebar to view detailed information.")
