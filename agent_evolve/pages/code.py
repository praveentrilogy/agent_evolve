"""
Code page for Agent Evolve Dashboard
"""

import streamlit as st
import sqlite3
import pandas as pd
import os

st.set_page_config(
    page_title="Code - Agent Evolve",
    page_icon="💻",
    layout="wide"
)

# Add Agent Evolve title at the top of sidebar
st.sidebar.title("🧬 Agent Evolve")

# Database info
db_path = os.environ.get('AGENT_EVOLVE_DB_PATH', 'data/graph.db')
if os.path.exists(db_path):
    st.sidebar.success(f"✅ Database connected")
    st.sidebar.caption(f"Path: {db_path}")
else:
    st.sidebar.error(f"❌ Database not found")
    st.sidebar.caption(f"Expected: {db_path}")


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
            ORDER BY call_count DESC, function_name ASC
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




# Main code page content
st.header("💻 Code")

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
        class_methods = sum(1 for f in functions if f['class_name'])
        st.metric("Class Methods", class_methods)
    with col4:
        regular_functions = sum(1 for f in functions if not f['class_name'])
        st.metric("Functions", regular_functions)
    
    st.divider()
    
    # Create table data without checkboxes first
    table_data = []
    for func in functions:
        display_name = func['full_name'] if func['class_name'] else func['name']
        
        table_data.append({
            "ID": func['id'],
            "Evolve": func.get('marked_for_evolution', 0) == 1,
            "Function": display_name,
            "Class": func['class_name'] or "-",
            "Module": func['module_name'] or "Unknown",
            "File": os.path.basename(func['filename']),
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
            )
        },
        disabled=["Function", "Class", "Module", "File", "Call Count", "First Seen", "Last Seen"],
        hide_index=True,
        use_container_width=True
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
                conn.execute('''
                    UPDATE functions 
                    SET marked_for_evolution = ? 
                    WHERE id = ?
                ''', [1 if current_state else 0, function_id])
                conn.commit()
                conn.close()
                
                # Show feedback
                function_name = functions[i]['full_name'] if functions[i]['class_name'] else functions[i]['name']
                if current_state:
                    st.success(f"✅ Marked '{function_name}' for evolution")
                else:
                    st.info(f"ℹ️ Unmarked '{function_name}' for evolution")
                    
                st.rerun()
                
            except Exception as e:
                st.error(f"Error updating function: {e}")
    
    # Detailed view selector
    st.divider()
    st.subheader("Detailed View")
    
    selected_function = st.selectbox(
        "Select a function to view details:",
        options=[f['full_name'] if f['class_name'] else f['name'] for f in functions],
        index=0 if functions else None
    )
    
    if selected_function:
        func = next(f for f in functions if (f['full_name'] if f['class_name'] else f['name']) == selected_function)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if func['signature']:
                st.subheader("Signature")
                st.code(func['signature'], language='python')
            
            if func['docstring']:
                st.subheader("Documentation")
                st.write(func['docstring'])
        
        with col2:
            st.subheader("Details")
            st.write(f"**Function:** {func['name']}")
            if func['class_name']:
                st.write(f"**Class:** {func['class_name']}")
            st.write(f"**Module:** {func['module_name'] or 'Unknown'}")
            st.write(f"**File:** {os.path.basename(func['filename'])}")
            st.write(f"**Call Count:** {func['call_count']}")
            st.write(f"**First Seen:** {func['first_seen'][:19]}")
            st.write(f"**Last Seen:** {func['last_seen'][:19]}")