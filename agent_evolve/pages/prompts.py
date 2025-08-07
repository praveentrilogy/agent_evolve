"""
Prompts page for Agent Evolve Dashboard
"""

import streamlit as st
import sqlite3
import json
import pandas as pd
import os

st.set_page_config(
    page_title="Prompts - Agent Evolve",
    page_icon="📝",
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


def get_prompt_usage(prompt_id: str) -> list:
    """Get usage records for a specific prompt"""
    import sqlite3
    
    conn = sqlite3.connect('agent_rl/backend/data/graph.db')
    cursor = conn.execute('''
        SELECT 
            pu.timestamp,
            te.function_name,
            pu.variable_values,
            pu.rendered_content
        FROM prompt_usages pu
        LEFT JOIN trace_events te ON te.id = (
            SELECT id FROM trace_events 
            WHERE trace_id = pu.trace_id 
            AND event_type = 'call'
            ORDER BY timestamp DESC
            LIMIT 1
        )
        WHERE pu.prompt_id = ?
        ORDER BY pu.timestamp DESC
        LIMIT 10
    ''', (prompt_id,))
    
    usage_records = []
    for row in cursor.fetchall():
        timestamp, function_name, variable_values, rendered_content = row
        usage_records.append({
            'timestamp': timestamp,
            'function_name': function_name or 'unknown',
            'variable_values': variable_values,
            'rendered_content': rendered_content
        })
    
    conn.close()
    return usage_records

def load_prompts_from_db(db_path):
    """Load all prompts from the database."""
    try:
        if not os.path.exists(db_path):
            return []
            
        conn = sqlite3.connect(db_path)
        cursor = conn.execute('''
            SELECT id, prompt_name, prompt_type, definition_location, full_code, 
                   content, variables, function_signature, enum_values, 
                   created_at, last_seen, usage_count, marked_for_evolution
            FROM prompts 
            ORDER BY definition_location ASC, prompt_name ASC
        ''')
        
        prompts = []
        for row in cursor.fetchall():
            prompt_id, name, ptype, location, full_code, content, variables_json, \
            func_sig, enum_json, created_at, last_seen, usage_count, marked_for_evolution = row
            
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
                'usage_count': usage_count,
                'marked_for_evolution': marked_for_evolution
            })
        
        conn.close()
        return prompts
        
    except Exception as e:
        st.error(f"Error loading prompts: {e}")
        return []



# Main prompts page content
st.header("📝 Prompts")

prompts = load_prompts_from_db(db_path)

if not prompts:
    st.warning("No prompts found in the database.")
    st.info("Prompts will appear here after running your traced application.")
else:
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Prompts", len(prompts))
    with col2:
        total_usage = sum(p['usage_count'] for p in prompts)
        st.metric("Total Usage", total_usage)
    with col3:
        template_count = sum(1 for p in prompts if p['type'] == 'template')
        st.metric("Templates", template_count)
    with col4:
        constant_count = sum(1 for p in prompts if p['type'] == 'constant')
        st.metric("Constants", constant_count)
    
    st.divider()
    
    # Create table data without checkboxes first
    table_data = []
    for prompt in prompts:
        variables_str = ", ".join([f"{k}({v})" for k, v in prompt['variables'].items()]) if prompt['variables'] else "-"
        preview = prompt['content'][:100] + "..." if len(prompt['content']) > 100 else prompt['content']
        
        table_data.append({
            "ID": prompt['id'],
            "Evolve": prompt.get('marked_for_evolution', 0) == 1,
            "Name": prompt['name'],
            "Type": prompt['type'],
            "Usage Count": prompt['usage_count'],
            "Variables": variables_str,
            "Location": prompt['definition_location'],
            "Content Preview": preview,
            "First Seen": prompt['created_at'][:19],
            "Last Seen": prompt['last_seen'][:19]
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
            "Name": st.column_config.TextColumn(
                "Name",
                help="Prompt name",
                max_chars=50,
            ),
            "Type": st.column_config.TextColumn(
                "Type",
                help="Prompt type",
                max_chars=20,
            ),
            "Usage Count": st.column_config.NumberColumn(
                "Usage Count",
                help="Number of times used",
                format="%d",
            ),
            "Variables": st.column_config.TextColumn(
                "Variables",
                help="Template variables",
                max_chars=50,
            ),
            "Location": st.column_config.TextColumn(
                "Location",
                help="File location",
                max_chars=100,
            ),
            "Content Preview": st.column_config.TextColumn(
                "Content Preview",
                help="Preview of prompt content",
                max_chars=150,
            ),
            "First Seen": st.column_config.TextColumn(
                "First Seen",
                help="First time seen",
                max_chars=20,
            ),
            "Last Seen": st.column_config.TextColumn(
                "Last Seen",
                help="Last time seen",
                max_chars=20,
            ),
        },
        disabled=["Name", "Type", "Usage Count", "Variables", "Location", "Content Preview", "First Seen", "Last Seen"],
        hide_index=True,
        use_container_width=True
    )
    
    # Detect changes and update database automatically
    for i, (_, row) in enumerate(edited_df.iterrows()):
        prompt_id = table_data[i]['ID']
        current_state = row['Evolve']
        original_state = prompts[i].get('marked_for_evolution', 0) == 1
        
        # If state changed, update database
        if current_state != original_state:
            try:
                conn = sqlite3.connect(db_path)
                conn.execute('''
                    UPDATE prompts 
                    SET marked_for_evolution = ? 
                    WHERE id = ?
                ''', [1 if current_state else 0, prompt_id])
                conn.commit()
                conn.close()
                
                # Show feedback
                if current_state:
                    st.success(f"✅ Marked '{prompts[i]['name']}' for evolution")
                else:
                    st.info(f"ℹ️ Unmarked '{prompts[i]['name']}' for evolution")
                    
                st.rerun()
                
            except Exception as e:
                st.error(f"Error updating prompt: {e}")
    
    # Detailed view selector
    st.divider()
    st.subheader("Detailed View")
    
    selected_prompt = st.selectbox(
        "Select a prompt to view details:",
        options=[p['name'] for p in prompts],
        index=0 if prompts else None
    )
    
    if selected_prompt:
        prompt = next(p for p in prompts if p['name'] == selected_prompt)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Content")
            st.code(prompt['content'], language='text')
            
            if prompt['full_code'] and not prompt['full_code'].startswith('<'):
                st.subheader("Source Code")
                st.code(prompt['full_code'], language='python')
        
        with col2:
            st.subheader("Details")
            st.write(f"**Type:** {prompt['type']}")
            st.write(f"**Location:** {prompt['definition_location']}")
            st.write(f"**Usage Count:** {prompt['usage_count']}")
            
            if prompt['variables']:
                st.write("**Template Variables:**")
                for var, var_type in prompt['variables'].items():
                    st.write(f"  - `{var}` ({var_type})")
            
            if prompt['function_signature']:
                st.write(f"**Signature:** `{prompt['function_signature']}`")
            
            st.write(f"**First Seen:** {prompt['created_at'][:19]}")
            st.write(f"**Last Seen:** {prompt['last_seen'][:19]}")
            
            # Add prompt usage details
            st.subheader("Recent Usage")
            usage_data = get_prompt_usage(prompt['id'])
            if usage_data:
                for usage in usage_data[:5]:  # Show last 5 usages
                    with st.expander(f"Used in {usage['function_name']} at {usage['timestamp'][:19]}"):
                        if usage['variable_values']:
                            st.write("**Variable Values:**")
                            import json
                            try:
                                var_values = json.loads(usage['variable_values'])
                                for var_name, var_value in var_values.items():
                                    st.code(f"{var_name} = {var_value}", language='python')
                            except:
                                st.text(usage['variable_values'])
                        
                        if usage['rendered_content']:
                            st.write("**Rendered Content:**")
                            st.text_area("", value=usage['rendered_content'], height=200, disabled=True, key=f"rendered_{usage['timestamp']}")
                        else:
                            st.write("No rendered content available.")
            else:
                st.write("No usage records found.")