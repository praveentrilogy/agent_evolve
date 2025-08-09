import streamlit as st
import sqlite3
import json
import pandas as pd
import os
import uuid
from datetime import datetime
from agent_evolve.training_data_generator import generate_training_data
from agent_evolve.evaluator_engine import auto_generate_evaluator_on_evolution

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
    
    conn = sqlite3.connect(db_path)
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

def load_evolve_queue(db_path):
    """Load the evolution queue from the database."""
    try:
        if not os.path.exists(db_path):
            return []
            
        conn = sqlite3.connect(db_path)
        cursor = conn.execute('''
            SELECT q.id, p.prompt_name, q.status, q.created_at, q.updated_at
            FROM prompt_evolve_queue q
            JOIN prompts p ON q.prompt_id = p.id
            ORDER BY q.created_at DESC
        ''')
        
        queue = []
        for row in cursor.fetchall():
            queue.append({
                'id': row[0],
                'prompt_name': row[1],
                'status': row[2],
                'created_at': row[3],
                'updated_at': row[4]
            })
        
        conn.close()
        return queue
        
    except Exception as e:
        st.error(f"Error loading evolution queue: {e}")
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
                cursor = conn.cursor()
                now = datetime.utcnow().isoformat()
                if current_state:
                    # Check if already in queue or running
                    cursor.execute("SELECT status FROM prompt_evolve_queue WHERE prompt_id = ?", (prompt_id,))
                    existing_queue_item = cursor.fetchone()

                    if existing_queue_item and existing_queue_item[0] in ['queued', 'running']:
                        st.warning(f"Prompt '{prompts[i]['name']}' is already in the queue or running (Status: {existing_queue_item[0]}).")
                    else:
                        # Add to queue or update status to queued if completed/failed
                        if existing_queue_item:
                            cursor.execute('''
                                UPDATE prompt_evolve_queue SET status = 'queued', updated_at = ? WHERE prompt_id = ?
                            ''', (now, prompt_id))
                        else:
                            cursor.execute('''
                                INSERT INTO prompt_evolve_queue (id, prompt_id, status, created_at, updated_at)
                                VALUES (?, ?, ?, ?, ?)
                            ''', (str(uuid.uuid4()), prompt_id, 'queued', now, now))
                            
                            # Auto-generate evaluator when marked for evolution
                            # Find project root by searching for .agent_evolve directory
                            import os
                            from pathlib import Path
                            current = Path.cwd()
                            project_root = None
                            while current != current.parent:
                                if (current / ".agent_evolve").exists():
                                    project_root = str(current)
                                    break
                                current = current.parent
                            if not project_root:
                                project_root = str(Path.cwd())
                            auto_generate_evaluator_on_evolution(db_path, prompt_id, 'prompt', project_root)
                    
                cursor.execute('''
                    UPDATE prompts 
                    SET marked_for_evolution = ? 
                    WHERE id = ?
                ''', [1 if current_state else 0, prompt_id])
                conn.commit()
                conn.close()
                    
                # Show feedback
                if current_state:
                    st.success(f"✅ Queued '{prompts[i]['name']}' for evolution")
                else:
                    st.info(f"ℹ️ Unmarked '{prompts[i]['name']}' for evolution")
                        
                st.rerun()
                    
            except Exception as e:
                st.error(f"Error updating prompt: {e}")
        
    st.markdown("💡 Use the checkboxes above to mark items for evolution, or use the sidebar to view detailed information.")