"""
Shared sidebar component for Agent Evolve Dashboard
"""

import streamlit as st
import sqlite3
import os

def load_evolution_targets(db_path):
    """Load all items marked for evolution from the database."""
    if not os.path.exists(db_path):
        return [], []
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Load prompts marked for evolution
        cursor = conn.execute('''
            SELECT id, prompt_name, prompt_type, usage_count
            FROM prompts 
            WHERE marked_for_evolution = 1
            ORDER BY prompt_name
        ''')
        prompts = [{'id': row[0], 'name': row[1], 'type': row[2], 'usage_count': row[3]} 
                   for row in cursor.fetchall()]
        
        # Load functions marked for evolution (if table exists)
        try:
            cursor = conn.execute('''
                SELECT id, full_function_name, filename, call_count
                FROM functions 
                WHERE marked_for_evolution = 1
                ORDER BY full_function_name
            ''')
            code_snippets = [{'id': row[0], 'name': row[1], 'path': row[2], 'usage_count': row[3]} 
                           for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            # Table doesn't exist or column doesn't exist yet
            code_snippets = []
        
        conn.close()
        return prompts, code_snippets
    except Exception as e:
        return [], []

def render_sidebar(page_type=None):
    """Render the consistent sidebar for all pages."""
    # Database info
    db_path = os.environ.get('AGENT_EVOLVE_DB_PATH', 'data/graph.db')
    
    # Load evolution targets
    evolution_prompts, evolution_code = load_evolution_targets(db_path)
    
    # Evolution Targets section at top of sidebar
    st.sidebar.title("🎯 Evolution Targets")
    
    if evolution_prompts or evolution_code:
        # Show relevant section based on page type
        if page_type == "prompts" and evolution_prompts:
            st.sidebar.subheader("📝 Prompts")
            for prompt in evolution_prompts:
                # Create a clickable link
                if st.sidebar.button(f"{prompt['name']}", key=f"prompt_{prompt['id']}", use_container_width=True, type="secondary"):
                    st.query_params.update({
                        "item_type": "prompt", 
                        "item_id": prompt['id'],
                        "item_name": prompt['name']
                    })
                    st.rerun()
        elif page_type == "code" and evolution_code:
            st.sidebar.subheader("💻 Code")
            for code in evolution_code:
                # Create a clickable link
                if st.sidebar.button(f"{code['name']}", key=f"code_{code['id']}", use_container_width=True, type="secondary"):
                    st.query_params.update({
                        "item_type": "code",
                        "item_id": code['id'],
                        "item_name": code['name']
                    })
                    st.rerun()
        else:
            # For other pages, show nothing
            st.sidebar.write("Navigate to Prompts or Code pages to see evolution targets.")
    else:
        st.sidebar.write("No items marked for evolution yet.")
    
    st.sidebar.divider()
    
    # Database connection status
    if os.path.exists(db_path):
        st.sidebar.success(f"✅ Database connected")
        st.sidebar.caption(f"Path: {db_path}")
    else:
        st.sidebar.error(f"❌ Database not found")
        st.sidebar.caption(f"Expected: {db_path}")
    
    return db_path

def get_selected_item():
    """Get the currently selected item from query parameters."""
    if "item_id" in st.query_params and "item_type" in st.query_params:
        return {
            "type": st.query_params["item_type"],
            "id": st.query_params["item_id"],
            "name": st.query_params.get("item_name", "Unknown")
        }
    return None