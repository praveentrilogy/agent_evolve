#!/usr/bin/env python3
"""
Agent Evolve Dashboard - Streamlit web interface for visualizing evolution results
"""
import streamlit as st
import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import difflib
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import subprocess
import sys

# Import tab modules
from tabs.overview_tab import render_overview_tab
from tabs.training_tab import render_training_tab
from tabs.evolution_tab import render_evolution_tab
from tabs.evaluator_tab import render_evaluator_tab
from tabs.evolved_code_tab import render_evolved_code_tab
from tabs.metrics_tab import render_metrics_tab
from tabs.config_tab import render_config_tab

def extract_tools(base_dir: str) -> bool:
    """Extract evolution targets from the current directory"""
    try:
        with st.spinner("🔄 Scanning your code for evolution targets..."):
            # Get the actual working directory (where the user launched the dashboard from)
            import os
            working_dir = os.getcwd()
            
            # Show debug info
            st.info(f"🔍 Scanning directory: {working_dir}")
            st.info(f"📁 Output directory: {os.path.abspath(base_dir)}")
            
            # Use the CLI command to extract tools
            result = subprocess.run([
                sys.executable, "-m", "agent_evolve", "extract", 
                "--path", working_dir,
                "--output-dir", os.path.abspath(base_dir)
            ], capture_output=True, text=True, cwd=working_dir)
            
            if result.returncode == 0:
                st.success("✅ Successfully extracted evolution targets!")
                if result.stdout:
                    st.info(result.stdout)
                st.info("🔄 Refreshing dashboard to show your new tools...")
                return True
            else:
                st.error(f"❌ Error extracting tools: {result.stderr}")
                if result.stdout:
                    st.warning(f"Output: {result.stdout}")
                return False
    except Exception as e:
        st.error(f"❌ Error running extraction: {str(e)}")
        return False

def show_getting_started_guide(base_dir: str):
    """Show comprehensive getting started guide when no tools are found"""
    st.info("👋 Welcome to Agent Evolve! Let's get started by setting up your evolution targets.")
    
    st.markdown("## 🚀 Quick Start Guide")
    
    # Step 1: Add decorators
    st.markdown("### 1️⃣ Add Evolution Decorators to Your Code")
    st.markdown("""
Add the `@evolve()` decorator to functions you want to optimize:
""")
    
    st.code('''
from agent_evolve import evolve

@evolve()
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment of the given text."""
    # Your implementation here
    if "good" in text.lower():
        return "positive"
    elif "bad" in text.lower():
        return "negative" 
    else:
        return "neutral"
''', language='python')
    
    # Step 2: Alternative with comments
    st.markdown("### 2️⃣ Alternative: Mark Functions with Comments")
    st.markdown("""
If you can't modify your code directly, use the special comment format:
""")
    
    st.code('''
#@evolve()
def calculate_price(base_price: float, tax_rate: float) -> float:
    """Calculate final price including tax."""
    return base_price * (1 + tax_rate)
''', language='python')
    
    # Step 3: Real examples
    st.markdown("### 3️⃣ Real Examples")
    
    with st.expander("📊 Data Processing Example", expanded=False):
        st.code('''
@evolve()
def clean_and_process_data(raw_data: list) -> list:
    """Clean and process raw data for analysis."""
    cleaned = []
    for item in raw_data:
        if item and item.strip():
            cleaned.append(item.strip().lower())
    return cleaned
''', language='python')
    
    with st.expander("🤖 AI Agent Example", expanded=False):
        st.code('''
@evolve()
def generate_response(user_input: str, context: str) -> str:
    """Generate a response based on user input and context."""
    prompt = f"Context: {context}\\nUser: {user_input}\\nAssistant:"
    # Your LLM call here
    return "I understand your question about " + user_input
''', language='python')
    
    # Step 4: Extract targets  
    st.markdown("### 4️⃣ Extract Your Evolution Targets")
    st.markdown("Click the button below to scan your codebase for functions marked with `@evolve()` or `#@evolve()`:")
    
    if st.button("🚀 Extract Evolution Targets", type="primary", help="Scan your code for @evolve() decorated functions"):
        if extract_tools(base_dir):
            st.rerun()
    
    # Step 5: Next steps
    st.markdown("### 5️⃣ Next Steps After Sync")
    st.markdown("""
After extraction, you'll see your functions in the sidebar. For each function you can:

1. **📋 Training Data** - Add test cases and examples
2. **🔧 Evaluator** - Define how to measure performance  
3. **⚙️ Config** - Set evolution parameters
4. **🧬 Evolution** - Run the optimization process
5. **📊 Metrics** - Track improvement over time
""")
    
    # Troubleshooting section
    with st.expander("🔧 Troubleshooting", expanded=False):
        st.markdown("""
**No functions found?**
- Make sure you've added `@evolve()` decorators or `#@evolve()` comments
- Check that your Python files are in the current directory or subdirectories
- Ensure your functions have docstrings (they're used for optimization)

**Import errors?**
- Install agent_evolve: `pip install -e .`
- Make sure your Python environment is set up correctly

**Permission errors?**
- Ensure you have write permissions in the current directory
- The tool will create a `.agent_evolve` directory for storing results

**Still having issues?**
- Check the console output for detailed error messages
- Make sure your code is syntactically correct Python
""")

def load_tool_data(base_dir: str = ".agent_evolve") -> Dict:
    """Load data for all evolved tools."""
    base_path = Path(base_dir)
    tools_data = {}
    
    if not base_path.exists():
        print(f"DEBUG: Base path {base_path} does not exist")
        return tools_data
    
    print(f"DEBUG: Loading tools from {base_path.absolute()}")
    
    # Skip non-tool directories
    skip_dirs = {'db', 'data', '__pycache__', '.git', 'logs', 'output', 'checkpoints', 'temp', 'tmp'}
    
    all_dirs = list(base_path.iterdir())
    print(f"DEBUG: Found {len(all_dirs)} items in {base_path}")
    
    for tool_dir in all_dirs:
        if not tool_dir.is_dir() or tool_dir.name in skip_dirs:
            continue
        
        tool_name = tool_dir.name
        tool_data = {
            'name': tool_name,
            'path': tool_dir,
            'has_evolution': False,
            'original_code': None,
            'best_code': None,
            'score_comparison': None,
            'checkpoints': [],
            'best_info': None
        }
        
        # Load basic tool files (training data, evaluator) regardless of evolution status
        # Load training data
        training_data_file = tool_dir / "training_data.json"
        if training_data_file.exists():
            with open(training_data_file, 'r') as f:
                tool_data['training_data'] = json.load(f)
        else:
            tool_data['training_data'] = None
        
        # Load evaluator code
        evaluator_file = tool_dir / "evaluator.py"
        if evaluator_file.exists():
            try:
                with open(evaluator_file, 'r', encoding='utf-8') as f:
                    tool_data['evaluator_code'] = f.read()
            except Exception as e:
                print(f"Error loading evaluator for {tool_name}: {e}")
                tool_data['evaluator_code'] = None
        else:
            tool_data['evaluator_code'] = None
        
        # Load original code if available
        evolve_target = tool_dir / "evolve_target.py"
        if evolve_target.exists():
            with open(evolve_target, 'r') as f:
                tool_data['original_code'] = f.read()
        
        # Check if tool has evolution results
        openevolve_output = tool_dir / "openevolve_output"
        if openevolve_output.exists():
            tool_data['has_evolution'] = True
            
            # Load checkpoint data
            checkpoints_dir = openevolve_output / "checkpoints"
            if checkpoints_dir.exists():
                checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint_')]
                if checkpoint_dirs:
                    def get_checkpoint_number(checkpoint_dir):
                        try:
                            return int(checkpoint_dir.name.split('_')[1])
                        except (IndexError, ValueError):
                            return 0
                    
                    checkpoint_dirs = sorted(checkpoint_dirs, key=get_checkpoint_number)
                    
                    for checkpoint_dir in checkpoint_dirs:
                        checkpoint_num = get_checkpoint_number(checkpoint_dir)
                        best_info_file = checkpoint_dir / "best_program_info.json"
                        
                        if best_info_file.exists():
                            try:
                                with open(best_info_file, 'r') as f:
                                    best_info = json.load(f)
                                
                                tool_data['checkpoints'].append({
                                    'checkpoint': checkpoint_num,
                                    'info': best_info
                                })
                            except Exception as e:
                                print(f"Error loading checkpoint {checkpoint_num}: {e}")
        
        tools_data[tool_name] = tool_data
    
    return tools_data

def main():
    st.set_page_config(
        page_title="Agent Evolve Dashboard",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 Agent Evolve Dashboard")
    st.markdown("Evolution tracking and analysis for your AI agents")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Get base directory from environment variable or use default
    import os
    default_base_dir = os.getenv('AGENT_EVOLVE_BASE_DIR', '.agent_evolve')
    base_dir = st.sidebar.text_input("Base Directory", value=default_base_dir)
    
    # Load data
    with st.spinner("Loading evolution data..."):
        tools_data = load_tool_data(base_dir)
    
    if not tools_data:
        show_getting_started_guide(base_dir)
        return
    
    # Show all tools
    evolved_tools = tools_data
    
    # Sidebar tool selection
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.header("Evolution Targets")
    with col2:
        if st.button("🔄", help="Sync - Re-extract all tools", key="sync_tools"):
            extract_tools(base_dir)
            st.rerun()
    
    if not evolved_tools:
        st.sidebar.warning("No evolution targets found")
        if st.sidebar.button("🚀 Get Started - Sync Tools", key="get_started_sync", help="Extract evolution targets from your code"):
            extract_tools(base_dir)
            st.rerun()
        selected_tool = None
    else:
        # Create tool selection in sidebar
        tool_options = list(evolved_tools.keys())
        selected_tool = st.sidebar.radio(
            "Tools",
            options=tool_options,
            label_visibility="hidden"
        )
        
        # Show tool status in sidebar
        if selected_tool:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Tool Status")
            if evolved_tools[selected_tool]['score_comparison']:
                avg_improvement = (
                    evolved_tools[selected_tool]['score_comparison']['best_version']['average'] - 
                    evolved_tools[selected_tool]['score_comparison']['original_version']['average']
                )
                improvement_color = "green" if avg_improvement > 0 else "red" if avg_improvement < 0 else "gray"
                st.sidebar.markdown(f"**Improvement:** <span style='color:{improvement_color}'>{avg_improvement:+.3f}</span>", 
                                   unsafe_allow_html=True)
            else:
                st.sidebar.markdown("**Improvement:** N/A")
            
            if evolved_tools[selected_tool]['best_info']:
                generation = evolved_tools[selected_tool]['best_info'].get('generation', 'N/A')
                st.sidebar.markdown(f"**Best Generation:** {generation}")
    
    # Main content header
    if selected_tool:
        st.header(f"📊 {selected_tool}")
    else:
        st.header("📊 Evolution Results")
        if not evolved_tools:
            return
    
    if not selected_tool:
        st.info("👈 Select a tool from the sidebar to view its evolution results")
        return
    
    tool_data = evolved_tools[selected_tool]
    
    # Create tabs for different views
    overview_tab, training_tab, evaluator_tab, evolution_tab, evolved_code_tab, metrics_tab, config_tab = st.tabs([
        "📈 Overview", "📋 Training Data", "🔧 Evaluator", "🧬 Evolution", "🔍 Evolved Code", "📈 Metrics", "⚙️ Config"
    ])
    
    # Render each tab using the modular functions
    with overview_tab:
        render_overview_tab(tool_data, selected_tool)
    
    with training_tab:
        render_training_tab(tool_data, selected_tool)
    
    with evaluator_tab:
        render_evaluator_tab(tool_data, selected_tool)
    
    with evolution_tab:
        render_evolution_tab(tool_data, selected_tool)
    
    with evolved_code_tab:
        render_evolved_code_tab(tool_data, selected_tool)
    
    with metrics_tab:
        render_metrics_tab(tool_data, selected_tool)
    
    with config_tab:
        render_config_tab(tool_data, selected_tool)

if __name__ == "__main__":
    main()