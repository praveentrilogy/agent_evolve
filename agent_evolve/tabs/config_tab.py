import streamlit as st
import yaml
from pathlib import Path

def render_config_tab(tool_data, selected_tool):
    """Render the Configuration tab content"""
    st.subheader("OpenEvolve Configuration")
    st.markdown("Configure the evolution process settings, metrics, and system prompts.")
    
    # Load existing OpenEvolve config if available
    config_file = Path(tool_data['path']) / "openevolve_output" / "config.yaml"
    config_data = {}
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            st.success("✅ Existing OpenEvolve configuration loaded")
        except Exception as e:
            st.error(f"❌ Error loading config: {e}")
    else:
        st.info("📝 No existing OpenEvolve configuration found. Create a new one below.")
    
    # Simple configuration form
    st.markdown("### Basic Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        population_size = st.number_input(
            "Population Size", 
            min_value=1, max_value=100, 
            value=config_data.get('genetic_algorithm', {}).get('population_size', 10),
            help="Number of candidate programs in each generation"
        )
        
        max_generations = st.number_input(
            "Max Generations", 
            min_value=1, max_value=1000, 
            value=config_data.get('genetic_algorithm', {}).get('num_generations', 50),
            help="Maximum number of evolution generations"
        )
    
    with col2:
        model_name = st.selectbox(
            "AI Model",
            options=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0,
            help="AI model used for code evolution"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0, max_value=2.0,
            value=config_data.get('llm', {}).get('temperature', 0.7),
            help="Controls randomness in AI responses"
        )
    
    # Save Configuration
    if st.button("💾 Save Configuration", type="primary"):
        new_config = {
            'language': 'python',
            'fitness': {
                'feature_dimensions': {
                    'correctness': {'description': 'How correct the function output is'},
                    'efficiency': {'description': 'How efficient the function is'},
                    'readability': {'description': 'How readable the code is'}
                }
            },
            'genetic_algorithm': {
                'population_size': population_size,
                'num_generations': max_generations,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8
            },
            'llm': {
                'model': model_name,
                'temperature': temperature,
                'max_tokens': 2048
            }
        }
        
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False, indent=2)
            st.success(f"✅ Configuration saved to {config_file}")
            st.code(yaml.dump(new_config, default_flow_style=False, indent=2), language='yaml')
        except Exception as e:
            st.error(f"❌ Failed to save configuration: {e}")
    
    # Show current config if exists
    if config_data:
        with st.expander("📋 Current Configuration", expanded=False):
            st.json(config_data)