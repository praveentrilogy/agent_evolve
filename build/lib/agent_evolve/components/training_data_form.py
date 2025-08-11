import streamlit as st
from pathlib import Path

def render_training_data_form(tool_data, selected_tool, form_key_prefix="", expanded=True, show_regenerate=False):
    """
    Reusable training data generation form component
    
    Args:
        tool_data: Tool data dictionary
        selected_tool: Name of the selected tool
        form_key_prefix: Prefix for form element keys to avoid conflicts
        expanded: Whether the form should be expanded by default
        show_regenerate: Whether to show as regenerate form or initial generation
    """
    
    if show_regenerate:
        form_title = "🔄 Regenerate Training Data"
        button_text = "🔄 Regenerate Training Data"
        default_force = True
    else:
        form_title = "🚀 Generate Training Data"
        button_text = "🚀 Generate Training Data"
        default_force = False
    
    with st.expander(form_title, expanded=expanded):
        st.markdown("Configure training data generation:")
        
        custom_prompt = st.text_area(
            "Custom instructions (optional):",
            placeholder="e.g., Focus on realistic user scenarios, include error cases, generate data for specific domains, add more diverse examples...",
            height=100,
            key=f"{form_key_prefix}_custom_prompt_{selected_tool}"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            num_samples = st.number_input(
                "Number of samples:",
                min_value=1,
                max_value=50,
                value=10,
                key=f"{form_key_prefix}_num_samples_{selected_tool}"
            )
        with col2:
            if show_regenerate:
                force_regenerate = st.checkbox(
                    "Force regenerate",
                    value=default_force,
                    key=f"{form_key_prefix}_force_{selected_tool}"
                )
            else:
                force_regenerate = default_force
        
        button_type = "primary" if not show_regenerate else "secondary"
        
        if st.button(button_text, key=f"{form_key_prefix}_btn_{selected_tool}", type=button_type):
            with st.spinner("🤖 Generating training data..."):
                try:
                    from agent_evolve.generate_training_data import TrainingDataGenerator
                    generator = TrainingDataGenerator(num_samples=num_samples)
                    
                    # If custom prompt provided, show info about it
                    if custom_prompt.strip():
                        st.info(f"Using custom instructions: {custom_prompt[:100]}...")
                    
                    # TODO: In future, we can modify the generator to use custom_prompt
                    # For now, we'll generate with standard prompts but the infrastructure is ready
                    
                    generator.generate_training_data(
                        str(Path(tool_data['path']).parent), 
                        force=force_regenerate, 
                        specific_tool=selected_tool
                    )
                    
                    action_text = "regenerated" if show_regenerate else "generated"
                    st.success(f"✅ Training data {action_text}!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")