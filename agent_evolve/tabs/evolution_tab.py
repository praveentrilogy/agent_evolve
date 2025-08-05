import streamlit as st
from pathlib import Path

def render_evolution_tab(tool_data, selected_tool):
    """Render the Evolution tab content"""
    st.subheader("Evolution Control")
    
    # Get available checkpoints
    available_checkpoints = []
    openevolve_output = Path(tool_data['path']) / "openevolve_output"
    if openevolve_output.exists():
        checkpoints_dir = openevolve_output / "checkpoints"
        if checkpoints_dir.exists():
            # Find all checkpoint directories
            checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint_')]
            for checkpoint_dir in checkpoint_dirs:
                try:
                    checkpoint_num = int(checkpoint_dir.name.split('_')[1])
                    available_checkpoints.append(checkpoint_num)
                except (IndexError, ValueError):
                    continue
        
        # Remove duplicates and sort
        available_checkpoints = sorted(list(set(available_checkpoints)))
    
    # Evolution controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Create selectbox for checkpoints
        if available_checkpoints:
            checkpoint_options = ["None (start from beginning)"]
            for cp in available_checkpoints:
                checkpoint_options.append(f"{cp}")
            
            selected_checkpoint_str = st.selectbox(
                "Start from checkpoint:",
                options=checkpoint_options,
                index=0,
                help="Choose a checkpoint to resume from, or None to start from beginning"
            )
            
            # Extract the numeric value (None means no checkpoint)
            if selected_checkpoint_str.startswith("None"):
                start_checkpoint = None
            else:
                start_checkpoint = int(selected_checkpoint_str)
        else:
            # No checkpoints available
            st.selectbox(
                "Start from checkpoint:",
                options=["None (no checkpoints available)"],
                index=0,
                disabled=True,
                help="No checkpoints found - will start from beginning"
            )
            start_checkpoint = None
    
    with col2:
        num_iterations = st.number_input(
            "Number of iterations:",
            min_value=1,
            max_value=100,
            value=20,
            help="How many evolution iterations to run (modifies the OpenEvolve config)"
        )
    
    # Build command for user to copy
    cmd_parts = ["agent-evolve", "evolve", selected_tool]
    
    # Only add checkpoint argument if one is specified
    if start_checkpoint is not None:
        cmd_parts.extend(["--checkpoint", str(start_checkpoint)])
    
    # Add iterations parameter if different from default
    if num_iterations != 20:
        cmd_parts.extend(["--iterations", str(num_iterations)])
    
    command = " ".join(cmd_parts)
    
    # Show the command to run
    st.subheader("💻 Run Evolution Command")
    st.markdown("Copy and paste these commands into your terminal to run evolution:")
    
    full_command = f"cd {Path.cwd()}\n{command}"
    st.code(full_command, language="bash")
    
    st.markdown("---")
    
    # Show additional info
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Tool**: {selected_tool}")
        if start_checkpoint is not None:
            st.info(f"**Starting from checkpoint**: {start_checkpoint}")
        else:
            st.info("**Starting from**: Beginning (no checkpoint)")
    
    with col2:
        st.info(f"**Iterations**: {num_iterations}")
        st.info(f"**Working directory**: {Path.cwd()}")