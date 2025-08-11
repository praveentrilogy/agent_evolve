import streamlit as st
import os
from pathlib import Path
from components.training_data_form import render_training_data_form

def render_overview_tab(tool_data, selected_tool):
    """Render the Overview tab content"""
    st.subheader("Evolution Pipeline")
    
    # Step 1: Training Data
    st.markdown("### 1️⃣ Training Data")
    if tool_data.get('training_data'):
        st.success(f"✅ Training data available ({len(tool_data['training_data'])} samples)")
        # Show regenerate form
        render_training_data_form(
            tool_data=tool_data,
            selected_tool=selected_tool,
            form_key_prefix="overview_regen",
            expanded=False,
            show_regenerate=True
        )
    else:
        st.warning("❌ No training data")
        # Show initial generation form
        render_training_data_form(
            tool_data=tool_data,
            selected_tool=selected_tool,
            form_key_prefix="overview_gen",
            expanded=True,
            show_regenerate=False
        )
    
    # Step 2: Evaluator
    st.markdown("### 2️⃣ Evaluator")
    if tool_data.get('evaluator_code'):
        st.success("✅ Evaluator exists")
    else:
        if tool_data.get('training_data'):
            st.warning("❌ No evaluator")
            if st.button("🚀 Generate Evaluator", key=f"gen_eval_{selected_tool}"):
                with st.spinner("🤖 Generating evaluator..."):
                    try:
                        from agent_evolve.evaluator_engine import auto_generate_evaluator_for_extracted_tool
                        if not os.getenv('OPENAI_API_KEY'):
                            st.error("❌ OPENAI_API_KEY required")
                        else:
                            # Use the new file-based approach with OpenEvolve format
                            tool_path = Path(tool_data['path'])
                            project_root = str(tool_path.parent.parent)
                            
                            success = auto_generate_evaluator_for_extracted_tool(
                                tool_dir_path=str(tool_path),
                                project_root=project_root
                            )
                            
                            if success:
                                # Check if evaluator was actually created
                                evaluator_file = tool_path / "evaluator.py"
                                if evaluator_file.exists():
                                    # Also regenerate config to use the correct metrics from the evaluator
                                    try:
                                        from agent_evolve.generate_openevolve_configs import OpenEvolveConfigGenerator
                                        config_generator = OpenEvolveConfigGenerator(str(tool_path.parent))
                                        config_generator._generate_single_config(tool_path, selected_tool)
                                        st.info("🔄 Updated config with evaluator metrics")
                                    except Exception as config_e:
                                        st.warning(f"⚠️ Config update failed: {config_e}")
                                    
                                    st.success("✅ Evaluator generated!")
                                    st.rerun()
                                else:
                                    st.error("❌ Evaluator generation failed - file not created")
                            else:
                                st.error("❌ Failed to generate evaluator")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.info("⏸️ Generate training data first")
    
    # Step 3: Config
    st.markdown("### 3️⃣ OpenEvolve Config")
    config_file = Path(tool_data['path']) / "openevolve_config.yaml"
    if config_file.exists():
        st.success("✅ Config file exists")
    else:
        if tool_data.get('evaluator_code'):
            st.warning("❌ No config file")
            if st.button("🚀 Generate Config", key=f"gen_config_{selected_tool}"):
                with st.spinner("🤖 Generating config..."):
                    try:
                        from agent_evolve.generate_openevolve_configs import OpenEvolveConfigGenerator
                        generator = OpenEvolveConfigGenerator(str(Path(tool_data['path']).parent))
                        generator._generate_single_config(Path(tool_data['path']), selected_tool)
                        st.success("✅ Config generated!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        else:
            st.info("⏸️ Generate evaluator first")
    
    # Step 4: Evolution
    st.markdown("### 4️⃣ Run Evolution")
    if tool_data.get('has_evolution'):
        st.success("✅ Evolution has been run")
        # Show metrics if available
        if tool_data['score_comparison']:
            col1, col2 = st.columns(2)
            with col1:
                avg_improvement = (
                    tool_data['score_comparison']['best_version']['average'] - 
                    tool_data['score_comparison']['original_version']['average']
                )
                improvement_color = "green" if avg_improvement > 0 else "red" if avg_improvement < 0 else "gray"
                st.markdown(f"**Average Improvement:** <span style='color:{improvement_color}'>{avg_improvement:+.3f}</span>", 
                           unsafe_allow_html=True)
            with col2:
                best_info = tool_data.get('best_info', {})
                if best_info:
                    st.markdown(f"**Best Generation:** {best_info.get('generation', 'N/A')}")
    else:
        if config_file.exists():
            st.warning("❌ Evolution not yet run")
            if st.button("🚀 Run Evolution", key=f"run_evolution_{selected_tool}"):
                st.info("💻 Run the following command in your terminal:")
                st.code(f"agent-evolve evolve {selected_tool}")
                st.markdown("⚠️ **Note:** Evolution runs in the background. Monitor progress in the terminal and refresh this page to see results.")
        else:
            st.info("⏸️ Generate config first")