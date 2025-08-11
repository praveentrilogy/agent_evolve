import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List

@st.dialog("🤖 Improve Training Data with AI")
def show_training_data_improvement_modal(selected_tool: str, current_training_data: List[Dict], evaluator_code: str):
    """Show AI training data improvement modal dialog"""
    
    # Check if we have valid training data
    if not current_training_data:
        st.error("❌ No training data found. Please generate training data first.")
        if st.button("Close"):
            st.rerun()
        return
    
    st.markdown("Describe how you want to improve or expand the training data:")
    
    # Show current training data preview
    with st.expander("Current Training Data Preview", expanded=False):
        st.json(current_training_data[:3])  # Show first 3 samples
        st.caption(f"Total samples: {len(current_training_data)}")
    
    improvement_instructions = st.text_area(
        "Improvement instructions:",
        placeholder="e.g., Add more diverse examples, generate 5 additional samples, focus on edge cases, improve data quality, etc.",
        height=120,
        key=f"modal_training_improvement_{selected_tool}"
    )
    
    # Option to specify number of samples
    col1, col2 = st.columns([2, 1])
    with col1:
        action_type = st.radio(
            "Action:",
            ["Improve existing data", "Generate additional samples", "Replace with better data"],
            key=f"training_action_{selected_tool}"
        )
    
    with col2:
        if action_type == "Generate additional samples":
            num_additional = st.number_input(
                "Additional samples:",
                min_value=1,
                max_value=20,
                value=5,
                key=f"num_additional_{selected_tool}"
            )
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("Cancel", key=f"modal_training_cancel_{selected_tool}"):
            st.rerun()
    
    with col1:
        if st.button("🚀 Generate Improved Training Data", type="primary", key=f"modal_training_submit_{selected_tool}"):
            if improvement_instructions:
                with st.spinner("🤖 Generating improved training data..."):
                    try:
                        import openai
                        client = openai.OpenAI()
                        
                        # Create training data improvement prompt
                        if action_type == "Generate additional samples":
                            action_instruction = f"Generate {num_additional} additional high-quality training samples"
                        elif action_type == "Replace with better data":
                            action_instruction = f"Replace the current training data with {len(current_training_data)} better quality samples"
                        else:
                            action_instruction = "Improve the existing training data samples"
                        
                        training_prompt = f"""You are an expert at creating high-quality training data for AI systems.

CURRENT TRAINING DATA:
```json
{json.dumps(current_training_data, indent=2)}
```

EVALUATOR CODE CONTEXT:
```python
{evaluator_code[:1000]}...
```

USER REQUEST:
{improvement_instructions}

TASK: {action_instruction}

REQUIREMENTS:
1. Maintain the same JSON structure as existing data
2. Ensure data is diverse, realistic, and high-quality
3. Focus on variety in scenarios, contexts, and use cases
4. Avoid repetitive patterns or artificial examples
5. Make sure data is relevant to the evaluator's purpose

For "Generate additional samples": Return ONLY the new samples as a JSON array
For "Improve existing" or "Replace": Return the complete improved dataset as a JSON array

Return ONLY valid JSON, no explanations or markdown."""
                        
                        response = client.chat.completions.create(
                            model="gpt-5",
                            messages=[{"role": "user", "content": training_prompt}]
                            # Note: gpt-5 only supports default temperature (1.0)
                        )
                        
                        if not response or not response.choices or len(response.choices) == 0:
                            raise ValueError("OpenAI API returned empty response")
                        
                        improved_data_text = response.choices[0].message.content
                        if not improved_data_text:
                            raise ValueError("OpenAI API returned empty content")
                        
                        improved_data_text = improved_data_text.strip()
                        
                        # Clean up any markdown formatting
                        if improved_data_text.startswith('```json'):
                            improved_data_text = improved_data_text[7:]
                        elif improved_data_text.startswith('```'):
                            improved_data_text = improved_data_text[3:]
                        if improved_data_text.endswith('```'):
                            improved_data_text = improved_data_text[:-3]
                        improved_data_text = improved_data_text.strip()
                        
                        # Parse the improved data
                        improved_data = json.loads(improved_data_text)
                        
                        # Handle different action types
                        if action_type == "Generate additional samples":
                            final_data = current_training_data + improved_data
                        else:
                            final_data = improved_data
                        
                        # Debug info
                        st.info(f"Generated {len(improved_data)} samples. Total: {len(final_data)}")
                        
                        # Show a preview of the changes
                        with st.expander("Preview of Improved Training Data", expanded=True):
                            if action_type == "Generate additional samples":
                                st.markdown("**New samples added:**")
                                st.json(improved_data[:3] if len(improved_data) > 3 else improved_data)
                            else:
                                st.markdown("**Improved training data:**")
                                st.json(final_data[:3] if len(final_data) > 3 else final_data)
                        
                        # Store the improved data for display in main tab
                        st.session_state[f'improved_training_data_{selected_tool}'] = final_data
                        st.session_state[f'show_training_diff_{selected_tool}'] = True
                        st.success("✅ Improved training data generated! Check the Training Data tab to see the changes and apply them.")
                        
                        # Close modal automatically
                        st.rerun()
                        
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Error parsing generated JSON: {e}")
                    except Exception as e:
                        st.error(f"❌ Error generating improved training data: {e}")
            else:
                st.warning("Please provide improvement instructions")

def render_training_tab(tool_data, selected_tool):
    """Render the Training Data tab content"""
    # Header with improve button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Training Data")
    with col2:
        if st.button("🤖 Improve with AI", key=f"improve_training_top_{selected_tool}"):
            # Get current training data and evaluator code
            current_training_data = tool_data.get('training_data') or []
            evaluator_code = tool_data.get('evaluator_code') or ''
            # Show the modal
            show_training_data_improvement_modal(selected_tool, current_training_data, evaluator_code)
    
    # Check if we should show training data diff view
    if st.session_state.get(f'show_training_diff_{selected_tool}', False):
        st.subheader("🔄 AI Improved Training Data - Review Changes")
        
        # Get the improved training data
        improved_training_data = st.session_state.get(f'improved_training_data_{selected_tool}', [])
        original_training_data = tool_data.get('training_data', [])
        
        if not improved_training_data:
            st.error("❌ No improved training data found in session state. Please try generating again.")
            del st.session_state[f'show_training_diff_{selected_tool}']
            st.rerun()
            return
        
        # Show side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Training Data**")
            st.json(original_training_data)
            st.caption(f"Total samples: {len(original_training_data)}")
        
        with col2:
            st.markdown("**Improved Training Data**") 
            st.json(improved_training_data)
            st.caption(f"Total samples: {len(improved_training_data)}")
            
            # Apply button at bottom right
            st.markdown("")  # Add some space
            col_left, col_right = st.columns([3, 1])
            with col_right:
                if st.button("✅ Apply Changes", type="primary", key=f"apply_training_diff_{selected_tool}"):
                    # Save the improved training data directly to file
                    training_data_path = Path(tool_data['path']) / "training_data.json"
                    
                    print(f"[APPLY TRAINING DIFF] Saving to: {training_data_path}")
                    print(f"[APPLY TRAINING DIFF] Data samples: {len(improved_training_data)}")
                    
                    try:
                        with open(training_data_path, 'w') as f:
                            json.dump(improved_training_data, f, indent=2)
                        
                        print(f"[APPLY TRAINING DIFF] File saved successfully")
                        st.success(f"✅ Training data changes applied and saved to {training_data_path}")
                        
                        # Update the in-memory data
                        tool_data['training_data'] = improved_training_data
                        
                        # Clear the diff flags to return to regular view
                        del st.session_state[f'show_training_diff_{selected_tool}']
                        del st.session_state[f'improved_training_data_{selected_tool}']
                        
                        # Refresh the page
                        st.rerun()
                        
                    except Exception as e:
                        print(f"[APPLY TRAINING DIFF] Error: {e}")
                        st.error(f"❌ Error saving: {e}")
            
            with col_left:
                if st.button("❌ Discard Changes", key=f"discard_training_diff_{selected_tool}"):
                    # Clear the diff flags without saving
                    del st.session_state[f'show_training_diff_{selected_tool}']
                    del st.session_state[f'improved_training_data_{selected_tool}']
                    st.rerun()
        
        return  # Don't show the regular training data view when in diff mode
    
    if tool_data['training_data']:
        # Display training data as a table
        training_df = pd.DataFrame(tool_data['training_data'])
        
        # Flatten nested dictionaries for better display
        flattened_data = []
        for i, item in enumerate(tool_data['training_data']):
            flattened_item = {'Index': i + 1}
            
            def flatten_dict(d, prefix=''):
                for key, value in d.items():
                    if isinstance(value, dict):
                        flatten_dict(value, f"{prefix}{key}.")
                    else:
                        flattened_item[f"{prefix}{key}"] = value
            
            flatten_dict(item)
            flattened_data.append(flattened_item)
        
        if flattened_data:
            flattened_df = pd.DataFrame(flattened_data)
            # Convert all columns to string to avoid Arrow type conversion issues
            flattened_df = flattened_df.astype(str)
            st.dataframe(flattened_df, use_container_width=True)
            
            # Show summary statistics
            st.subheader("Training Data Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Samples", len(tool_data['training_data']))
            with col2:
                st.metric("Columns", len(flattened_df.columns) - 1)  # -1 for Index column
        else:
            st.warning("Could not parse training data structure")
    else:
        st.warning("No training data available")