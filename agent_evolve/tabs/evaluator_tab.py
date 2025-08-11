import streamlit as st
import difflib
import json
import re
from pathlib import Path
from typing import Dict, List

def create_code_diff_html_for_display(original: str, evolved: str, show_original: bool = True) -> str:
    """Create HTML diff with red/green highlighting for side-by-side display"""
    if not original or not evolved:
        return "<p>Code not available</p>"
    
    original_lines = original.splitlines()
    evolved_lines = evolved.splitlines()
    
    # Use SequenceMatcher to find differences
    matcher = difflib.SequenceMatcher(None, original_lines, evolved_lines)
    
    html_lines = []
    html_lines.append('<div style="font-family: \'Courier New\', monospace; font-size: 12px; line-height: 1.4; background-color: #f8f9fa; padding: 10px; border-radius: 4px; max-height: 600px; overflow-y: auto;">')
    
    if show_original:
        # Show original code with deletions highlighted in red
        line_num = 1
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for i in range(i1, i2):
                    html_lines.append(f'<div style="padding: 1px 4px;"><span style="color: #666; margin-right: 10px; user-select: none;">{line_num:3d}</span><span style="color: #24292e;">{original_lines[i]}</span></div>')
                    line_num += 1
            elif tag == 'delete':
                for i in range(i1, i2):
                    html_lines.append(f'<div style="background-color: #ffeef0; border-left: 3px solid #d73a49; padding: 1px 4px;"><span style="color: #666; margin-right: 10px; user-select: none;">{line_num:3d}</span><span style="color: #d73a49;"><del>{original_lines[i]}</del></span></div>')
                    line_num += 1
            elif tag == 'replace':
                for i in range(i1, i2):
                    html_lines.append(f'<div style="background-color: #ffeef0; border-left: 3px solid #d73a49; padding: 1px 4px;"><span style="color: #666; margin-right: 10px; user-select: none;">{line_num:3d}</span><span style="color: #d73a49;"><del>{original_lines[i]}</del></span></div>')
                    line_num += 1
            # Skip 'insert' for original view
    else:
        # Show evolved code with additions highlighted in green
        line_num = 1
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for j in range(j1, j2):
                    html_lines.append(f'<div style="padding: 1px 4px;"><span style="color: #666; margin-right: 10px; user-select: none;">{line_num:3d}</span><span style="color: #24292e;">{evolved_lines[j]}</span></div>')
                    line_num += 1
            elif tag == 'insert':
                for j in range(j1, j2):
                    html_lines.append(f'<div style="background-color: #e6ffed; border-left: 3px solid #28a745; padding: 1px 4px;"><span style="color: #666; margin-right: 10px; user-select: none;">{line_num:3d}</span><span style="color: #28a745;"><strong>{evolved_lines[j]}</strong></span></div>')
                    line_num += 1
            elif tag == 'replace':
                for j in range(j1, j2):
                    html_lines.append(f'<div style="background-color: #e6ffed; border-left: 3px solid #28a745; padding: 1px 4px;"><span style="color: #666; margin-right: 10px; user-select: none;">{line_num:3d}</span><span style="color: #28a745;"><strong>{evolved_lines[j]}</strong></span></div>')
                    line_num += 1
            # Skip 'delete' for evolved view
    
    html_lines.append('</div>')
    return ''.join(html_lines)

@st.dialog("🤖 Improve Evaluator with AI")
def show_ai_improvement_modal(selected_tool: str, current_code: str, training_data: List[Dict]):
    """Show AI improvement modal dialog"""
    
    # Check if we have valid code
    if not current_code:
        st.error("❌ No evaluator code found. Please generate an evaluator first.")
        if st.button("Close"):
            st.rerun()
        return
    
    st.markdown("Describe how you want to improve the evaluator:")
    
    # Debug info
    with st.expander("Current Code Preview (first 500 chars)", expanded=False):
        preview_code = current_code[:500] + "..." if len(current_code) > 500 else current_code
        st.code(preview_code, language="python")
    
    improvement_instructions = st.text_area(
        "Improvement instructions:",
        placeholder="e.g., Make the evaluation more strict, focus on specific aspects, add new metrics, fix errors, etc.",
        height=120,
        key=f"modal_improvement_instructions_{selected_tool}"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("Cancel", key=f"modal_cancel_{selected_tool}"):
            st.rerun()
    
    with col1:
        if st.button("🚀 Generate Improved Evaluator", type="primary", key=f"modal_submit_{selected_tool}"):
            if improvement_instructions:
                with st.spinner("🤖 Generating improved evaluator..."):
                    try:
                        import openai
                        client = openai.OpenAI()
                        
                        # Create improvement prompt
                        improvement_prompt = f"""You are an expert Python developer improving an OpenEvolve evaluator.

CURRENT EVALUATOR CODE:
```python
{current_code}
```

TRAINING DATA SAMPLES (first 3):
```json
{json.dumps(training_data[:3] if training_data else [], indent=2)}
```

USER IMPROVEMENT REQUEST:
{improvement_instructions}

REQUIREMENTS:
1. The evaluator MUST have this exact function signature: def evaluate(program) -> dict:
2. Use raw OpenAI API (openai.OpenAI()), NOT langchain
3. Include robust JSON parsing with error handling
4. Return a dictionary with metric scores between 0.0 and 1.0
5. Apply the user's improvement instructions while maintaining functionality
6. Include proper error handling and logging

Generate the complete improved evaluator code.
Return ONLY Python code, no explanations or markdown."""
                        
                        response = client.chat.completions.create(
                            model="gpt-5",
                            messages=[{"role": "user", "content": improvement_prompt}],
                        )
                        
                        improved_code = response.choices[0].message.content.strip()
                        
                        # Clean up any markdown formatting
                        if improved_code.startswith('```python'):
                            improved_code = improved_code[9:]
                        elif improved_code.startswith('```'):
                            improved_code = improved_code[3:]
                        if improved_code.endswith('```'):
                            improved_code = improved_code[:-3]
                        improved_code = improved_code.strip()
                        
                        # Debug info
                        st.info(f"Generated {len(improved_code)} characters of improved code")
                        
                        # Show a preview of the changes
                        with st.expander("Preview of Improved Code (first 500 chars)", expanded=True):
                            st.code(improved_code[:500] + "..." if len(improved_code) > 500 else improved_code, language="python")
                        
                        # Store the improved code for display in main tab
                        st.session_state[f'improved_evaluator_{selected_tool}'] = improved_code
                        st.session_state[f'show_diff_{selected_tool}'] = True
                        st.success("✅ Improved evaluator generated! Check the Evaluator tab to see the diff and apply changes.")
                        
                        # Close modal automatically
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error generating improved evaluator: {e}")
            else:
                st.warning("Please provide improvement instructions")

def render_evaluator_tab(tool_data, selected_tool):
    """Render the Evaluator tab content"""
    # Header with improve button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Evaluator Code")
    with col2:
        if st.button("🤖 Improve with AI", key=f"improve_ai_top_{selected_tool}"):
            # Store tool data in session state for modal access
            st.session_state['current_tool_data'] = tool_data
            
            # Get current evaluator code - prioritize what's in the editor (session state)
            current_code = None
            
            # First try to get from session state (what's currently in the editor)
            if f'evaluator_editor_{selected_tool}' in st.session_state:
                current_code = st.session_state[f'evaluator_editor_{selected_tool}']
            
            # If not in session state, try to get from tool_data
            if not current_code:
                current_code = tool_data.get('evaluator_code')
            
            # If still not found, try to reload from file
            if not current_code:
                evaluator_file = Path(tool_data['path']) / "evaluator.py"
                if evaluator_file.exists():
                    try:
                        with open(evaluator_file, 'r', encoding='utf-8') as f:
                            current_code = f.read()
                    except Exception as e:
                        print(f"[ERROR] Failed to read evaluator file: {e}")
            
            # Load training data
            training_data = tool_data.get('training_data', [])
            # Show the modal
            show_ai_improvement_modal(selected_tool, current_code, training_data)
    
    # Check if there's a pending apply from the modal
    if st.session_state.get(f'pending_apply_{selected_tool}'):
        print(f"[MAIN] Processing pending apply for {selected_tool}")
        
        # Get the improved code
        improved_code = st.session_state[f'pending_apply_{selected_tool}']
        print(f"[MAIN] Applying {len(improved_code)} chars to editor")
        
        # Update the editor session state
        st.session_state[f'evaluator_editor_{selected_tool}'] = improved_code
        
        # Clear the pending flag
        del st.session_state[f'pending_apply_{selected_tool}']
        
        # Show success message
        st.success("✅ Evaluator has been improved with AI! The code has been updated in the editor below. Click 'Save Evaluator' to save to file.")
    
    if tool_data['evaluator_code']:
        # Check if we should show diff view
        if st.session_state.get(f'show_diff_{selected_tool}', False):
            st.subheader("🔄 AI Improved Code - Review Changes")
            
            # Get the improved code with debugging
            improved_code = st.session_state.get(f'improved_evaluator_{selected_tool}', '')
            original_code = tool_data['evaluator_code']
            
            if not improved_code:
                st.error("❌ No improved code found in session state. Please try generating again.")
                # Clear the flag and return to regular view
                del st.session_state[f'show_diff_{selected_tool}']
                st.rerun()
                return
            
            # Show side-by-side diff with additions/deletions highlighted
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Code**")
                # Create diff HTML for original (deletions in red)
                diff_html_original = create_code_diff_html_for_display(original_code, improved_code, show_original=True)
                st.markdown(diff_html_original, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Improved Code**")
                # Create diff HTML for improved (additions in green)
                diff_html_improved = create_code_diff_html_for_display(original_code, improved_code, show_original=False)
                st.markdown(diff_html_improved, unsafe_allow_html=True)
                
                # Apply button at bottom right
                st.markdown("")  # Add some space
                col_left, col_right = st.columns([3, 1])
                with col_right:
                    if st.button("✅ Apply Changes", type="primary", key=f"apply_diff_{selected_tool}"):
                        # Save the improved code directly to file
                        evaluator_path = Path(tool_data['path']) / "evaluator.py"
                        
                        try:
                            with open(evaluator_path, 'w') as f:
                                f.write(improved_code)
                            
                            st.success(f"✅ Changes applied and saved to {evaluator_path}")
                            
                            # Update the in-memory data so the regular editor shows the new code
                            tool_data['evaluator_code'] = improved_code
                            
                            # Update the editor session state with the new code
                            st.session_state[f'evaluator_editor_{selected_tool}'] = improved_code
                            
                            # Clear the diff flags to return to regular editor
                            del st.session_state[f'show_diff_{selected_tool}']
                            del st.session_state[f'improved_evaluator_{selected_tool}']
                            
                            # Refresh the page
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error saving: {e}")
                
                with col_left:
                    if st.button("❌ Discard Changes", key=f"discard_diff_{selected_tool}"):
                        # Clear the diff flags without saving
                        del st.session_state[f'show_diff_{selected_tool}']
                        del st.session_state[f'improved_evaluator_{selected_tool}']
                        st.rerun()
            
            return  # Don't show the regular editor when in diff mode
        
        # Regular code editor
        try:
            # Try to use streamlit-ace for better code editing experience
            from streamlit_ace import st_ace
            
            # Get the current value - check for improved code first
            if st.session_state.get(f'improved_evaluator_{selected_tool}'):
                current_value = st.session_state[f'improved_evaluator_{selected_tool}']
                # Clear the improved code after using it
                del st.session_state[f'improved_evaluator_{selected_tool}']
            else:
                current_value = st.session_state.get(f'evaluator_editor_{selected_tool}', tool_data['evaluator_code'])
            
            edited_code = st_ace(
                value=current_value,
                language='python',
                theme='github',
                key=f"evaluator_editor_{selected_tool}",
                font_size=14,
                show_gutter=True,
                show_print_margin=True,
                wrap=False,
                auto_update=True,
                annotations=None,
                height=600
            )
            
            # Note about the Apply button
            st.caption("💡 Use the 'Save Evaluator' button below to save changes to file")
        except ImportError:
            # Fallback to text_area if ace editor not installed
            st.info("💡 Install streamlit-ace for a better code editing experience: pip install streamlit-ace")
            
            # Get the current value - check for improved code first
            if st.session_state.get(f'improved_evaluator_{selected_tool}'):
                current_value = st.session_state[f'improved_evaluator_{selected_tool}']
                # Clear the improved code after using it
                del st.session_state[f'improved_evaluator_{selected_tool}']
            else:
                current_value = st.session_state.get(f'evaluator_editor_{selected_tool}', tool_data['evaluator_code'])
            
            edited_code = st.text_area(
                "Edit evaluator code:",
                value=current_value,
                height=600,
                key=f"evaluator_editor_{selected_tool}"
            )
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Evaluator", type="primary"):
                evaluator_path = Path(tool_data['path']) / "evaluator.py"
                
                try:
                    with open(evaluator_path, 'w') as f:
                        f.write(edited_code)
                    st.success(f"✅ Evaluator saved to {evaluator_path}")
                    # Update the in-memory data
                    tool_data['evaluator_code'] = edited_code
                except Exception as e:
                    st.error(f"❌ Error saving evaluator: {e}")
        
        with col2:
            if st.button("🔍 Validate Evaluator"):
                try:
                    # Try to compile the code
                    compile(edited_code, '<string>', 'exec')
                    
                    # Check for required function
                    if 'def evaluate(' in edited_code:
                        st.success("✅ Evaluator syntax is valid and contains evaluate() function")
                    else:
                        st.error("❌ Missing required 'def evaluate(' function")
                        
                except SyntaxError as e:
                    st.error(f"❌ Syntax error: {e}")
                except Exception as e:
                    st.error(f"❌ Validation error: {e}")
        
        with col3:
            if st.button("🤖 Improve with AI"):
                # Get current evaluator code - prioritize what's in the editor (session state)
                current_code = None
                
                # First try to get from session state (what's currently in the editor)
                if f'evaluator_editor_{selected_tool}' in st.session_state:
                    current_code = st.session_state[f'evaluator_editor_{selected_tool}']
                
                # If not in session state, try to get from tool_data
                if not current_code:
                    current_code = tool_data.get('evaluator_code')
                
                # If still not found, try to reload from file
                if not current_code:
                    evaluator_file = Path(tool_data['path']) / "evaluator.py"
                    if evaluator_file.exists():
                        try:
                            with open(evaluator_file, 'r', encoding='utf-8') as f:
                                current_code = f.read()
                        except Exception as e:
                            print(f"[ERROR] Failed to read evaluator file: {e}")
                
                # Load training data
                training_data = tool_data.get('training_data', [])
                # Show the modal
                show_ai_improvement_modal(selected_tool, current_code, training_data)
        
        # Show code statistics at the bottom
        st.markdown("---")
        st.subheader("Code Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        lines = edited_code.split('\n')
        with col1:
            st.metric("Lines of Code", len(lines))
        with col2:
            st.metric("Characters", len(edited_code))
        
        # Count functions
        functions = re.findall(r'def\s+(\w+)', edited_code)
        with col3:
            st.metric("Functions", len(functions))
        
        # Show function list
        with col4:
            if functions:
                st.markdown("**Functions found:**")
                for func in functions:
                    st.text(f"• {func}")
    else:
        st.warning("❌ No evaluator found")
        st.info("💡 Generate training data first, then you can create an evaluator")