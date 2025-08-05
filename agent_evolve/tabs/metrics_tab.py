import streamlit as st
import json
from pathlib import Path

def render_metrics_tab(tool_data, selected_tool):
    """Render the Metrics tab content"""
    st.subheader("Metrics & Evolution Timeline")
    st.info("📊 This tab shows evolution progress and performance metrics over time.")
    
    # Load initial scores for timeline
    initial_scores = {}
    initial_score_file = Path(tool_data['path']) / "initial_score.json"
    if initial_score_file.exists():
        try:
            with open(initial_score_file, 'r') as f:
                initial_scores = json.load(f)
            st.success("✅ Found initial baseline scores")
            
            # Show initial scores
            st.markdown("### 📈 Initial Program Scores")
            col1, col2, col3 = st.columns(3)
            metrics = [k for k in initial_scores.keys() if k != 'combined_score']
            
            for i, (metric, score) in enumerate(initial_scores.items()):
                if metric == 'combined_score':
                    continue
                with [col1, col2, col3][i % 3]:
                    st.metric(metric.title(), f"{score:.3f}")
                    
        except Exception as e:
            st.error(f"❌ Error loading initial scores: {e}")
    else:
        st.warning("⚠️ No initial scores found. Run evolution first to see metrics.")
    
    # Show checkpoint data if available
    if tool_data.get('checkpoints'):
        st.markdown("### 🏁 Evolution Checkpoints")
        checkpoint_data = tool_data['checkpoints']
        
        # Create simple timeline
        checkpoint_nums = [cp['checkpoint'] for cp in checkpoint_data]
        st.write(f"📊 Found {len(checkpoint_data)} checkpoints: {', '.join(map(str, checkpoint_nums))}")
        
        # Show latest checkpoint info
        if checkpoint_data:
            latest = max(checkpoint_data, key=lambda x: x['checkpoint'])
            st.markdown(f"**Latest Checkpoint:** {latest['checkpoint']}")
            
            latest_info = latest.get('info', {})
            if 'metrics' in latest_info:
                col1, col2, col3 = st.columns(3)
                for i, (metric, score) in enumerate(latest_info['metrics'].items()):
                    if metric == 'combined_score':
                        continue
                    with [col1, col2, col3][i % 3]:
                        improvement = score - initial_scores.get(metric, 0) if initial_scores else 0
                        st.metric(metric.title(), f"{score:.3f}", delta=f"{improvement:+.3f}")
    else:
        st.info("🔄 No evolution checkpoints found. Start evolution to see progress timeline.")