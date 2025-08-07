import streamlit as st
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go

def display_metrics_chart(score_data: Dict) -> None:
    """Display a radar chart of metrics comparison."""
    if not score_data:
        st.warning("No score data available")
        return
    
    original_scores = score_data.get('original_version', {}).get('scores', {})
    best_scores = score_data.get('best_version', {}).get('scores', {})
    
    if not original_scores or not best_scores:
        st.warning("Incomplete score data")
        return
    
    # Create radar chart
    metrics = list(original_scores.keys())
    original_values = [original_scores[m] for m in metrics]
    best_values = [best_scores[m] for m in metrics]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=original_values,
        theta=metrics,
        fill='toself',
        name='Original Version',
        line_color='red',
        fillcolor='rgba(255, 0, 0, 0.1)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=best_values,
        theta=metrics,
        fill='toself',
        name='Best Evolved Version',
        line_color='green',
        fillcolor='rgba(0, 255, 0, 0.1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0]
            )),
        showlegend=True,
        title="Performance Metrics Comparison"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_evolution_timeline(checkpoints: List[Dict]) -> None:
    """Display evolution progress over checkpoints."""
    if not checkpoints:
        st.warning("No checkpoint data available")
        return
    
    # Extract metrics data for timeline
    timeline_data = []
    for cp in checkpoints:
        checkpoint_num = cp['checkpoint']
        metrics = cp['info'].get('metrics', {})
        
        for metric_name, value in metrics.items():
            timeline_data.append({
                'Checkpoint': checkpoint_num,
                'Metric': metric_name,
                'Score': value
            })
    
    if not timeline_data:
        st.warning("No metrics data in checkpoints")
        return
    
    df = pd.DataFrame(timeline_data)
    
    fig = px.line(df, x='Checkpoint', y='Score', color='Metric',
                  title='Evolution Progress Over Checkpoints',
                  markers=True)
    
    fig.update_layout(
        xaxis_title="Checkpoint",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.0])
    )
    
    st.plotly_chart(fig, use_container_width=True)


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
    
    # Show checkpoint data and timeline chart
    if tool_data.get('checkpoints'):
        st.markdown("### 🏁 Evolution Timeline")
        checkpoint_data = tool_data['checkpoints']
        
        # Display evolution timeline chart
        display_evolution_timeline(checkpoint_data)
        
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
    
    # Show radar chart comparison if score data exists
    score_file = Path(tool_data['path']) / "score_data.json"
    if score_file.exists():
        try:
            with open(score_file, 'r') as f:
                score_data = json.load(f)
            
            st.markdown("### 📊 Performance Comparison")
            display_metrics_chart(score_data)
            
        except Exception as e:
            st.error(f"❌ Error loading score comparison data: {e}")
    
    # If we have both initial scores and checkpoints, create comparison
    elif initial_scores and tool_data.get('checkpoints'):
        # Build score comparison from available data
        checkpoint_data = tool_data['checkpoints']
        if checkpoint_data:
            latest = max(checkpoint_data, key=lambda x: x['checkpoint'])
            latest_metrics = latest.get('info', {}).get('metrics', {})
            
            if latest_metrics:
                score_data = {
                    'original_version': {
                        'scores': {k: v for k, v in initial_scores.items() if k != 'combined_score'},
                        'average': initial_scores.get('combined_score', 0)
                    },
                    'best_version': {
                        'scores': {k: v for k, v in latest_metrics.items() if k != 'combined_score'},
                        'average': latest_metrics.get('combined_score', 0)
                    }
                }
                
                st.markdown("### 📊 Performance Comparison")
                display_metrics_chart(score_data)