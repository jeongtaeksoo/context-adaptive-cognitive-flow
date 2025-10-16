"""
Context-Adaptive Cognitive Flow (CACF) System - Research Implementation
Multi-Agent AI System for Cognitive Activation in Older Adults
"""

import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from perception_engine import PerceptionEngine
from context_hub import ContextHub
from response_engine import ResponseEngine
from feedback_loop import FeedbackLoop
from utils import load_sample_data
from database_models import DatabaseManager
from agent_orchestrator import AgentOrchestrator
import uuid
from datetime import datetime

# Initialize database manager
@st.cache_resource
def get_db_manager():
    """Get or create database manager instance"""
    import os
    if not os.environ.get('DATABASE_URL'):
        return None
    try:
        db_manager = DatabaseManager()
        db_manager.init_db()
        return db_manager
    except:
        return None

# Initialize session state
if 'user_ability' not in st.session_state:
    st.session_state.user_ability = 0.5
if 'engagement_history' not in st.session_state:
    st.session_state.engagement_history = []
if 'context_history' not in st.session_state:
    st.session_state.context_history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
    st.session_state.session_created = False
if 'agent_strategy' not in st.session_state:
    st.session_state.agent_strategy = 'highest_priority'
if 'show_all_agents' not in st.session_state:
    st.session_state.show_all_agents = True

def main():
    st.set_page_config(
        page_title="Context-Adaptive Cognitive Flow System",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Context-Adaptive Cognitive Flow System")
    st.markdown("**Multi-Agent AI System for Cognitive Activation in Older Adults**")
    st.markdown("---")
    
    # Database connection
    db_manager = get_db_manager()
    if db_manager and not st.session_state.session_created:
        try:
            db_manager.create_session(
                session_id=st.session_state.session_id,
                user_id="research_participant",
                session_metadata={"created_at": datetime.now().isoformat()}
            )
            st.session_state.session_created = True
        except:
            pass
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Agent Strategy Selection
    st.session_state.agent_strategy = st.sidebar.selectbox(
        "Response Selection Strategy",
        ["highest_priority", "weighted_blend", "round_robin"],
        index=["highest_priority", "weighted_blend", "round_robin"].index(st.session_state.agent_strategy),
        help="Strategy for selecting/combining agent responses"
    )
    
    st.session_state.show_all_agents = st.sidebar.checkbox(
        "Show Individual Agent Outputs",
        value=st.session_state.show_all_agents,
        help="Display responses from all three agents"
    )
    
    # Equation Reference Panel
    st.sidebar.markdown("---")
    with st.sidebar.expander("📐 Equation Reference", expanded=False):
        st.markdown("### Paper Equations → UI Mapping")
        
        st.markdown("**Stage I: Perception (Eq. 2-3)**")
        st.code("D_t = f_perception(x_behavioral, x_voice, x_performance, x_temporal)")
        st.caption("→ Multimodal Input Section")
        
        st.markdown("**Stage II: Context (Eq. 4-6)**")
        st.code("C_t = α_persona ⊙ D_t\nVA_t = f_VA(C_t)")
        st.caption("→ Agent Context Triggers")
        
        st.markdown("**Stage III: Response (Eq. 7-9)**")
        st.code("P(θ,β) = 1/(1+exp(-a(θ-β)))\nL_cog = f_load(C_t, β)\nπ = priority_score")
        st.caption("→ Agent Priority Metrics")
        
        st.markdown("**Stage IV: Feedback (Eq. 10-11, 22-23)**")
        st.code("θ_{t+1} = 0.7·θ_t + 0.3·θ̂_t\nE_t = Σ w_i·factor_i\nP_engage = Σ(e^(-0.1(n-i))·s_i)/Σ(e^(-0.1(n-i)))")
        st.caption("→ System Metrics & Trends")
        
        st.markdown("---")
        st.caption("**Key Parameters:**")
        st.caption("• α=0.7 (EMA smoothing)")
        st.caption("• λ=0.1 (engagement decay)")
        st.caption("• a=1.0 (IRT discrimination)")
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Session: {st.session_state.session_id}")
    st.sidebar.caption(f"Interactions: {len(st.session_state.engagement_history)}")
    
    # Initialize system components
    perception_engine = PerceptionEngine()
    context_hub = ContextHub(use_ml_embeddings=False)
    response_engine = ResponseEngine()
    feedback_loop = FeedbackLoop()
    orchestrator = AgentOrchestrator()
    
    # Main Demo Interface
    st.header("📊 Interactive Demonstration")
    
    # Load sample data
    sample_data = load_sample_data()
    
    # Input Section
    st.subheader("1️⃣ Select Scenario")
    
    scenario_key = st.selectbox(
        "Choose a test scenario:",
        list(sample_data.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    input_data = sample_data[scenario_key]
    
    # Display input data
    with st.expander("📋 View Multimodal Input Data", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("**Behavioral**")
            st.write(f"• Activity: {input_data['behavioral']['activity_level']:.2f}")
            st.write(f"• Social: {input_data['behavioral']['social_engagement']:.2f}")
        
        with col2:
            st.write("**Voice**")
            st.write(f"• Clarity: {input_data['voice']['speech_clarity']:.2f}")
            st.write(f"• Emotion: {input_data['voice']['emotional_tone']:+.2f}")
        
        with col3:
            st.write("**Performance**")
            st.write(f"• Response Time: {input_data['performance']['response_time']:.1f}s")
            st.write(f"• Accuracy: {input_data['performance']['accuracy']:.2f}")
        
        with col4:
            st.write("**Temporal**")
            st.write(f"• Time: {input_data['temporal']['time_of_day'].title()}")
            st.write(f"• Day: {input_data['temporal']['day_of_week'].title()}")
    
    st.markdown("---")
    
    # Add 4-stage flow visualization
    with st.expander("🔄 View 4-Stage CACF Processing Flow", expanded=False):
        st.markdown("### Context-Adaptive Cognitive Flow Architecture")
        
        # Create minimal flowchart using Plotly
        fig_flow = go.Figure()
        
        # Define boxes (x, y, width, height, label, color)
        boxes = [
            # Stage I: D_t
            (0.05, 0.45, 0.12, 0.15, "D_t | Multimodal<br>Input", "#f5f5f5", "#333"),
            # Stage II: C_t
            (0.25, 0.45, 0.12, 0.15, "C_t | Context<br>Recognition", "#e8f5e9", "#4caf50"),
            # Multi-Agent box (larger)
            (0.42, 0.25, 0.24, 0.55, "Multi-Agent Processing", "#f3e5f5", "#9c27b0"),
            # Agents inside
            (0.45, 0.15, 0.08, 0.10, "Companion", "#ede7f6", "#673ab7"),
            (0.45, 0.45, 0.08, 0.10, "Teacher", "#ede7f6", "#673ab7"),
            (0.45, 0.70, 0.08, 0.10, "Coach", "#ede7f6", "#673ab7"),
            # Stage III: R_t
            (0.72, 0.45, 0.12, 0.15, "R_t | Response<br>Generation", "#fff9c4", "#fbc02d"),
            # Stage IV: F_t
            (0.90, 0.45, 0.12, 0.15, "F_t | Feedback<br>Adaptation", "#f5f5f5", "#333"),
        ]
        
        # Draw boxes
        for x, y, w, h, label, fill_color, line_color in boxes:
            # Skip multi-agent box outline for now (we'll add it separately)
            if "Multi-Agent" in label:
                # Draw multi-agent container
                fig_flow.add_shape(
                    type="rect",
                    x0=x, y0=y, x1=x+w, y1=y+h,
                    line=dict(color=line_color, width=2, dash="dash"),
                    fillcolor=fill_color,
                    opacity=0.3,
                    layer="below"
                )
                fig_flow.add_annotation(
                    x=x+w/2, y=y+h+0.03,
                    text=label,
                    showarrow=False,
                    font=dict(size=9, color=line_color),
                    xanchor="center"
                )
            else:
                # Draw regular boxes
                fig_flow.add_shape(
                    type="rect",
                    x0=x, y0=y, x1=x+w, y1=y+h,
                    line=dict(color=line_color, width=2),
                    fillcolor=fill_color,
                )
                fig_flow.add_annotation(
                    x=x+w/2, y=y+h/2,
                    text=label,
                    showarrow=False,
                    font=dict(size=10, color="#333"),
                    xanchor="center",
                    yanchor="middle"
                )
        
        # Draw arrows
        arrows = [
            (0.17, 0.525, 0.25, 0.525),  # D_t -> C_t
            (0.37, 0.525, 0.42, 0.525),  # C_t -> Multi-Agent
            (0.53, 0.20, 0.72, 0.50),    # Companion -> R_t
            (0.53, 0.50, 0.72, 0.525),   # Teacher -> R_t
            (0.53, 0.75, 0.72, 0.55),    # Coach -> R_t
            (0.84, 0.525, 0.90, 0.525),  # R_t -> F_t
        ]
        
        for x0, y0, x1, y1 in arrows:
            fig_flow.add_annotation(
                x=x1, y=y1,
                ax=x0, ay=y0,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor="#666",
            )
        
        # Update layout
        fig_flow.update_xaxes(range=[0, 1.05], showgrid=False, zeroline=False, visible=False)
        fig_flow.update_yaxes(range=[0, 1], showgrid=False, zeroline=False, visible=False)
        fig_flow.update_layout(
            showlegend=False,
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='white',
            title=dict(
                text="Data Flow: D_t → C_t → R_t → F_t",
                font=dict(size=14),
                x=0.5,
                xanchor='center'
            )
        )
        
        st.plotly_chart(fig_flow, use_container_width=True)
        
        st.markdown("---")
        
        # Stage descriptions with equations
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**Stage I: Perception**")
            st.caption("Eq. 2-3: D_t = normalize(multimodal_features)")
        with col2:
            st.markdown("**Stage II: Context**")
            st.caption("Eq. 4-6: C_t = α_persona ⊙ D_t")
        with col3:
            st.markdown("**Stage III: Response**")
            st.caption("Eq. 7-9: R_t = IRT(θ, β, C_t)")
        with col4:
            st.markdown("**Stage IV: Feedback**")
            st.caption("Eq. 10-11: θ_{t+1} = EMA(θ_t, α=0.7)")
    
    st.markdown("---")
    
    # Process Button
    if st.button("🚀 Process with Multi-Agent System", type="primary", use_container_width=True):
        
        # Stage I: Perception
        with st.spinner("Processing multimodal input..."):
            processed_input = perception_engine.process_input(input_data)
        
        # Multi-Agent Processing
        with st.spinner("Invoking Teacher, Companion, and Coach agents..."):
            agent_context = {
                'difficulty_level': 0.5,
                'time_of_day': input_data['temporal'].get('time_of_day', 'afternoon'),
                'task_type': 'cognitive task'
            }
            
            import time
            start_time = time.time()
            all_agent_outputs = orchestrator.invoke_agents_parallel(processed_input, agent_context)
            invocation_time = time.time() - start_time
        
        st.success(f"✅ Multi-agent processing complete ({invocation_time*1000:.0f}ms)")
        
        # Display Individual Agent Outputs
        if st.session_state.show_all_agents:
            st.markdown("---")
            st.subheader("2️⃣ Individual Agent Responses")
            
            agent_config = {
                'teacher': {'emoji': '👨‍🏫', 'title': 'Teacher Agent', 'color': '#4A90E2'},
                'companion': {'emoji': '🤝', 'title': 'Companion Agent', 'color': '#E27B4A'},
                'coach': {'emoji': '💪', 'title': 'Coach Agent', 'color': '#7BC043'}
            }
            
            # Add Agent State Radar Charts
            with st.expander("📊 View Agent State Diagrams (Radar Charts)", expanded=False):
                st.markdown("### Multi-Dimensional Agent State Analysis")
                st.caption("Equation parameters: Priority (π), Cognitive Load (L_cog), Valence (V), Arousal (A), Success Probability P(θ,β)")
                
                # Create radar chart for each agent
                fig_radar = go.Figure()
                
                for agent_name, agent_output in all_agent_outputs.items():
                    config = agent_config.get(agent_name, {'emoji': '🤖', 'title': agent_name.title(), 'color': '#666666'})
                    metadata = agent_output['metadata']
                    
                    # Extract metrics for radar chart
                    priority = metadata.get('priority_score', 0.0)
                    cognitive_load = metadata.get('estimated_cognitive_load', 0.5)
                    success_prob = metadata.get('success_probability', 0.5)
                    context_alignment = metadata.get('context_alignment', 0.5)
                    
                    # For demonstration, compute valence/arousal approximations
                    # In real implementation, these would come from context_state
                    valence_approx = (priority - 0.5) * 2  # Map priority to [-1, 1]
                    arousal_approx = cognitive_load  # Use cognitive load as arousal proxy
                    
                    categories = ['Priority (π)', 'Context Alignment', 'Success P(θ,β)', 
                                  'Valence (V)', 'Arousal (A)']
                    values = [priority, context_alignment, success_prob, 
                             (valence_approx + 1) / 2, arousal_approx]  # Normalize to [0,1]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values + [values[0]],  # Close the polygon
                        theta=categories + [categories[0]],
                        fill='toself',
                        name=f"{config['emoji']} {config['title']}",
                        line=dict(color=config['color'], width=2),
                        fillcolor=config['color'],
                        opacity=0.5
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    showlegend=True,
                    height=500,
                    title="Agent State Comparison (Normalized [0,1])"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Add interpretation
                st.caption("**Interpretation:** Higher values indicate stronger agent activation. Priority (π) determines response selection in 'highest_priority' mode.")
            
            # Display agent cards
            cols = st.columns(3)
            
            for idx, (agent_name, agent_output) in enumerate(all_agent_outputs.items()):
                with cols[idx]:
                    config = agent_config.get(agent_name, {'emoji': '🤖', 'title': agent_name.title()})
                    
                    st.markdown(f"### {config['emoji']} {config['title']}")
                    
                    # Trigger context
                    trigger = agent_output.get('trigger', 'N/A')
                    st.caption(f"**Context:** {trigger}")
                    
                    # Priority score with equation reference
                    priority = agent_output['metadata'].get('priority_score', 0.0)
                    st.metric("Priority (π)", f"{priority:.3f}", help="Eq. 9: Priority score for response selection")
                    
                    # Additional metrics
                    metadata = agent_output['metadata']
                    if 'success_probability' in metadata:
                        st.caption(f"P(θ,β) = {metadata['success_probability']:.2f}")
                    if 'estimated_cognitive_load' in metadata:
                        st.caption(f"L_cog = {metadata['estimated_cognitive_load']:.2f}")
                    
                    # Response
                    st.info(agent_output['response'])
            
            st.markdown("---")
        
        # Add Response Selection Flow Visualization
        with st.expander("🔀 View Response Selection Flow", expanded=False):
            st.markdown("### Multi-Agent Parallel Processing & Selection")
            
            # Create flow diagram
            col_flow1, col_flow2 = st.columns([2, 1])
            
            with col_flow1:
                # Build network diagram showing parallel processing
                priorities = {name: output['metadata'].get('priority_score', 0) for name, output in all_agent_outputs.items()}
                
                # Create bar chart showing priorities
                fig_flow = go.Figure()
                
                colors_map = {'teacher': '#4A90E2', 'companion': '#E27B4A', 'coach': '#7BC043'}
                
                fig_flow.add_trace(go.Bar(
                    x=list(priorities.keys()),
                    y=list(priorities.values()),
                    marker=dict(color=[colors_map.get(k, '#666') for k in priorities.keys()]),
                    text=[f"{v:.3f}" for v in priorities.values()],
                    textposition='auto',
                ))
                
                fig_flow.update_layout(
                    title=f"Agent Priority Scores (Strategy: {st.session_state.agent_strategy})",
                    xaxis_title="Agent",
                    yaxis_title="Priority Score (π)",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig_flow, use_container_width=True)
            
            with col_flow2:
                st.markdown("**Selection Logic:**")
                
                if st.session_state.agent_strategy == 'highest_priority':
                    st.info(f"✅ Select agent with max(π)")
                    max_agent = max(priorities, key=priorities.get)
                    st.metric("Selected", max_agent.capitalize())
                    
                elif st.session_state.agent_strategy == 'weighted_blend':
                    st.info("✅ Blend responses weighted by π_i / Σπ")
                    total = sum(priorities.values())
                    for agent, priority in priorities.items():
                        weight = priority / total if total > 0 else 0
                        st.caption(f"{agent}: {weight:.1%}")
                        
                elif st.session_state.agent_strategy == 'round_robin':
                    st.info("✅ Rotate agents for balanced interaction")
                    st.caption("Equal opportunity regardless of π")
            
            # Processing timeline
            st.markdown("---")
            st.markdown("**Processing Timeline:**")
            timeline_data = {
                'Stage': ['Input → Perception', 'Perception → Context', 'Context → Agents (Parallel)', 
                          'Agents → Selection', 'Selection → Output'],
                'Duration': ['~5ms', '~3ms', f'{invocation_time*1000:.0f}ms', '~2ms', '~1ms']
            }
            st.dataframe(pd.DataFrame(timeline_data), hide_index=True, use_container_width=True)
        
        # Aggregate responses
        aggregated_output = orchestrator.aggregate_responses(all_agent_outputs, st.session_state.agent_strategy)
        
        # Update user model
        feedback_metrics = feedback_loop.update_user_model(
            st.session_state.user_ability,
            0.5,
            input_data["performance"]["accuracy"]
        )
        st.session_state.user_ability = feedback_metrics["updated_ability"]
        
        # Save to database
        if db_manager:
            try:
                db_manager.save_interaction(
                    session_id=st.session_state.session_id,
                    persona="Multi-Agent",
                    input_data=input_data,
                    context_state={'context_dimensions': {}, 'valence_arousal': {}},
                    response_data={'content': aggregated_output['response']},
                    feedback_metrics=feedback_metrics
                )
            except:
                pass
        
        # Display Final Selected Response
        st.subheader("3️⃣ Final System Response")
        
        attribution = aggregated_output.get('attribution', {})
        primary_agent = attribution.get('primary_agent', 'Unknown')
        agent_emoji_map = {'teacher': '👨‍🏫', 'companion': '🤝', 'coach': '💪'}
        
        # Response header with agent badge
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"**{agent_emoji_map.get(primary_agent, '🤖')} {primary_agent.capitalize()} Agent Selected**")
            st.caption(f"Strategy: {st.session_state.agent_strategy.replace('_', ' ').title()}")
        with col_b:
            st.metric("Priority", f"{aggregated_output['metadata'].get('priority_score', 0):.2f}")
        
        # Show supporting agents if blended
        if 'supporting_agents' in attribution and attribution['supporting_agents']:
            supporting = ', '.join([a.capitalize() for a in attribution['supporting_agents']])
            st.caption(f"🔗 Supporting agents: {supporting}")
        
        # Final response
        st.success(aggregated_output['response'])
        
        st.markdown("---")
        
        # System Metrics with Equation References
        st.subheader("4️⃣ System Metrics")
        
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric(
                "User Ability (θ)", 
                f"{st.session_state.user_ability:.3f}",
                delta=f"{feedback_metrics.get('ability_change', 0):+.3f}",
                help="Eq. 10: θ_{t+1} = α·θ_t + (1-α)·θ̂_t with α=0.7 (EMA update)"
            )
        with metric_cols[1]:
            st.metric(
                "Engagement (E_t)", 
                f"{feedback_metrics['engagement']:.3f}",
                help="Eq. 11, 22: E_t = Σ w_i·factor_i (weighted engagement score)"
            )
        with metric_cols[2]:
            st.metric(
                "Long-term (P_engage)", 
                f"{feedback_metrics.get('long_term_engagement', feedback_metrics['engagement']):.3f}",
                help="Eq. 23: P_engage = Σ(w_i·s_i)/Σ(w_i) where w_i=exp(-λ·(n-i)), λ=0.1"
            )
        with metric_cols[3]:
            st.metric(
                "Processing Time", 
                f"{invocation_time*1000:.0f}ms",
                help=f"Parallel agent invocation: {len(all_agent_outputs)} agents via ThreadPoolExecutor"
            )
        
        # Long-term feedback tracking
        st.session_state.engagement_history.append({
            'engagement': feedback_metrics["engagement"],
            'long_term_engagement': feedback_metrics.get("long_term_engagement", feedback_metrics["engagement"]),
            'ability': st.session_state.user_ability,
            'interaction': len(st.session_state.engagement_history) + 1
        })
        
        # Enhanced Long-term Trends Visualization
        if len(st.session_state.engagement_history) > 1:
            with st.expander("📈 Long-term Feedback Trends (θ_{t+1} & P_engage)", expanded=False):
                st.markdown("### Longitudinal Analysis: User Ability & Engagement Evolution")
                st.caption("**Equation 10:** θ_{t+1} = α·θ_t + (1-α)·θ̂_t (α=0.7) | **Equation 23:** P_engage = Σ(w_i·s_i)/Σ(w_i) where w_i=exp(-λ·(n-i)), λ=0.1")
                
                engagement_data = pd.DataFrame(st.session_state.engagement_history)
                
                # Create dual-axis plot
                fig_trends = go.Figure()
                
                # User ability (θ) trend
                fig_trends.add_trace(go.Scatter(
                    x=engagement_data['interaction'],
                    y=engagement_data['ability'],
                    mode='lines+markers',
                    name='User Ability (θ)',
                    line=dict(color='#4A90E2', width=3),
                    marker=dict(size=8),
                    yaxis='y1'
                ))
                
                # Engagement (E_t) trend
                fig_trends.add_trace(go.Scatter(
                    x=engagement_data['interaction'],
                    y=engagement_data['engagement'],
                    mode='lines+markers',
                    name='Engagement (E_t)',
                    line=dict(color='#7BC043', width=2, dash='dot'),
                    marker=dict(size=6),
                    yaxis='y1'
                ))
                
                # Long-term engagement (P_engage) trend
                fig_trends.add_trace(go.Scatter(
                    x=engagement_data['interaction'],
                    y=engagement_data['long_term_engagement'],
                    mode='lines+markers',
                    name='Long-term Engagement (P_engage)',
                    line=dict(color='#E27B4A', width=2),
                    marker=dict(size=6),
                    yaxis='y1'
                ))
                
                # Add target zone
                fig_trends.add_hrect(
                    y0=0.6, y1=0.8, 
                    fillcolor='rgba(123,192,67,0.1)', 
                    line_width=0,
                    annotation_text="Optimal Zone",
                    annotation_position="top left"
                )
                
                fig_trends.update_layout(
                    title="User Model Evolution: θ_{t+1} and P_engage Over Sessions",
                    xaxis=dict(title="Interaction Number (t)", dtick=1),
                    yaxis=dict(title="Score [0, 1]", range=[0, 1]),
                    height=400,
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                
                st.plotly_chart(fig_trends, use_container_width=True)
                
                # Statistical summary
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                
                with col_stats1:
                    ability_trend = engagement_data['ability'].iloc[-1] - engagement_data['ability'].iloc[0]
                    st.metric(
                        "Ability Change (Δθ)", 
                        f"{ability_trend:+.3f}",
                        delta=f"{ability_trend*100:+.1f}%",
                        help="Total change in user ability from first to last interaction"
                    )
                
                with col_stats2:
                    avg_engagement = engagement_data['engagement'].mean()
                    st.metric(
                        "Avg E_t", 
                        f"{avg_engagement:.3f}",
                        help="Mean engagement score across all interactions"
                    )
                
                with col_stats3:
                    current_p_engage = engagement_data['long_term_engagement'].iloc[-1]
                    st.metric(
                        "Current P_engage", 
                        f"{current_p_engage:.3f}",
                        help="Current long-term engagement with exponential weighting (λ=0.1)"
                    )
                
                # Correlation analysis
                if len(engagement_data) >= 3:
                    st.markdown("---")
                    st.markdown("**Correlation Analysis:**")
                    
                    correlation = engagement_data['ability'].corr(engagement_data['engagement'])
                    
                    col_corr1, col_corr2 = st.columns(2)
                    with col_corr1:
                        st.metric(
                            "θ ↔ E_t Correlation", 
                            f"{correlation:.3f}",
                            help="Pearson correlation between user ability and engagement"
                        )
                    
                    with col_corr2:
                        if correlation > 0.5:
                            st.success("✅ Strong positive relationship: Higher ability → Higher engagement")
                        elif correlation < -0.5:
                            st.warning("⚠️ Negative relationship: System may need recalibration")
                        else:
                            st.info("ℹ️ Moderate relationship: Continue monitoring")
    
    # Footer
    st.markdown("---")
    st.caption("💡 **Academic Research Demonstration** - Context-Adaptive Cognitive Flow System")
    st.caption(f"Session: {st.session_state.session_id} | Interactions: {len(st.session_state.engagement_history)}")

if __name__ == "__main__":
    main()
