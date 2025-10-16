"""
Context-Adaptive Cognitive Flow (CACF) System - Publication UI

This implementation serves as a minimal academic demonstration of the Context-Adaptive 
Cognitive Flow (CACF) model described in Section 3.2.2. It is not a clinical or commercial 
application but a conceptual prototype for research reproducibility.

Theoretical Framework:
    S_t = {D_t, C_t, R_t, F_t}
    
    Stage I   (D_t): Multimodal data acquisition   → perception_engine.py
    Stage II  (C_t): Context recognition           → context_hub.py  
    Stage III (R_t): Response strategy generation  → response_engine.py
    Stage IV  (F_t): Feedback metric accumulation  → feedback_loop.py

For minimal academic demonstration, see: cacf_demo.py
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

# Initialize database manager (silent)
@st.cache_resource
def get_db_manager():
    """Get or create database manager instance (silent for publication version)"""
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
    
    # Silent database connection
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
    
    # Sidebar - Simplified Configuration
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
    
    # Display input data in a clean format
    with st.expander("📋 View Multimodal Input Data", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("**Behavioral**")
            st.write(f"• Activity: {input_data['behavioral']['activity_level']:.2f}")
            st.write(f"• Social: {input_data['behavioral']['social_engagement']:.2f}")
            st.write(f"• Sleep: {input_data['behavioral'].get('sleep_quality', 'N/A')}")
        
        with col2:
            st.write("**Voice**")
            st.write(f"• Clarity: {input_data['voice']['speech_clarity']:.2f}")
            st.write(f"• Emotion: {input_data['voice']['emotional_tone']:+.2f}")
            st.write(f"• Rate: {input_data['voice'].get('speaking_rate', 'N/A')}")
        
        with col3:
            st.write("**Performance**")
            st.write(f"• Response Time: {input_data['performance']['response_time']:.1f}s")
            st.write(f"• Accuracy: {input_data['performance']['accuracy']:.2f}")
            st.write(f"• Cognitive Load: {input_data['performance'].get('cognitive_load', 'N/A')}")
        
        with col4:
            st.write("**Temporal**")
            st.write(f"• Time: {input_data['temporal']['time_of_day'].title()}")
            st.write(f"• Day: {input_data['temporal']['day_of_week'].title()}")
    
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
            
            cols = st.columns(3)
            
            for idx, (agent_name, agent_output) in enumerate(all_agent_outputs.items()):
                with cols[idx]:
                    config = agent_config.get(agent_name, {'emoji': '🤖', 'title': agent_name.title()})
                    
                    st.markdown(f"### {config['emoji']} {config['title']}")
                    
                    # Trigger context
                    trigger = agent_output.get('trigger', 'N/A')
                    st.caption(f"**Context:** {trigger}")
                    
                    # Priority score
                    priority = agent_output['metadata'].get('priority_score', 0.0)
                    st.metric("Priority", f"{priority:.2f}")
                    
                    # Response
                    st.info(agent_output['response'])
            
            st.markdown("---")
        
        # Aggregate responses
        aggregated_output = orchestrator.aggregate_responses(all_agent_outputs, st.session_state.agent_strategy)
        
        # Update user model
        feedback_metrics = feedback_loop.update_user_model(
            st.session_state.user_ability,
            0.5,
            input_data["performance"]["accuracy"]
        )
        st.session_state.user_ability = feedback_metrics["updated_ability"]
        
        # Save to database (silent)
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
        
        # System Metrics
        st.subheader("4️⃣ System Metrics")
        
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("User Ability (θ)", f"{st.session_state.user_ability:.3f}")
        with metric_cols[1]:
            st.metric("Engagement", f"{feedback_metrics['engagement']:.3f}")
        with metric_cols[2]:
            st.metric("Agents Invoked", len(all_agent_outputs))
        with metric_cols[3]:
            st.metric("Processing Time", f"{invocation_time*1000:.0f}ms")
        
        # Engagement visualization
        st.session_state.engagement_history.append({
            'engagement': feedback_metrics["engagement"],
            'long_term_engagement': feedback_metrics.get("long_term_engagement", feedback_metrics["engagement"])
        })
        
        if len(st.session_state.engagement_history) > 1:
            with st.expander("📈 Engagement Trend Visualization", expanded=False):
                engagement_data = pd.DataFrame(st.session_state.engagement_history)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=engagement_data['engagement'],
                    mode='lines+markers',
                    name='Engagement',
                    line=dict(color='#4A90E2', width=2)
                ))
                fig.update_layout(
                    title="Engagement Score Over Time",
                    xaxis_title="Interaction",
                    yaxis_title="Engagement Score",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption("💡 **Academic Research Demonstration** - Context-Adaptive Cognitive Flow System with Multi-Agent Architecture")
    st.caption(f"Session: {st.session_state.session_id} | Interactions: {len(st.session_state.engagement_history)}")

if __name__ == "__main__":
    main()
