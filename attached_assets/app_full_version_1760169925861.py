import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from perception_engine import PerceptionEngine
from context_hub import ContextHub
from response_engine import ResponseEngine
from feedback_loop import FeedbackLoop
from utils import load_sample_data, visualize_context_state, create_engagement_chart, safe_serialize
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
    except Exception as e:
        print(f"Database initialization failed: {e}")
        return None

# Initialize session state
if 'user_ability' not in st.session_state:
    st.session_state.user_ability = 0.5  # θ parameter
if 'engagement_history' not in st.session_state:
    st.session_state.engagement_history = []
if 'context_history' not in st.session_state:
    st.session_state.context_history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
    st.session_state.session_created = False
if 'db_logging_enabled' not in st.session_state:
    st.session_state.db_logging_enabled = True
if 'multi_agent_mode' not in st.session_state:
    st.session_state.multi_agent_mode = True  # Default to multi-agent
if 'agent_strategy' not in st.session_state:
    st.session_state.agent_strategy = 'highest_priority'
if 'show_all_agents' not in st.session_state:
    st.session_state.show_all_agents = True  # Default to showing all agents

def main():
    st.set_page_config(
        page_title="Context-Adaptive Cognitive Flow System",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Context-Adaptive Cognitive Flow System")
    st.markdown("**Generative AI-Based Daily Life Application for Cognitive Activation in Older Adults**")
    
    # Get database manager
    db_manager = get_db_manager()
    
    # Sidebar for system configuration
    st.sidebar.header("System Configuration")
    
    # Session management
    st.sidebar.subheader("Session Management")
    st.sidebar.text(f"Session ID: {st.session_state.session_id}")
    
    # Create session in database if not already done
    if db_manager and not st.session_state.session_created:
        try:
            db_manager.create_session(
                session_id=st.session_state.session_id,
                user_id="research_participant",
                session_metadata={"created_at": datetime.now().isoformat()}
            )
            st.session_state.session_created = True
        except Exception as e:
            st.sidebar.warning(f"Database session creation failed: {e}")
    
    # Database logging toggle and status
    if db_manager:
        st.session_state.db_logging_enabled = st.sidebar.checkbox(
            "Enable Database Logging",
            value=st.session_state.db_logging_enabled,
            help="Save all interactions to database for longitudinal analysis"
        )
        st.sidebar.success("✅ Database connected")
    else:
        st.session_state.db_logging_enabled = False
        st.sidebar.warning("⚠️ Database not configured (DATABASE_URL missing)")
    
    if st.sidebar.button("🔄 New Session"):
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
        st.session_state.session_created = False
        st.session_state.user_ability = 0.5
        st.session_state.engagement_history = []
        st.session_state.context_history = []
        st.rerun()
    
    st.sidebar.divider()
    
    # Multi-agent system controls
    st.sidebar.subheader("Agent Configuration")
    
    st.session_state.multi_agent_mode = st.sidebar.checkbox(
        "🤖 Multi-Agent Mode",
        value=st.session_state.multi_agent_mode,
        help="Enable parallel processing with Teacher, Companion, and Coach agents"
    )
    
    if st.session_state.multi_agent_mode:
        st.session_state.agent_strategy = st.sidebar.selectbox(
            "Response Strategy",
            ["highest_priority", "weighted_blend", "round_robin"],
            index=["highest_priority", "weighted_blend", "round_robin"].index(st.session_state.agent_strategy),
            help="How to select/combine responses from multiple agents"
        )
        
        st.session_state.show_all_agents = st.sidebar.checkbox(
            "☑️ Show All Agent Responses",
            value=st.session_state.show_all_agents,
            help="Display individual outputs from all three agents before showing the final selected response"
        )
        selected_persona = "Multi-Agent"  # Not used in multi-agent mode
    else:
        selected_persona = st.sidebar.selectbox(
            "Select AI Persona",
            ["Teacher", "Companion", "Coach"],
            help="Choose the AI agent persona for interaction"
        )
    
    difficulty_level = st.sidebar.slider(
        "Base Difficulty Level",
        0.1, 1.0, 0.5,
        help="Initial difficulty setting (will be adapted based on user performance)"
    )
    
    st.sidebar.divider()
    st.sidebar.subheader("Advanced Settings")
    
    # ML embeddings toggle (requires sentence-transformers package)
    import os
    use_ml = st.sidebar.checkbox(
        "Use ML Embeddings (Experimental)",
        value=os.environ.get('USE_ML_EMBEDDINGS', 'false').lower() == 'true',
        help="Enable sentence-transformers for context encoding (requires package installation)"
    )
    
    # Initialize system components
    perception_engine = PerceptionEngine()
    context_hub = ContextHub(use_ml_embeddings=use_ml)
    response_engine = ResponseEngine()
    feedback_loop = FeedbackLoop()
    orchestrator = AgentOrchestrator()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Interactive Demo", 
        "📊 System Visualization", 
        "📜 Session History",
        "🔧 Component Analysis",
        "📚 Documentation"
    ])
    
    with tab1:
        st.header("Interactive System Demonstration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Configuration")
            
            # Load sample data
            sample_data = load_sample_data()
            
            # User input options
            input_method = st.radio(
                "Choose input method:",
                ["Use Sample Data", "Custom Input", "🎤 Voice Input"]
            )
            
            if input_method == "Use Sample Data":
                scenario_key = st.selectbox(
                    "Select scenario:",
                    list(sample_data.keys())
                )
                input_data = sample_data[scenario_key]
                st.json(input_data)
            
            elif input_method == "Custom Input":
                st.write("**Behavioral Input**")
                activity_level = st.slider("Activity Level", 0.0, 1.0, 0.5)
                social_engagement = st.slider("Social Engagement", 0.0, 1.0, 0.5)
                
                st.write("**Voice Features**")
                speech_clarity = st.slider("Speech Clarity", 0.0, 1.0, 0.7)
                emotional_tone = st.slider("Emotional Tone", -1.0, 1.0, 0.0)
                
                st.write("**Performance Data**")
                response_time = st.slider("Response Time (seconds)", 1.0, 10.0, 3.0)
                accuracy = st.slider("Task Accuracy", 0.0, 1.0, 0.8)
                
                input_data = {
                    "behavioral": {
                        "activity_level": activity_level,
                        "social_engagement": social_engagement
                    },
                    "voice": {
                        "speech_clarity": speech_clarity,
                        "emotional_tone": emotional_tone
                    },
                    "performance": {
                        "response_time": response_time,
                        "accuracy": accuracy
                    },
                    "temporal": {
                        "time_of_day": "afternoon",
                        "day_of_week": "tuesday"
                    }
                }
            
            elif input_method == "🎤 Voice Input":
                st.write("**Voice Input Processing**")
                st.info("📱 Record your voice to analyze speech patterns and emotional tone")
                
                # Voice recording widget
                audio_file = st.audio_input("Record your message", key="voice_input")
                
                if audio_file:
                    st.audio(audio_file)
                    
                    # Process voice input for speech features
                    import speech_recognition as sr
                    
                    with st.spinner("🔊 Analyzing voice features..."):
                        try:
                            recognizer = sr.Recognizer()
                            
                            # Convert audio file to AudioFile
                            audio_data = sr.AudioFile(audio_file)
                            
                            with audio_data as source:
                                audio = recognizer.record(source)
                            
                            # Transcribe speech
                            try:
                                transcribed_text = recognizer.recognize_google(audio)
                                st.success(f"**Transcription:** {transcribed_text}")
                                
                                # Analyze speech features from audio
                                # Simulate feature extraction (in real implementation, use actual audio analysis)
                                speech_clarity = min(1.0, len(transcribed_text.split()) / 20)  # More words = clearer
                                emotional_tone = 0.3 if "good" in transcribed_text.lower() or "great" in transcribed_text.lower() else -0.2
                                
                            except sr.UnknownValueError:
                                st.warning("Could not understand audio - using default values")
                                transcribed_text = "unclear audio"
                                speech_clarity = 0.3
                                emotional_tone = -0.3
                            except sr.RequestError as e:
                                st.error(f"Speech recognition error: {e}")
                                transcribed_text = "recognition error"
                                speech_clarity = 0.5
                                emotional_tone = 0.0
                            
                            # Simulated voice features (in production, extract from actual audio analysis)
                            st.write("**Extracted Voice Features:**")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Speech Clarity", f"{speech_clarity:.2f}")
                            with col_b:
                                st.metric("Emotional Tone", f"{emotional_tone:+.2f}")
                            
                            # Use default behavioral and performance metrics with voice data
                            input_data = {
                                "behavioral": {
                                    "activity_level": 0.6,
                                    "social_engagement": 0.7
                                },
                                "voice": {
                                    "speech_clarity": float(speech_clarity),
                                    "emotional_tone": float(emotional_tone),
                                    "transcribed_text": transcribed_text
                                },
                                "performance": {
                                    "response_time": 3.0,
                                    "accuracy": 0.75
                                },
                                "temporal": {
                                    "time_of_day": datetime.now().strftime("%p").lower().replace("am", "morning").replace("pm", "afternoon"),
                                    "day_of_week": datetime.now().strftime("%A").lower()
                                }
                            }
                            
                        except Exception as e:
                            st.error(f"Voice processing error: {str(e)[:200]}")
                            # Fallback to default data
                            input_data = sample_data["afternoon_moderate_fatigue"]
                else:
                    st.info("👆 Click above to record your voice")
                    # Use default data until voice is recorded
                    input_data = sample_data["afternoon_moderate_fatigue"]
        
        with col2:
            st.subheader("System Processing & Output")
            
            if st.button("🚀 Process Input", type="primary"):
                # Stage I: Multimodal Data Sensing
                st.write("**Stage I: Multimodal Data Sensing**")
                processed_input = perception_engine.process_input(input_data)
                st.success(f"✅ Input vector xt processed: {len(processed_input)} features")
                
                if st.session_state.multi_agent_mode:
                    # Multi-Agent Processing Mode
                    st.write("**Stage II-IV: Multi-Agent Orchestration**")
                    
                    # Prepare context for agents
                    agent_context = {
                        'difficulty_level': difficulty_level,
                        'time_of_day': input_data['temporal'].get('time_of_day', 'afternoon'),
                        'task_type': 'cognitive task'
                    }
                    
                    # First, invoke all agents in parallel to get individual outputs
                    import time
                    start_time = time.time()
                    all_agent_outputs = orchestrator.invoke_agents_parallel(processed_input, agent_context)
                    invocation_time = time.time() - start_time
                    
                    st.success(f"✅ {len(all_agent_outputs)} agents invoked in {invocation_time*1000:.1f}ms")
                    
                    # Display all agent responses if enabled
                    if st.session_state.show_all_agents:
                        st.divider()
                        st.subheader("🤖 All Agent Responses")
                        
                        # Agent emoji mapping
                        agent_emojis = {
                            'teacher': '👨‍🏫',
                            'companion': '🤝', 
                            'coach': '💪'
                        }
                        
                        # Display each agent's response in columns
                        cols = st.columns(3)
                        
                        for idx, (agent_name, agent_output) in enumerate(all_agent_outputs.items()):
                            with cols[idx]:
                                emoji = agent_emojis.get(agent_name, '🤖')
                                agent_title = agent_name.capitalize()
                                
                                with st.expander(f"{emoji} **{agent_title}Agent**", expanded=True):
                                    # Display trigger
                                    st.caption(f"**Trigger:** {agent_output.get('trigger', 'N/A')}")
                                    
                                    # Display priority
                                    priority = agent_output['metadata'].get('priority_score', 0.0)
                                    st.metric("Priority Score", f"{priority:.2f}")
                                    
                                    # Display response
                                    st.write("**Response:**")
                                    st.info(agent_output['response'])
                        
                        st.divider()
                    
                    # Now aggregate responses using the selected strategy
                    aggregated_output = orchestrator.aggregate_responses(all_agent_outputs, st.session_state.agent_strategy)
                    
                    # Add orchestration metadata
                    aggregated_output['orchestration'] = {
                        'num_agents_invoked': len(all_agent_outputs),
                        'invocation_time_ms': invocation_time * 1000,
                        'all_agent_priorities': {
                            name: output['metadata'].get('priority_score', 0.0)
                            for name, output in all_agent_outputs.items()
                        }
                    }
                    
                    # Extract response and metadata
                    response_text = aggregated_output['response']
                    response_metadata = aggregated_output['metadata']
                    attribution = aggregated_output.get('attribution', {})
                    trigger = aggregated_output.get('trigger', '')
                    
                    # Update difficulty if teacher agent made adjustments
                    if 'new_difficulty' in aggregated_output:
                        difficulty_level = aggregated_output['new_difficulty']
                    
                    # Compute feedback using traditional feedback loop
                    feedback_metrics = feedback_loop.update_user_model(
                        st.session_state.user_ability,
                        difficulty_level,
                        input_data["performance"]["accuracy"]
                    )
                    st.session_state.user_ability = feedback_metrics["updated_ability"]
                    
                    # Create response dict for compatibility
                    response = {
                        'content': response_text,
                        'difficulty': difficulty_level,
                        'multi_agent': True,
                        'attribution': attribution,
                        'trigger': trigger,
                        'orchestration': aggregated_output['orchestration'],
                        'all_agent_outputs': all_agent_outputs  # Store for potential future use
                    }
                    
                    # Create context state for recording (approximate from agent data)
                    context_state = {
                        'context_dimensions': {
                            'cognitive_state': response_metadata.get('cognitive_load', 0.5) if response_metadata['agent'] == 'Teacher' else 0.5,
                            'emotional_state': response_metadata.get('valence', 0.0) if response_metadata['agent'] == 'Companion' else 0.0,
                            'engagement_level': response_metadata.get('priority_score', 0.5)
                        },
                        'valence_arousal': {
                            'valence': response_metadata.get('valence', 0.0),
                            'arousal': response_metadata.get('arousal', 0.0)
                        }
                    }
                    
                else:
                    # Single Persona Mode (Traditional Flow)
                    st.write("**Stage II: Context Recognition**")
                    context_state = context_hub.compute_context(processed_input, selected_persona)
                    st.success(f"✅ Context state computed for {selected_persona}")
                    
                    st.write("**Stage III: Response Generation**")
                    response = response_engine.generate_response(
                        context_state, 
                        selected_persona,
                        st.session_state.user_ability,
                        difficulty_level
                    )
                    st.success("✅ Adaptive response generated")
                    
                    st.write("**Stage IV: Feedback & Adaptation**")
                    feedback_metrics = feedback_loop.update_user_model(
                        st.session_state.user_ability,
                        response["difficulty"],
                        input_data["performance"]["accuracy"]
                    )
                    st.session_state.user_ability = feedback_metrics["updated_ability"]
                
                # Create complete interaction record
                interaction_record = {
                    'timestamp': datetime.now().isoformat(),
                    'persona': selected_persona,
                    'input_data': input_data,
                    'context_dimensions': context_state['context_dimensions'],
                    'valence_arousal': context_state['valence_arousal'],
                    'response': response,
                    'feedback_metrics': feedback_metrics,
                    'user_ability': st.session_state.user_ability
                }
                
                st.session_state.context_history.append(interaction_record)
                st.session_state.engagement_history.append({
                    'engagement': feedback_metrics["engagement"],
                    'long_term_engagement': feedback_metrics.get("long_term_engagement", feedback_metrics["engagement"])
                })
                
                # Save to database if logging enabled and database manager available
                if st.session_state.db_logging_enabled and db_manager:
                    try:
                        # Save interaction
                        db_manager.save_interaction(
                            session_id=st.session_state.session_id,
                            persona=selected_persona,
                            input_data=input_data,
                            context_state=context_state,
                            response_data=response,
                            feedback_metrics=feedback_metrics
                        )
                        
                        # Save ability update
                        db_manager.save_ability_update(
                            session_id=st.session_state.session_id,
                            ability_data=feedback_metrics,
                            context_state=context_state
                        )
                        
                        # Save engagement metrics
                        db_manager.save_engagement_metrics(
                            session_id=st.session_state.session_id,
                            engagement_data=feedback_metrics
                        )
                        
                        # Save adaptation signals if any
                        if 'adaptation_signals' in feedback_metrics:
                            for component, signals in feedback_metrics['adaptation_signals'].items():
                                if signals and component != 'global_system':
                                    db_manager.save_adaptation_event(
                                        session_id=st.session_state.session_id,
                                        component=component,
                                        adaptation_type='parameter_adjustment',
                                        adaptation_data=signals,
                                        reason=f"Adaptation triggered by {component} signals"
                                    )
                        
                        st.success("✅ User model updated (saved to database)")
                    except Exception as e:
                        st.warning(f"⚠️ User model updated (database save failed: {str(e)[:100]})")
                else:
                    st.success("✅ User model updated")
                
                # Display results
                # Show different header based on whether all agents were displayed
                if response.get('multi_agent', False) and st.session_state.show_all_agents:
                    st.subheader("▶️ Final Selected Response")
                else:
                    st.subheader("🎯 Generated Response")
                
                # Show multi-agent attribution if applicable
                if response.get('multi_agent', False):
                    attribution = response.get('attribution', {})
                    primary_agent = attribution.get('primary_agent', 'Unknown')
                    
                    # Show agent badge
                    agent_emoji = {'teacher': '👨‍🏫', 'companion': '🤝', 'coach': '💪'}.get(primary_agent, '🤖')
                    st.markdown(f"**{agent_emoji} {primary_agent.capitalize()} Agent** ({attribution.get('selection_reason', 'Selected')})")
                    
                    # Show contextual trigger
                    if 'trigger' in response:
                        st.caption(f"📊 Trigger: {response['trigger']}")
                    
                    # Show supporting agents if blended
                    if 'supporting_agents' in attribution and attribution['supporting_agents']:
                        supporting = ', '.join([a.capitalize() for a in attribution['supporting_agents']])
                        st.caption(f"🔗 Supporting: {supporting}")
                
                st.info(response["content"])
                
                st.subheader("📈 System Metrics")
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("User Ability (θ)", f"{st.session_state.user_ability:.3f}")
                with metrics_col2:
                    st.metric("Response Difficulty", f"{response['difficulty']:.3f}")
                with metrics_col3:
                    st.metric("Engagement Score", f"{feedback_metrics['engagement']:.3f}")
                
                # Show orchestration metrics for multi-agent mode
                if response.get('multi_agent', False) and 'orchestration' in response:
                    orch = response['orchestration']
                    st.caption(f"⚙️ Orchestration: {orch['num_agents_invoked']} agents invoked | {orch['invocation_time_ms']:.1f}ms | Priorities: {orch['all_agent_priorities']}")
        
        # Current session export
        if st.session_state.context_history:
            st.divider()
            st.subheader("📥 Export Current Session")
            
            exp_col1, exp_col2 = st.columns(2)
            
            with exp_col1:
                # Comprehensive JSON export with all session data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_session_data = safe_serialize({
                    'export_metadata': {
                        'export_time': datetime.now().isoformat(),
                        'session_id': st.session_state.session_id,
                        'total_interactions': len(st.session_state.context_history)
                    },
                    'user_model': {
                        'current_ability': st.session_state.user_ability,
                        'ability_history': [ctx.get('user_ability', st.session_state.user_ability) 
                                          for ctx in st.session_state.context_history]
                    },
                    'interactions': [{
                        'interaction_num': i + 1,
                        'timestamp': ctx.get('timestamp'),
                        'persona': ctx.get('persona'),
                        'input_data': ctx.get('input_data'),
                        'context_state': ctx.get('context_dimensions'),
                        'response': ctx.get('response'),
                        'feedback_metrics': ctx.get('feedback_metrics'),
                        'valence_arousal': ctx.get('valence_arousal')
                    } for i, ctx in enumerate(st.session_state.context_history)],
                    'engagement_metrics': st.session_state.engagement_history,
                    'context_evolution': st.session_state.context_history
                })
                
                json_export = json.dumps(current_session_data, indent=2)
                
                st.download_button(
                    label="📋 Download Complete JSON",
                    data=json_export,
                    file_name=f"session_{st.session_state.session_id[:8]}_{timestamp}.json",
                    mime="application/json",
                    key="json_export_current"
                )
            
            with exp_col2:
                # Comprehensive CSV export for research analysis
                csv_rows = []
                for i, ctx in enumerate(st.session_state.context_history):
                    row = {
                        'interaction_num': i + 1,
                        'session_id': st.session_state.session_id,
                        'timestamp': ctx.get('timestamp', datetime.now().isoformat()),
                        'persona': ctx.get('persona', 'Unknown'),
                        'user_ability': ctx.get('user_ability', st.session_state.user_ability),
                    }
                    
                    # Add context dimensions
                    if 'context_dimensions' in ctx:
                        for k, v in ctx['context_dimensions'].items():
                            row[f'context_{k}'] = safe_serialize(v)
                    
                    # Add complete input data (behavioral, voice, performance, temporal)
                    if 'input_data' in ctx:
                        input_data = ctx['input_data']
                        
                        # Behavioral inputs
                        if 'behavioral' in input_data:
                            for k, v in input_data['behavioral'].items():
                                row[f'input_behavioral_{k}'] = safe_serialize(v)
                        
                        # Voice inputs
                        if 'voice' in input_data:
                            for k, v in input_data['voice'].items():
                                row[f'input_voice_{k}'] = safe_serialize(v)
                        
                        # Performance inputs
                        if 'performance' in input_data:
                            for k, v in input_data['performance'].items():
                                row[f'input_performance_{k}'] = safe_serialize(v)
                        
                        # Temporal context
                        if 'temporal' in input_data:
                            row['time_of_day'] = input_data['temporal'].get('time_of_day', '')
                            row['day_of_week'] = input_data['temporal'].get('day_of_week', '')
                    
                    # Add engagement data (handle both dict and float formats)
                    if i < len(st.session_state.engagement_history):
                        eng = st.session_state.engagement_history[i]
                        if isinstance(eng, dict):
                            row['engagement_score'] = eng.get('engagement', 0)
                            row['long_term_engagement'] = eng.get('long_term_engagement', 0)
                        else:
                            # Legacy float format
                            row['engagement_score'] = float(eng)
                            row['long_term_engagement'] = float(eng)
                    
                    # Add valence-arousal
                    if 'valence_arousal' in ctx:
                        row['valence'] = ctx['valence_arousal'].get('valence', 0)
                        row['arousal'] = ctx['valence_arousal'].get('arousal', 0)
                    
                    # Add response metrics
                    if 'response' in ctx:
                        row['response_difficulty'] = ctx['response'].get('difficulty', 0)
                        row['response_content'] = ctx['response'].get('content', '')[:200]  # Truncate for CSV
                    
                    # Add feedback metrics and adaptation signals
                    if 'feedback_metrics' in ctx:
                        fm = ctx['feedback_metrics']
                        row['performance_score'] = fm.get('performance', 0)
                        
                        # Add adaptation signals if present
                        if 'adaptation_signals' in fm:
                            for component, signals in fm['adaptation_signals'].items():
                                if signals and isinstance(signals, dict):
                                    for sig_key, sig_val in signals.items():
                                        row[f'adaptation_{component}_{sig_key}'] = safe_serialize(sig_val)
                    
                    csv_rows.append(safe_serialize(row))
                
                if csv_rows:
                    session_csv = pd.DataFrame(csv_rows).to_csv(index=False)
                else:
                    session_csv = "No data available\n"
                
                st.download_button(
                    label="💾 Download Research CSV",
                    data=session_csv,
                    file_name=f"session_{st.session_state.session_id[:8]}_{timestamp}.csv",
                    mime="text/csv",
                    key="csv_export_current"
                )
    
    with tab2:
        st.header("System Visualization")
        
        if st.session_state.context_history:
            # Context state visualization
            st.subheader("Context State Evolution")
            
            # Extract numeric context dimensions for visualization
            context_data = []
            for i, ctx in enumerate(st.session_state.context_history):
                row = {'Interaction': i + 1}
                row.update(ctx['context_dimensions'])
                context_data.append(row)
            
            context_df = pd.DataFrame(context_data)
            
            # Create line chart with properly formatted data
            fig = go.Figure()
            for col in context_df.columns:
                if col != 'Interaction':
                    fig.add_trace(go.Scatter(
                        x=context_df['Interaction'],
                        y=context_df[col],
                        mode='lines+markers',
                        name=col.replace('_', ' ').title()
                    ))
            
            fig.update_layout(
                title="Context Dimensions Over Time",
                xaxis_title="Interaction Number",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Engagement history
            if st.session_state.engagement_history:
                st.subheader("Engagement Metrics")
                engagement_fig = create_engagement_chart(st.session_state.engagement_history)
                st.plotly_chart(engagement_fig, use_container_width=True)
            
            # Valence-Arousal mapping
            st.subheader("Valence-Arousal State")
            if len(st.session_state.context_history) > 0:
                latest_context = st.session_state.context_history[-1]
                valence_arousal_fig = visualize_context_state(latest_context)
                st.plotly_chart(valence_arousal_fig, use_container_width=True)
        else:
            st.info("Run the interactive demo to generate visualizations")
    
    with tab3:
        st.header("📜 Session History & Longitudinal Analysis")
        
        if db_manager:
            # Session selector
            st.subheader("Session Selection")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Get all sessions
                all_sessions = db_manager.get_all_sessions()
                
                if all_sessions:
                    session_options = [f"{s['session_id']} ({s['total_interactions']} interactions)" 
                                     for s in all_sessions]
                    selected_idx = st.selectbox(
                        "Select Session:",
                        range(len(session_options)),
                        format_func=lambda i: session_options[i],
                        index=0 if st.session_state.session_id == all_sessions[0]['session_id'] else 0
                    )
                    selected_session = all_sessions[selected_idx]
                    
                    # Session details
                    st.info(f"**Session ID:** {selected_session['session_id']}  \n"
                           f"**Start Time:** {selected_session['start_time']}  \n"
                           f"**Total Interactions:** {selected_session['total_interactions']}  \n"
                           f"**Status:** {'Active' if selected_session['is_active'] else 'Ended'}")
                else:
                    st.warning("No sessions found in database")
                    selected_session = None
            
            with col2:
                if st.button("🔄 Refresh Sessions"):
                    st.rerun()
            
            # Export functionality
            if selected_session and selected_session['total_interactions'] > 0:
                st.divider()
                st.subheader("📥 Export Session Data")
                
                # Fetch all data once for all exports
                interactions = db_manager.get_session_history(selected_session['session_id'])
                ability_data = db_manager.get_ability_evolution(selected_session['session_id'])
                engagement_data = db_manager.get_engagement_history(selected_session['session_id'])
                adaptations = db_manager.get_adaptation_history(selected_session['session_id'])
                
                export_col1, export_col2, export_col3 = st.columns(3)
                
                with export_col1:
                    # Comprehensive CSV for research
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_rows = []
                    
                    if interactions:
                        for i, interaction in enumerate(interactions):
                            # Base row with all comprehensive research fields
                            row = safe_serialize({
                                'interaction_num': i + 1,
                                'session_id': selected_session['session_id'],
                                'timestamp': interaction.get('timestamp'),
                                'persona': interaction.get('persona'),
                                'user_ability': interaction.get('user_ability'),
                                'task_difficulty': interaction.get('response_difficulty'),
                                'engagement_score': interaction.get('engagement_score'),
                                'performance_score': interaction.get('performance_score'),
                                'valence': interaction.get('valence'),
                                'arousal': interaction.get('arousal'),
                                'response_content': interaction.get('response_content', '')[:200]
                            })
                            
                            # Add context dimensions with prefix
                            if 'context_dimensions' in interaction and interaction['context_dimensions']:
                                ctx_dims = safe_serialize(interaction['context_dimensions'])
                                for k, v in ctx_dims.items():
                                    row[f'context_{k}'] = v
                            
                            # Add input data fields
                            if 'input_data' in interaction and interaction['input_data']:
                                input_data = safe_serialize(interaction['input_data'])
                                if 'temporal' in input_data:
                                    row['time_of_day'] = input_data['temporal'].get('time_of_day')
                                    row['day_of_week'] = input_data['temporal'].get('day_of_week')
                                if 'behavioral' in input_data:
                                    for k, v in input_data['behavioral'].items():
                                        row[f'input_behavioral_{k}'] = v
                                if 'performance' in input_data:
                                    for k, v in input_data['performance'].items():
                                        row[f'input_performance_{k}'] = v
                            
                            # Add ability evolution if available
                            if ability_data and i < len(ability_data):
                                row['ability_at_interaction'] = ability_data[i].get('user_ability')
                            
                            # Add long-term engagement if available
                            if engagement_data and i < len(engagement_data):
                                row['long_term_engagement'] = engagement_data[i].get('long_term_engagement')
                            
                            csv_rows.append(row)
                    
                    if csv_rows:
                        csv_df = pd.DataFrame(csv_rows)
                        csv_data = csv_df.to_csv(index=False)
                    else:
                        csv_data = "No data available\n"
                    
                    st.download_button(
                        label="💾 Download Research CSV",
                        data=csv_data,
                        file_name=f"research_session_{selected_session['session_id'][:8]}_{timestamp}.csv",
                        mime="text/csv",
                        key="csv_download_history"
                    )
                
                with export_col2:
                    # Comprehensive JSON export
                    export_data = safe_serialize({
                        'export_metadata': {
                            'export_time': datetime.now().isoformat(),
                            'export_version': '1.0'
                        },
                        'session_info': {
                            'session_id': selected_session['session_id'],
                            'start_time': selected_session['start_time'],
                            'total_interactions': selected_session['total_interactions'],
                            'is_active': selected_session['is_active']
                        },
                        'interactions': interactions or [],
                        'ability_evolution': ability_data or [],
                        'engagement_history': engagement_data or [],
                        'system_adaptations': adaptations or []
                    })
                    
                    json_data = json.dumps(export_data, indent=2)
                    
                    st.download_button(
                        label="📋 Download Complete JSON",
                        data=json_data,
                        file_name=f"complete_session_{selected_session['session_id'][:8]}_{timestamp}.json",
                        mime="application/json",
                        key="json_download_history"
                    )
                
                with export_col3:
                    st.markdown("**📊 Chart Export Guide**")
                    st.info("To export publication-ready figures:\n\n"
                           "1. View charts in tabs below\n"
                           "2. Hover over chart\n"
                           "3. Click camera icon (📷)\n"
                           "4. Or right-click → 'Save image'\n\n"
                           "Plotly exports as high-DPI PNG suitable for publications.")
            
            # Display session data
            if selected_session and selected_session['total_interactions'] > 0:
                st.divider()
                
                # Tabs for different data views
                hist_tab1, hist_tab2, hist_tab3, hist_tab4 = st.tabs([
                    "📊 Ability Evolution",
                    "💬 Interactions",
                    "📈 Engagement Metrics",
                    "⚙️ Adaptations"
                ])
                
                with hist_tab1:
                    # User ability evolution
                    st.subheader("User Ability (θ) Evolution")
                    ability_history = db_manager.get_ability_evolution(selected_session['session_id'])
                    
                    if ability_history:
                        ability_df = pd.DataFrame(ability_history)
                        
                        # Create ability evolution chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(ability_df) + 1)),
                            y=ability_df['user_ability'],
                            mode='lines+markers',
                            name='User Ability (θ)',
                            line=dict(color='blue', width=3)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(ability_df) + 1)),
                            y=ability_df['task_difficulty'],
                            mode='lines',
                            name='Task Difficulty',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title="User Ability vs Task Difficulty Over Time",
                            xaxis_title="Interaction Number",
                            yaxis_title="Value",
                            yaxis=dict(range=[0, 1]),
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Initial Ability", f"{ability_df['user_ability'].iloc[0]:.3f}")
                        with col2:
                            st.metric("Current Ability", f"{ability_df['user_ability'].iloc[-1]:.3f}")
                        with col3:
                            total_change = ability_df['user_ability'].iloc[-1] - ability_df['user_ability'].iloc[0]
                            st.metric("Total Change", f"{total_change:+.3f}")
                        
                        # Show data table
                        if st.checkbox("Show Ability Data Table"):
                            st.dataframe(ability_df, use_container_width=True)
                    else:
                        st.info("No ability evolution data available for this session")
                
                with hist_tab2:
                    # Interaction history
                    st.subheader("Interaction History")
                    interactions = db_manager.get_session_history(selected_session['session_id'])
                    
                    if interactions:
                        for i, interaction in enumerate(interactions, 1):
                            with st.expander(f"Interaction {i} - {interaction['persona']} (Ability: {interaction['user_ability']:.3f})"):
                                col1, col2 = st.columns([1, 1])
                                
                                with col1:
                                    st.write("**Input Data:**")
                                    st.json(interaction['input_data'])
                                    
                                    st.write("**Context Dimensions:**")
                                    st.json(interaction['context_dimensions'])
                                
                                with col2:
                                    st.write("**Response:**")
                                    st.info(interaction['response_content'])
                                    
                                    st.write("**Metrics:**")
                                    metrics_data = {
                                        "Difficulty": interaction['response_difficulty'],
                                        "Engagement": interaction['engagement_score'],
                                        "Performance": interaction['performance_score'],
                                        "Valence": interaction['valence'],
                                        "Arousal": interaction['arousal']
                                    }
                                    st.json(metrics_data)
                    else:
                        st.info("No interaction data available for this session")
                
                with hist_tab3:
                    # Engagement metrics
                    st.subheader("Engagement Metrics Over Time")
                    engagement_history = db_manager.get_engagement_history(selected_session['session_id'])
                    
                    if engagement_history:
                        engagement_df = pd.DataFrame(engagement_history)
                        
                        # Create engagement chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(engagement_df) + 1)),
                            y=engagement_df['engagement_score'],
                            mode='lines+markers',
                            name='Engagement Score',
                            line=dict(color='green', width=2)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(engagement_df) + 1)),
                            y=engagement_df['long_term_engagement'],
                            mode='lines',
                            name='Long-term Engagement',
                            line=dict(color='purple', dash='dash')
                        ))
                        
                        fig.add_hline(y=0.7, line_dash="dot", line_color="green", 
                                    annotation_text="High Threshold")
                        fig.add_hline(y=0.3, line_dash="dot", line_color="orange",
                                    annotation_text="Low Threshold")
                        
                        fig.update_layout(
                            title="Engagement Trends",
                            xaxis_title="Interaction Number",
                            yaxis_title="Engagement Score",
                            yaxis=dict(range=[0, 1]),
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_engagement = engagement_df['engagement_score'].mean()
                            st.metric("Average Engagement", f"{avg_engagement:.3f}")
                        with col2:
                            current_lte = engagement_df['long_term_engagement'].iloc[-1]
                            st.metric("Long-term Engagement", f"{current_lte:.3f}")
                        with col3:
                            max_engagement = engagement_df['engagement_score'].max()
                            st.metric("Peak Engagement", f"{max_engagement:.3f}")
                    else:
                        st.info("No engagement data available for this session")
                
                with hist_tab4:
                    # System adaptations
                    st.subheader("System Adaptation Events")
                    adaptations = db_manager.get_adaptation_history(selected_session['session_id'])
                    
                    if adaptations:
                        for i, adapt in enumerate(adaptations, 1):
                            with st.expander(f"Adaptation {i} - {adapt['component']} ({adapt['adaptation_type']})"):
                                st.write(f"**Timestamp:** {adapt['timestamp']}")
                                st.write(f"**Component:** {adapt['component']}")
                                st.write(f"**Type:** {adapt['adaptation_type']}")
                                if adapt['reason']:
                                    st.write(f"**Reason:** {adapt['reason']}")
                                st.write("**Adaptation Data:**")
                                st.json(adapt['adaptation_data'])
                    else:
                        st.info("No adaptation events recorded for this session")
        else:
            st.warning("⚠️ Database not configured. Session history requires DATABASE_URL to be set.")
            st.info("Session history tracking allows you to:\n"
                   "- View all past interactions\n"
                   "- Analyze user ability evolution over time\n"
                   "- Track engagement metrics\n"
                   "- Review system adaptations\n\n"
                   "Configure DATABASE_URL environment variable to enable this feature.")
    
    with tab4:
        st.header("Component Analysis")
        
        # Equation mapping
        st.subheader("Equation-to-Module Mapping")
        equation_mapping = {
            "Stage I - Perception Engine": {
                "Equations": "2, 3, 13-15",
                "Description": "Multimodal input processing",
                "Implementation": "perception_engine.py"
            },
            "Stage II - Context Hub": {
                "Equations": "4-6, 16-18", 
                "Description": "Context recognition with attention mechanism",
                "Implementation": "context_hub.py"
            },
            "Stage III - Response Engine": {
                "Equations": "7-9, 19-21",
                "Description": "IRT-based adaptive response generation",
                "Implementation": "response_engine.py"
            },
            "Stage IV - Feedback Loop": {
                "Equations": "10-11, 22-23",
                "Description": "User ability updates and engagement tracking",
                "Implementation": "feedback_loop.py"
            }
        }
        
        for stage, details in equation_mapping.items():
            with st.expander(stage):
                st.write(f"**Equations:** {details['Equations']}")
                st.write(f"**Description:** {details['Description']}")
                st.write(f"**Implementation:** {details['Implementation']}")
        
        # System architecture diagram
        st.subheader("System Architecture")
        st.code("""
        Stage I: Multimodal Input → Perception Engine
                           ↓
        Stage II: Context Hub (with Attention Mechanism)
                           ↓
        Stage III: Response Engine (IRT-based Adaptation)
                           ↓
        Stage IV: Feedback Loop → Updated User Model
                           ↓ (iterative)
                    Back to Context Hub
        """, language="text")
    
    with tab5:
        st.header("Documentation")
        
        st.markdown("""
        ## Project Overview
        
        This application demonstrates a context-adaptive cognitive flow system designed to enhance 
        cognitive activation in older adults through personalized AI interactions.
        
        ## System Components
        
        ### 1. Perception Engine (Stage I)
        - **Purpose**: Process multimodal input data
        - **Equations**: 2, 3, 13-15
        - **Features**: Behavioral patterns, voice analysis, performance metrics, temporal context
        
        ### 2. Context Hub (Stage II)
        - **Purpose**: Recognize user context and emotional state
        - **Equations**: 4-6, 16-18
        - **Features**: Attention mechanism, valence-arousal mapping, persona-specific processing
        
        ### 3. Response Engine (Stage III)
        - **Purpose**: Generate adaptive responses based on context
        - **Equations**: 7-9, 19-21
        - **Features**: IRT-based difficulty adjustment, policy mapping, persona alignment
        
        ### 4. Feedback Loop (Stage IV)
        - **Purpose**: Update user model and track engagement
        - **Equations**: 10-11, 22-23
        - **Features**: EMA-based ability updates, engagement calculation
        
        ## Usage Instructions
        
        1. **Interactive Demo**: Use the first tab to process inputs and see real-time system responses
        2. **System Visualization**: Monitor context evolution and engagement metrics
        3. **Component Analysis**: Explore the equation-to-module mapping and system architecture
        
        ## Citation Guidelines
        
        When referencing this implementation in academic work, please cite:
        
        ```
        Context-Adaptive Cognitive Flow System: A Modular Python Implementation
        for Generative AI-Based Daily Life Applications in Older Adult Care
        ```
        
        ## Technical Notes
        
        - All neural network operations are simulated using mathematical functions
        - No actual machine learning training is performed
        - Sample data represents realistic scenarios for older adult interactions
        """)

if __name__ == "__main__":
    main()
