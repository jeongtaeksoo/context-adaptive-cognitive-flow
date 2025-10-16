# Context-Adaptive Cognitive Flow (CACF) System

## Overview

This is a research implementation of the Context-Adaptive Cognitive Flow (CACF) System - a multi-agent AI platform designed for cognitive activation in older adults. The system implements a 4-stage processing pipeline based on academic research, processing multimodal inputs through perception, context recognition, adaptive response generation, and feedback loops.

The application uses a multi-agent architecture with three specialized AI personas (Teacher, Companion, Coach) that adapt their responses based on user context, emotional state, and performance metrics. It's built as a Streamlit web application with optional database persistence.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Architecture Pattern: 4-Stage Pipeline

The system implements a sequential processing pipeline where each stage transforms data for the next:

1. **Stage I - Perception Engine** (`perception_engine.py`): Multimodal data acquisition
   - Extracts features from behavioral, voice, performance, and temporal inputs
   - Normalizes disparate data sources into unified feature vectors
   - Implements attention weighting based on signal quality

2. **Stage II - Context Hub** (`context_hub.py`): Persona-specific context recognition
   - Applies persona-specific attention weights to features
   - Maps inputs to valence-arousal emotional space (Russell's Circumplex Model)
   - Generates context vectors using neural network simulation or ML embeddings

3. **Stage III - Response Engine** (`response_engine.py`): Adaptive response generation
   - Uses Item Response Theory (IRT) for difficulty adjustment
   - Calculates success probabilities and cognitive load
   - Generates persona-appropriate responses with adaptive pacing

4. **Stage IV - Feedback Loop** (`feedback_loop.py`): User ability adaptation
   - Updates user ability estimates using Exponential Moving Average (EMA)
   - Tracks engagement metrics across interactions
   - Enables longitudinal learning and personalization

### Multi-Agent System Design

The **Agent Orchestrator** (`agent_orchestrator.py`) coordinates three specialized agents:

- **Teacher Agent** (`teacher_agent.py`): Focuses on cognitive activation, performance metrics, and educational progression
- **Companion Agent** (`companion_agent.py`): Emphasizes emotional support, social connection, and valence-arousal state
- **Coach Agent** (`coach_agent.py`): Concentrates on behavioral patterns, activity recommendations, and lifestyle optimization

**Selection Strategies**:
- `highest_priority`: Selects agent with best context match
- `weighted_blend`: Combines responses from multiple agents
- `round_robin`: Rotates between agents for variety

### Frontend Architecture

**Streamlit Application** (`app.py`):
- Web-based UI with real-time interaction processing
- Scenario selection from predefined contexts (morning energy, afternoon fatigue, etc.)
- Visualization of context states, engagement history, and agent responses
- Session state management for continuous user experience
- Toggle between single-agent and multi-agent modes

**Key UI Components**:
- Input data cards showing all modality values
- Context state visualization (valence-arousal mapping, cognitive load)
- Agent response cards with priority scores and reasoning
- Engagement history charts with longitudinal tracking
- System architecture diagram showing data flow

### Data Flow Architecture

```
User Input → Perception Engine → Context Hub → Response Engine → User Display
                                                       ↓
                                                Feedback Loop
                                                       ↓
                                            User Ability Update
```

**State Management**:
- Session-based state in Streamlit (`st.session_state`)
- User ability parameter (theta) persisted across interactions
- Engagement and context history stored in memory
- Optional database persistence for longitudinal analysis

### Mathematical Framework

The system implements specific equations from the academic paper:

- **Equations 2-3**: Feature extraction and normalization
- **Equations 4-6**: Attention weighting and valence-arousal mapping
- **Equations 7-9**: IRT-based difficulty adjustment
- **Equations 10-11, 22-23**: EMA-based ability updates and engagement tracking

Key parameters:
- IRT discrimination: `a = 1.0`
- EMA alpha: `α = 0.7` (ability updates)
- Target success probability: `P* = 0.7`
- Cognitive load weights: `β₁=0.4, β₂=0.35, β₃=0.25`

### Utility Functions

`utils.py` provides mathematical operations:
- Sigmoid and tanh activation functions with temperature control
- Softmax for probability distributions
- Feature normalization and exponential moving averages
- IRT probability calculations
- Sample data loading for demonstrations

## External Dependencies

### Required Python Packages

**Core Dependencies**:
- `streamlit` (>=1.50.0): Web application framework
- `numpy` (>=2.3.3): Numerical computations and array operations
- `pandas` (>=2.3.3): Data manipulation and analysis
- `plotly` (>=6.3.1): Interactive visualizations
- `scipy` (>=1.16.2): Scientific computing functions

### Database Integration (Optional)

**PostgreSQL via SQLAlchemy**:
- `sqlalchemy` (>=2.0.43): ORM for database operations
- `psycopg2-binary` (>=2.9.10): PostgreSQL adapter

**Database Models** (`database_models.py`):
- `UserSession`: Tracks research sessions with metadata
- `Interaction`: Logs individual system processing cycles
- Stores multimodal inputs, context states, responses, and feedback metrics

**Connection**: Requires `DATABASE_URL` environment variable for PostgreSQL connection. System gracefully degrades to in-memory operation if unavailable.

### Optional ML Enhancements

**Sentence Transformers** (experimental, commented out):
- `sentence-transformers` (>=2.2.0): For semantic text embeddings
- `torch` (>=2.0.0): PyTorch backend for neural models

The system can use ML embeddings for context encoding if `sentence-transformers` is available, otherwise falls back to rule-based encoding.

### Sample Data

Predefined scenarios in `sample_input.json`:
- Morning high energy
- Afternoon moderate fatigue  
- Evening relaxed
- Various behavioral and performance states

### Deployment Configuration

The application is designed for:
- Streamlit Cloud deployment (primary)
- Local development with `streamlit run app.py`
- Research demonstration with minimal dependencies
- Academic reproducibility with documented equations

**Entry Points**:
- `main.py`: Programmatic launch with configuration
- `app.py`: Direct Streamlit execution
- `cacf_demo.py`: Minimal academic demonstration without UI