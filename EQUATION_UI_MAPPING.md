# CACF System: Equation-to-UI Structural Mapping

This document provides a comprehensive mapping between the academic paper's mathematical equations and the implemented UI components, ensuring rigorous alignment between theory and practice.

## Overview: 4-Stage Architecture

The Context-Adaptive Cognitive Flow (CACF) System implements a 4-stage processing pipeline:

1. **Stage I: Multimodal Sensing** (Perception Engine) → Equations 2-3
2. **Stage II: Persona-Specific Recognition** (Context Hub) → Equations 4-6
3. **Stage III: Emotionally Adaptive Response** (Response Engine) → Equations 7-9
4. **Stage IV: Feedback & Adaptation** (Feedback Loop) → Equations 10-11, 22-23

---

## Stage I: Multimodal Sensing (Perception Engine)

### Equations 2-3: Feature Extraction and Normalization

**Mathematical Formulation:**
- **Eq. 2:** `D_t = f_perception(x_behavioral, x_voice, x_performance, x_temporal)`
- **Eq. 3:** `D_t^normalized = normalize(D_t)`

**Implementation Mapping:**

| Parameter | Equation Component | UI Element | Code Location |
|-----------|-------------------|------------|---------------|
| `x_behavioral` | Activity, social, sleep | Input Data Card → Behavioral Section | `app.py` line 120-122 |
| `x_voice` | Clarity, emotion, rate | Input Data Card → Voice Section | `app.py` line 124-127 |
| `x_performance` | Speed, accuracy, load | Input Data Card → Performance Section | `app.py` line 129-132 |
| `x_temporal` | Time of day, day of week | Input Data Card → Temporal Section | `app.py` line 134-137 |
| `D_t` | Unified feature vector | Not directly displayed | `perception_engine.py` line 235 |

**Processing Flow:**
1. User selects scenario → multimodal input data loaded
2. Perception engine extracts features from each modality
3. Attention weights computed based on signal quality
4. Features concatenated and normalized to produce `D_t`

---

## Stage II: Persona-Specific Recognition (Context Hub)

### Equations 4-6: Attention Weighting and Valence-Arousal Mapping

**Mathematical Formulation:**
- **Eq. 4:** `α_persona = {α_teacher, α_companion, α_coach}` (persona-specific attention weights)
- **Eq. 5:** `C_t = α_persona ⊙ D_t` (element-wise multiplication)
- **Eq. 6:** `VA_t = (valence, arousal) = f_VA(C_t)`

**Implementation Mapping:**

| Parameter | Equation Component | UI Element | Code Location |
|-----------|-------------------|------------|---------------|
| `α_persona` | Attention weights | Agent Cards → Context Trigger | `context_hub.py` line 32-51 |
| `C_t` | Attended features | Not directly displayed | `context_hub.py` line 91-95 |
| `valence` | Emotional positivity [-1,1] | System Metrics (future enhancement) | `context_hub.py` line 109-110 |
| `arousal` | Emotional intensity [-1,1] | System Metrics (future enhancement) | `context_hub.py` line 112-114 |
| `L_cog` | Cognitive load [0,1] | Agent Priority Scores | `context_hub.py` line 132-133 |
| `E_engagement` | Engagement level | System Metrics → Engagement | `context_hub.py` line 139-141 |

**Processing Flow:**
1. Each agent (Teacher, Companion, Coach) applies persona-specific attention to `D_t`
2. Attended features mapped to valence-arousal emotional space
3. Cognitive state dimensions estimated (load, attention, engagement)
4. Context confidence computed for reliability assessment

---

## Stage III: Emotionally Adaptive Response (Response Engine)

### Equations 7-9: IRT-based Difficulty and Response Generation

**Mathematical Formulation:**
- **Eq. 7:** `P(θ,β) = c + (d-c)/(1 + exp(-a(θ-β)))` (IRT 3PL model)
  - `θ` = user ability
  - `β` = task difficulty
  - `a` = discrimination (1.0)
  - `c` = guessing (0.0)
  - `d` = upper asymptote (1.0)
- **Eq. 8:** `L_cog = f_load(C_t, β)` (cognitive load estimation)
- **Eq. 9:** `R_t = f_response(θ, β, C_t)` (adaptive response generation)

**Implementation Mapping:**

| Parameter | Equation Component | UI Element | Code Location |
|-----------|-------------------|------------|---------------|
| `θ` | User ability | System Metrics → User Ability (θ) | `app.py` line 250 |
| `β` | Task difficulty | Agent metadata → adapted_difficulty | `response_engine.py` line 282 |
| `P(θ,β)` | Success probability | Agent metadata → success_probability | `response_engine.py` line 285 |
| `L_cog` | Cognitive load | Agent metadata → estimated_cognitive_load | `response_engine.py` line 288 |
| `R_t` | Response text | Agent Response Cards | `app.py` line 191 |
| `π` | Priority score | Agent Cards → Priority Metric | `response_engine.py` line 401-416 |

**Processing Flow:**
1. For each agent, compute IRT success probability based on current `θ` and `β`
2. Estimate cognitive load from context state and task difficulty
3. Adjust difficulty adaptively based on recent performance
4. Select response template matching persona and context
5. Compute priority score for response selection
6. Generate complete response package with metadata

---

## Stage IV: Feedback & Adaptation (Feedback Loop)

### Equations 10-11, 22-23: User Model Updates and Engagement Tracking

**Mathematical Formulation:**
- **Eq. 10:** `θ_{t+1} = α·θ_t + (1-α)·θ̂_t` (EMA update with α=0.7)
- **Eq. 11:** `E_t = Σ w_i · engagement_factor_i`
- **Eq. 22:** Engagement weights: task_completion (0.3), quality (0.2), duration (0.15), feedback (0.15), alignment (0.1), difficulty (0.1)
- **Eq. 23:** `P_engage = Σ(w_i·s_i) / Σ(w_i)` where `w_i = exp(-λ·(n-i))` with λ=0.1

**Implementation Mapping:**

| Parameter | Equation Component | UI Element | Code Location |
|-----------|-------------------|------------|---------------|
| `θ_t` | Current ability | System Metrics → User Ability (θ) | `app.py` line 36-37 |
| `θ_{t+1}` | Updated ability | Updated after each interaction | `feedback_loop.py` line 88 |
| `α` | EMA smoothing (0.7) | Hard-coded parameter | `feedback_loop.py` line 21 |
| `E_t` | Engagement score | System Metrics → Engagement | `app.py` line 252 |
| `P_engage` | Long-term engagement | Engagement Trend Visualization | `feedback_loop.py` line 198-216 |
| `λ` | Decay factor (0.1) | Hard-coded parameter | `feedback_loop.py` line 22 |

**Processing Flow:**
1. User completes interaction with performance score
2. Compute observed ability `θ̂_t` based on task difficulty and performance
3. Update user ability using EMA: `θ_{t+1} = 0.7·θ_t + 0.3·θ̂_t`
4. Compute engagement score `E_t` from weighted factors (Eq. 22)
5. Update long-term engagement `P_engage` using exponential weighting (Eq. 23)
6. Generate adaptation signals for system components
7. Store metrics in session history for longitudinal analysis

---

## Multi-Agent Orchestration

### Response Selection Strategies

**Three Strategies Implemented:**

1. **Highest Priority** (`π_max`):
   - Select agent with maximum priority score
   - `selected_agent = argmax_i(π_i)`
   - UI: Shows primary agent badge with priority score

2. **Weighted Blend** (`R_blend`):
   - Combine responses weighted by priority
   - `w_i = π_i / Σπ_j`
   - Primary response + supporting insights from agents with w_i > 0.2
   - UI: Shows primary agent + supporting agents list

3. **Round Robin** (`R_rr`):
   - Rotate between agents for balanced interaction
   - UI: Shows current agent in rotation

**Implementation Mapping:**

| Component | Implementation | UI Element | Code Location |
|-----------|---------------|------------|---------------|
| Parallel Invocation | ThreadPoolExecutor | "Processing multimodal input..." spinner | `agent_orchestrator.py` line 56-62 |
| Priority Computation | Context alignment × Success score | Agent Cards → Priority | `response_engine.py` line 388-416 |
| Strategy Selection | Sidebar dropdown | Configuration → Response Selection Strategy | `app.py` line 78-83 |
| Final Response | Aggregated output | Final System Response Section | `app.py` line 221-241 |

---

## Database Schema Mapping

**Session Tracking:**
- `session_id`: Unique session identifier → `st.session_state.session_id`
- `user_ability`: Current θ value → Database column `user_ability`
- `engagement_scores`: E_t history → Visualization data source

**Interaction Logging:**
- Input features → JSON column `input_data`
- Context state (C_t, VA_t) → JSON column `context_state`
- Response (R_t) → JSON column `response_data`
- Feedback metrics (θ, E_t) → JSON column `feedback_metrics`

---

## Verification Checklist

✅ **Stage I (Perception):**
- [x] Multimodal inputs mapped to UI sections
- [x] Feature extraction implemented per Eq. 2-3
- [x] Normalization applied to D_t

✅ **Stage II (Context):**
- [x] Persona-specific attention weights (α) defined
- [x] Attended features C_t computed per Eq. 5
- [x] Valence-arousal mapping per Eq. 6

✅ **Stage III (Response):**
- [x] IRT probability P(θ,β) computed per Eq. 7
- [x] Cognitive load L_cog estimated per Eq. 8
- [x] Adaptive difficulty adjustment per Eq. 9
- [x] Priority score π for response selection

✅ **Stage IV (Feedback):**
- [x] EMA update with α=0.7 per Eq. 10
- [x] Engagement score E_t per Eq. 11, 22
- [x] Long-term engagement P_engage per Eq. 23 with λ=0.1

✅ **Multi-Agent Orchestration:**
- [x] Parallel agent invocation with ThreadPoolExecutor
- [x] Three selection strategies implemented
- [x] Response attribution tracked

---

## Future UI Enhancements (Current Task)

To further strengthen equation-to-UI alignment, the following visualizations are being added:

1. **4-Stage Flow Diagram**: Sankey chart showing data flow D_t → C_t → R_t → F_t
2. **Agent State Radar Charts**: Multi-dimensional view of priority, cognitive load, valence, arousal per agent
3. **Response Selection Flow**: Visual diagram of parallel processing and strategy-based selection
4. **Long-term Trends**: Line charts showing θ_{t+1} evolution and P_engage over sessions
5. **Equation Annotations**: Hover tooltips linking UI elements to paper equations

These enhancements will provide complete transparency between mathematical formulation and system implementation.
