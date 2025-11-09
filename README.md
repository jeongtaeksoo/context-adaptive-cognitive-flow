# Context-Adaptive Cognitive Flow for Persona-Driven Interaction

A Python implementation of the context-adaptive cognitive flow framework for cognitive rehabilitation in older adults. This system employs a context-adaptive cognitive flow that dynamically adjusts to each user's emotional state, behavioral patterns, and cognitive ability.

## Overview

To enable emotionally meaningful and cognitively relevant engagement, the system implements a 4-stage adaptive pipeline with three collaborative persona agents (Coach, Teacher, Companion) that work together to provide personalized cognitive interventions.

### Clinical Validation Results
- **41% higher retention** compared to non-adaptive systems
- **62% reduction** in computational load via selective attention
- **80% accuracy** in emotion recognition (Russell's circumplex model)

## Architecture

### Four-Stage Framework

1. **Stage I: Multimodal Data Sensing**
   - Captures behavioral, vocal, performance, and temporal features
   - Selective attention reduces computational load by 62%

2. **Stage II: Persona-Specific Context Recognition**
   - **Eq.(1)**: `L_cog = 0.4*(Δt_resp/t̄_base) + 0.35*e_rate + 0.25*σ_att²`
   - Provides clinically interpretable load levels in [0, 2]

3. **Stage III: Emotionally Adaptive Response Strategy**
   - **Eq.(2)**: `P_t = 1/(1 + exp(-a*(θ_t - b_t)))`
   - **Eq.(3)**: `b_{t+1} = clip[0,3](b_t + η*(P_t - P*))`
   - Maintains success probability at P* ≈ 0.85 (optimal learning zone)
   - Adaptive response delay prevents frustration

4. **Stage IV: Feedback and Iterative Adaptation**
   - **Eq.(4)**: `θ_{t+1} = 0.7*θ_t + 0.3*θ̂_t`
   - 0.7/0.3 weighting accounts for "good days" and "bad days"
   - Accommodates day-to-day cognitive variability

### Persona Agents

- **Coach**: Receives difficulty state, delivers motivational feedback
- **Teacher**: Exclusively controls adaptive difficulty regulation (θ_t, b_t, P_t)
- **Companion**: Receives difficulty state, provides emotional support (Russell's circumplex model)

## Quick Start

```bash
# Run the simulation
python context_adaptive_cognitive_flow/main.py
```

The simulation will:
- Execute 10 time steps of adaptive cognitive flow
- Print detailed state information for each step
- Show persona agent responses (Coach, Companion, Teacher)
- Generate visualization saved as `cognitive_flow_simulation.png`

## Project Structure

```
context_adaptive_cognitive_flow/
├── main.py                    # Main simulation entry point
├── agents/
│   ├── coach.py            # Coach persona agent
│   ├── teacher.py          # Teacher persona agent
│   └── companion.py        # Companion persona agent    
└── modules/
    ├── sensing.py            # Stage I: Multimodal sensing
    ├── context_recognition.py # Stage II: Cognitive load (Eq.1)
    ├── response_strategy.py   # Stage III: Adaptation (Eq.2-3)
    └── feedback_loop.py       # Stage IV: User ability update (Eq.4)
```

## Sample Output

```
[t=00] L_cog=0.81 | P_t=0.71 | b_t=1.20 | θ_t=1.50 | Delay=4.1s | Emotion=(+0.6,-0.1)
       [Coach] Step 0 | Establishing baseline...
       [Companion] Emotion=content (V=+0.64, A=-0.08) | L_cog=0.81
       [Teacher] Difficulty decreasing by 0.021 | User Ability Δ=+0.003 | Performance=71.09%
```

## Key Equations Implemented

- **Eq.1** (Cognitive Load): `L_cog = 0.4*(Δt_resp/t̄_base) + 0.35*e_rate + 0.25*σ_att²`
Weights (0.4, 0.35, 0.25) were set to reflect the relative influence of response latency, error rate, and attention variability.
- **Eq.2** (Performance): `P_t = 1/(1 + exp(-a*(θ_t - b_t)))`
  - θ_t: user ability, b_t: item difficulty
- **Eq.3** (Difficulty Update): `b_{t+1} = clip[0,3](b_t + η*(P_t - P*))`
  - P* ≈ 0.85 (optimal challenge zone for learning)
- **Eq.4** (User Ability Update): `θ_{t+1} = 0.7*θ_t + 0.3*θ̂_t`
  - 0.7/0.3 weighting balances stability vs. responsiveness
- **Delay Formula**: `t_delay = clip[1.5,5.0](2.5 * max(0.5, 1 + 0.8*L_cog))`
  - Neither frustratingly fast nor patronizingly slow

## Dependencies

- Python 3.11+
- NumPy (mathematical computations)
- Matplotlib (visualization)

## Features

✅ Deterministic simulation (reproducible results)  
✅ Mathematical fidelity to research equations  
✅ Modular architecture with clean separation of concerns  
✅ Three collaborative persona agents  
✅ Real-time visualization of cognitive flow dynamics  
✅ Academic-style documentation with equation references  

## References

Based on the paper: *"Designing a Generative AI Framework for Cognitive Intervention in Older Adults"*

**GitHub Repository**: https://github.com/jeongtaeksoo/context-adaptive-cognitive-flow

The complete implementation of this clinically-validated pipeline includes multimodal perception, persona-specific recognition, adaptive response generation, and iterative feedback loops with interactive demonstrations.
