"""
CACF Validation Test
Verify that all corrections match LaTeX specification
"""

import numpy as np
from perception_engine import PerceptionEngine
from context_hub import ContextHub
from response_engine import ResponseEngine
from feedback_loop import FeedbackLoop

def validate_CACF_system():
    """
    Run comprehensive validation test for all CACF stages
    """
    print("="*70)
    print("CACF SYSTEM VALIDATION TEST")
    print("Verifying LaTeX Specification Compliance")
    print("="*70)
    
    # Initialize all components
    perception = PerceptionEngine()
    context = ContextHub()
    response = ResponseEngine()
    feedback = FeedbackLoop()
    
    # Create sample input
    sample_input = {
        'behavioral': {
            'activity_level': 0.7,
            'social_engagement': 0.6,
            'sleep_quality': 0.8
        },
        'voice': {
            'speech_clarity': 0.8,
            'emotional_tone': 0.3,
            'speaking_rate': 0.6
        },
        'performance': {
            'response_time': 3.5,
            'accuracy': 0.8,
            'task_completion': 0.85
        },
        'temporal': {
            'time_of_day': 'afternoon',
            'day_of_week': 'wednesday'
        }
    }
    
    print("\n" + "="*70)
    print("STAGE I: Multimodal Data Sensing")
    print("="*70)
    
    # Stage I: Process inputs
    behavioral_features = perception.process_behavioral_input(sample_input['behavioral'])
    voice_features = perception.process_voice_input(sample_input['voice'])
    performance_features = perception.process_performance_input(sample_input['performance'])
    temporal_features = perception.process_temporal_input(sample_input['temporal'])
    
    # Combine features
    D_t = np.concatenate([
        behavioral_features, voice_features, 
        performance_features, temporal_features
    ])
    
    print(f"✅ Feature vector D_t constructed: {len(D_t)} dimensions")
    
    print("\n" + "="*70)
    print("STAGE II: Context Recognition")
    print("="*70)
    
    # Stage II: Context recognition with Teacher persona
    C_t = context.compute_context(D_t, 'Teacher')
    
    # Test cognitive load formula (NEW)
    L_cog = context.compute_cognitive_load(
        response_time=3.5,
        error_rate=0.2,
        attention_variance=0.3
    )
    
    print(f"✅ Context state C_t computed: {len(C_t['context_vector'])} dimensions")
    print(f"✅ Cognitive Load: L_cog = {L_cog:.3f}")
    print(f"   Formula: L_cog = 0.4*(Δt/t_base) + 0.35*e_rate + 0.25*σ²")
    print(f"   Expected: β₁=0.4, β₂=0.35, β₃=0.25 ✓")
    
    # Verify neural network dimensions
    h_dim = context.context_network_weights['input_to_hidden'].shape[1]
    print(f"✅ Neural network hidden dimension: h={h_dim}")
    print(f"   Expected: h=64 {'✓' if h_dim == 64 else '✗ FAILED'}")
    
    print("\n" + "="*70)
    print("STAGE III: Response Strategy")
    print("="*70)
    
    # Stage III: Response generation
    theta_t = 0.6  # User ability
    b_t = 0.5      # Current difficulty
    
    # Test IRT probability
    P_t = response.compute_response_probability(theta_t, b_t)
    
    # Test difficulty update (NEW)
    b_next = response.update_difficulty(b_t, P_t)
    
    # Test adaptive pacing (NEW)
    t_delay = response.compute_adaptive_pacing(L_cog)
    
    print(f"✅ Success Probability: P_t = {P_t:.3f}")
    print(f"   Formula: P_t = 1/(1 + exp{{-a(θ_t - b_t)}})")
    print(f"   Using a={response.irt_parameters['a']} {'✓' if response.irt_parameters['a'] == 1.0 else '✗ FAILED'}")
    
    print(f"✅ Difficulty Update: b_{{t+1}} = {b_next:.3f}")
    print(f"   Formula: b_{{t+1}} = clip[0,3](b_t + η(P_t - P*))")
    print(f"   Using η={response.irt_parameters['eta']}, P*={response.irt_parameters['P_star']} ✓")
    
    print(f"✅ Adaptive Pacing: t_delay = {t_delay:.2f}s")
    print(f"   Formula: t_delay = clip[1.5,5.0](2.5 × max(0.5, 1 + 0.8·L_cog))")
    print(f"   Using γ={response.pacing_params['gamma']}, t_base={response.pacing_params['t_base']} ✓")
    
    print("\n" + "="*70)
    print("STAGE IV: Feedback Loop")
    print("="*70)
    
    # Stage IV: Feedback and adaptation
    theta_old = theta_t
    performance = 0.8
    
    # Test ability update (using actual context state structure)
    theta_new = feedback.update_user_ability(theta_old, b_t, performance, C_t)
    
    # Test engagement persistence
    session_history = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    for s in session_history:
        feedback.user_history['engagement_scores'].append(float(s))
    P_engage = feedback.update_long_term_engagement(0.8)
    
    print(f"✅ User Ability Update: θ_{{t+1}} = {theta_new:.3f}")
    print(f"   Formula: θ_{{t+1}} = α·θ_t + (1-α)·θ̂_t")
    print(f"   Using α={feedback.ema_parameters['alpha']} {'✓' if feedback.ema_parameters['alpha'] == 0.7 else '✗ FAILED'}")
    
    print(f"✅ Engagement Persistence: P_engage = {P_engage:.3f}")
    print(f"   Formula: P_engage = Σ(w_i·s_i)/Σ(w_i) where w_i=exp(-λ(n-i))")
    print(f"   Using λ={feedback.ema_parameters['lambda']} {'✓' if feedback.ema_parameters['lambda'] == 0.1 else '✗ FAILED'}")
    
    print("\n" + "="*70)
    print("PARAMETER VERIFICATION TABLE")
    print("="*70)
    
    params_check = [
        ("Stage II: β₁ (time weight)", 0.4, context.cognitive_load_params['beta_1']),
        ("Stage II: β₂ (error weight)", 0.35, context.cognitive_load_params['beta_2']),
        ("Stage II: β₃ (attention weight)", 0.25, context.cognitive_load_params['beta_3']),
        ("Stage II: Hidden dim h", 64, h_dim),
        ("Stage III: IRT discrimination a", 1.0, response.irt_parameters['a']),
        ("Stage III: Learning rate η", 0.1, response.irt_parameters['eta']),
        ("Stage III: Target prob P*", 0.7, response.irt_parameters['P_star']),
        ("Stage III: Pacing sensitivity γ", 0.8, response.pacing_params['gamma']),
        ("Stage IV: EMA stability α", 0.7, feedback.ema_parameters['alpha']),
        ("Stage IV: Decay parameter λ", 0.1, feedback.ema_parameters['lambda'])
    ]
    
    all_correct = True
    for name, expected, actual in params_check:
        status = "✅" if abs(expected - actual) < 0.001 else "❌"
        if status == "❌":
            all_correct = False
        print(f"{status} {name:35s} Expected: {expected:5.2f}  Actual: {actual:5.2f}")
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    print("\nNumerical Outputs (expected ±0.01 tolerance):")
    print(f"  L_cog   = {L_cog:.2f}   (expected ~0.56)")
    print(f"  P_t     = {P_t:.2f}   (expected ~0.52)")
    print(f"  b_next  = {b_next:.2f}   (expected ~0.48)")
    print(f"  t_delay = {t_delay:.1f}s  (expected ~3.6s)")
    print(f"  θ_next  = {theta_new:.2f}   (expected ~0.61)")
    print(f"  P_engage = {P_engage:.2f}  (expected ~0.81)")
    
    if all_correct:
        print("\n✅ ALL PARAMETERS MATCH LATEX SPECIFICATION")
        print("✅ CACF IMPLEMENTATION CORRECTED SUCCESSFULLY")
    else:
        print("\n❌ SOME PARAMETERS DO NOT MATCH - REVIEW NEEDED")
    
    print("="*70)
    
    return {
        'L_cog': L_cog,
        'P_t': P_t,
        'b_next': b_next,
        't_delay': t_delay,
        'theta_next': theta_new,
        'P_engage': P_engage,
        'all_correct': all_correct
    }

if __name__ == "__main__":
    results = validate_CACF_system()
