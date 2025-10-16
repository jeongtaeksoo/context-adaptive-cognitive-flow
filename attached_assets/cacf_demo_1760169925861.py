"""
Context-Adaptive Cognitive Flow (CACF) System - Minimal Academic Demonstration

This implementation serves as a minimal academic demonstration of the Context-Adaptive 
Cognitive Flow (CACF) model described in Section 3.2.2. It is not a clinical or commercial 
application but a conceptual prototype for research reproducibility.

Paper Reference: Context-Adaptive Cognitive Flow System for Cognitive Activation in Older Adults
Section 3.2.2 - Multi-Agent Architecture Implementation

Theoretical Framework: S_t = {D_t, C_t, R_t, F_t}
"""

import numpy as np

# Theoretical Variable Mapping (Section 3.2.2)
# D_t = multimodal_input  | C_t = context_vector  | R_t = response_plan  | F_t = feedback_metrics

class CACF_System:
    def __init__(self):
        self.theta = 0.5  # User ability θ_t
        self.alpha = 0.3  # EMA parameter
    
    # Stage I: D_t - Multimodal Data Sensing (Eq. 2, 3)
    def stage_D(self, scenario):
        """D_t: Acquire and normalize multimodal inputs"""
        D_t = np.array([
            scenario['behavioral'].get('activity_level', 0.5),
            scenario['behavioral'].get('social_engagement', 0.5),
            scenario['voice'].get('speech_clarity', 0.7),
            scenario['voice'].get('emotional_tone', 0.0),
            scenario['performance'].get('response_time', 3.0) / 10.0,
            scenario['performance'].get('accuracy', 0.75),
            {'morning': 1.0, 'afternoon': 0.5, 'evening': 0.0}.get(scenario['temporal']['time_of_day'], 0.5)
        ])
        return D_t
    
    # Stage II: C_t - Context Recognition (Eq. 4-6)
    def stage_C(self, D_t, persona):
        """C_t: Recognize context with persona-specific attention"""
        # Persona-specific attention weights (Eq. 4 - simplified)
        weights = {'teacher': [0.2, 0.2, 0.1, 0.1, 0.5, 0.5, 0.2],
                   'companion': [0.2, 0.3, 0.5, 0.5, 0.2, 0.2, 0.2],
                   'coach': [0.5, 0.5, 0.2, 0.2, 0.2, 0.3, 0.2]}[persona]
        
        C_t = D_t * np.array(weights)  # Attention mechanism (Eq. 5)
        valence = np.tanh(np.mean(C_t[:4]))  # Valence-arousal mapping (Eq. 6)
        arousal = 1 / (1 + np.exp(-np.mean(C_t[4:])))
        return C_t, {'valence': valence, 'arousal': arousal}
    
    # Stage III: R_t - Response Generation (Eq. 7-9)
    def stage_R(self, C_t, persona, va):
        """R_t: Generate adaptive response strategy"""
        cognitive_load = 1.0 - np.mean(C_t)
        
        if persona == 'teacher':
            msg = 'Challenge!' if cognitive_load < 0.3 else 'Break it down!' if cognitive_load > 0.7 else 'Keep going!'
            return f"Teacher: {msg} (θ={self.theta:.2f})"
        elif persona == 'companion':
            v = va['valence']
            msg = 'Good spirits!' if v > 0.3 else "I'm here." if v < -0.2 else 'How are you?'
            return f"Companion: {msg} (v={v:.2f})"
        else:
            activity = C_t[0]
            msg = 'Great momentum!' if activity > 0.7 else 'Try a walk!' if activity < 0.3 else 'Gentle movement!'
            return f"Coach: {msg} (a={activity:.2f})"
    
    # Stage IV: F_t - Feedback Loop (Eq. 10-11)
    def stage_F(self, accuracy, difficulty=0.5):
        """F_t: Update user model with feedback"""
        theta_obs = difficulty + (accuracy - 0.5) * 2.0  # Observed ability
        self.theta = self.alpha * theta_obs + (1 - self.alpha) * self.theta  # EMA update (Eq. 10)
        engagement = 1 / (1 + np.exp(-(self.theta - difficulty))) * accuracy  # Engagement (Eq. 11)
        return {'theta': self.theta, 'engagement': engagement}
    
    # Complete CACF Cycle
    def execute_cacf(self, scenario, persona='teacher'):
        """Execute S_t = {D_t, C_t, R_t, F_t}"""
        D_t = self.stage_D(scenario)
        C_t, va = self.stage_C(D_t, persona)
        R_t = self.stage_R(C_t, persona, va)
        F_t = self.stage_F(scenario['performance']['accuracy'])
        return {'D_t': D_t, 'C_t': C_t, 'R_t': R_t, 'F_t': F_t, 'va': va}

# Demonstration
if __name__ == "__main__":
    cacf = CACF_System()
    
    # Example scenario
    scenario = {
        "behavioral": {"activity_level": 0.8, "social_engagement": 0.7},
        "voice": {"speech_clarity": 0.85, "emotional_tone": 0.6},
        "performance": {"response_time": 2.5, "accuracy": 0.9},
        "temporal": {"time_of_day": "morning"}
    }
    
    print("=" * 70)
    print("CACF Demonstration - Section 3.2.2")
    print("=" * 70)
    
    for persona in ['teacher', 'companion', 'coach']:
        S_t = cacf.execute_cacf(scenario, persona)
        print(f"\n{persona.upper()}")
        print(f"  D_t: {len(S_t['D_t'])} features | C_t: v={S_t['va']['valence']:.2f}, a={S_t['va']['arousal']:.2f}")
        print(f"  R_t: {S_t['R_t']}")
        print(f"  F_t: θ={S_t['F_t']['theta']:.3f}, E={S_t['F_t']['engagement']:.3f}")
    
    print(f"\n{'='*70}")
    print("Closed-loop: F_t → updated θ → influences next cycle")
    print("=" * 70)
