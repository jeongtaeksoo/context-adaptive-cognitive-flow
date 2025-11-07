"""
Stage III: Emotionally Adaptive Response Strategy

To prevent frustration while maintaining engagement—a critical balance in cognitive 
interventions for older adults—we developed an adaptive difficulty system that maintains 
success probability within the therapeutic window.

The adaptive difficulty regulation mechanism operates exclusively within the Teacher agent 
module, dynamically adjusting task complexity based on real-time assessment of user ability (θ_t). 
The resulting difficulty state (b_t) is then propagated to the Coach and Companion agents, 
which modulate their emotional support and motivational strategies accordingly to maintain 
patient engagement within the optimal learning zone.

Crucially, response timing adapts based on age-related processing speed changes.
This formulation ensures responses are neither frustratingly fast nor patronizingly slow, 
with bounds empirically validated through older adult usability studies.
"""

import numpy as np


class ResponseStrategy:
    """
    Stage III: Emotionally Adaptive Response Strategy
    
    Implements adaptive difficulty control and emotional state estimation.
    Equations (2-3) control task difficulty based on user performance probability.
    
    Reference: Eq.(2-3) from paper
    """
    
    def __init__(self, a: float = 3.0, P_star: float = 0.85, eta: float = 0.15):
        """
        Initialize response strategy with control parameters.
        
        Args:
            a: Sigmoid steepness parameter (sensitivity to difficulty mismatch)
            P_star: Target success probability at approximately 0.85, 
                   the optimal challenge zone for learning
            eta: Bias adaptation learning rate
        """
        self.a = a
        self.P_star = P_star
        self.eta = eta
        
    def compute_performance_probability(self, theta_t: float, b_t: float) -> float:
        """
        Compute expected user performance probability via sigmoid function.
        
        Eq.(2): P_t = 1 / (1 + exp(-a*(θ_t - b_t)))
        
        Interpretation:
        - θ_t: User ability parameter
        - b_t: Item difficulty
        - When θ_t > b_t: task is easier than capacity → high P_t
        - When θ_t < b_t: task exceeds capacity → low P_t
        
        Args:
            theta_t: Current user ability estimate
            b_t: Current item difficulty
            
        Returns:
            P_t: Performance probability ∈ [0, 1]
        """
        exponent = -self.a * (theta_t - b_t)
        P_t = 1.0 / (1.0 + np.exp(exponent))
        return P_t
    
    def update_difficulty_bias(self, b_t: float, P_t: float) -> float:
        """
        Update task difficulty bias to maintain optimal challenge.
        
        Eq.(3): b_{t+1} = clip_[0,3](b_t + η*(P_t - P*))
        
        Adaptive rule:
        - If P_t > P*: task too easy → increase bias (harder)
        - If P_t < P*: task too hard → decrease bias (easier)
        
        Args:
            b_t: Current difficulty bias
            P_t: Current performance probability
            
        Returns:
            b_{t+1}: Updated difficulty bias, clipped to [0, 3]
        """
        delta_b = self.eta * (P_t - self.P_star)
        b_next = b_t + delta_b
        b_next = np.clip(b_next, 0.0, 3.0)
        return b_next
    
    def estimate_emotional_state(self, L_cog: float) -> tuple:
        """
        Estimate emotional valence and arousal from cognitive load.
        
        Companion agent uses this for empathetic response generation.
        
        Mapping heuristic:
        - Low L_cog → positive valence, low arousal (boredom)
        - Optimal L_cog → positive valence, moderate arousal (flow state)
        - High L_cog → negative valence, high arousal (stress)
        
        Args:
            L_cog: Cognitive load index
            
        Returns:
            (valence, arousal): Emotional state in [-1, 1] × [-1, 1]
        """
        if L_cog < 0.5:
            valence = 0.3 - 0.5 * (0.5 - L_cog)
            arousal = -0.4 + 0.3 * L_cog
        elif L_cog <= 1.5:
            valence = 0.6 - 0.2 * (L_cog - 1.0)
            arousal = -0.2 + 0.4 * (L_cog - 0.5)
        else:
            overstimulation = L_cog - 1.5
            valence = 0.4 - 0.8 * overstimulation
            arousal = 0.2 + 0.5 * overstimulation
            
        valence = np.clip(valence, -1.0, 1.0)
        arousal = np.clip(arousal, -1.0, 1.0)
        
        return (valence, arousal)
    
    def compute_response_delay(self, L_cog: float) -> float:
        """
        Compute adaptive system response delay based on cognitive load.
        
        Formula: t_delay = clip_[1.5, 5.0](2.5 * max(0.5, 1 + 0.8*L_cog))
        
        Rationale: Higher cognitive load → longer delay to prevent overwhelm
        
        Args:
            L_cog: Cognitive load index
            
        Returns:
            t_delay: Response delay in seconds
        """
        inner_term = max(0.5, 1.0 + 0.8 * L_cog)
        t_delay = 2.5 * inner_term
        t_delay = np.clip(t_delay, 1.5, 5.0)
        return t_delay
