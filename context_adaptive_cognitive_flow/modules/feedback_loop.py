"""
Stage IV: Feedback and Iterative Adaptation

The system accommodates day-to-day variability common in older adults through adaptive 
user modeling. The 0.7/0.3 weighting balances stability against responsiveness, accounting 
for both "good days" and "bad days" that characterize cognitive aging.

Validation: 41% increase in engagement retention over 6-week intervention.
"""

import numpy as np


class FeedbackLoop:
    """
    Stage IV: Feedback and Iterative Adaptation
    
    Implements recursive capacity estimation via exponential smoothing.
    Enables long-term personalization as user skill evolves.
    
    Reference: Eq.(4) from paper
    """
    
    def __init__(self, alpha: float = 0.7):
        """
        Initialize feedback loop with smoothing parameter.
        
        Args:
            alpha: Exponential smoothing weight (memory vs. adaptation tradeoff)
                   α=0.7 favors historical estimate (stable)
                   α=0.3 favors new observation (responsive)
        """
        self.alpha = alpha
        self.beta = 1.0 - alpha
        
    def update_capacity(self, theta_t: float, theta_hat_t: float) -> float:
        """
        Update user capacity estimate via exponential moving average.
        
        Eq.(4): θ_{t+1} = 0.7*θ_t + 0.3*θ̂_t
        
        Where:
        - θ_t: Current capacity estimate (smoothed history)
        - θ̂_t: New capacity observation (from recent performance)
        - θ_{t+1}: Updated capacity estimate
        
        Clinical interpretation:
        - Gradual updates prevent overreaction to transient fluctuations
        - Balances stability with responsiveness to genuine skill changes
        - Supports detection of learning curves and fatigue patterns
        
        Args:
            theta_t: Current capacity estimate
            theta_hat_t: Observed capacity from recent performance
            
        Returns:
            theta_{t+1}: Updated capacity estimate
        """
        theta_next = self.alpha * theta_t + self.beta * theta_hat_t
        return theta_next
    
    def estimate_observed_capacity(self, L_cog: float, P_t: float, b_t: float) -> float:
        """
        Estimate observed capacity θ̂_t from current performance metrics.
        
        Inverse inference: given L_cog and P_t, estimate underlying capacity.
        
        Heuristic used here:
        θ̂_t ≈ b_t + k*(P_t - 0.5) - m*L_cog
        
        Where k amplifies performance signal and m penalizes high cognitive load.
        
        Args:
            L_cog: Cognitive load index
            P_t: Performance probability
            b_t: Current difficulty bias
            
        Returns:
            theta_hat_t: Observed capacity estimate
        """
        performance_adjustment = 1.2 * (P_t - 0.5)
        
        load_penalty = 0.3 * (L_cog - 1.0)
        
        theta_hat_t = b_t + performance_adjustment - load_penalty
        
        theta_hat_t = np.clip(theta_hat_t, 0.0, 5.0)
        
        return theta_hat_t
    
    def adapt(self, theta_t: float, L_cog: float, P_t: float, b_t: float) -> float:
        """
        Full adaptation cycle: observe performance → estimate capacity → update.
        
        Args:
            theta_t: Current capacity estimate
            L_cog: Cognitive load index
            P_t: Performance probability
            b_t: Current difficulty bias
            
        Returns:
            theta_{t+1}: Updated capacity estimate
        """
        theta_hat_t = self.estimate_observed_capacity(L_cog, P_t, b_t)
        
        theta_next = self.update_capacity(theta_t, theta_hat_t)
        
        return theta_next
