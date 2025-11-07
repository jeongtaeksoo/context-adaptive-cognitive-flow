"""
Teacher Agent: Adaptive Difficulty Regulation

The adaptive difficulty regulation mechanism operates exclusively within the Teacher agent 
module, dynamically adjusting task complexity based on real-time assessment of user ability (θ_t).

Responsibilities:
- Exclusive control of difficulty adaptation (θ_t, b_t, P_t)
- Maintain success probability at approximately 0.85 (optimal learning zone)
- Adjust task difficulty to prevent frustration while maintaining engagement

Collaboration:
- Propagates item difficulty state (b_t) to Companion and Coach agents
- Provides user ability estimates (θ_t) for coordinated support strategies
"""

from typing import Dict, Any


class TeacherAgent:
    """
    Teacher Persona: Adaptive Pedagogical Scaffolding
    
    Primary responsibility: maintain user in optimal challenge zone
    through dynamic difficulty adjustment (Eq. 2-3).
    """
    
    def __init__(self):
        """Initialize teacher agent with pedagogical parameters."""
        self.name = "Teacher"
        self.role = "Difficulty Adaptation"
        
    def respond(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate teacher response based on current cognitive state.
        
        Decision logic:
        - L_cog < 0.5: "Task too easy, increasing challenge"
        - 0.5 ≤ L_cog ≤ 1.5: "Good pace, maintaining difficulty"  
        - L_cog > 1.5: "Reducing difficulty, you're working hard"
        
        Args:
            state: Dictionary containing L_cog, P_t, b_t, theta_t
            
        Returns:
            Teacher response with message and adjusted parameters
        """
        L_cog = state['L_cog']
        P_t = state['P_t']
        b_t = state['b_t']
        theta_t = state['theta_t']
        
        if L_cog < 0.5:
            message = "Let's try something more challenging."
            recommendation = "increase_difficulty"
        elif L_cog <= 1.5:
            if P_t >= 0.7:
                message = "You're doing well at this level."
                recommendation = "maintain_difficulty"
            else:
                message = "Take your time, you're learning."
                recommendation = "slight_decrease"
        else:
            message = "Let's slow down a bit."
            recommendation = "decrease_difficulty"
            
        return {
            'agent': self.name,
            'message': message,
            'recommendation': recommendation,
            'adjusted_b_t': state.get('b_next', b_t),
            'user_ability_estimate': theta_t
        }
    
    def update(self, L_cog: float, P_t: float, b_t: float, 
               b_next: float, theta_t: float, theta_next: float) -> str:
        """
        Provide detailed update on difficulty adaptation.
        
        Args:
            L_cog: Current cognitive load
            P_t: Performance probability
            b_t: Current item difficulty
            b_next: Next item difficulty (from Eq.3)
            theta_t: Current user ability
            theta_next: Updated user ability (from Eq.4)
            
        Returns:
            Formatted teacher update message
        """
        difficulty_change = b_next - b_t
        ability_change = theta_next - theta_t
        
        if abs(difficulty_change) < 0.01:
            diff_msg = "maintaining"
        elif difficulty_change > 0:
            diff_msg = f"increasing by {difficulty_change:.3f}"
        else:
            diff_msg = f"decreasing by {abs(difficulty_change):.3f}"
            
        return (f"[Teacher] Difficulty {diff_msg} | "
                f"User Ability Δ={ability_change:+.3f} | "
                f"Performance={P_t:.2%}")
