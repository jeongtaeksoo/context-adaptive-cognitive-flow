"""
Coach Agent: Motivational Reinforcement

Receives difficulty state from Teacher agent and modulates motivational strategies 
accordingly to maintain patient engagement within the optimal learning zone.

Responsibilities:
- Monitor behavioral trends across sessions
- Provide progress feedback and milestone recognition
- Deliver motivational encouragement based on performance

Collaboration:
- Receives difficulty state (b_t) and user ability (θ_t) from Teacher agent
- Coordinates with Companion for comprehensive support delivery
"""

from typing import Dict, Any, List


class CoachAgent:
    """
    Coach Persona: Behavioral Feedback and Motivation
    
    Primary responsibility: track progress trends and provide
    motivational reinforcement to sustain engagement.
    """
    
    def __init__(self):
        """Initialize coach agent with behavioral tracking."""
        self.name = "Coach"
        self.role = "Behavioral Feedback"
        self.performance_history: List[float] = []
        self.ability_history: List[float] = []
        
    def respond(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate coach response based on performance trends.
        
        Feedback strategy:
        - Track P_t and θ_t over time
        - Recognize improvements and milestones
        - Provide specific behavioral encouragement
        
        Args:
            state: Dictionary containing P_t, theta_t, t (time step)
            
        Returns:
            Coach response with motivational feedback
        """
        P_t = state['P_t']
        theta_t = state['theta_t']
        t = state['t']
        
        self.performance_history.append(P_t)
        self.ability_history.append(theta_t)
        
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
            self.ability_history.pop(0)
        
        if t == 0:
            message = "Welcome! Let's work together on this."
            feedback_type = "introduction"
        elif len(self.ability_history) >= 3:
            recent_trend = self.ability_history[-1] - self.ability_history[-3]
            if recent_trend > 0.1:
                message = "Great progress! Your skills are improving."
                feedback_type = "positive_trend"
            elif recent_trend < -0.1:
                message = "Everyone has ups and downs. Keep going!"
                feedback_type = "encouragement"
            else:
                message = "You're building consistency. Well done."
                feedback_type = "maintenance"
        else:
            if P_t >= 0.75:
                message = "Excellent work on this task!"
                feedback_type = "achievement"
            elif P_t >= 0.5:
                message = "You're on the right track."
                feedback_type = "support"
            else:
                message = "This is challenging, but you're trying."
                feedback_type = "encouragement"
                
        return {
            'agent': self.name,
            'message': message,
            'feedback_type': feedback_type,
            'sessions_tracked': len(self.performance_history)
        }
    
    def analyze_progress(self) -> str:
        """
        Analyze long-term progress trends.
        
        Returns:
            Progress analysis summary
        """
        if len(self.ability_history) < 2:
            return "Establishing baseline..."
        
        initial_ability = self.ability_history[0]
        current_ability = self.ability_history[-1]
        improvement = current_ability - initial_ability
        
        avg_performance = sum(self.performance_history) / len(self.performance_history)
        
        return (f"User Ability Δ={improvement:+.3f} | "
                f"Avg Performance={avg_performance:.2%}")
    
    def update(self, P_t: float, theta_t: float, t: int) -> str:
        """
        Provide behavioral progress update.
        
        Args:
            P_t: Performance probability
            theta_t: Current user ability estimate
            t: Time step
            
        Returns:
            Formatted coach update message
        """
        progress_summary = self.analyze_progress()
        
        return f"[Coach] Step {t} | {progress_summary}"
