"""
Companion Agent: Emotional Support

Receives difficulty state from Teacher agent and modulates emotional support strategies 
accordingly to maintain patient engagement within the optimal learning zone.

Responsibilities:
- Monitor emotional state from cognitive load patterns
- Employ valence-arousal mapping (Russell's circumplex model, 80% accuracy)
- Provide emotional validation and encouragement
- Detect stress or frustration signals

Collaboration:
- Receives difficulty state (b_t) propagated from Teacher agent
- Coordinates with Coach for comprehensive support delivery
"""

from typing import Dict, Any, Tuple


class CompanionAgent:
    """
    Companion Persona: Emotional Support and Empathy
    
    Primary responsibility: maintain positive emotional engagement
    through affective response and validation.
    """
    
    def __init__(self):
        """Initialize companion agent with empathetic parameters."""
        self.name = "Companion"
        self.role = "Emotional Support"
        self.emotion_history = []
        
    def respond(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate companion response based on emotional state.
        
        Emotional mapping from cognitive load:
        - Valence: positive when L_cog optimal, negative when overstimulated
        - Arousal: increases with cognitive load
        
        Args:
            state: Dictionary containing L_cog, valence, arousal
            
        Returns:
            Companion response with emotional validation message
        """
        L_cog = state['L_cog']
        valence = state['valence']
        arousal = state['arousal']
        
        self.emotion_history.append((valence, arousal))
        if len(self.emotion_history) > 10:
            self.emotion_history.pop(0)
        
        if valence >= 0.4:
            if arousal > 0.2:
                message = "You're really engaged! I can see your focus."
                mood = "energized"
            else:
                message = "You seem calm and comfortable."
                mood = "relaxed"
        elif valence >= 0.0:
            if arousal > 0.0:
                message = "This is challenging, but you're managing well."
                mood = "focused"
            else:
                message = "How are you feeling? Let me know if you need a break."
                mood = "neutral"
        else:
            if arousal > 0.3:
                message = "I notice this is difficult. Remember to breathe."
                mood = "stressed"
            else:
                message = "It's okay to take your time."
                mood = "withdrawn"
                
        return {
            'agent': self.name,
            'message': message,
            'mood': mood,
            'emotion': (valence, arousal)
        }
    
    def estimate_emotion(self, valence: float, arousal: float) -> str:
        """
        Classify emotional state from valence-arousal coordinates.
        
        Russell's circumplex model:
        - (+V, +A): excited, alert
        - (+V, -A): calm, relaxed
        - (-V, +A): stressed, anxious
        - (-V, -A): sad, withdrawn
        
        Args:
            valence: Emotional valence [-1, 1]
            arousal: Emotional arousal [-1, 1]
            
        Returns:
            Emotional state label
        """
        if valence >= 0:
            if arousal >= 0:
                return "engaged"
            else:
                return "content"
        else:
            if arousal >= 0:
                return "frustrated"
            else:
                return "fatigued"
    
    def update(self, valence: float, arousal: float, L_cog: float) -> str:
        """
        Provide emotional state update with validation.
        
        Args:
            valence: Emotional valence
            arousal: Emotional arousal  
            L_cog: Cognitive load index
            
        Returns:
            Formatted companion update message
        """
        emotion_label = self.estimate_emotion(valence, arousal)
        
        valence_str = f"{valence:+.2f}".replace('+', '+') if valence >= 0 else f"{valence:.2f}"
        arousal_str = f"{arousal:+.2f}".replace('+', '+') if arousal >= 0 else f"{arousal:.2f}"
        
        return (f"[Companion] Emotion={emotion_label} "
                f"(V={valence_str}, A={arousal_str}) | "
                f"L_cog={L_cog:.2f}")
