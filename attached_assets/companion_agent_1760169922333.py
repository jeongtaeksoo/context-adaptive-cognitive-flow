"""
CompanionAgent - Specialized agent for emotional support and empathetic interaction
Maps valence-arousal space from voice/emotion and provides context-appropriate responses
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from utils import sigmoid

class CompanionAgent:
    """
    Companion persona agent that focuses on emotional well-being through
    empathetic responses based on valence-arousal emotional state.
    """
    
    def __init__(self):
        """Initialize CompanionAgent with emotional mapping parameters"""
        
        # Valence-arousal quadrants and their interpretations
        self.emotional_quadrants = {
            'happy': {'valence': (0, 1), 'arousal': (0, 1)},      # High valence, high arousal
            'calm': {'valence': (0, 1), 'arousal': (-1, 0)},      # High valence, low arousal
            'anxious': {'valence': (-1, 0), 'arousal': (0, 1)},   # Low valence, high arousal
            'sad': {'valence': (-1, 0), 'arousal': (-1, 0)}       # Low valence, low arousal
        }
        
        # Empathetic response templates by emotional state
        self.response_templates = {
            'happy': [
                "It's wonderful to see you in such good spirits! Let's enjoy this moment together.",
                "Your positive energy is infectious! What would you like to share today?",
                "I'm so glad you're feeling great! Let's make the most of this wonderful mood."
            ],
            'calm': [
                "You seem peaceful and content. This is a perfect time to reflect or relax.",
                "I sense a nice calm energy from you. How can I support your tranquility today?",
                "It's lovely to share this serene moment with you. What's on your mind?"
            ],
            'anxious': [
                "I notice you might be feeling a bit on edge. I'm here for you. Let's take this one step at a time.",
                "It seems like there's some tension. Would you like to talk about what's bothering you?",
                "I'm here to support you through this. Take a deep breath, and let's work through this together."
            ],
            'sad': [
                "I can sense you're feeling down. Please know that I'm here to listen and support you.",
                "It's okay to feel this way. I'm here with you, and we can get through this together.",
                "Your feelings are valid. Let's take things gently today. How can I help?"
            ]
        }
        
        # Voice feature weights for emotional inference
        self.voice_emotion_weights = {
            'pitch_variance': 0.3,
            'speech_rate': 0.25,
            'energy': 0.25,
            'tone_quality': 0.2
        }
        
    def map_valence_arousal(self, voice_features: np.ndarray, behavioral_features: np.ndarray) -> Tuple[float, float]:
        """
        Map input features to valence-arousal emotional space
        
        Args:
            voice_features: Voice/acoustic features (pitch, energy, etc.)
            behavioral_features: Behavioral patterns
            
        Returns:
            Tuple of (valence, arousal) in [-1, 1] range
        """
        # Extract voice emotional indicators
        if len(voice_features) >= 4:
            pitch_var = voice_features[0]
            speech_rate = voice_features[1]
            energy = voice_features[2]
            tone_quality = voice_features[3]
        else:
            # Fallback to neutral
            pitch_var = speech_rate = energy = tone_quality = 0.5
            
        # Compute arousal (high energy/variance = high arousal)
        arousal_score = (
            self.voice_emotion_weights['pitch_variance'] * pitch_var +
            self.voice_emotion_weights['speech_rate'] * speech_rate +
            self.voice_emotion_weights['energy'] * energy
        )
        arousal = (arousal_score - 0.5) * 2  # Scale to [-1, 1]
        
        # Compute valence (positive tone/low tension = positive valence)
        valence_score = (
            self.voice_emotion_weights['tone_quality'] * tone_quality +
            0.3 * (1 - abs(pitch_var - 0.5) * 2) +  # Moderate variance is positive
            0.2 * np.mean(behavioral_features[:3]) if len(behavioral_features) >= 3 else 0.5
        )
        valence = (valence_score - 0.5) * 2  # Scale to [-1, 1]
        
        # Clip to valid range
        valence = np.clip(valence, -1.0, 1.0)
        arousal = np.clip(arousal, -1.0, 1.0)
        
        return float(valence), float(arousal)
        
    def identify_emotional_state(self, valence: float, arousal: float) -> str:
        """
        Identify emotional quadrant from valence-arousal coordinates
        
        Args:
            valence: Valence value in [-1, 1]
            arousal: Arousal value in [-1, 1]
            
        Returns:
            Emotional state label
        """
        if valence >= 0 and arousal >= 0:
            return 'happy'
        elif valence >= 0 and arousal < 0:
            return 'calm'
        elif valence < 0 and arousal >= 0:
            return 'anxious'
        else:  # valence < 0 and arousal < 0
            return 'sad'
            
    def generate_empathetic_response(self, emotional_state: str, valence: float, arousal: float, context: Dict[str, Any]) -> str:
        """
        Generate empathetic response based on emotional state
        
        Args:
            emotional_state: Identified emotional quadrant
            valence: Valence value
            arousal: Arousal value
            context: Additional context
            
        Returns:
            Empathetic response text
        """
        # Select base response from templates
        templates = self.response_templates.get(emotional_state, self.response_templates['calm'])
        base_response = np.random.choice(templates)
        
        # Add context-specific support if available
        time_of_day = context.get('time_of_day', 'day')
        
        if emotional_state in ['anxious', 'sad']:
            # Offer additional support for negative emotions
            if time_of_day == 'morning':
                base_response += " Starting the day can be challenging, but I believe in you."
            elif time_of_day == 'evening':
                base_response += " You've made it through the day. That's something to be proud of."
        elif emotional_state in ['happy', 'calm']:
            # Reinforce positive emotions
            if time_of_day == 'morning':
                base_response += " What a great way to start the day!"
            elif time_of_day == 'evening':
                base_response += " It's wonderful to end the day on such a positive note."
                
        return base_response
        
    def calculate_priority(self, valence: float, arousal: float) -> float:
        """
        Calculate priority score for companion intervention
        Higher priority for negative emotions that need support
        
        Args:
            valence: Valence value
            arousal: Arousal value
            
        Returns:
            Priority score [0, 1]
        """
        # Negative emotions get higher priority
        emotional_distress = max(0, -valence)  # 0 to 1, higher when valence is negative
        
        # High arousal (anxiety) increases priority
        arousal_factor = max(0, arousal) * 0.5  # 0 to 0.5
        
        # Combined priority
        priority = min(emotional_distress + arousal_factor, 1.0)
        
        return float(priority)
        
    def process_input(self, xt: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main processing method for CompanionAgent
        
        Args:
            xt: Input feature vector from perception engine
            context: Optional context dictionary
            
        Returns:
            Dict containing response, metadata, and contextual triggers
        """
        if context is None:
            context = {}
            
        # Extract voice and behavioral features
        behavioral_features = xt[:7] if len(xt) > 7 else np.array([0.5] * 7)
        voice_features = xt[7:14] if len(xt) > 14 else np.array([0.5] * 7)
        
        # Map to valence-arousal space
        valence, arousal = self.map_valence_arousal(voice_features, behavioral_features)
        
        # Identify emotional state
        emotional_state = self.identify_emotional_state(valence, arousal)
        
        # Generate empathetic response
        response_text = self.generate_empathetic_response(emotional_state, valence, arousal, context)
        
        # Calculate priority
        priority_score = self.calculate_priority(valence, arousal)
        
        # Compile metadata
        metadata = {
            'agent': 'Companion',
            'emotional_state': emotional_state,
            'valence': float(valence),
            'arousal': float(arousal),
            'priority_score': priority_score
        }
        
        # Contextual trigger information
        trigger = f"Valence={valence:.2f}, Arousal={arousal:.2f}, State={emotional_state}"
        
        return {
            'response': response_text,
            'metadata': metadata,
            'trigger': trigger
        }
