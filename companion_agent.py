"""
Companion Agent - Specialized agent focusing on emotional support and social connection
Emphasizes valence-arousal emotional state and interpersonal engagement
"""

import numpy as np
from typing import Dict, List, Any, Optional
from utils import sigmoid

class CompanionAgent:
    """
    Companion agent specialized in emotional support and social connection.
    Focuses on valence-arousal emotional state and interpersonal well-being.
    """
    
    def __init__(self):
        """Initialize companion agent with emotional support parameters"""
        
        # Companion-specific priority weights
        self.priority_weights = {
            'emotional_state': 0.4,      # Weight on valence-arousal state
            'social_engagement': 0.3,    # Weight on social connection
            'voice_emotional_tone': 0.2, # Weight on voice emotion indicators
            'behavioral_wellness': 0.1   # Weight on overall behavioral wellness
        }
        
        # Valence-arousal quadrant definitions (Russell's Circumplex Model)
        self.emotion_quadrants = {
            'high_valence_high_arousal': {'valence': (0.3, 1.0), 'arousal': (0.3, 1.0)},    # Excited, happy
            'high_valence_low_arousal': {'valence': (0.3, 1.0), 'arousal': (-1.0, 0.3)},   # Calm, content
            'low_valence_high_arousal': {'valence': (-1.0, -0.3), 'arousal': (0.3, 1.0)},  # Anxious, stressed  
            'low_valence_low_arousal': {'valence': (-1.0, -0.3), 'arousal': (-1.0, 0.3)}   # Sad, depressed
        }
        
        # Response templates for different emotional states
        self.response_templates = {
            'high_valence_high_arousal': [
                "I can hear the excitement in your voice! Your energy is wonderful. What's bringing you such joy today?",
                "You sound absolutely delighted! I love sharing in your enthusiasm. Tell me more about what's making you feel so great!",
                "Your positive energy is infectious! It's wonderful to experience this happiness with you. How can we keep this momentum going?"
            ],
            'high_valence_low_arousal': [
                "You seem peaceful and content today. There's something beautiful about this calm satisfaction you're experiencing.",
                "I sense a lovely serenity in you right now. These moments of quiet contentment are so precious, aren't they?",
                "You sound wonderfully at ease. This gentle happiness you're feeling - it's one of life's real treasures."
            ],
            'low_valence_high_arousal': [
                "I can sense some tension or worry in your voice. I'm here with you - you don't have to face this alone.",
                "It sounds like you might be feeling anxious or stressed. That's completely understandable. Let's take this one step at a time together.",
                "I hear that you're going through something challenging right now. I want you to know that I'm here to support you through this."
            ],
            'low_valence_low_arousal': [
                "I can hear that you're having a difficult time. Sometimes life feels heavy, and that's okay. I'm here to sit with you through this.",
                "Your voice tells me you might be feeling down today. These feelings are valid, and you're not alone in experiencing them.",
                "I sense you're going through a tough patch. It's okay to feel this way - healing isn't always linear, and I'm here for you."
            ],
            'neutral_emotional_state': [
                "How are you feeling today? I'm here to listen and spend this time together with you.",
                "It's good to be here with you. What's on your mind? I'm ready to hear whatever you'd like to share.",
                "I'm glad we have this moment together. Is there anything you'd like to talk about or explore today?"
            ],
            'social_connection_low': [
                "I noticed it's been quiet on the social front lately. Sometimes we need our own space, and that's perfectly fine.",
                "It seems like you might be spending more time on your own recently. I'm here to provide some companionship if you'd like.",
                "Whether you're choosing solitude or it's just how things have been lately, remember that connection can happen in many ways - including right here with me."
            ],
            'social_connection_high': [
                "It sounds like you've been having some wonderful social interactions! Those connections really do feed the soul, don't they?",
                "I can hear that you've been engaging with others in meaningful ways. Social connection is such a vital part of well-being.",
                "Your social engagement seems to be flourishing - that's fantastic! These relationships you're nurturing are so valuable."
            ]
        }
        
        # Emotional validation phrases
        self.validation_phrases = [
            "Your feelings are completely valid and understandable.",
            "It's natural to experience these emotions - you're human.",
            "Thank you for sharing this with me - it takes courage to be vulnerable.",
            "I appreciate you letting me into your emotional world.",
            "Whatever you're feeling right now, it's okay to feel it."
        ]
        
        # Emotional support strategies
        self.support_strategies = {
            'active_listening': "I hear you saying...",
            'emotional_validation': "It makes complete sense that you would feel...",
            'gentle_encouragement': "You're doing better than you might realize...",
            'perspective_offering': "Sometimes it can help to remember...",
            'presence_offering': "I'm here with you in this moment..."
        }
        
    def analyze_emotional_state(self, feature_vector: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze emotional state from valence-arousal mapping and voice features
        
        Args:
            feature_vector: Input features from perception engine
            context: Context information including valence-arousal
            
        Returns:
            Emotional state analysis
        """
        # Extract valence-arousal from context if available
        valence_arousal = context.get('valence_arousal', {'valence': 0.0, 'arousal': 0.0})
        valence = valence_arousal['valence']
        arousal = valence_arousal['arousal']
        
        # Extract voice emotional features (indices 3-5: clarity, emotion, rate)
        voice_features = feature_vector[3:6]
        voice_clarity = voice_features[0]
        voice_emotion = voice_features[1]  # This was normalized to [0,1] from [-1,1]
        voice_rate = voice_features[2]
        
        # Convert voice emotion back to [-1,1] scale for consistency
        voice_emotion_bipolar = (voice_emotion * 2) - 1
        
        # Determine emotional quadrant
        emotion_quadrant = self._classify_emotion_quadrant(valence, arousal)
        
        # Estimate emotional intensity
        emotional_intensity = np.sqrt(valence**2 + arousal**2)  # Distance from neutral
        
        # Voice-emotion consistency check
        voice_valence_consistency = 1.0 - abs(valence - voice_emotion_bipolar)
        
        # Overall emotional confidence based on signal consistency
        emotional_confidence = (voice_clarity + voice_valence_consistency) / 2
        
        return {
            'valence': float(valence),
            'arousal': float(arousal),
            'emotion_quadrant': emotion_quadrant,
            'emotional_intensity': float(emotional_intensity),
            'voice_emotion': float(voice_emotion_bipolar),
            'voice_clarity': float(voice_clarity),
            'emotional_confidence': float(emotional_confidence),
            'voice_emotion_consistency': float(voice_valence_consistency)
        }
    
    def _classify_emotion_quadrant(self, valence: float, arousal: float) -> str:
        """
        Classify emotion into Russell's Circumplex Model quadrants
        
        Args:
            valence: Emotional valence [-1, 1]
            arousal: Emotional arousal [-1, 1]
            
        Returns:
            Emotion quadrant classification
        """
        if valence >= 0.3:
            if arousal >= 0.3:
                return 'high_valence_high_arousal'  # Excited, energetic
            else:
                return 'high_valence_low_arousal'   # Calm, content
        elif valence <= -0.3:
            if arousal >= 0.3:
                return 'low_valence_high_arousal'   # Anxious, stressed
            else:
                return 'low_valence_low_arousal'    # Sad, tired
        else:
            return 'neutral_emotional_state'        # Neutral emotions
    
    def analyze_social_engagement(self, feature_vector: np.ndarray, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze social engagement patterns
        
        Args:
            feature_vector: Input features
            context: Context information
            
        Returns:
            Social engagement analysis
        """
        # Extract social engagement feature (index 1)
        social_engagement_level = feature_vector[1]
        
        # Extract behavioral activity (index 0) - social activity correlation
        activity_level = feature_vector[0]
        
        # Voice clarity can indicate social interaction quality
        voice_clarity = feature_vector[3]
        
        # Estimate social wellness
        social_wellness = (social_engagement_level * 0.5 + 
                         activity_level * 0.3 + 
                         voice_clarity * 0.2)
        
        # Social connection trend (simplified - would need historical data)
        # For now, use current engagement as proxy
        social_trend = 'stable'
        if social_engagement_level > 0.7:
            social_trend = 'increasing'
        elif social_engagement_level < 0.3:
            social_trend = 'decreasing'
            
        return {
            'social_engagement_level': float(social_engagement_level),
            'social_wellness': float(social_wellness),
            'social_trend': social_trend,
            'isolation_risk': float(1.0 - social_engagement_level)
        }
    
    def compute_companion_priority(self, emotional_analysis: Dict[str, Any], 
                                 social_analysis: Dict[str, float], 
                                 context: Dict[str, Any]) -> float:
        """
        Compute priority score for companion agent intervention
        
        Args:
            emotional_analysis: Emotional state analysis
            social_analysis: Social engagement analysis
            context: Current context
            
        Returns:
            Priority score [0, 1]
        """
        # High priority for emotional distress (negative valence, high intensity)
        emotional_distress = 0.0
        if emotional_analysis['valence'] < -0.2:  # Negative emotions
            emotional_distress = abs(emotional_analysis['valence']) * emotional_analysis['emotional_intensity']
            
        emotional_priority = self.priority_weights['emotional_state'] * (
            emotional_distress + 
            (1.0 - emotional_analysis['emotional_confidence']) * 0.3  # Unclear emotions also need attention
        )
        
        # High priority for social isolation
        social_priority = self.priority_weights['social_engagement'] * (
            social_analysis['isolation_risk'] * 
            (1.5 if social_analysis['social_trend'] == 'decreasing' else 1.0)
        )
        
        # Voice emotion indicators
        voice_priority = self.priority_weights['voice_emotional_tone'] * (
            1.0 - emotional_analysis['voice_emotion_consistency']
        )
        
        # Behavioral wellness concerns
        behavioral_features = [emotional_analysis['voice_clarity']]  # Could expand this
        behavioral_priority = self.priority_weights['behavioral_wellness'] * (
            1.0 - np.mean(behavioral_features)
        )
        
        total_priority = emotional_priority + social_priority + voice_priority + behavioral_priority
        
        # Boost priority for extreme emotional states (very positive or negative)
        if emotional_analysis['emotional_intensity'] > 0.7:
            total_priority *= 1.2
            
        return max(0.0, min(1.0, total_priority))
    
    def select_response_template(self, emotional_analysis: Dict[str, Any], 
                               social_analysis: Dict[str, float]) -> str:
        """
        Select appropriate response template based on emotional and social analysis
        
        Args:
            emotional_analysis: Emotional state analysis
            social_analysis: Social engagement analysis
            
        Returns:
            Selected response template
        """
        # Priority: social isolation, then emotional quadrant
        if social_analysis['isolation_risk'] > 0.7:
            if social_analysis['social_trend'] == 'decreasing':
                template_category = 'social_connection_low'
            else:
                template_category = emotional_analysis['emotion_quadrant']
        elif social_analysis['social_engagement_level'] > 0.8:
            template_category = 'social_connection_high'
        else:
            # Use emotional quadrant for template selection
            template_category = emotional_analysis['emotion_quadrant']
            
        # Select from appropriate template category
        templates = self.response_templates.get(template_category, 
                                              self.response_templates['neutral_emotional_state'])
        selected_template = np.random.choice(templates)
        
        # Add emotional validation for negative states
        if emotional_analysis['valence'] < -0.2:
            validation = np.random.choice(self.validation_phrases)
            selected_template += f" {validation}"
            
        return selected_template
    
    def generate_trigger_context(self, emotional_analysis: Dict[str, Any], 
                               social_analysis: Dict[str, float]) -> str:
        """
        Generate context description for why companion agent was triggered
        
        Args:
            emotional_analysis: Emotional analysis
            social_analysis: Social analysis
            
        Returns:
            Human-readable trigger context
        """
        valence = emotional_analysis['valence']
        arousal = emotional_analysis['arousal']
        social_level = social_analysis['social_engagement_level']
        
        # Primary trigger identification
        if social_analysis['isolation_risk'] > 0.7:
            return f"Social isolation detected (engagement: {social_level:.1%})"
        elif valence < -0.5:
            return f"Negative emotional state (valence: {valence:+.2f}, arousal: {arousal:+.2f})"
        elif valence > 0.5 and arousal > 0.3:
            return f"High positive emotion sharing opportunity (valence: {valence:+.2f})"
        elif emotional_analysis['emotional_intensity'] > 0.6:
            return f"High emotional intensity detected ({emotional_analysis['emotion_quadrant']})"
        else:
            return f"General emotional support ({emotional_analysis['emotion_quadrant']})"
    
    def process_input(self, feature_vector: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for companion agent
        
        Args:
            feature_vector: Input feature vector from perception engine
            context: Context dictionary from context hub
            
        Returns:
            Companion agent response package
        """
        # Analyze emotional state
        emotional_analysis = self.analyze_emotional_state(feature_vector, context)
        
        # Analyze social engagement
        social_analysis = self.analyze_social_engagement(feature_vector, context)
        
        # Compute priority score
        priority_score = self.compute_companion_priority(emotional_analysis, social_analysis, context)
        
        # Select appropriate response
        response_text = self.select_response_template(emotional_analysis, social_analysis)
        
        # Generate trigger context
        trigger_context = self.generate_trigger_context(emotional_analysis, social_analysis)
        
        # Compile response package
        return {
            'response': response_text,
            'metadata': {
                'agent': 'Companion',
                'priority_score': priority_score,
                'emotional_analysis': emotional_analysis,
                'social_analysis': social_analysis,
                'support_focus': emotional_analysis['emotion_quadrant']
            },
            'trigger': trigger_context
        }
