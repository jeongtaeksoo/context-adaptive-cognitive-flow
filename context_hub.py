"""
Context Hub - Stage II: Context Recognition and Persona Adaptation
Implements Equations 4-6 for attention weighting and valence-arousal mapping
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from utils import sigmoid, softmax, exponential_moving_average

class ContextHub:
    """
    Context recognition system that applies persona-specific attention weighting
    and maps features to valence-arousal emotional space.
    """
    
    def __init__(self, use_ml_embeddings: bool = False):
        """Initialize context hub with persona definitions and ML options"""
        
        self.use_ml_embeddings = use_ml_embeddings
        
        # Initialize ML encoder if available
        self.sentence_encoder = None
        if use_ml_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("Warning: sentence-transformers not available, using rule-based encoding")
                self.use_ml_embeddings = False
        
        # Persona-specific attention weights (Equation 4)
        self.persona_attention_weights = {
            'teacher': {
                'behavioral_activity': 0.2, 'behavioral_social': 0.2, 'behavioral_sleep': 0.1,
                'voice_clarity': 0.1, 'voice_emotion': 0.1, 'voice_rate': 0.1,
                'performance_speed': 0.5, 'performance_accuracy': 0.5, 'performance_load': 0.4,
                'temporal_time': 0.2, 'temporal_day': 0.1
            },
            'companion': {
                'behavioral_activity': 0.2, 'behavioral_social': 0.4, 'behavioral_sleep': 0.3,
                'voice_clarity': 0.3, 'voice_emotion': 0.5, 'voice_rate': 0.2,
                'performance_speed': 0.2, 'performance_accuracy': 0.2, 'performance_load': 0.3,
                'temporal_time': 0.2, 'temporal_day': 0.2
            },
            'coach': {
                'behavioral_activity': 0.5, 'behavioral_social': 0.4, 'behavioral_sleep': 0.4,
                'voice_clarity': 0.2, 'voice_emotion': 0.3, 'voice_rate': 0.2,
                'performance_speed': 0.3, 'performance_accuracy': 0.3, 'performance_load': 0.2,
                'temporal_time': 0.3, 'temporal_day': 0.4
            }
        }
        
        # Valence-arousal mapping parameters (Equation 6)
        self.va_mapping = {
            'valence_weights': np.array([0.1, 0.3, 0.2, 0.0, 0.4, 0.1, 0.2, 0.3, -0.1, 0.1, 0.0]),
            'arousal_weights': np.array([0.3, 0.2, -0.2, 0.2, 0.1, 0.3, 0.4, 0.2, 0.3, 0.2, 0.1])
        }
        
        # Context state tracking
        self.context_history = []
        self.current_context = None
        
    def apply_attention_mechanism(self, feature_vector: np.ndarray, persona: str) -> np.ndarray:
        """
        Apply persona-specific attention weights to feature vector (Equation 5)
        
        Args:
            feature_vector: Input feature vector from perception engine
            persona: Selected persona ('teacher', 'companion', 'coach')
            
        Returns:
            Attention-weighted feature vector
        """
        if persona not in self.persona_attention_weights:
            persona = 'teacher'  # Default fallback
            
        # Get attention weights for persona
        attention_weights = self.persona_attention_weights[persona]
        
        # Convert to numpy array in correct order
        feature_names = [
            'behavioral_activity', 'behavioral_social', 'behavioral_sleep',
            'voice_clarity', 'voice_emotion', 'voice_rate',
            'performance_speed', 'performance_accuracy', 'performance_load',
            'temporal_time', 'temporal_day'
        ]
        
        weights_array = np.array([attention_weights[name] for name in feature_names])
        
        # Apply attention mechanism
        attended_features = feature_vector * weights_array
        
        # Normalize to maintain magnitude
        attended_features = attended_features / (np.sum(weights_array) + 1e-8)
        
        return attended_features
    
    def map_to_valence_arousal(self, attended_features: np.ndarray) -> Dict[str, float]:
        """
        Map attended features to valence-arousal space (Equation 6)
        
        Args:
            attended_features: Attention-weighted feature vector
            
        Returns:
            Dictionary containing valence and arousal values [-1, 1]
        """
        # Compute valence (emotional positivity)
        valence_raw = np.dot(attended_features, self.va_mapping['valence_weights'])
        valence = np.tanh(valence_raw)  # Bound to [-1, 1]
        
        # Compute arousal (emotional intensity)  
        arousal_raw = np.dot(attended_features, self.va_mapping['arousal_weights'])
        arousal = np.tanh(arousal_raw)  # Bound to [-1, 1]
        
        return {'valence': float(valence), 'arousal': float(arousal)}
    
    def estimate_cognitive_state(self, attended_features: np.ndarray) -> Dict[str, float]:
        """
        Estimate cognitive state dimensions from features
        
        Args:
            attended_features: Attention-weighted feature vector
            
        Returns:
            Dictionary containing cognitive state estimates
        """
        # Performance-based features (indices 6-8)
        performance_features = attended_features[6:9]
        
        # Cognitive load (inverse of performance quality)
        cognitive_load = 1.0 - np.mean(performance_features)
        cognitive_load = max(0.0, min(1.0, cognitive_load))
        
        # Attention level (based on response time and accuracy)
        attention_level = (performance_features[0] + performance_features[1]) / 2
        
        # Engagement level (combination of multiple factors)
        behavioral_engagement = np.mean(attended_features[0:2])  # Activity + social
        voice_engagement = attended_features[3]  # Voice clarity
        engagement_level = (behavioral_engagement + voice_engagement) / 2
        
        return {
            'cognitive_load': float(cognitive_load),
            'attention_level': float(attention_level), 
            'engagement_level': float(engagement_level),
            'cognitive_state': float(1.0 - cognitive_load)  # Overall cognitive state
        }
    
    def encode_textual_context(self, text_data: Optional[str] = None) -> np.ndarray:
        """
        Encode textual context using ML embeddings or rule-based approach
        
        Args:
            text_data: Optional text input (e.g., transcribed speech)
            
        Returns:
            Context encoding vector
        """
        if text_data and self.use_ml_embeddings and self.sentence_encoder:
            # Use sentence transformer for semantic encoding
            embedding = self.sentence_encoder.encode([text_data])
            return embedding[0][:64]  # Truncate to manageable size
        else:
            # Rule-based encoding for keywords
            context_keywords = {
                'positive': ['good', 'great', 'happy', 'well', 'fine', 'excellent'],
                'negative': ['bad', 'tired', 'difficult', 'hard', 'confused', 'stressed'],
                'cognitive': ['think', 'remember', 'focus', 'concentrate', 'understand'],
                'social': ['family', 'friends', 'talk', 'visit', 'together', 'alone']
            }
            
            if not text_data:
                return np.zeros(16)  # Default encoding
                
            text_lower = text_data.lower()
            encoding = np.zeros(16)
            
            # Encode presence of different keyword categories
            for i, (category, keywords) in enumerate(context_keywords.items()):
                for keyword in keywords:
                    if keyword in text_lower:
                        encoding[i*4:(i+1)*4] += 0.25  # Distribute across 4 dimensions per category
                        
            return encoding
    
    def compute_context_confidence(self, context_dimensions: Dict[str, float]) -> float:
        """
        Compute confidence in current context estimation
        
        Args:
            context_dimensions: Dictionary of context dimension values
            
        Returns:
            Context confidence score [0, 1]
        """
        # Base confidence on consistency of estimates
        values = list(context_dimensions.values())
        
        # Higher confidence for moderate values (not extreme)
        extremeness = np.mean([abs(v - 0.5) for v in values])
        consistency_bonus = 1.0 - extremeness
        
        # Higher confidence for non-zero variance (informative)
        if len(values) > 1:
            variance = np.var(values)
            information_bonus = min(0.3, variance * 2)
        else:
            information_bonus = 0.1
            
        confidence = 0.5 + 0.3 * consistency_bonus + information_bonus
        return min(1.0, max(0.0, confidence))
    
    def process_context(self, feature_vector: np.ndarray, persona: str, 
                       additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main context processing pipeline
        
        Args:
            feature_vector: Feature vector from perception engine
            persona: Selected persona for attention weighting
            additional_data: Optional additional context data
            
        Returns:
            Complete context state dictionary
        """
        # Apply persona-specific attention
        attended_features = self.apply_attention_mechanism(feature_vector, persona)
        
        # Map to valence-arousal space
        valence_arousal = self.map_to_valence_arousal(attended_features)
        
        # Estimate cognitive dimensions
        cognitive_dimensions = self.estimate_cognitive_state(attended_features)
        
        # Encode textual context if available
        text_data = None
        if additional_data:
            text_data = additional_data.get('transcribed_text') or additional_data.get('text_input')
        textual_encoding = self.encode_textual_context(text_data)
        
        # Compute attention weights used
        attention_weights = {name: weight for name, weight in 
                           zip(['behavioral_activity', 'behavioral_social', 'behavioral_sleep',
                               'voice_clarity', 'voice_emotion', 'voice_rate', 
                               'performance_speed', 'performance_accuracy', 'performance_load',
                               'temporal_time', 'temporal_day'],
                               self.persona_attention_weights[persona].values())}
        
        # Assemble complete context state
        context_state = {
            'persona': persona,
            'attended_features': attended_features,
            'valence_arousal': valence_arousal,
            'context_dimensions': cognitive_dimensions,
            'textual_encoding': textual_encoding,
            'attention_weights': attention_weights,
            'context_confidence': self.compute_context_confidence(cognitive_dimensions)
        }
        
        # Update context history
        self.context_history.append(context_state)
        if len(self.context_history) > 10:  # Keep last 10 contexts
            self.context_history.pop(0)
            
        self.current_context = context_state
        
        return context_state
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get summary of current context state for visualization
        
        Returns:
            Simplified context summary
        """
        if not self.current_context:
            return {'status': 'no_context'}
            
        return {
            'persona': self.current_context['persona'],
            'valence': self.current_context['valence_arousal']['valence'],
            'arousal': self.current_context['valence_arousal']['arousal'],
            'cognitive_load': self.current_context['context_dimensions']['cognitive_load'],
            'engagement_level': self.current_context['context_dimensions']['engagement_level'],
            'confidence': self.current_context['context_confidence']
        }
