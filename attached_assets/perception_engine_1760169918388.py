"""
Perception Engine - Stage I: Multimodal Data Sensing
Implements Equations 2, 3, 13-15 for processing behavioral, voice, performance, and temporal inputs
"""

import numpy as np
from typing import Dict, List, Any
import json

class PerceptionEngine:
    """
    Multimodal data sensing engine that processes various input modalities
    and creates unified feature vectors for context recognition.
    """
    
    def __init__(self):
        """Initialize the perception engine with feature extractors"""
        self.behavioral_weights = {
            'activity_level': 0.3,
            'social_engagement': 0.4,
            'sleep_quality': 0.3
        }
        
        self.voice_weights = {
            'speech_clarity': 0.4,
            'emotional_tone': 0.3,
            'speaking_rate': 0.3
        }
        
        self.performance_weights = {
            'response_time': -0.4,  # Negative because slower is worse
            'accuracy': 0.6
        }
        
        self.temporal_mappings = {
            'morning': [1.0, 0.0, 0.0, 0.0],
            'afternoon': [0.0, 1.0, 0.0, 0.0], 
            'evening': [0.0, 0.0, 1.0, 0.0],
            'night': [0.0, 0.0, 0.0, 1.0]
        }
        
        self.day_mappings = {
            'monday': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'tuesday': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'wednesday': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            'thursday': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            'friday': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            'saturday': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            'sunday': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        }
    
    def process_behavioral_input(self, behavioral_data: Dict[str, float]) -> List[float]:
        """
        Process behavioral input patterns
        
        Research Paper Reference: Equation 2
        x_behav = f(activity_level, social_engagement, sleep_quality, ...)
        
        Implements weighted combination of behavioral features to create
        unified behavioral representation for context recognition.
        
        Args:
            behavioral_data: Dictionary containing behavioral metrics
            
        Returns:
            List of processed behavioral features
        """
        features = []
        
        # Extract core behavioral features
        activity_level = behavioral_data.get('activity_level', 0.5)
        social_engagement = behavioral_data.get('social_engagement', 0.5)
        sleep_quality = behavioral_data.get('sleep_quality', 0.5)
        
        # Apply weighted combination (simulating learned feature extraction)
        behavioral_score = (
            activity_level * self.behavioral_weights['activity_level'] +
            social_engagement * self.behavioral_weights['social_engagement'] +
            sleep_quality * self.behavioral_weights['sleep_quality']
        )
        
        # Create feature vector with raw values and computed score
        features.extend([activity_level, social_engagement, sleep_quality, behavioral_score])
        
        # Add interaction features (Equation 3 - multimodal interactions)
        interaction_features = [
            activity_level * social_engagement,  # Activity-social interaction
            np.tanh(behavioral_score),  # Non-linear activation
            1.0 if behavioral_score > 0.6 else 0.0  # Threshold feature
        ]
        
        features.extend(interaction_features)
        return features
    
    def process_voice_input(self, voice_data: Dict[str, float]) -> List[float]:
        """
        Process voice and speech features
        
        Research Paper Reference: Equation 13
        x_voice = STT_features + prosodic_analysis + emotional_markers
        
        Extracts speech-to-text features, prosodic patterns, and emotional markers
        from voice input for emotional state analysis in Stage II.
        
        Args:
            voice_data: Dictionary containing voice analysis results
            
        Returns:
            List of processed voice features
        """
        features = []
        
        # Extract voice characteristics
        speech_clarity = voice_data.get('speech_clarity', 0.7)
        emotional_tone = voice_data.get('emotional_tone', 0.0)  # -1 to 1 scale
        speaking_rate = voice_data.get('speaking_rate', 0.5)
        
        # Compute weighted voice score
        voice_score = (
            speech_clarity * self.voice_weights['speech_clarity'] +
            abs(emotional_tone) * self.voice_weights['emotional_tone'] +
            speaking_rate * self.voice_weights['speaking_rate']
        )
        
        # Create voice feature vector
        features.extend([speech_clarity, emotional_tone, speaking_rate, voice_score])
        
        # Prosodic and emotional analysis features
        prosodic_features = [
            np.exp(-abs(emotional_tone)),  # Emotional stability
            1.0 if speech_clarity > 0.6 else 0.0,  # Clear speech indicator
            np.sin(speaking_rate * np.pi)  # Rhythmic pattern feature
        ]
        
        features.extend(prosodic_features)
        return features
    
    def process_performance_input(self, performance_data: Dict[str, float]) -> List[float]:
        """
        Process performance and cognitive metrics (Equation 14)
        x_perf = cognitive_load + response_accuracy + task_completion_time
        
        Args:
            performance_data: Dictionary containing performance metrics
            
        Returns:
            List of processed performance features
        """
        features = []
        
        # Extract performance metrics
        response_time = performance_data.get('response_time', 3.0)
        accuracy = performance_data.get('accuracy', 0.8)
        cognitive_load = performance_data.get('cognitive_load', 0.5)
        
        # Normalize response time (assuming 1-10 second range)
        normalized_rt = max(0.0, min(1.0, (10.0 - response_time) / 9.0))
        
        # Compute performance score
        performance_score = (
            normalized_rt * abs(self.performance_weights['response_time']) +
            accuracy * self.performance_weights['accuracy']
        )
        
        # Create performance feature vector
        features.extend([normalized_rt, accuracy, cognitive_load, performance_score])
        
        # Cognitive state indicators
        cognitive_features = [
            1.0 if accuracy > 0.8 else 0.0,  # High performance indicator
            np.exp(-response_time / 3.0),  # Speed efficiency
            cognitive_load * accuracy  # Load-adjusted performance
        ]
        
        features.extend(cognitive_features)
        return features
    
    def process_temporal_input(self, temporal_data: Dict[str, str]) -> List[float]:
        """
        Process temporal context features (Equation 15)
        x_temp = time_of_day + day_of_week + seasonal_factors + circadian_alignment
        
        Args:
            temporal_data: Dictionary containing temporal context
            
        Returns:
            List of processed temporal features
        """
        features = []
        
        # Time of day encoding (one-hot)
        time_of_day = temporal_data.get('time_of_day', 'afternoon').lower()
        time_features = self.temporal_mappings.get(time_of_day, [0.0, 1.0, 0.0, 0.0])
        features.extend(time_features)
        
        # Day of week encoding (one-hot)
        day_of_week = temporal_data.get('day_of_week', 'tuesday').lower()
        day_features = self.day_mappings.get(day_of_week, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        features.extend(day_features)
        
        # Circadian and contextual features
        circadian_features = [
            1.0 if time_of_day in ['morning', 'afternoon'] else 0.0,  # Daytime indicator
            1.0 if day_of_week in ['saturday', 'sunday'] else 0.0,  # Weekend indicator
            np.sin(2 * np.pi * hash(time_of_day) / (2**32)),  # Cyclic time representation
        ]
        
        features.extend(circadian_features)
        return features
    
    def process_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Main processing function that integrates all modalities
        Implements the complete multimodal sensing pipeline
        
        Args:
            input_data: Dictionary containing all input modalities
            
        Returns:
            Unified feature vector xt combining all modalities
        """
        # Process each modality
        behavioral_features = self.process_behavioral_input(
            input_data.get('behavioral', {})
        )
        
        voice_features = self.process_voice_input(
            input_data.get('voice', {})
        )
        
        performance_features = self.process_performance_input(
            input_data.get('performance', {})
        )
        
        temporal_features = self.process_temporal_input(
            input_data.get('temporal', {})
        )
        
        # Combine all features into unified vector xt
        # This represents Equation 2: xt = [x_behav, x_voice, x_perf, x_temp]
        unified_features = np.concatenate([
            behavioral_features,
            voice_features, 
            performance_features,
            temporal_features
        ])
        
        # Apply normalization and feature scaling
        normalized_features = self._normalize_features(unified_features)
        
        return normalized_features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize feature vector to prevent any single modality from dominating
        
        Args:
            features: Raw feature vector
            
        Returns:
            Normalized feature vector
        """
        # Apply min-max normalization
        features = np.array(features)
        
        # Prevent division by zero
        feat_min = np.min(features)
        feat_max = np.max(features) 
        feat_range = feat_max - feat_min
        
        if feat_range > 0:
            normalized = (features - feat_min) / feat_range
        else:
            normalized = features
            
        return normalized
    
    def get_feature_dimension(self) -> int:
        """
        Return the expected dimension of the output feature vector
        
        Returns:
            Integer representing feature vector dimension
        """
        # Behavioral: 7, Voice: 7, Performance: 7, Temporal: 14
        return 35
    
    def get_feature_names(self) -> List[str]:
        """
        Return descriptive names for each feature dimension
        
        Returns:
            List of feature names for interpretability
        """
        behavioral_names = [
            'activity_level', 'social_engagement', 'sleep_quality', 'behavioral_score',
            'activity_social_interaction', 'behavioral_activation', 'behavioral_threshold'
        ]
        
        voice_names = [
            'speech_clarity', 'emotional_tone', 'speaking_rate', 'voice_score',
            'emotional_stability', 'clear_speech_indicator', 'rhythmic_pattern'
        ]
        
        performance_names = [
            'response_speed', 'accuracy', 'cognitive_load', 'performance_score',
            'high_performance_indicator', 'speed_efficiency', 'load_adjusted_performance'
        ]
        
        temporal_names = [
            'morning', 'afternoon', 'evening', 'night',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'daytime_indicator', 'weekend_indicator', 'cyclic_time'
        ]
        
        return behavioral_names + voice_names + performance_names + temporal_names
