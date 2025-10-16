"""
Perception Engine - Stage I: Multimodal Data Acquisition and Processing
Implements Equations 2-3 for feature extraction and normalization
"""

import numpy as np
from typing import Dict, List, Any, Optional
from utils import sigmoid, normalize_features

class PerceptionEngine:
    """
    Multimodal perception engine that processes behavioral, voice, performance,
    and temporal inputs into normalized feature vectors.
    """
    
    def __init__(self):
        """Initialize perception engine with processing parameters"""
        
        # Feature extraction parameters
        self.feature_weights = {
            'behavioral': {'activity_level': 1.0, 'social_engagement': 1.0, 'sleep_quality': 0.8},
            'voice': {'speech_clarity': 1.0, 'emotional_tone': 1.0, 'speaking_rate': 0.7},
            'performance': {'response_time': 1.0, 'accuracy': 1.0, 'cognitive_load': 0.9},
            'temporal': {'time_of_day': 0.6, 'day_of_week': 0.4}
        }
        
        # Normalization ranges for different modalities
        self.normalization_ranges = {
            'response_time': (0.5, 10.0),  # seconds
            'speaking_rate': (80, 200),    # words per minute
            'cognitive_load': (0.0, 1.0)
        }
        
    def extract_behavioral_features(self, behavioral_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract and normalize behavioral features
        
        Args:
            behavioral_data: Dictionary containing behavioral metrics
            
        Returns:
            Normalized behavioral feature vector
        """
        features = []
        
        # Activity level (already normalized 0-1)
        activity = behavioral_data.get('activity_level', 0.5)
        features.append(activity)
        
        # Social engagement (already normalized 0-1)
        social = behavioral_data.get('social_engagement', 0.5)
        features.append(social)
        
        # Sleep quality (convert to 0-1 if needed)
        sleep = behavioral_data.get('sleep_quality', 0.7)
        if isinstance(sleep, str):
            sleep_mapping = {'poor': 0.2, 'fair': 0.5, 'good': 0.8, 'excellent': 1.0}
            sleep = sleep_mapping.get(sleep.lower(), 0.5)
        features.append(sleep)
        
        return np.array(features)
    
    def extract_voice_features(self, voice_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract and normalize voice features
        
        Args:
            voice_data: Dictionary containing voice metrics
            
        Returns:
            Normalized voice feature vector
        """
        features = []
        
        # Speech clarity (already normalized 0-1)
        clarity = voice_data.get('speech_clarity', 0.7)
        features.append(clarity)
        
        # Emotional tone (normalized -1 to 1, convert to 0-1)
        tone = voice_data.get('emotional_tone', 0.0)
        tone_normalized = (tone + 1) / 2  # Convert [-1,1] to [0,1]
        features.append(tone_normalized)
        
        # Speaking rate (normalize from words per minute)
        rate = voice_data.get('speaking_rate', 120)
        min_rate, max_rate = self.normalization_ranges['speaking_rate']
        rate_normalized = (rate - min_rate) / (max_rate - min_rate)
        rate_normalized = max(0, min(1, rate_normalized))
        features.append(rate_normalized)
        
        return np.array(features)
    
    def extract_performance_features(self, performance_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract and normalize performance features
        
        Args:
            performance_data: Dictionary containing performance metrics
            
        Returns:
            Normalized performance feature vector
        """
        features = []
        
        # Response time (normalize and invert - lower time is better)
        response_time = performance_data.get('response_time', 3.0)
        min_time, max_time = self.normalization_ranges['response_time']
        time_normalized = 1.0 - (response_time - min_time) / (max_time - min_time)
        time_normalized = max(0, min(1, time_normalized))
        features.append(time_normalized)
        
        # Accuracy (already normalized 0-1)
        accuracy = performance_data.get('accuracy', 0.75)
        features.append(accuracy)
        
        # Cognitive load (lower is better, so invert)
        cognitive_load = performance_data.get('cognitive_load', 0.5)
        load_inverted = 1.0 - cognitive_load
        features.append(load_inverted)
        
        return np.array(features)
    
    def extract_temporal_features(self, temporal_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract and encode temporal features
        
        Args:
            temporal_data: Dictionary containing temporal information
            
        Returns:
            Encoded temporal feature vector
        """
        features = []
        
        # Time of day encoding (circular encoding for continuous time)
        time_of_day = temporal_data.get('time_of_day', 'afternoon')
        time_mapping = {
            'morning': 0.8,
            'afternoon': 0.6, 
            'evening': 0.4,
            'night': 0.2
        }
        time_encoded = time_mapping.get(time_of_day.lower(), 0.5)
        features.append(time_encoded)
        
        # Day of week encoding (weekday vs weekend pattern)
        day_of_week = temporal_data.get('day_of_week', 'tuesday')
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
        day_encoded = 0.7 if day_of_week.lower() in weekdays else 0.4
        features.append(day_encoded)
        
        return np.array(features)
    
    def compute_attention_weights(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute attention weights for different modalities based on signal quality
        
        Args:
            features: Dictionary of feature vectors by modality
            
        Returns:
            Attention weights for each modality
        """
        weights = {}
        
        # Behavioral attention (higher for consistent patterns)
        behavioral_variance = np.var(features['behavioral'])
        weights['behavioral'] = sigmoid(1.0 - behavioral_variance)
        
        # Voice attention (higher for clear speech)
        voice_clarity = features['voice'][0]  # First feature is clarity
        weights['voice'] = voice_clarity
        
        # Performance attention (higher for good performance)
        performance_mean = np.mean(features['performance'])
        weights['performance'] = performance_mean
        
        # Temporal attention (constant moderate weight)
        weights['temporal'] = 0.6
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            # Fallback to equal weights
            weights = {k: 0.25 for k in weights.keys()}
            
        return weights
    
    def process_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Main processing pipeline that converts multimodal input to feature vector
        
        Args:
            input_data: Dictionary containing all input modalities
            
        Returns:
            Unified feature vector for downstream processing
        """
        # Extract features from each modality
        features = {}
        
        if 'behavioral' in input_data:
            features['behavioral'] = self.extract_behavioral_features(input_data['behavioral'])
        else:
            features['behavioral'] = np.array([0.5, 0.5, 0.5])  # Default values
            
        if 'voice' in input_data:
            features['voice'] = self.extract_voice_features(input_data['voice'])
        else:
            features['voice'] = np.array([0.7, 0.5, 0.5])  # Default values
            
        if 'performance' in input_data:
            features['performance'] = self.extract_performance_features(input_data['performance'])
        else:
            features['performance'] = np.array([0.5, 0.75, 0.5])  # Default values
            
        if 'temporal' in input_data:
            features['temporal'] = self.extract_temporal_features(input_data['temporal'])
        else:
            features['temporal'] = np.array([0.6, 0.7])  # Default values
        
        # Compute attention weights
        attention_weights = self.compute_attention_weights(features)
        
        # Concatenate weighted features
        weighted_features = []
        for modality, feature_vector in features.items():
            weight = attention_weights[modality]
            weighted_vector = feature_vector * weight
            weighted_features.extend(weighted_vector)
        
        # Convert to numpy array and apply final normalization
        unified_features = np.array(weighted_features)
        unified_features = normalize_features(unified_features)
        
        return unified_features
    
    def get_feature_names(self) -> List[str]:
        """
        Get descriptive names for each feature dimension
        
        Returns:
            List of feature names corresponding to the output vector
        """
        return [
            'behavioral_activity', 'behavioral_social', 'behavioral_sleep',
            'voice_clarity', 'voice_emotion', 'voice_rate',
            'performance_speed', 'performance_accuracy', 'performance_load',
            'temporal_time', 'temporal_day'
        ]
