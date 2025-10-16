"""
Utility Functions - Mathematical and helper functions for CACF system
Implements core mathematical operations used across all system components
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import json
from datetime import datetime

def sigmoid(x: Union[float, np.ndarray], temperature: float = 1.0) -> Union[float, np.ndarray]:
    """
    Sigmoid activation function with temperature parameter
    σ(x) = 1 / (1 + exp(-x/T))
    
    Args:
        x: Input value(s)
        temperature: Temperature parameter for controlling steepness
        
    Returns:
        Sigmoid-transformed value(s)
    """
    return 1.0 / (1.0 + np.exp(-x / temperature))

def tanh_scaled(x: Union[float, np.ndarray], scale: float = 1.0) -> Union[float, np.ndarray]:
    """
    Scaled hyperbolic tangent function
    tanh(x * scale)
    
    Args:
        x: Input value(s) 
        scale: Scaling factor
        
    Returns:
        Scaled tanh values in range [-1, 1]
    """
    return np.tanh(x * scale)

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Softmax function with temperature parameter
    
    Args:
        x: Input array
        temperature: Temperature parameter
        
    Returns:
        Softmax probabilities
    """
    exp_x = np.exp((x - np.max(x)) / temperature)
    return exp_x / np.sum(exp_x)

def irt_probability(ability: float, difficulty: float, discrimination: float = 1.0, 
                   guessing: float = 0.0, upper_asymptote: float = 1.0) -> float:
    """
    Item Response Theory (IRT) probability computation
    P(θ,β) = c + (d-c) / (1 + exp(-a(θ-β)))
    
    Args:
        ability: User ability parameter θ
        difficulty: Item difficulty parameter β
        discrimination: Item discrimination parameter a
        guessing: Guessing parameter c (3PL model)
        upper_asymptote: Upper asymptote parameter d (4PL model)
        
    Returns:
        Probability of correct response [0, 1]
    """
    exponent = -discrimination * (ability - difficulty)
    probability = guessing + (upper_asymptote - guessing) / (1.0 + np.exp(exponent))
    return max(0.0, min(1.0, probability))

def adaptive_difficulty(current_difficulty: float, success_rate: float, 
                       target_success_rate: float = 0.7, step_size: float = 0.1,
                       min_difficulty: float = 0.1, max_difficulty: float = 0.9) -> float:
    """
    Adaptive difficulty adjustment based on success rate
    
    Args:
        current_difficulty: Current difficulty level
        success_rate: Observed success rate
        target_success_rate: Target success rate
        step_size: Maximum adjustment step size
        min_difficulty: Minimum allowed difficulty
        max_difficulty: Maximum allowed difficulty
        
    Returns:
        Adjusted difficulty level
    """
    error = success_rate - target_success_rate
    
    # Adjust difficulty inversely to success rate
    if error > 0.1:  # Too easy
        adjustment = min(step_size, error * 0.5)
    elif error < -0.1:  # Too hard
        adjustment = max(-step_size, error * 0.5)
    else:
        adjustment = error * 0.1  # Fine tuning
        
    new_difficulty = current_difficulty + adjustment
    return max(min_difficulty, min(max_difficulty, new_difficulty))

def exponential_moving_average(values: List[float], alpha: float = 0.3) -> float:
    """
    Compute exponential moving average
    EMA_t = α * x_t + (1-α) * EMA_{t-1}
    
    Args:
        values: List of values (most recent last)
        alpha: Smoothing factor [0, 1]
        
    Returns:
        Exponential moving average
    """
    if not values:
        return 0.0
    
    ema = values[0]
    for value in values[1:]:
        ema = alpha * value + (1 - alpha) * ema
        
    return ema

def weighted_average(values: List[float], weights: List[float]) -> float:
    """
    Compute weighted average
    
    Args:
        values: List of values
        weights: List of weights (same length as values)
        
    Returns:
        Weighted average
    """
    if not values or not weights or len(values) != len(weights):
        return 0.0
        
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    weight_sum = sum(weights)
    
    return weighted_sum / weight_sum if weight_sum > 0 else 0.0

def normalize_features(features: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize feature vector using specified method
    
    Args:
        features: Input feature array
        method: Normalization method ('minmax', 'zscore', 'unit')
        
    Returns:
        Normalized feature array
    """
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(features)
        max_val = np.max(features)
        if max_val > min_val:
            return (features - min_val) / (max_val - min_val)
        else:
            return features
    elif method == 'zscore':
        # Z-score normalization (mean=0, std=1)
        mean_val = np.mean(features)
        std_val = np.std(features)
        if std_val > 0:
            return (features - mean_val) / std_val
        else:
            return features - mean_val
    elif method == 'unit':
        # Unit vector normalization
        norm = np.linalg.norm(features)
        if norm > 0:
            return features / norm
        else:
            return features
    else:
        return features

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity [-1, 1]
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return np.dot(vec1, vec2) / (norm1 * norm2)

def smooth_signal(signal: List[float], window_size: int = 3) -> List[float]:
    """
    Apply moving average smoothing to signal
    
    Args:
        signal: Input signal values
        window_size: Size of smoothing window
        
    Returns:
        Smoothed signal
    """
    if len(signal) < window_size:
        return signal
        
    smoothed = []
    for i in range(len(signal)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(signal), i + window_size // 2 + 1)
        window_mean = np.mean(signal[start_idx:end_idx])
        smoothed.append(window_mean)
        
    return smoothed

def load_sample_data() -> Dict[str, Any]:
    """
    Load sample multimodal data scenarios for demonstration
    
    Returns:
        Dictionary of sample data scenarios
    """
    sample_scenarios = {
        'high_engagement_morning': {
            'behavioral': {
                'activity_level': 0.8,
                'social_engagement': 0.7,
                'sleep_quality': 0.9
            },
            'voice': {
                'speech_clarity': 0.9,
                'emotional_tone': 0.6,
                'speaking_rate': 140
            },
            'performance': {
                'response_time': 2.1,
                'accuracy': 0.92,
                'cognitive_load': 0.3
            },
            'temporal': {
                'time_of_day': 'morning',
                'day_of_week': 'tuesday'
            }
        },
        'moderate_engagement_afternoon': {
            'behavioral': {
                'activity_level': 0.6,
                'social_engagement': 0.5,
                'sleep_quality': 0.7
            },
            'voice': {
                'speech_clarity': 0.75,
                'emotional_tone': 0.1,
                'speaking_rate': 120
            },
            'performance': {
                'response_time': 3.5,
                'accuracy': 0.78,
                'cognitive_load': 0.5
            },
            'temporal': {
                'time_of_day': 'afternoon',
                'day_of_week': 'friday'
            }
        },
        'low_engagement_evening': {
            'behavioral': {
                'activity_level': 0.3,
                'social_engagement': 0.2,
                'sleep_quality': 0.4
            },
            'voice': {
                'speech_clarity': 0.6,
                'emotional_tone': -0.4,
                'speaking_rate': 100
            },
            'performance': {
                'response_time': 5.2,
                'accuracy': 0.62,
                'cognitive_load': 0.8
            },
            'temporal': {
                'time_of_day': 'evening',
                'day_of_week': 'monday'
            }
        },
        'challenging_task_scenario': {
            'behavioral': {
                'activity_level': 0.7,
                'social_engagement': 0.6,
                'sleep_quality': 0.8
            },
            'voice': {
                'speech_clarity': 0.8,
                'emotional_tone': -0.1,
                'speaking_rate': 110
            },
            'performance': {
                'response_time': 6.8,
                'accuracy': 0.45,
                'cognitive_load': 0.9
            },
            'temporal': {
                'time_of_day': 'afternoon',
                'day_of_week': 'wednesday'
            }
        },
        'social_isolation_scenario': {
            'behavioral': {
                'activity_level': 0.4,
                'social_engagement': 0.1,
                'sleep_quality': 0.5
            },
            'voice': {
                'speech_clarity': 0.7,
                'emotional_tone': -0.3,
                'speaking_rate': 95
            },
            'performance': {
                'response_time': 4.0,
                'accuracy': 0.70,
                'cognitive_load': 0.6
            },
            'temporal': {
                'time_of_day': 'morning',
                'day_of_week': 'saturday'
            }
        }
    }
    
    return sample_scenarios

def visualize_context_state(context_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare context state data for visualization
    
    Args:
        context_state: Context state from context hub
        
    Returns:
        Visualization-ready data structure
    """
    if not context_state:
        return {'error': 'No context state provided'}
    
    valence_arousal = context_state.get('valence_arousal', {'valence': 0, 'arousal': 0})
    context_dims = context_state.get('context_dimensions', {})
    
    return {
        'valence_arousal': {
            'valence': float(valence_arousal.get('valence', 0)),
            'arousal': float(valence_arousal.get('arousal', 0)),
            'quadrant': classify_va_quadrant(valence_arousal.get('valence', 0), 
                                           valence_arousal.get('arousal', 0))
        },
        'cognitive_dimensions': {
            'cognitive_load': float(context_dims.get('cognitive_load', 0.5)),
            'attention_level': float(context_dims.get('attention_level', 0.5)),
            'engagement_level': float(context_dims.get('engagement_level', 0.5)),
            'cognitive_state': float(context_dims.get('cognitive_state', 0.5))
        },
        'confidence': float(context_state.get('context_confidence', 0.5)),
        'persona': context_state.get('persona', 'unknown')
    }

def classify_va_quadrant(valence: float, arousal: float) -> str:
    """
    Classify valence-arousal values into emotional quadrants
    
    Args:
        valence: Emotional valence [-1, 1]
        arousal: Emotional arousal [-1, 1]
        
    Returns:
        Quadrant classification string
    """
    if valence >= 0 and arousal >= 0:
        return 'high_valence_high_arousal'  # Excited, happy
    elif valence >= 0 and arousal < 0:
        return 'high_valence_low_arousal'   # Calm, content
    elif valence < 0 and arousal >= 0:
        return 'low_valence_high_arousal'   # Anxious, stressed
    else:
        return 'low_valence_low_arousal'    # Sad, tired

def create_engagement_chart(engagement_history: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Create engagement trend chart data for visualization
    
    Args:
        engagement_history: List of engagement score dictionaries
        
    Returns:
        Chart data structure
    """
    if not engagement_history:
        return {'error': 'No engagement history provided'}
    
    return {
        'x': list(range(len(engagement_history))),
        'y': [entry.get('engagement', 0.5) for entry in engagement_history],
        'long_term_y': [entry.get('long_term_engagement', 0.5) for entry in engagement_history],
        'title': 'Engagement Score Over Time',
        'x_label': 'Interaction Number',
        'y_label': 'Engagement Score'
    }

def safe_serialize(obj: Any) -> str:
    """
    Safely serialize object to JSON, handling numpy types
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string representation
    """
    def convert_numpy(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, (np.int32, np.int64)):
            return int(item)
        elif isinstance(item, (np.float32, np.float64)):
            return float(item)
        elif isinstance(item, dict):
            return {key: convert_numpy(value) for key, value in item.items()}
        elif isinstance(item, list):
            return [convert_numpy(value) for value in item]
        else:
            return item
    
    try:
        converted_obj = convert_numpy(obj)
        return json.dumps(converted_obj, indent=2)
    except Exception as e:
        return f"Serialization error: {str(e)}"

def validate_feature_vector(features: np.ndarray, expected_length: int = 11) -> bool:
    """
    Validate feature vector format and content
    
    Args:
        features: Feature vector to validate
        expected_length: Expected vector length
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(features, np.ndarray):
        return False
    
    if len(features) != expected_length:
        return False
    
    # Check for NaN or infinite values
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        return False
    
    # Check reasonable range [0, 1] for most features
    if np.any(features < 0) or np.any(features > 1):
        return False
    
    return True

def compute_system_health_metrics(user_history: Dict[str, List]) -> Dict[str, float]:
    """
    Compute overall system health metrics from user interaction history
    
    Args:
        user_history: User interaction history from feedback loop
        
    Returns:
        System health metrics
    """
    if not user_history or not any(user_history.values()):
        return {'status': 'no_data'}
    
    # Ability progression
    ability_estimates = user_history.get('ability_estimates', [])
    if len(ability_estimates) >= 2:
        ability_trend = (ability_estimates[-1] - ability_estimates[0]) / len(ability_estimates)
        ability_stability = 1.0 - np.std(ability_estimates[-5:]) if len(ability_estimates) >= 5 else 1.0
    else:
        ability_trend = 0.0
        ability_stability = 1.0
    
    # Engagement trends
    engagement_scores = user_history.get('engagement_scores', [])
    if len(engagement_scores) >= 2:
        engagement_trend = (engagement_scores[-1] - engagement_scores[0]) / len(engagement_scores)
        engagement_stability = 1.0 - np.std(engagement_scores[-5:]) if len(engagement_scores) >= 5 else 1.0
    else:
        engagement_trend = 0.0
        engagement_stability = 1.0
    
    # System adaptation effectiveness
    adaptation_events = user_history.get('adaptation_events', [])
    adaptation_frequency = len(adaptation_events) / max(1, len(ability_estimates))
    
    return {
        'ability_trend': float(ability_trend),
        'ability_stability': float(ability_stability),
        'engagement_trend': float(engagement_trend), 
        'engagement_stability': float(engagement_stability),
        'adaptation_frequency': float(adaptation_frequency),
        'total_interactions': len(ability_estimates),
        'system_health_score': float(np.mean([ability_stability, engagement_stability, 
                                            min(1.0, 1.0 - abs(engagement_trend))]))
    }

def log_interaction_event(event_type: str, data: Dict[str, Any], 
                         timestamp: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Create standardized log entry for system interactions
    
    Args:
        event_type: Type of event (e.g., 'user_input', 'agent_response', 'adaptation')
        data: Event data dictionary
        timestamp: Optional timestamp (defaults to now)
        
    Returns:
        Standardized log entry
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return {
        'timestamp': timestamp.isoformat(),
        'event_type': event_type,
        'data': safe_serialize(data),
        'system_version': '1.0.0'
    }
