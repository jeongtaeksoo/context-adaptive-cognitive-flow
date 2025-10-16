"""
Utility functions for the Context-Adaptive Cognitive Flow System
Shared mathematical functions, data processing utilities, and visualization helpers
"""

import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Tuple
import random

def sigmoid(x):
    """
    Compute sigmoid activation function with numerical stability
    
    Args:
        x: Input array or scalar
        
    Returns:
        Sigmoid-activated values
    """
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax function with numerical stability
    
    Args:
        x: Input array
        
    Returns:
        Normalized probability distribution
    """
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def item_response_probability(ability: float, difficulty: float, 
                            discrimination: float = 1.0, guessing: float = 0.0) -> float:
    """
    Compute Item Response Theory (IRT) success probability
    P(θ, β) = γ + (1-γ) * sigmoid(α*(θ - β))
    
    Args:
        ability: User ability parameter θ
        difficulty: Item difficulty parameter β  
        discrimination: Item discrimination parameter α
        guessing: Guessing parameter γ
        
    Returns:
        Probability of successful response [0, 1]
    """
    logit = discrimination * (ability - difficulty)
    probability = guessing + (1 - guessing) * sigmoid(logit)
    return float(probability)

def exponential_moving_average(values: List[float], alpha: float = 0.3) -> float:
    """
    Compute exponential moving average
    
    Args:
        values: List of values (most recent last)
        alpha: Learning rate [0, 1]
        
    Returns:
        EMA value
    """
    if not values:
        return 0.0
        
    ema = values[0]
    for value in values[1:]:
        ema = alpha * value + (1 - alpha) * ema
        
    return ema

def compute_attention_weights(features: np.ndarray, attention_params: Dict[str, float]) -> np.ndarray:
    """
    Compute attention weights for different feature modalities
    
    Args:
        features: Input feature vector
        attention_params: Parameters for attention computation
        
    Returns:
        Attention-weighted feature vector
    """
    # Simple attention mechanism based on feature magnitudes
    feature_magnitudes = np.abs(features)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(feature_magnitudes)
    
    # Apply attention to features
    attended_features = features * attention_weights
    
    return attended_features

def generate_persona_response(template: str, context_state: Dict[str, Any], 
                            strategy: Dict[str, Any]) -> str:
    """
    Generate persona-specific response using template and context
    
    Args:
        template: Response template with placeholders
        context_state: Current user context
        strategy: Response strategy configuration
        
    Returns:
        Generated response text
    """
    # Extract context information for template filling
    context_dims = context_state['context_dimensions']
    
    # Determine skill area based on context
    if context_dims['cognitive_state'] > 0.6:
        skill_area = "problem-solving skills"
    elif context_dims['engagement_level'] > 0.6:
        skill_area = "attention and focus"
    else:
        skill_area = "memory and recall"
        
    # Determine performance metric
    if context_dims['cognitive_state'] > context_dims['engagement_level']:
        performance_metric = "cognitive performance"
    else:
        performance_metric = "engagement level"
        
    # Fill template with context-appropriate values
    response = template.format(
        skill_area=skill_area,
        performance_metric=performance_metric
    )
    
    # Add persona-specific modifications
    persona = strategy['persona']
    if persona == 'Teacher' and context_dims['difficulty_preference'] > 0.7:
        response += " I believe you're ready for more advanced challenges!"
    elif persona == 'Companion' and context_dims['social_need'] > 0.6:
        response += " I enjoy our conversations together."
    elif persona == 'Coach' and context_dims['engagement_level'] < 0.4:
        response += " Remember, every small step counts toward your goals!"
        
    return response

def load_sample_data() -> Dict[str, Any]:
    """
    Generate realistic sample data for system demonstration
    
    Returns:
        Dictionary containing various input scenarios
    """
    sample_scenarios = {
        "morning_high_energy": {
            "behavioral": {
                "activity_level": 0.8,
                "social_engagement": 0.7,
                "sleep_quality": 0.9
            },
            "voice": {
                "speech_clarity": 0.85,
                "emotional_tone": 0.6,
                "speaking_rate": 0.7
            },
            "performance": {
                "response_time": 2.5,
                "accuracy": 0.9,
                "cognitive_load": 0.6
            },
            "temporal": {
                "time_of_day": "morning",
                "day_of_week": "monday"
            }
        },
        
        "afternoon_moderate_fatigue": {
            "behavioral": {
                "activity_level": 0.5,
                "social_engagement": 0.4,
                "sleep_quality": 0.6
            },
            "voice": {
                "speech_clarity": 0.7,
                "emotional_tone": -0.2,
                "speaking_rate": 0.5
            },
            "performance": {
                "response_time": 4.2,
                "accuracy": 0.7,
                "cognitive_load": 0.8
            },
            "temporal": {
                "time_of_day": "afternoon", 
                "day_of_week": "wednesday"
            }
        },
        
        "evening_relaxed": {
            "behavioral": {
                "activity_level": 0.3,
                "social_engagement": 0.6,
                "sleep_quality": 0.7
            },
            "voice": {
                "speech_clarity": 0.8,
                "emotional_tone": 0.3,
                "speaking_rate": 0.4
            },
            "performance": {
                "response_time": 3.8,
                "accuracy": 0.75,
                "cognitive_load": 0.4
            },
            "temporal": {
                "time_of_day": "evening",
                "day_of_week": "friday"
            }
        },
        
        "weekend_social": {
            "behavioral": {
                "activity_level": 0.6,
                "social_engagement": 0.9,
                "sleep_quality": 0.8
            },
            "voice": {
                "speech_clarity": 0.9,
                "emotional_tone": 0.8,
                "speaking_rate": 0.8
            },
            "performance": {
                "response_time": 3.0,
                "accuracy": 0.85,
                "cognitive_load": 0.5
            },
            "temporal": {
                "time_of_day": "afternoon",
                "day_of_week": "saturday"
            }
        },
        
        "challenging_day": {
            "behavioral": {
                "activity_level": 0.2,
                "social_engagement": 0.3,
                "sleep_quality": 0.4
            },
            "voice": {
                "speech_clarity": 0.5,
                "emotional_tone": -0.6,
                "speaking_rate": 0.3
            },
            "performance": {
                "response_time": 6.5,
                "accuracy": 0.5,
                "cognitive_load": 0.9
            },
            "temporal": {
                "time_of_day": "morning",
                "day_of_week": "tuesday"
            }
        }
    }
    
    return sample_scenarios

def visualize_context_state(context_state: Dict[str, Any]) -> go.Figure:
    """
    Create visualization of context state in valence-arousal space
    
    Args:
        context_state: Current context state dictionary
        
    Returns:
        Plotly figure showing valence-arousal mapping
    """
    valence = context_state['valence_arousal']['valence']
    arousal = context_state['valence_arousal']['arousal']
    
    # Create scatter plot in valence-arousal space
    fig = go.Figure()
    
    # Add background quadrants
    fig.add_shape(
        type="rect",
        x0=-1, y0=0, x1=0, y1=1,
        fillcolor="lightblue", opacity=0.2,
        line_width=0
    )
    fig.add_shape(
        type="rect", 
        x0=0, y0=0, x1=1, y1=1,
        fillcolor="lightgreen", opacity=0.2,
        line_width=0
    )
    fig.add_shape(
        type="rect",
        x0=-1, y0=-1, x1=0, y1=0,
        fillcolor="lightyellow", opacity=0.2,
        line_width=0
    )
    fig.add_shape(
        type="rect",
        x0=0, y0=-1, x1=1, y1=0,
        fillcolor="lightcoral", opacity=0.2,
        line_width=0
    )
    
    # Add current state point
    fig.add_trace(go.Scatter(
        x=[valence],
        y=[arousal],
        mode='markers',
        marker=dict(size=15, color='red'),
        name='Current State',
        text=[f'Valence: {valence:.2f}<br>Arousal: {arousal:.2f}'],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Add quadrant labels
    fig.add_annotation(x=-0.5, y=0.5, text="Anxious/Frustrated", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0.5, y=0.5, text="Energetic/Happy", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=-0.5, y=-0.5, text="Sad/Withdrawn", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0.5, y=-0.5, text="Calm/Content", showarrow=False, font=dict(size=10))
    
    fig.update_layout(
        title="Valence-Arousal State Mapping",
        xaxis_title="Valence (Negative ← → Positive)",
        yaxis_title="Arousal (Low ← → High)",
        xaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinewidth=2),
        yaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinewidth=2),
        width=500,
        height=500
    )
    
    return fig

def create_engagement_chart(engagement_history: List) -> go.Figure:
    """
    Create chart showing engagement over time
    
    Args:
        engagement_history: List of historical engagement scores (can be floats or dicts)
        
    Returns:
        Plotly figure showing engagement trends
    """
    fig = go.Figure()
    
    x_values = list(range(1, len(engagement_history) + 1))
    
    # Extract engagement scores (handle both float and dict formats)
    engagement_scores = []
    long_term_scores = []
    for item in engagement_history:
        if isinstance(item, dict):
            engagement_scores.append(item.get('engagement', 0))
            long_term_scores.append(item.get('long_term_engagement', item.get('engagement', 0)))
        else:
            engagement_scores.append(float(item))
            long_term_scores.append(float(item))
    
    # Add engagement line
    fig.add_trace(go.Scatter(
        x=x_values,
        y=engagement_scores,
        mode='lines+markers',
        name='Engagement Score',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Add long-term engagement if available
    if long_term_scores and long_term_scores != engagement_scores:
        fig.add_trace(go.Scatter(
            x=x_values,
            y=long_term_scores,
            mode='lines',
            name='Long-term Engagement',
            line=dict(color='purple', dash='dash', width=2)
        ))
    
    # Add trend line
    if len(engagement_scores) > 2:
        z = np.polyfit(x_values, engagement_scores, 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=p(x_values),
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dot', width=2)
        ))
    
    # Add threshold line
    fig.add_hline(
        y=0.7, 
        line_dash="dot", 
        line_color="green",
        annotation_text="High Engagement Threshold"
    )
    
    fig.add_hline(
        y=0.3,
        line_dash="dot", 
        line_color="orange",
        annotation_text="Low Engagement Threshold"
    )
    
    fig.update_layout(
        title="Engagement Score Over Time",
        xaxis_title="Interaction Number",
        yaxis_title="Engagement Score",
        yaxis=dict(range=[0, 1]),
        showlegend=True
    )
    
    return fig

def save_sample_input():
    """
    Save sample input data to JSON file for external use
    """
    sample_data = load_sample_data()
    
    with open('sample_input.json', 'w') as f:
        json.dump(sample_data, f, indent=2)

def load_json_data(file_path: str) -> Dict[str, Any]:
    """
    Load JSON data from file with error handling
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data dictionary or empty dict if error
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize vector to unit length
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    else:
        return vector

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity [-1, 1]
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 > 0 and norm2 > 0:
        return dot_product / (norm1 * norm2)
    else:
        return 0.0

def generate_interaction_id() -> str:
    """
    Generate unique interaction ID
    
    Returns:
        Unique string identifier
    """
    return f"interaction_{random.randint(100000, 999999)}"

def validate_input_data(input_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate input data structure and ranges
    
    Args:
        input_data: Input data dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_keys = ['behavioral', 'voice', 'performance', 'temporal']
    
    # Check top-level structure
    for key in required_keys:
        if key not in input_data:
            return False, f"Missing required key: {key}"
    
    # Check behavioral data
    behavioral = input_data['behavioral']
    for key in ['activity_level', 'social_engagement']:
        if key not in behavioral:
            return False, f"Missing behavioral key: {key}"
        if not (0 <= behavioral[key] <= 1):
            return False, f"Behavioral {key} must be in range [0, 1]"
    
    # Check voice data
    voice = input_data['voice']
    for key in ['speech_clarity', 'emotional_tone']:
        if key not in voice:
            return False, f"Missing voice key: {key}"
    
    if not (0 <= voice['speech_clarity'] <= 1):
        return False, "Speech clarity must be in range [0, 1]"
    if not (-1 <= voice['emotional_tone'] <= 1):
        return False, "Emotional tone must be in range [-1, 1]"
    
    # Check performance data
    performance = input_data['performance']
    for key in ['response_time', 'accuracy']:
        if key not in performance:
            return False, f"Missing performance key: {key}"
    
    if not (0 <= performance['accuracy'] <= 1):
        return False, "Accuracy must be in range [0, 1]"
    if performance['response_time'] <= 0:
        return False, "Response time must be positive"
    
    # Check temporal data
    temporal = input_data['temporal']
    valid_times = ['morning', 'afternoon', 'evening', 'night']
    valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    
    if temporal.get('time_of_day', '').lower() not in valid_times:
        return False, f"Invalid time_of_day. Must be one of: {valid_times}"
    if temporal.get('day_of_week', '').lower() not in valid_days:
        return False, f"Invalid day_of_week. Must be one of: {valid_days}"
    
    return True, "Input data is valid"

# Mathematical utility functions for advanced operations
def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function"""
    return np.maximum(0, x)

def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation function"""
    return np.tanh(x)

def gaussian_kernel(x: float, y: float, sigma: float = 1.0) -> float:
    """Gaussian kernel function"""
    return np.exp(-((x - y) ** 2) / (2 * sigma ** 2))

def moving_average(values: List[float], window_size: int = 3) -> List[float]:
    """Compute moving average of values"""
    if len(values) < window_size:
        return values
    
    moving_averages = []
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        moving_averages.append(sum(window) / window_size)
    
    return moving_averages

def z_score_normalize(values: List[float]) -> List[float]:
    """Normalize values using z-score"""
    if not values:
        return values
        
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    if std_val > 0:
        return [(x - mean_val) / std_val for x in values]
    else:
        return [0.0] * len(values)

def safe_serialize(obj: Any) -> Any:
    """
    Safely serialize data for JSON/CSV export, handling numpy, Decimal, datetime, etc.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable version of the object
    """
    import decimal
    import datetime
    
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, decimal.Decimal):
        return float(obj)
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    else:
        return obj
