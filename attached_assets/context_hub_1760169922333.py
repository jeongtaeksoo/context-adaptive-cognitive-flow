"""
Context Hub - Stage II: Persona-Specific Context Recognition
Implements Equations 4-6, 16-18 for context modeling with attention mechanism and valence-arousal mapping
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from utils import sigmoid, softmax, compute_attention_weights

class ContextHub:
    """
    Context recognition engine that processes multimodal features and generates
    persona-specific context representations with attention mechanisms.
    """
    
    def __init__(self, use_ml_embeddings: bool = False):
        """
        Initialize context hub with persona-specific parameters
        
        Args:
            use_ml_embeddings: If True, use sentence-transformers for context encoding.
                             If False, use mock neural network simulation (default)
        """
        self.use_ml_embeddings = use_ml_embeddings
        self.ml_model = None
        
        # Load ML model if requested
        if use_ml_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.ml_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Loaded sentence-transformers model: all-MiniLM-L6-v2")
            except ImportError:
                print("⚠️  sentence-transformers not available, falling back to mock encodings")
                self.use_ml_embeddings = False
        
        # Persona-specific attention weights (Equation 16)
        self.persona_attention_weights = {
            'Teacher': {
                'behavioral': 0.2, 'voice': 0.3, 'performance': 0.4, 'temporal': 0.1
            },
            'Companion': {
                'behavioral': 0.4, 'voice': 0.4, 'performance': 0.1, 'temporal': 0.1  
            },
            'Coach': {
                'behavioral': 0.3, 'voice': 0.2, 'performance': 0.3, 'temporal': 0.2
            }
        }
        
        # Context dimension mappings (Equation 17)
        self.context_dimensions = {
            'cognitive_state': 0,
            'emotional_state': 1, 
            'engagement_level': 2,
            'difficulty_preference': 3,
            'social_need': 4,
            'attention_focus': 5
        }
        
        # Valence-arousal mapping parameters (Equation 18)
        self.valence_arousal_weights = np.array([
            [0.8, 0.2, -0.1, 0.5, 0.3, 0.1],  # Valence weights per context dimension
            [0.3, 0.7, 0.6, -0.2, 0.4, 0.8]   # Arousal weights per context dimension  
        ])
        
        # CORRECTED: Neural network dimensions h=64 per LaTeX spec
        self.context_network_weights = {
            'input_to_hidden': np.random.normal(0, 0.1, (35, 64)),  # 35 input features to 64 hidden
            'hidden_to_context': np.random.normal(0, 0.1, (64, 6))  # 64 hidden to 6 context dimensions
        }
        
        # ADDED: Cognitive load parameters (NASA-TLX adapted) - LaTeX spec
        self.cognitive_load_params = {
            'beta_1': 0.4,   # Response time weight β₁
            'beta_2': 0.35,  # Error rate weight β₂
            'beta_3': 0.25,  # Attention variance weight β₃
            't_base': 3.0    # Baseline response time (seconds)
        }
        
        # Persona-specific context transformation matrices
        self.persona_transforms = {
            'Teacher': np.random.normal(0, 0.1, (6, 6)),
            'Companion': np.random.normal(0, 0.1, (6, 6)),
            'Coach': np.random.normal(0, 0.1, (6, 6))
        }
        
    def compute_attention_mechanism(self, features: np.ndarray, persona: str) -> np.ndarray:
        """
        Compute attention weights for different feature modalities (Equation 4)
        α_i = softmax(W_attention * features_i + b_attention)
        
        Args:
            features: Input feature vector from perception engine
            persona: Selected AI persona (Teacher/Companion/Coach)
            
        Returns:
            Attention-weighted feature representation
        """
        # Split features by modality (assuming known split points)
        behavioral_features = features[:7]
        voice_features = features[7:14] 
        performance_features = features[14:21]
        temporal_features = features[21:]
        
        # Get persona-specific attention preferences
        attention_prefs = self.persona_attention_weights[persona]
        
        # Compute attention scores for each modality
        behavioral_score = np.mean(behavioral_features) * attention_prefs['behavioral']
        voice_score = np.mean(voice_features) * attention_prefs['voice'] 
        performance_score = np.mean(performance_features) * attention_prefs['performance']
        temporal_score = np.mean(temporal_features) * attention_prefs['temporal']
        
        attention_scores = np.array([behavioral_score, voice_score, performance_score, temporal_score])
        
        # Apply softmax to get normalized attention weights
        attention_weights = softmax(attention_scores)
        
        # Apply attention weights to modality features
        weighted_behavioral = behavioral_features * attention_weights[0]
        weighted_voice = voice_features * attention_weights[1]
        weighted_performance = performance_features * attention_weights[2] 
        weighted_temporal = temporal_features * attention_weights[3]
        
        # Combine attention-weighted features
        attended_features = np.concatenate([
            weighted_behavioral, weighted_voice, weighted_performance, weighted_temporal
        ])
        
        return attended_features
        
    def compute_context_vector(self, attended_features: np.ndarray, persona: str) -> np.ndarray:
        """
        Compute context vector using lightweight neural network (Equation 5)
        C_t = f_context(α_t ⊙ x_t)
        
        Note: This implementation uses mock neural network simulation. When use_ml_embeddings=True,
        the ml_model can be integrated here to generate embeddings from textual representations
        of the attended features, then mapped to context dimensions.
        
        Args:
            attended_features: Attention-weighted input features
            persona: Selected AI persona
            
        Returns:
            Context vector representing current user state
        """
        # TODO: Optional ML integration point
        # if self.use_ml_embeddings and self.ml_model:
        #     feature_text = self._features_to_text(attended_features)
        #     embeddings = self.ml_model.encode([feature_text])[0]
        #     context_raw = self._embeddings_to_context(embeddings)
        # else:
        #     # Use mock neural network (current implementation)
        
        # Forward pass through mock neural network
        # Hidden layer computation
        hidden_activations = sigmoid(
            np.dot(attended_features, self.context_network_weights['input_to_hidden'])
        )
        
        # Context layer computation  
        context_raw = np.dot(hidden_activations, self.context_network_weights['hidden_to_context'])
        
        # Apply persona-specific transformation
        persona_transform = self.persona_transforms[persona]
        context_vector = sigmoid(np.dot(context_raw, persona_transform))
        
        return context_vector
        
    def compute_valence_arousal(self, context_vector: np.ndarray) -> Tuple[float, float]:
        """
        Map context vector to valence-arousal space (Equation 6 & 18)
        [valence, arousal] = W_va * C_t + b_va
        
        Args:
            context_vector: 6-dimensional context representation
            
        Returns:
            Tuple of (valence, arousal) values in [-1, 1] range
        """
        # Compute valence and arousal using linear transformation
        valence_arousal = np.dot(self.valence_arousal_weights, context_vector)
        
        # Apply tanh to constrain to [-1, 1] range
        valence = np.tanh(valence_arousal[0])
        arousal = np.tanh(valence_arousal[1])
        
        return valence, arousal
        
    def compute_context(self, input_features: np.ndarray, persona: str) -> Dict[str, Any]:
        """
        Main context computation pipeline integrating all components
        
        Args:
            input_features: Processed multimodal features from perception engine
            persona: Selected AI persona (Teacher/Companion/Coach)
            
        Returns:
            Dictionary containing complete context state representation
        """
        # Stage 1: Attention mechanism (Equation 4)
        attended_features = self.compute_attention_mechanism(input_features, persona)
        
        # Stage 2: Context vector computation (Equation 5) 
        context_vector = self.compute_context_vector(attended_features, persona)
        
        # Stage 3: Valence-arousal mapping (Equation 6)
        valence, arousal = self.compute_valence_arousal(context_vector)
        
        # Create interpretable context state
        context_state = {
            'persona': persona,
            'context_vector': context_vector.tolist(),
            'context_dimensions': {
                'cognitive_state': float(context_vector[0]),
                'emotional_state': float(context_vector[1]),
                'engagement_level': float(context_vector[2]), 
                'difficulty_preference': float(context_vector[3]),
                'social_need': float(context_vector[4]),
                'attention_focus': float(context_vector[5])
            },
            'valence_arousal': {
                'valence': float(valence),
                'arousal': float(arousal) 
            },
            'attention_weights': self._get_modality_attention(attended_features, input_features),
            'context_summary': self._generate_context_summary(context_vector, valence, arousal)
        }
        
        return context_state
    
    def compute_cognitive_load(self, response_time: float, error_rate: float, 
                              attention_variance: float) -> float:
        """
        ADDED: Cognitive load estimation using NASA-TLX adapted formula (LaTeX spec)
        L_cog = β₁·(Δt_resp/t_base) + β₂·e_rate + β₃·σ_att²
        
        Args:
            response_time: Response time in seconds
            error_rate: Error frequency [0, 1]
            attention_variance: Attention variance metric
        
        Returns:
            Cognitive load L_cog ∈ [0, 2]
        """
        params = self.cognitive_load_params
        
        # Component 1: Response time (normalized by baseline)
        time_component = params['beta_1'] * (response_time / params['t_base'])
        
        # Component 2: Error rate
        error_component = params['beta_2'] * error_rate
        
        # Component 3: Attention variance squared
        attention_component = params['beta_3'] * (attention_variance ** 2)
        
        # Combine components
        L_cog = time_component + error_component + attention_component
        
        # Bound to [0, 2] approximately (per LaTeX spec)
        L_cog = np.clip(L_cog, 0.0, 2.0)
        
        return L_cog
        
    def _get_modality_attention(self, attended_features: np.ndarray, 
                               original_features: np.ndarray) -> Dict[str, float]:
        """
        Compute attention contribution of each modality for interpretability
        
        Args:
            attended_features: Features after attention weighting
            original_features: Original input features
            
        Returns:
            Dictionary with attention weights per modality
        """
        # Compute relative attention by comparing attended vs original features
        behavioral_attention = np.mean(attended_features[:7]) / (np.mean(original_features[:7]) + 1e-8)
        voice_attention = np.mean(attended_features[7:14]) / (np.mean(original_features[7:14]) + 1e-8) 
        performance_attention = np.mean(attended_features[14:21]) / (np.mean(original_features[14:21]) + 1e-8)
        temporal_attention = np.mean(attended_features[21:]) / (np.mean(original_features[21:]) + 1e-8)
        
        # Normalize to sum to 1
        total_attention = behavioral_attention + voice_attention + performance_attention + temporal_attention
        
        return {
            'behavioral': float(behavioral_attention / total_attention),
            'voice': float(voice_attention / total_attention),
            'performance': float(performance_attention / total_attention), 
            'temporal': float(temporal_attention / total_attention)
        }
        
    def _generate_context_summary(self, context_vector: np.ndarray, 
                                 valence: float, arousal: float) -> str:
        """
        Generate human-readable summary of context state
        
        Args:
            context_vector: 6-dimensional context representation
            valence: Emotional valence [-1, 1]
            arousal: Emotional arousal [-1, 1]
            
        Returns:
            Natural language summary of user's current context
        """
        # Interpret context dimensions
        cognitive_level = "high" if context_vector[0] > 0.6 else "moderate" if context_vector[0] > 0.3 else "low"
        emotional_state = "positive" if context_vector[1] > 0.5 else "neutral" if context_vector[1] > 0.2 else "subdued"
        engagement = "engaged" if context_vector[2] > 0.6 else "moderately engaged" if context_vector[2] > 0.3 else "disengaged"
        
        # Interpret valence-arousal
        if valence > 0.3 and arousal > 0.3:
            mood = "energetic and positive"
        elif valence > 0.3 and arousal < -0.3:
            mood = "calm and content"
        elif valence < -0.3 and arousal > 0.3:
            mood = "anxious or frustrated"
        elif valence < -0.3 and arousal < -0.3:
            mood = "tired or withdrawn"
        else:
            mood = "balanced"
            
        summary = f"User shows {cognitive_level} cognitive activity, {emotional_state} emotional state, and appears {engagement}. Current mood: {mood}."
        
        return summary
        
    def update_persona_preferences(self, persona: str, feedback_score: float):
        """
        Adaptive update of persona-specific parameters based on user feedback
        
        Args:
            persona: AI persona being updated
            feedback_score: User satisfaction/engagement score [0, 1]
        """
        # Simple adaptation: adjust attention weights based on feedback
        learning_rate = 0.1
        
        if feedback_score > 0.7:
            # Positive feedback: slightly increase current attention pattern
            for modality in self.persona_attention_weights[persona]:
                current_weight = self.persona_attention_weights[persona][modality]
                self.persona_attention_weights[persona][modality] += learning_rate * current_weight * 0.1
        elif feedback_score < 0.3:
            # Negative feedback: slightly randomize attention pattern  
            for modality in self.persona_attention_weights[persona]:
                noise = np.random.normal(0, 0.05)
                self.persona_attention_weights[persona][modality] += noise
                
        # Ensure weights remain normalized
        total_weight = sum(self.persona_attention_weights[persona].values())
        for modality in self.persona_attention_weights[persona]:
            self.persona_attention_weights[persona][modality] /= total_weight
