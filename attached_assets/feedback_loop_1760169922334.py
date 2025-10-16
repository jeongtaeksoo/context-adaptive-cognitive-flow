"""
Feedback Loop - Stage IV: Feedback and Iterative Adaptation
Implements Equations 10-11, 22-23 for user ability updates and engagement tracking
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from utils import sigmoid, exponential_moving_average

class FeedbackLoop:
    """
    Feedback and adaptation system that updates user models and tracks engagement
    using Exponential Moving Average (EMA) and engagement metrics.
    """
    
    def __init__(self):
        """Initialize feedback loop with adaptation parameters"""
        
        # CORRECTED: EMA parameters per LaTeX spec (Equation 10)
        self.ema_parameters = {
            'alpha': 0.7,     # FIXED: α=0.7 for ability updates (was 0.3)
            'lambda': 0.1     # ADDED: λ=0.1 for engagement decay weighting
        }
        
        # Engagement computation weights (Equation 22)
        self.engagement_weights = {
            'task_completion': 0.3,
            'response_quality': 0.2, 
            'interaction_duration': 0.15,
            'positive_feedback': 0.15,
            'context_alignment': 0.1,
            'difficulty_appropriateness': 0.1
        }
        
        # Long-term tracking parameters (Equation 23)
        self.tracking_parameters = {
            'window_size': 10,  # Number of recent interactions to consider
            'decay_factor': 0.95,  # Exponential decay for historical data
            'min_engagement_threshold': 0.3,  # Minimum engagement for positive adaptation
            'adaptation_sensitivity': 0.1  # How quickly the system adapts
        }
        
        # User model state tracking
        self.user_history = {
            'ability_estimates': [],
            'engagement_scores': [], 
            'performance_metrics': [],
            'context_states': [],
            'adaptation_events': []
        }
        
    def update_user_ability(self, current_ability: float, task_difficulty: float, 
                           performance_score: float, context_state: Dict[str, Any]) -> float:
        """
        Update user ability estimate using EMA (Equation 10)
        θ_t = α * θ_observed + (1-α) * θ_t-1
        
        Args:
            current_ability: Previous ability estimate θ_t-1
            task_difficulty: Difficulty of completed task β
            performance_score: User performance on task [0, 1]
            context_state: Current context information
            
        Returns:
            Updated ability estimate θ_t
        """
        # Compute observed ability based on IRT model
        # If user performed well on difficult task, observed ability is high
        if performance_score > 0.5:
            # Success case: observed ability is above task difficulty
            observed_ability = task_difficulty + (performance_score - 0.5) * 0.4
        else:
            # Failure case: observed ability is below task difficulty  
            observed_ability = task_difficulty - (0.5 - performance_score) * 0.6
            
        # Context-based adjustment
        cognitive_state = context_state['context_dimensions']['cognitive_state']
        engagement_level = context_state['context_dimensions']['engagement_level']
        
        # Adjust observed ability based on context quality
        context_multiplier = (cognitive_state + engagement_level) / 2
        observed_ability *= (0.8 + 0.4 * context_multiplier)  # Scale between 0.8x and 1.2x
        
        # CORRECTED: Apply EMA update with correct formula order
        # θ_{t+1} = α·θ_t + (1-α)·θ̂_t where α=0.7
        alpha = self.ema_parameters['alpha']
        updated_ability = alpha * current_ability + (1 - alpha) * observed_ability
        
        # Constrain to reasonable bounds [0, 1]
        updated_ability = max(0.0, min(1.0, updated_ability))
        
        # Store in history
        self.user_history['ability_estimates'].append(updated_ability)
        
        return updated_ability
    
    def compute_engagement_score(self, interaction_data: Dict[str, Any], 
                               context_state: Dict[str, Any], 
                               response_data: Dict[str, Any]) -> float:
        """
        Compute current engagement score (Equation 11 & 22)
        E_t = Σ w_i * engagement_factor_i
        
        Args:
            interaction_data: Data about current interaction
            context_state: Current user context
            response_data: Information about system response
            
        Returns:
            Engagement score [0, 1]
        """
        engagement_factors = {}
        
        # Task completion factor
        completion_rate = interaction_data.get('task_completion_rate', 0.8)
        engagement_factors['task_completion'] = completion_rate
        
        # Response quality factor (based on performance)
        response_quality = interaction_data.get('response_quality', 0.7)
        engagement_factors['response_quality'] = response_quality
        
        # Interaction duration factor (normalized)
        interaction_time = interaction_data.get('interaction_duration_minutes', 5.0)
        # Optimal interaction time is around 5-15 minutes
        if interaction_time < 2:
            duration_score = interaction_time / 2.0  # Too short
        elif interaction_time > 20:
            duration_score = max(0.3, 1.0 - (interaction_time - 20) / 30.0)  # Too long
        else:
            duration_score = min(1.0, interaction_time / 10.0)  # Good range
        engagement_factors['interaction_duration'] = duration_score
        
        # Positive feedback factor
        positive_feedback = interaction_data.get('positive_feedback_count', 0) / max(1, interaction_data.get('total_feedback_count', 1))
        engagement_factors['positive_feedback'] = positive_feedback
        
        # Context alignment factor (from response system)
        context_alignment = response_data.get('context_alignment', 0.5)
        engagement_factors['context_alignment'] = context_alignment
        
        # Difficulty appropriateness (based on success probability)
        success_prob = response_data.get('success_probability', 0.5)
        # Optimal success probability is around 0.6-0.8 (challenging but achievable)
        if 0.6 <= success_prob <= 0.8:
            difficulty_appropriateness = 1.0
        elif success_prob < 0.6:
            difficulty_appropriateness = success_prob / 0.6  # Too difficult
        else:
            difficulty_appropriateness = max(0.5, 2.0 - success_prob / 0.4)  # Too easy
        engagement_factors['difficulty_appropriateness'] = difficulty_appropriateness
        
        # Compute weighted engagement score
        engagement_score = sum(
            self.engagement_weights[factor] * value
            for factor, value in engagement_factors.items()
        )
        
        # Apply context-based modulation
        valence = context_state['valence_arousal']['valence']
        arousal = context_state['valence_arousal']['arousal']
        
        # Positive valence increases engagement
        valence_boost = max(0, valence) * 0.1
        
        # Moderate arousal is optimal for engagement
        arousal_modifier = 1.0 - 0.3 * abs(arousal)  # Peak at arousal=0, decreases with |arousal|
        
        final_engagement = (engagement_score + valence_boost) * arousal_modifier
        final_engagement = max(0.0, min(1.0, final_engagement))
        
        # Store in history
        self.user_history['engagement_scores'].append(final_engagement)
        
        return final_engagement
    
    def update_long_term_engagement(self, current_engagement: float) -> float:
        """
        CORRECTED: Update long-term engagement using exponential weighting (Equation 23)
        P_engage = Σ(w_i·s_i) / Σ(w_i) where w_i = exp(-λ·(n-i))
        
        Args:
            current_engagement: Current interaction engagement score
            
        Returns:
            Updated long-term engagement estimate
        """
        lambda_param = self.ema_parameters['lambda']  # FIXED: Use λ=0.1
        
        # Get previous engagement scores
        if self.user_history['engagement_scores']:
            recent_scores = self.user_history['engagement_scores'][-self.tracking_parameters['window_size']:]
            n = len(recent_scores)
            
            # CORRECTED: Compute weights w_i = exp(-λ·(n-i)) where i=1..n
            weights = [np.exp(-lambda_param * (n - i)) for i in range(1, n + 1)]
            
            # Weighted average
            weighted_sum = sum(w * s for w, s in zip(weights, recent_scores))
            weight_sum = sum(weights)
            
            long_term_engagement = weighted_sum / weight_sum if weight_sum > 0 else current_engagement
        else:
            long_term_engagement = current_engagement
            
        return long_term_engagement
    
    def compute_adaptation_signals(self, engagement_score: float, 
                                 user_ability: float, context_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute signals for system adaptation based on feedback
        
        Args:
            engagement_score: Current engagement level
            user_ability: Updated user ability estimate
            context_state: Current context state
            
        Returns:
            Dictionary containing adaptation signals for different system components
        """
        adaptation_signals = {
            'perception_engine': {},
            'context_hub': {},
            'response_engine': {},
            'global_system': {}
        }
        
        # Perception engine adaptations
        if engagement_score < self.tracking_parameters['min_engagement_threshold']:
            adaptation_signals['perception_engine'] = {
                'increase_sensitivity': True,
                'focus_attention': context_state['attention_weights']
            }
            
        # Context hub adaptations
        context_confidence = np.mean(list(context_state['context_dimensions'].values()))
        if context_confidence < 0.4:
            adaptation_signals['context_hub'] = {
                'recalibrate_persona': True,
                'adjust_attention_weights': True
            }
            
        # Response engine adaptations
        ability_change = 0.0
        if len(self.user_history['ability_estimates']) >= 2:
            ability_change = (self.user_history['ability_estimates'][-1] - 
                            self.user_history['ability_estimates'][-2])
                            
        if abs(ability_change) > 0.1:  # Significant ability change
            adaptation_signals['response_engine'] = {
                'update_difficulty_calibration': True,
                'ability_trend': 'increasing' if ability_change > 0 else 'decreasing'
            }
            
        # Global system adaptations
        long_term_engagement = self.update_long_term_engagement(engagement_score)
        
        if long_term_engagement < 0.4:  # Sustained low engagement
            adaptation_signals['global_system'] = {
                'major_recalibration_needed': True,
                'consider_persona_switch': True,
                'reduce_system_complexity': True
            }
        elif long_term_engagement > 0.8:  # Sustained high engagement
            adaptation_signals['global_system'] = {
                'system_performing_well': True,
                'can_increase_complexity': True,
                'maintain_current_approach': True
            }
            
        return adaptation_signals
    
    def update_user_model(self, current_ability: float, task_difficulty: float, 
                         performance_score: float, interaction_data: Dict[str, Any] = None,
                         context_state: Dict[str, Any] = None, 
                         response_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main feedback processing pipeline integrating all components
        
        Args:
            current_ability: Current user ability estimate
            task_difficulty: Difficulty of completed task
            performance_score: User performance score [0, 1]
            interaction_data: Optional interaction metadata
            context_state: Optional current context state
            response_data: Optional response generation data
            
        Returns:
            Complete feedback update package
        """
        # Default values for optional parameters
        if interaction_data is None:
            interaction_data = {
                'task_completion_rate': performance_score,
                'response_quality': performance_score,
                'interaction_duration_minutes': 5.0,
                'positive_feedback_count': 1 if performance_score > 0.6 else 0,
                'total_feedback_count': 1
            }
            
        if context_state is None:
            context_state = {
                'context_dimensions': {
                    'cognitive_state': 0.5,
                    'emotional_state': 0.5,
                    'engagement_level': performance_score
                },
                'valence_arousal': {'valence': 0.0, 'arousal': 0.0},
                'attention_weights': {'behavioral': 0.25, 'voice': 0.25, 'performance': 0.25, 'temporal': 0.25}
            }
            
        if response_data is None:
            response_data = {
                'context_alignment': 0.5,
                'success_probability': 0.6
            }
        
        # Update user ability (Equation 10)
        updated_ability = self.update_user_ability(
            current_ability, task_difficulty, performance_score, context_state
        )
        
        # Compute engagement score (Equations 11 & 22)
        engagement_score = self.compute_engagement_score(
            interaction_data, context_state, response_data
        )
        
        # Update long-term engagement tracking (Equation 23)
        long_term_engagement = self.update_long_term_engagement(engagement_score)
        
        # Compute adaptation signals
        adaptation_signals = self.compute_adaptation_signals(
            engagement_score, updated_ability, context_state
        )
        
        # Store complete interaction record
        interaction_record = {
            'timestamp': len(self.user_history['ability_estimates']),
            'ability_before': current_ability,
            'ability_after': updated_ability,
            'task_difficulty': task_difficulty,
            'performance_score': performance_score,
            'engagement_score': engagement_score,
            'long_term_engagement': long_term_engagement
        }
        
        self.user_history['performance_metrics'].append(interaction_record)
        self.user_history['context_states'].append(context_state)
        
        # Create feedback package
        feedback_package = {
            'updated_ability': updated_ability,
            'ability_change': updated_ability - current_ability,
            'engagement': engagement_score,
            'long_term_engagement': long_term_engagement,
            'adaptation_signals': adaptation_signals,
            'interaction_record': interaction_record,
            'system_performance': {
                'ability_tracking_confidence': self._compute_tracking_confidence(),
                'engagement_trend': self._compute_engagement_trend(),
                'adaptation_effectiveness': self._compute_adaptation_effectiveness()
            }
        }
        
        return feedback_package
    
    def _compute_tracking_confidence(self) -> float:
        """
        Compute confidence in ability tracking based on stability of estimates
        
        Returns:
            Confidence score [0, 1]
        """
        if len(self.user_history['ability_estimates']) < 3:
            return 0.5  # Insufficient data
            
        recent_estimates = self.user_history['ability_estimates'][-5:]  # Last 5 estimates
        variance = np.var(recent_estimates)
        
        # Lower variance indicates higher confidence
        confidence = max(0.0, min(1.0, 1.0 - variance * 5.0))
        
        return confidence
    
    def _compute_engagement_trend(self) -> str:
        """
        Compute trend in engagement scores
        
        Returns:
            Trend description: 'increasing', 'decreasing', 'stable'
        """
        if len(self.user_history['engagement_scores']) < 3:
            return 'stable'
            
        recent_scores = self.user_history['engagement_scores'][-5:]
        
        # Simple linear trend computation
        x = np.arange(len(recent_scores))
        coeffs = np.polyfit(x, recent_scores, 1)
        slope = coeffs[0]
        
        if slope > 0.05:
            return 'increasing'
        elif slope < -0.05:
            return 'decreasing'
        else:
            return 'stable'
    
    def _compute_adaptation_effectiveness(self) -> float:
        """
        Compute how effective system adaptations have been
        
        Returns:
            Effectiveness score [0, 1]
        """
        if len(self.user_history['engagement_scores']) < 5:
            return 0.5
            
        # Compare recent engagement to earlier engagement
        early_engagement = np.mean(self.user_history['engagement_scores'][:3])
        recent_engagement = np.mean(self.user_history['engagement_scores'][-3:])
        
        # Effectiveness is improvement in engagement
        improvement = recent_engagement - early_engagement
        effectiveness = max(0.0, min(1.0, 0.5 + improvement))
        
        return effectiveness
    
    def get_user_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary of user model state
        
        Returns:
            Dictionary containing user model summary
        """
        if not self.user_history['ability_estimates']:
            return {'status': 'no_data'}
            
        current_ability = self.user_history['ability_estimates'][-1]
        current_engagement = self.user_history['engagement_scores'][-1] if self.user_history['engagement_scores'] else 0.5
        
        summary = {
            'current_ability': current_ability,
            'current_engagement': current_engagement,
            'total_interactions': len(self.user_history['ability_estimates']),
            'ability_trend': self._compute_engagement_trend(),
            'tracking_confidence': self._compute_tracking_confidence(),
            'adaptation_effectiveness': self._compute_adaptation_effectiveness(),
            'long_term_engagement': self.update_long_term_engagement(current_engagement) if current_engagement else 0.5
        }
        
        return summary
