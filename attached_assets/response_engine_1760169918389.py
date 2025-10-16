"""
Response Engine - Stage III: Adaptive Response Strategy
Implements Equations 7-9, 19-21 for IRT-based difficulty adjustment and response generation
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import json
from utils import sigmoid, item_response_probability, generate_persona_response

class ResponseEngine:
    """
    Adaptive response generation engine using Item Response Theory (IRT) 
    and persona-specific response strategies.
    """
    
    def __init__(self):
        """Initialize response engine with IRT parameters and response templates"""
        
        # CORRECTED: IRT model parameters per LaTeX spec
        self.irt_parameters = {
            'a': 1.0,      # FIXED: Discrimination parameter (was 1.5)
            'eta': 0.1,    # Learning rate η for difficulty adjustment
            'P_star': 0.7  # Target success probability P*
        }
        
        # Adaptive pacing parameters (LaTeX spec)
        self.pacing_params = {
            't_base': 2.5,  # Base delay (seconds)
            'gamma': 0.8,   # Cognitive load sensitivity γ
            't_min': 1.5,   # Minimum delay
            't_max': 5.0    # Maximum delay
        }
        
        # Track difficulty history
        self.difficulty_history = []
        
        # Persona-specific response strategies (Equation 20)
        self.response_strategies = {
            'Teacher': {
                'instruction_weight': 0.8,
                'encouragement_weight': 0.6,
                'difficulty_adaptation_rate': 0.3,
                'feedback_detail_level': 0.8
            },
            'Companion': {
                'social_weight': 0.9,
                'emotional_support_weight': 0.8, 
                'difficulty_adaptation_rate': 0.1,
                'conversation_continuity': 0.7
            },
            'Coach': {
                'motivation_weight': 0.9,
                'challenge_weight': 0.7,
                'difficulty_adaptation_rate': 0.4,
                'goal_orientation': 0.8
            }
        }
        
        # Response content templates for different scenarios
        self.response_templates = {
            'Teacher': {
                'high_performance': [
                    "Excellent work! You're demonstrating strong {skill_area}. Let's try something a bit more challenging.",
                    "Well done! Your {performance_metric} shows great improvement. Ready for the next level?",
                    "Outstanding! I can see you've mastered this concept. Time for a new learning adventure."
                ],
                'moderate_performance': [
                    "You're making good progress! Let's practice this concept a bit more before moving forward.",
                    "Nice effort! I think with a little more practice, you'll have this mastered.",
                    "Good work! Let me explain this part differently to help clarify the concept."
                ],
                'low_performance': [
                    "No worries, learning takes time. Let's break this down into smaller, easier steps.",
                    "That's okay! Let's try a simpler approach to this problem.",
                    "Don't worry about getting it perfect. Let's focus on understanding the basics first."
                ]
            },
            'Companion': {
                'positive_mood': [
                    "You seem to be in great spirits today! What's been making you happy?",
                    "I love seeing you so upbeat! Would you like to share what's on your mind?",
                    "Your positive energy is wonderful! How has your day been going?"
                ],
                'neutral_mood': [
                    "How are you feeling today? I'm here to chat about whatever interests you.",
                    "It's nice to spend time with you. What would you like to talk about?",
                    "I'm enjoying our time together. Is there anything special you'd like to discuss?"
                ],
                'subdued_mood': [
                    "I'm here for you. Would you like to talk about how you're feeling?",
                    "Sometimes it helps to share what's on your mind. I'm listening.",
                    "I notice you might be feeling a bit down. I'm here to support you."
                ]
            },
            'Coach': {
                'high_engagement': [
                    "Fantastic energy! You're really pushing yourself today. Let's set an even bigger goal!",
                    "I love your determination! You're showing real growth. What challenge should we tackle next?",
                    "Your commitment is impressive! Let's channel this motivation into something exciting."
                ],
                'moderate_engagement': [
                    "You're doing well! Let's find what really motivates you and build on that.",
                    "Good effort! I believe you have more potential to unlock. What drives you?",
                    "Nice work! Let's discover what gets you most excited and energized."
                ],
                'low_engagement': [
                    "Every journey starts with a single step. What small goal can we achieve today?",
                    "Let's find something that sparks your interest. What activities do you enjoy?",
                    "No pressure! Let's start with something fun and see where it leads us."
                ]
            }
        }
    
    def compute_difficulty_level(self, user_ability: float, context_state: Dict[str, Any], 
                                persona: str) -> float:
        """
        Compute adaptive difficulty using IRT model (Equation 7 & 19)
        P(θ, β) = γ + (1-γ) * sigmoid(α*(θ - β))
        
        Args:
            user_ability: Current user ability estimate (θ)
            context_state: Current context from context hub
            persona: Selected AI persona
            
        Returns:
            Optimal difficulty level for current context
        """
        # Extract context-based difficulty modifiers
        cognitive_state = context_state['context_dimensions']['cognitive_state']
        engagement_level = context_state['context_dimensions']['engagement_level']
        difficulty_preference = context_state['context_dimensions']['difficulty_preference']
        
        # Get persona-specific adaptation rate
        adaptation_rate = self.response_strategies[persona]['difficulty_adaptation_rate']
        
        # Base difficulty from IRT optimal challenge point
        # Optimal difficulty is slightly above current ability (zone of proximal development)
        base_difficulty = user_ability + 0.2
        
        # Context-based adjustments
        cognitive_adjustment = (cognitive_state - 0.5) * 0.3  # ±0.15 max
        engagement_adjustment = (engagement_level - 0.5) * 0.2  # ±0.1 max  
        preference_adjustment = (difficulty_preference - 0.5) * 0.4  # ±0.2 max
        
        # Valence-arousal adjustments
        valence = context_state['valence_arousal']['valence']
        arousal = context_state['valence_arousal']['arousal']
        
        # Higher positive valence allows for more challenge
        valence_adjustment = valence * 0.1
        # Moderate arousal is optimal; too high or low reduces difficulty
        arousal_adjustment = -abs(arousal) * 0.1
        
        # Combine all adjustments
        total_adjustment = (
            cognitive_adjustment + 
            engagement_adjustment + 
            preference_adjustment +
            valence_adjustment + 
            arousal_adjustment
        ) * adaptation_rate
        
        # Compute final difficulty
        difficulty = base_difficulty + total_adjustment
        
        # Constrain to reasonable bounds [0.1, 0.9]
        difficulty = max(0.1, min(0.9, difficulty))
        
        return difficulty
    
    def compute_response_probability(self, user_ability: float, difficulty: float) -> float:
        """
        CORRECTED: Compute success probability using IRT model (Equation 8)
        P_t = 1/(1 + exp{-a(θ_t - b_t)})
        
        Args:
            user_ability: User's current ability level (θ)
            difficulty: Task difficulty level (β)
            
        Returns:
            Probability of successful response
        """
        return item_response_probability(
            user_ability, 
            difficulty,
            self.irt_parameters['a'],  # FIXED: Using a=1.0
            0.0  # No guessing parameter in LaTeX spec
        )
    
    def update_difficulty(self, b_current: float, P_observed: float) -> float:
        """
        ADDED: Difficulty adaptation per LaTeX spec
        b_{t+1} = clip[0,3](b_t + η(P_t - P*))
        
        Args:
            b_current: Current difficulty b_t
            P_observed: Observed success probability P_t
        
        Returns:
            Updated difficulty b_{t+1} ∈ [0, 3]
        """
        eta = self.irt_parameters['eta']
        P_star = self.irt_parameters['P_star']
        
        # Compute adjustment
        adjustment = eta * (P_observed - P_star)
        
        # Update difficulty
        b_next = b_current + adjustment
        
        # Clip to bounds [0, 3]
        b_next = np.clip(b_next, 0.0, 3.0)
        
        # Store in history
        self.difficulty_history.append(b_next)
        
        return b_next
    
    def compute_adaptive_pacing(self, L_cog: float) -> float:
        """
        ADDED: Adaptive pacing per LaTeX spec
        t_delay = clip[1.5,5.0](2.5 × max(0.5, 1 + 0.8·L_cog))
        
        Args:
            L_cog: Cognitive load ∈ [0, 2]
        
        Returns:
            Response delay t_delay in seconds ∈ [1.5, 5.0]
        """
        t_base = self.pacing_params['t_base']
        gamma = self.pacing_params['gamma']
        t_min = self.pacing_params['t_min']
        t_max = self.pacing_params['t_max']
        
        # Compute delay multiplier
        multiplier = max(0.5, 1.0 + gamma * L_cog)
        
        # Compute delay
        t_delay = t_base * multiplier
        
        # Clip to bounds
        t_delay = np.clip(t_delay, t_min, t_max)
        
        return t_delay
    
    def select_response_strategy(self, context_state: Dict[str, Any], 
                               persona: str, difficulty: float) -> Dict[str, Any]:
        """
        Select appropriate response strategy based on context and persona (Equation 9 & 21)
        π(Ct, Ht) → at
        
        Args:
            context_state: Current user context
            persona: Selected AI persona
            difficulty: Computed difficulty level
            
        Returns:
            Response strategy configuration
        """
        # Extract context dimensions
        cognitive_state = context_state['context_dimensions']['cognitive_state']
        emotional_state = context_state['context_dimensions']['emotional_state'] 
        engagement_level = context_state['context_dimensions']['engagement_level']
        
        valence = context_state['valence_arousal']['valence']
        arousal = context_state['valence_arousal']['arousal']
        
        # Determine performance category based on context
        if cognitive_state > 0.6 and engagement_level > 0.6:
            performance_category = 'high_performance'
        elif cognitive_state > 0.3 and engagement_level > 0.3:
            performance_category = 'moderate_performance'  
        else:
            performance_category = 'low_performance'
            
        # Determine mood category for Companion
        if valence > 0.3:
            mood_category = 'positive_mood'
        elif valence > -0.3:
            mood_category = 'neutral_mood'
        else:
            mood_category = 'subdued_mood'
            
        # Determine engagement category for Coach
        if engagement_level > 0.6:
            engagement_category = 'high_engagement'
        elif engagement_level > 0.3:
            engagement_category = 'moderate_engagement'
        else:
            engagement_category = 'low_engagement'
            
        # Select response strategy based on persona
        if persona == 'Teacher':
            strategy_key = performance_category
            response_style = 'instructional'
        elif persona == 'Companion':
            strategy_key = mood_category
            response_style = 'conversational'
        else:  # Coach
            strategy_key = engagement_category
            response_style = 'motivational'
            
        strategy = {
            'persona': persona,
            'strategy_key': strategy_key,
            'response_style': response_style,
            'difficulty_level': difficulty,
            'context_summary': context_state['context_summary'],
            'adaptation_parameters': self.response_strategies[persona]
        }
        
        return strategy
    
    def generate_response_content(self, strategy: Dict[str, Any], 
                                context_state: Dict[str, Any]) -> str:
        """
        Generate actual response content using selected strategy
        
        Args:
            strategy: Response strategy configuration
            context_state: Current user context
            
        Returns:
            Generated response text
        """
        persona = strategy['persona']
        strategy_key = strategy['strategy_key']
        
        # Get appropriate template
        templates = self.response_templates[persona][strategy_key]
        
        # Select template based on context (simple selection for demo)
        template_index = int(context_state['context_dimensions']['attention_focus'] * len(templates))
        template_index = min(template_index, len(templates) - 1)
        
        selected_template = templates[template_index]
        
        # Generate context-aware content
        response_content = generate_persona_response(
            template=selected_template,
            context_state=context_state,
            strategy=strategy
        )
        
        return response_content
    
    def generate_response(self, context_state: Dict[str, Any], persona: str, 
                         user_ability: float, base_difficulty: float) -> Dict[str, Any]:
        """
        Main response generation pipeline integrating all components
        
        Args:
            context_state: Current context from context hub
            persona: Selected AI persona
            user_ability: Current user ability estimate
            base_difficulty: Base difficulty preference
            
        Returns:
            Complete response package with content and metadata
        """
        # Compute adaptive difficulty (Equation 7)
        adaptive_difficulty = self.compute_difficulty_level(
            user_ability, context_state, persona
        )
        
        # Blend with base difficulty preference
        final_difficulty = 0.7 * adaptive_difficulty + 0.3 * base_difficulty
        
        # Compute success probability (Equation 8)
        success_probability = self.compute_response_probability(
            user_ability, final_difficulty
        )
        
        # Select response strategy (Equation 9)
        response_strategy = self.select_response_strategy(
            context_state, persona, final_difficulty
        )
        
        # Generate response content
        response_content = self.generate_response_content(
            response_strategy, context_state
        )
        
        # Create complete response package
        response_package = {
            'content': response_content,
            'persona': persona,
            'difficulty': final_difficulty,
            'success_probability': success_probability,
            'strategy': response_strategy,
            'context_alignment': self._compute_context_alignment(context_state, response_strategy),
            'adaptation_info': {
                'base_difficulty': base_difficulty,
                'adaptive_difficulty': adaptive_difficulty,
                'final_difficulty': final_difficulty,
                'ability_difficulty_gap': user_ability - final_difficulty
            }
        }
        
        return response_package
    
    def _compute_context_alignment(self, context_state: Dict[str, Any], 
                                  strategy: Dict[str, Any]) -> float:
        """
        Compute how well the response strategy aligns with current context
        
        Args:
            context_state: Current user context
            strategy: Selected response strategy
            
        Returns:
            Alignment score [0, 1]
        """
        # Simple alignment computation based on context-strategy matching
        persona = strategy['persona']
        
        if persona == 'Teacher':
            # Teacher alignment based on cognitive state and learning readiness
            cognitive_alignment = context_state['context_dimensions']['cognitive_state']
            engagement_alignment = context_state['context_dimensions']['engagement_level']
            alignment = (cognitive_alignment + engagement_alignment) / 2
            
        elif persona == 'Companion':
            # Companion alignment based on social need and emotional state
            social_alignment = context_state['context_dimensions']['social_need']
            emotional_alignment = (context_state['context_dimensions']['emotional_state'] + 1) / 2  # Normalize
            alignment = (social_alignment + emotional_alignment) / 2
            
        else:  # Coach
            # Coach alignment based on engagement and motivation potential
            engagement_alignment = context_state['context_dimensions']['engagement_level']
            # High arousal can indicate readiness for motivation
            arousal_alignment = min(1.0, abs(context_state['valence_arousal']['arousal']) + 0.5)
            alignment = (engagement_alignment + arousal_alignment) / 2
        
        return float(alignment)
    
    def update_response_parameters(self, feedback_score: float, persona: str,
                                  context_state: Dict[str, Any]):
        """
        Update response generation parameters based on user feedback
        
        Args:
            feedback_score: User satisfaction score [0, 1]
            persona: AI persona being updated
            context_state: Context when response was generated
        """
        learning_rate = 0.05
        
        # Update persona-specific strategy weights based on feedback
        if feedback_score > 0.7:
            # Positive feedback: reinforce current strategy weights
            for param in self.response_strategies[persona]:
                if isinstance(self.response_strategies[persona][param], float):
                    current_value = self.response_strategies[persona][param]
                    self.response_strategies[persona][param] += learning_rate * current_value * 0.1
                    
        elif feedback_score < 0.3:
            # Negative feedback: slightly adjust strategy weights
            for param in self.response_strategies[persona]:
                if isinstance(self.response_strategies[persona][param], float):
                    noise = np.random.normal(0, 0.02)
                    self.response_strategies[persona][param] += noise
                    
        # Ensure weights stay within reasonable bounds
        for param in self.response_strategies[persona]:
            if isinstance(self.response_strategies[persona][param], float):
                self.response_strategies[persona][param] = max(
                    0.1, min(1.0, self.response_strategies[persona][param])
                )
