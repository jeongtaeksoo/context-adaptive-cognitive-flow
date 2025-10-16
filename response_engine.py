"""
Response Engine - Stage III: Adaptive Response Generation
Implements Equations 7-9 for IRT-based difficulty adjustment and response generation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from utils import sigmoid, irt_probability, adaptive_difficulty

class ResponseEngine:
    """
    Adaptive response generation engine using Item Response Theory (IRT)
    for difficulty adjustment and context-aware response selection.
    """
    
    def __init__(self):
        """Initialize response engine with IRT parameters and response templates"""
        
        # IRT model parameters (Equation 7)
        self.irt_params = {
            'discrimination': 1.0,  # a parameter - item discrimination
            'difficulty_base': 0.0,  # b parameter base - item difficulty 
            'guessing': 0.0,       # c parameter - guessing (3PL model)
            'upper_asymptote': 1.0  # d parameter - upper asymptote (4PL model)
        }
        
        # Cognitive load thresholds (Equation 8)
        self.load_thresholds = {
            'low': 0.3,      # Easy tasks
            'moderate': 0.6,  # Moderate tasks  
            'high': 0.8      # Challenging tasks
        }
        
        # Response adaptation parameters (Equation 9)
        self.adaptation_params = {
            'difficulty_step': 0.1,     # How much to adjust difficulty
            'min_difficulty': 0.1,      # Minimum difficulty level
            'max_difficulty': 0.9,      # Maximum difficulty level
            'target_success_rate': 0.7  # Optimal success probability
        }
        
        # Persona-specific response templates
        self.response_templates = {
            'teacher': {
                'high_performance': [
                    "Excellent work! Let's try something a bit more challenging to keep you engaged.",
                    "You're doing great! I think you're ready for the next level.",
                    "Outstanding! Your accuracy suggests we can increase the complexity."
                ],
                'moderate_performance': [
                    "Good job! Let's practice a bit more at this level to build confidence.",
                    "You're making progress. Let's consolidate these skills before moving forward.",
                    "Nice work! Let's try a few more exercises at this difficulty."
                ],
                'low_performance': [
                    "That's okay - let's break this down into smaller steps.",
                    "No worries! Let me simplify this task so you can succeed.",
                    "Let's try an easier approach that builds on what you know."
                ]
            },
            'companion': {
                'positive_emotion': [
                    "I can hear the joy in your voice! It's wonderful to see you engaged.",
                    "You sound happy today - that's great! How are you feeling overall?",
                    "Your positive energy is contagious! Tell me more about what's going well."
                ],
                'neutral_emotion': [
                    "How are you feeling today? I'm here to listen and support you.",
                    "It's good to spend time together. What would you like to talk about?",
                    "I'm here with you. Is there anything on your mind you'd like to share?"
                ],
                'negative_emotion': [
                    "I can sense you might be having a difficult time. I'm here for you.",
                    "It's completely normal to have challenging days. You're not alone.",
                    "I hear that things might be tough right now. Would you like to talk about it?"
                ]
            },
            'coach': {
                'high_activity': [
                    "I love seeing your activity level! Keep up this great momentum.",
                    "You're moving well today! This energy will serve you well in other areas too.",
                    "Fantastic activity pattern! Your body and mind are both benefiting."
                ],
                'moderate_activity': [
                    "Good movement today. What if we added just a little more activity?",
                    "You're doing well with staying active. Could we try one more small activity?",
                    "Nice job staying engaged! A brief walk might feel good right now."
                ],
                'low_activity': [
                    "Today might be a good day to start with something small and gentle.",
                    "Even small movements count! What feels manageable for you right now?",
                    "Let's find a simple activity that feels good and achievable today."
                ]
            }
        }
        
        # Task difficulty database
        self.task_difficulty_db = {
            'memory_recall': {'base_difficulty': 0.4, 'cognitive_load_factor': 0.6},
            'pattern_recognition': {'base_difficulty': 0.5, 'cognitive_load_factor': 0.5},
            'problem_solving': {'base_difficulty': 0.7, 'cognitive_load_factor': 0.8},
            'attention_task': {'base_difficulty': 0.3, 'cognitive_load_factor': 0.4},
            'executive_function': {'base_difficulty': 0.6, 'cognitive_load_factor': 0.7}
        }
        
    def compute_success_probability(self, user_ability: float, task_difficulty: float) -> float:
        """
        Compute probability of success using IRT model (Equation 7)
        P(θ,β) = c + (d-c) / (1 + exp(-a(θ-β)))
        
        Args:
            user_ability: User ability estimate θ
            task_difficulty: Task difficulty parameter β
            
        Returns:
            Probability of successful task completion [0, 1]
        """
        a = self.irt_params['discrimination']
        c = self.irt_params['guessing']  
        d = self.irt_params['upper_asymptote']
        
        # IRT 3-parameter logistic model
        probability = c + (d - c) * sigmoid(a * (user_ability - task_difficulty))
        
        return probability
    
    def estimate_cognitive_load(self, context_state: Dict[str, Any], task_difficulty: float) -> float:
        """
        Estimate cognitive load based on context and task difficulty (Equation 8)
        
        Args:
            context_state: Current context from context hub
            task_difficulty: Proposed task difficulty
            
        Returns:
            Estimated cognitive load [0, 1]
        """
        # Base load from task difficulty
        base_load = task_difficulty
        
        # Context-based adjustments
        context_dims = context_state.get('context_dimensions', {})
        
        # Current cognitive state affects load
        current_cognitive_state = context_dims.get('cognitive_state', 0.5)
        cognitive_modifier = 1.0 - current_cognitive_state
        
        # Engagement level affects perceived load
        engagement = context_dims.get('engagement_level', 0.5)
        engagement_modifier = 1.0 - (engagement * 0.3)  # High engagement reduces perceived load
        
        # Emotional state affects load
        valence_arousal = context_state.get('valence_arousal', {'valence': 0, 'arousal': 0})
        valence = valence_arousal['valence']
        arousal = valence_arousal['arousal']
        
        # Negative emotions increase cognitive load
        emotion_modifier = 1.0 + max(0, -valence * 0.2)
        
        # Very high or low arousal increases load
        arousal_modifier = 1.0 + abs(arousal) * 0.1
        
        # Combine all factors
        estimated_load = base_load * cognitive_modifier * engagement_modifier * emotion_modifier * arousal_modifier
        
        return max(0.0, min(1.0, estimated_load))
    
    def adjust_difficulty(self, current_difficulty: float, user_ability: float, 
                         recent_performance: List[float]) -> float:
        """
        Adapt task difficulty based on user performance (Equation 9)
        
        Args:
            current_difficulty: Current task difficulty level
            user_ability: Current user ability estimate
            recent_performance: List of recent performance scores
            
        Returns:
            Adjusted difficulty level
        """
        if not recent_performance:
            return current_difficulty
            
        # Calculate recent success rate
        recent_success_rate = np.mean(recent_performance)
        target_rate = self.adaptation_params['target_success_rate']
        
        # Determine adjustment direction and magnitude
        if recent_success_rate > target_rate + 0.1:
            # Too easy - increase difficulty
            adjustment = self.adaptation_params['difficulty_step']
        elif recent_success_rate < target_rate - 0.1:
            # Too hard - decrease difficulty  
            adjustment = -self.adaptation_params['difficulty_step']
        else:
            # Just right - small random adjustment for exploration
            adjustment = np.random.normal(0, 0.02)
            
        # Apply adjustment with bounds
        new_difficulty = current_difficulty + adjustment
        new_difficulty = max(self.adaptation_params['min_difficulty'], 
                           min(self.adaptation_params['max_difficulty'], new_difficulty))
        
        # Ensure difficulty is reasonable relative to user ability
        ability_gap = abs(new_difficulty - user_ability)
        if ability_gap > 0.4:  # Too far from ability level
            # Move difficulty closer to ability level
            new_difficulty = user_ability + 0.2 * np.sign(new_difficulty - user_ability)
            
        return new_difficulty
    
    def select_response_template(self, persona: str, context_state: Dict[str, Any], 
                               performance_score: float) -> str:
        """
        Select appropriate response template based on persona and context
        
        Args:
            persona: Active persona ('teacher', 'companion', 'coach')
            context_state: Current context state
            performance_score: Recent performance score
            
        Returns:
            Selected response template
        """
        templates = self.response_templates.get(persona, self.response_templates['teacher'])
        
        if persona == 'teacher':
            # Performance-based selection for teacher
            if performance_score > 0.8:
                category = 'high_performance'
            elif performance_score > 0.5:
                category = 'moderate_performance'  
            else:
                category = 'low_performance'
                
        elif persona == 'companion':
            # Emotion-based selection for companion
            valence = context_state.get('valence_arousal', {}).get('valence', 0)
            if valence > 0.3:
                category = 'positive_emotion'
            elif valence > -0.3:
                category = 'neutral_emotion'
            else:
                category = 'negative_emotion'
                
        elif persona == 'coach':
            # Activity-based selection for coach
            activity_level = context_state.get('context_dimensions', {}).get('engagement_level', 0.5)
            if activity_level > 0.7:
                category = 'high_activity'
            elif activity_level > 0.4:
                category = 'moderate_activity'
            else:
                category = 'low_activity'
        else:
            category = list(templates.keys())[0]  # Fallback
            
        # Randomly select from appropriate category
        selected_responses = templates.get(category, list(templates.values())[0])
        return np.random.choice(selected_responses)
    
    def generate_response(self, context_state: Dict[str, Any], user_ability: float,
                         current_difficulty: float, persona: str = 'teacher',
                         recent_performance: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Main response generation pipeline
        
        Args:
            context_state: Current context from context hub
            user_ability: Current user ability estimate θ
            current_difficulty: Current task difficulty β
            persona: Active persona for response generation
            recent_performance: Optional list of recent performance scores
            
        Returns:
            Complete response package
        """
        if recent_performance is None:
            recent_performance = [0.75]  # Default moderate performance
            
        # Adjust difficulty based on recent performance
        adapted_difficulty = self.adjust_difficulty(current_difficulty, user_ability, recent_performance)
        
        # Compute success probability for adapted task
        success_probability = self.compute_success_probability(user_ability, adapted_difficulty)
        
        # Estimate cognitive load
        cognitive_load = self.estimate_cognitive_load(context_state, adapted_difficulty)
        
        # Select appropriate response template
        latest_performance = recent_performance[-1] if recent_performance else 0.75
        response_text = self.select_response_template(persona, context_state, latest_performance)
        
        # Generate task recommendation
        task_recommendation = self.recommend_task(cognitive_load, adapted_difficulty, context_state)
        
        # Compute response metadata
        response_metadata = {
            'adapted_difficulty': adapted_difficulty,
            'success_probability': success_probability,
            'estimated_cognitive_load': cognitive_load,
            'persona': persona,
            'task_recommendation': task_recommendation,
            'context_alignment': self.compute_context_alignment(context_state, persona),
            'priority_score': self.compute_priority_score(context_state, persona, success_probability)
        }
        
        return {
            'response': response_text,
            'metadata': response_metadata,
            'adapted_difficulty': adapted_difficulty,
            'success_probability': success_probability
        }
    
    def recommend_task(self, cognitive_load: float, difficulty: float, 
                      context_state: Dict[str, Any]) -> str:
        """
        Recommend appropriate task based on cognitive load and context
        
        Args:
            cognitive_load: Estimated cognitive load
            difficulty: Adapted difficulty level
            context_state: Current context state
            
        Returns:
            Task recommendation string
        """
        # Select task type based on cognitive load
        if cognitive_load < self.load_thresholds['low']:
            task_types = ['attention_task', 'memory_recall']
        elif cognitive_load < self.load_thresholds['moderate']:
            task_types = ['pattern_recognition', 'memory_recall']
        else:
            task_types = ['problem_solving', 'executive_function']
            
        # Consider time of day from context
        time_context = context_state.get('attention_weights', {}).get('temporal_time', 0.5)
        if time_context > 0.7:  # Morning - more challenging tasks
            task_types = [t for t in task_types if self.task_difficulty_db[t]['base_difficulty'] > 0.4]
        elif time_context < 0.3:  # Evening - easier tasks
            task_types = [t for t in task_types if self.task_difficulty_db[t]['base_difficulty'] < 0.6]
            
        if not task_types:  # Fallback
            task_types = ['memory_recall']
            
        selected_task = np.random.choice(task_types)
        return f"{selected_task.replace('_', ' ').title()} (Difficulty: {difficulty:.1f})"
    
    def compute_context_alignment(self, context_state: Dict[str, Any], persona: str) -> float:
        """
        Compute how well the response aligns with current context
        
        Args:
            context_state: Current context state
            persona: Active persona
            
        Returns:
            Context alignment score [0, 1]
        """
        # Base alignment based on persona appropriateness
        context_dims = context_state.get('context_dimensions', {})
        
        if persona == 'teacher':
            # Teacher most appropriate when cognitive state is good
            cognitive_state = context_dims.get('cognitive_state', 0.5)
            alignment = cognitive_state
            
        elif persona == 'companion':
            # Companion most appropriate for emotional support
            valence = context_state.get('valence_arousal', {}).get('valence', 0)
            # High alignment when negative emotions (need support) or very positive (share joy)
            alignment = max(0.5, 1.0 - abs(valence))
            
        elif persona == 'coach':
            # Coach most appropriate when activity/engagement could be improved
            engagement = context_dims.get('engagement_level', 0.5)
            # High alignment when engagement is low or moderate (room for improvement)
            alignment = 1.0 - engagement if engagement < 0.7 else 0.7
        else:
            alignment = 0.5
            
        # Adjust for context confidence
        confidence = context_state.get('context_confidence', 0.5)
        alignment *= confidence
        
        return max(0.1, min(1.0, alignment))
    
    def compute_priority_score(self, context_state: Dict[str, Any], persona: str, 
                             success_probability: float) -> float:
        """
        Compute priority score for response selection
        
        Args:
            context_state: Current context state
            persona: Active persona
            success_probability: Computed success probability
            
        Returns:
            Priority score [0, 1]
        """
        # Base priority from context alignment
        context_alignment = self.compute_context_alignment(context_state, persona)
        
        # Bonus for optimal success probability
        optimal_success = self.adaptation_params['target_success_rate']
        success_score = 1.0 - abs(success_probability - optimal_success)
        
        # Context confidence bonus
        confidence = context_state.get('context_confidence', 0.5)
        
        # Weighted combination
        priority = (0.5 * context_alignment + 
                   0.3 * success_score + 
                   0.2 * confidence)
        
        return max(0.0, min(1.0, priority))
