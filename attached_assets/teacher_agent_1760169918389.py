"""
TeacherAgent - Specialized agent for educational guidance and cognitive load management
Estimates Lcog from performance metrics and adjusts task difficulty adaptively
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from utils import sigmoid, softmax

class TeacherAgent:
    """
    Teacher persona agent that focuses on cognitive activation through
    educational tasks with adaptive difficulty adjustment.
    """
    
    def __init__(self):
        """Initialize TeacherAgent with cognitive load parameters"""
        
        # Cognitive load estimation parameters
        self.lcog_weights = {
            'accuracy': 0.4,
            'response_time': 0.3,
            'consistency': 0.2,
            'complexity': 0.1
        }
        
        # Difficulty adjustment thresholds
        self.difficulty_thresholds = {
            'too_easy': 0.3,      # Lcog < 0.3: increase difficulty
            'optimal': (0.3, 0.7), # 0.3 <= Lcog <= 0.7: maintain
            'too_hard': 0.7       # Lcog > 0.7: decrease difficulty
        }
        
        # Response templates based on cognitive load
        self.response_templates = {
            'low_load': [
                "Great job! Let's try something more challenging to keep you engaged.",
                "You're doing excellent! Ready for the next level?",
                "Perfect! Let's explore a more complex task together."
            ],
            'optimal_load': [
                "You're doing well! This level seems just right for you.",
                "Excellent progress! Keep going at this pace.",
                "Nice work! You're in the optimal learning zone."
            ],
            'high_load': [
                "Let's take it one step at a time. I'm here to help.",
                "This is challenging, but you're making progress. Let's try a simpler approach.",
                "Good effort! Let's break this down into smaller steps."
            ]
        }
        
        # Task difficulty levels
        self.difficulty_levels = {
            'beginner': 0.3,
            'intermediate': 0.5,
            'advanced': 0.7,
            'expert': 0.9
        }
        
    def estimate_cognitive_load(self, performance_metrics: Dict[str, float]) -> float:
        """
        Estimate cognitive load (Lcog) from performance metrics
        Lcog = Σ(wi * mi) where mi are normalized metrics
        
        Args:
            performance_metrics: Dict with accuracy, response_time, etc.
            
        Returns:
            Cognitive load estimate in [0, 1] range
        """
        # Extract and normalize metrics
        accuracy = performance_metrics.get('accuracy', 0.5)
        response_time = performance_metrics.get('response_time', 0.5)
        consistency = performance_metrics.get('consistency', 0.5)
        complexity = performance_metrics.get('task_complexity', 0.5)
        
        # Normalize response time (inverse: faster = lower load)
        normalized_time = 1.0 - min(response_time, 1.0)
        
        # Compute weighted cognitive load
        lcog = (
            self.lcog_weights['accuracy'] * (1.0 - accuracy) +  # Lower accuracy = higher load
            self.lcog_weights['response_time'] * response_time +
            self.lcog_weights['consistency'] * (1.0 - consistency) +
            self.lcog_weights['complexity'] * complexity
        )
        
        # Ensure in valid range
        lcog = np.clip(lcog, 0.0, 1.0)
        
        return lcog
        
    def adjust_difficulty(self, lcog: float, current_difficulty: float) -> Tuple[float, str]:
        """
        Adjust task difficulty based on cognitive load
        
        Args:
            lcog: Current cognitive load estimate
            current_difficulty: Current difficulty level [0, 1]
            
        Returns:
            Tuple of (new_difficulty, adjustment_reason)
        """
        adjustment_step = 0.1
        
        if lcog < self.difficulty_thresholds['too_easy']:
            # Increase difficulty
            new_difficulty = min(current_difficulty + adjustment_step, 1.0)
            reason = "increasing_challenge"
        elif lcog > self.difficulty_thresholds['too_hard']:
            # Decrease difficulty
            new_difficulty = max(current_difficulty - adjustment_step, 0.1)
            reason = "reducing_complexity"
        else:
            # Maintain current level
            new_difficulty = current_difficulty
            reason = "optimal_zone"
            
        return new_difficulty, reason
        
    def generate_response(self, lcog: float, context: Dict[str, Any]) -> str:
        """
        Generate educational response based on cognitive load
        
        Args:
            lcog: Cognitive load estimate
            context: Additional context information
            
        Returns:
            Response text
        """
        # Select response category based on Lcog
        if lcog < self.difficulty_thresholds['too_easy']:
            templates = self.response_templates['low_load']
        elif lcog > self.difficulty_thresholds['too_hard']:
            templates = self.response_templates['high_load']
        else:
            templates = self.response_templates['optimal_load']
            
        # Select template (could be based on context or random)
        response = np.random.choice(templates)
        
        # Add task suggestion based on difficulty
        task_type = context.get('task_type', 'memory task')
        if lcog < 0.3:
            response += f" Let's try a more advanced {task_type}."
        elif lcog > 0.7:
            response += f" Let's practice with a simpler {task_type} first."
        else:
            response += f" Continue with the current {task_type}."
            
        return response
        
    def process_input(self, xt: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main processing method for TeacherAgent
        
        Args:
            xt: Input feature vector from perception engine
            context: Optional context dictionary with additional info
            
        Returns:
            Dict containing response, metadata, and contextual triggers
        """
        if context is None:
            context = {}
            
        # Extract performance metrics from input features
        # Assuming features contain performance data
        performance_metrics = {
            'accuracy': float(xt[14]) if len(xt) > 14 else 0.5,
            'response_time': float(xt[15]) if len(xt) > 15 else 0.5,
            'consistency': float(xt[16]) if len(xt) > 16 else 0.5,
            'task_complexity': float(xt[17]) if len(xt) > 17 else 0.5
        }
        
        # Estimate cognitive load
        lcog = self.estimate_cognitive_load(performance_metrics)
        
        # Get current difficulty
        current_difficulty = context.get('difficulty_level', 0.5)
        
        # Adjust difficulty based on Lcog
        new_difficulty, adjustment_reason = self.adjust_difficulty(lcog, current_difficulty)
        
        # Generate response
        response_text = self.generate_response(lcog, context)
        
        # Compile metadata
        metadata = {
            'agent': 'Teacher',
            'cognitive_load': float(lcog),
            'difficulty_adjustment': {
                'from': float(current_difficulty),
                'to': float(new_difficulty),
                'reason': adjustment_reason
            },
            'performance_metrics': performance_metrics,
            'priority_score': float(lcog)  # Higher Lcog = higher priority for teacher intervention
        }
        
        # Contextual trigger information
        trigger = f"Lcog={lcog:.2f}, Difficulty: {current_difficulty:.1f}→{new_difficulty:.1f}"
        
        return {
            'response': response_text,
            'metadata': metadata,
            'trigger': trigger,
            'new_difficulty': new_difficulty
        }
