"""
Teacher Agent - Specialized agent focusing on cognitive activation and performance-based adaptation
Emphasizes difficulty adjustment and educational guidance
"""

import numpy as np
from typing import Dict, List, Any, Optional
from utils import sigmoid, irt_probability

class TeacherAgent:
    """
    Teacher agent specialized in cognitive activation and adaptive difficulty management.
    Focuses on performance metrics and educational progression.
    """
    
    def __init__(self):
        """Initialize teacher agent with educational parameters"""
        
        # Teacher-specific priority weights
        self.priority_weights = {
            'performance_focus': 0.4,     # Weight on performance metrics
            'cognitive_load': 0.3,        # Weight on cognitive state
            'difficulty_appropriateness': 0.2,  # Weight on task difficulty match
            'learning_progression': 0.1   # Weight on learning trajectory
        }
        
        # Performance thresholds for different response types
        self.performance_thresholds = {
            'excellent': 0.9,    # Exceptional performance
            'good': 0.7,         # Good performance
            'moderate': 0.5,     # Moderate performance
            'needs_support': 0.3  # Needs additional support
        }
        
        # Cognitive load management
        self.cognitive_load_targets = {
            'optimal': (0.4, 0.7),      # Sweet spot for learning
            'too_easy': (0.0, 0.3),     # Under-challenged
            'too_hard': (0.8, 1.0)      # Over-challenged
        }
        
        # Response templates for different scenarios
        self.response_templates = {
            'excellent_performance': [
                "Outstanding work! Your performance shows you're ready for more complex challenges. Let's explore advanced concepts that will keep you engaged.",
                "Exceptional! You've mastered this level brilliantly. Time to unlock new cognitive territories - are you ready for the next adventure?",
                "Wow! Your accuracy and speed are impressive. Let's channel this momentum into exploring more sophisticated problems."
            ],
            'good_performance': [
                "Great job! You're demonstrating solid understanding. Let's build on this foundation with some interesting variations.",
                "Well done! Your consistent performance shows real learning is happening. Ready to take on a slightly more challenging version?",
                "Excellent progress! You're clearly grasping the concepts. Let's add one more layer of complexity to keep growing."
            ],
            'moderate_performance': [
                "Good effort! You're on the right track. Let's practice this approach a bit more to build your confidence.",
                "You're making progress! Sometimes learning happens in steps. Let's reinforce these skills before moving forward.",
                "Nice work! I can see you're thinking through this carefully. Let's consolidate what you've learned with a few more examples."
            ],
            'needs_support': [
                "That's perfectly okay - learning is a process! Let me break this down into smaller, more manageable pieces.",
                "No worries at all! Sometimes we need to approach things differently. Let me show you another way to think about this.",
                "Don't be discouraged - every expert was once a beginner! Let's simplify this and build up your confidence step by step."
            ],
            'cognitive_overload': [
                "I can see this might be feeling overwhelming. Let's take a step back and focus on the core concepts first.",
                "It looks like we might be trying to cover too much at once. Let me simplify this task so you can succeed.",
                "Let's reduce the complexity here. Sometimes the best learning happens when we focus on mastering one thing at a time."
            ],
            'under_challenged': [
                "You're handling this with ease! Your brain is ready for more stimulation. Let's add some interesting complexity.",
                "This seems too simple for your abilities! Let me present you with a puzzle that will really engage your mind.",
                "You're clearly capable of more! Time to stretch those cognitive muscles with a more demanding challenge."
            ]
        }
        
    def analyze_performance_pattern(self, feature_vector: np.ndarray, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze performance patterns from input features
        
        Args:
            feature_vector: Input features from perception engine
            context: Context information
            
        Returns:
            Performance analysis metrics
        """
        # Extract performance-related features (indices 6-8: speed, accuracy, cognitive load)
        performance_features = feature_vector[6:9]
        
        # Compute performance metrics
        response_speed = performance_features[0]      # Higher = faster response
        accuracy = performance_features[1]            # Higher = more accurate
        cognitive_efficiency = performance_features[2]  # Higher = less cognitive load
        
        # Overall performance score
        overall_performance = np.mean([response_speed, accuracy, cognitive_efficiency])
        
        # Performance consistency (based on feature variance)
        consistency = 1.0 - np.var(performance_features)
        consistency = max(0.0, consistency)
        
        # Learning indicator (combination of accuracy and efficiency)
        learning_indicator = (accuracy * 0.7 + cognitive_efficiency * 0.3)
        
        return {
            'overall_performance': float(overall_performance),
            'accuracy': float(accuracy),
            'response_speed': float(response_speed),
            'cognitive_efficiency': float(cognitive_efficiency),
            'consistency': float(consistency),
            'learning_indicator': float(learning_indicator)
        }
    
    def assess_cognitive_load(self, feature_vector: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess current cognitive load and appropriateness
        
        Args:
            feature_vector: Input features
            context: Context information
            
        Returns:
            Cognitive load assessment
        """
        # Extract cognitive load indicators
        performance_features = feature_vector[6:9]
        behavioral_features = feature_vector[0:3]  # Activity, social, sleep
        
        # Estimate current cognitive load
        # Lower performance + lower activity = higher cognitive load
        cognitive_load = 1.0 - (np.mean(performance_features) + np.mean(behavioral_features)) / 2
        cognitive_load = max(0.0, min(1.0, cognitive_load))
        
        # Determine load category
        if cognitive_load <= self.cognitive_load_targets['optimal'][0]:
            load_category = 'too_easy'
        elif cognitive_load >= self.cognitive_load_targets['optimal'][1]:
            load_category = 'too_hard'
        else:
            load_category = 'optimal'
            
        # Task difficulty from context
        current_difficulty = context.get('difficulty_level', 0.5)
        
        # Assess appropriateness
        if load_category == 'optimal':
            appropriateness = 1.0
        elif load_category == 'too_easy':
            appropriateness = 0.3  # Need to increase difficulty
        else:  # too_hard
            appropriateness = 0.2  # Need to decrease difficulty
            
        return {
            'cognitive_load': cognitive_load,
            'load_category': load_category,
            'difficulty_appropriateness': appropriateness,
            'recommended_difficulty_change': self._recommend_difficulty_change(load_category, current_difficulty)
        }
    
    def _recommend_difficulty_change(self, load_category: str, current_difficulty: float) -> float:
        """
        Recommend difficulty adjustment based on cognitive load assessment
        
        Args:
            load_category: Current cognitive load category
            current_difficulty: Current difficulty level
            
        Returns:
            Recommended difficulty change (positive = increase, negative = decrease)
        """
        if load_category == 'too_easy':
            # Increase difficulty, but don't jump too much
            max_increase = min(0.2, 0.9 - current_difficulty)
            return max_increase * 0.5  # Conservative increase
        elif load_category == 'too_hard':
            # Decrease difficulty to reduce cognitive load
            max_decrease = min(0.2, current_difficulty - 0.1)
            return -max_decrease * 0.7  # More aggressive decrease
        else:
            # Fine-tuning around optimal range
            return np.random.normal(0, 0.05)  # Small random adjustments
    
    def compute_teacher_priority(self, performance_analysis: Dict[str, float], 
                               cognitive_assessment: Dict[str, Any], 
                               context: Dict[str, Any]) -> float:
        """
        Compute priority score for teacher agent intervention
        
        Args:
            performance_analysis: Performance metrics
            cognitive_assessment: Cognitive load assessment
            context: Current context
            
        Returns:
            Priority score [0, 1]
        """
        # High priority when performance issues need addressing
        performance_priority = self.priority_weights['performance_focus'] * (
            1.0 - performance_analysis['overall_performance'] if performance_analysis['overall_performance'] < 0.6
            else performance_analysis['overall_performance']  # High performance also needs teacher attention
        )
        
        # High priority when cognitive load is suboptimal
        load_priority = self.priority_weights['cognitive_load'] * (
            1.0 - cognitive_assessment['difficulty_appropriateness']
        )
        
        # Priority based on how far difficulty is from optimal
        difficulty_priority = self.priority_weights['difficulty_appropriateness'] * (
            1.0 - abs(cognitive_assessment['recommended_difficulty_change']) / 0.2
        )
        
        # Learning progression priority (higher when consistent good performance)
        progression_priority = self.priority_weights['learning_progression'] * (
            performance_analysis['consistency'] * performance_analysis['learning_indicator']
        )
        
        total_priority = performance_priority + load_priority + difficulty_priority + progression_priority
        
        return max(0.0, min(1.0, total_priority))
    
    def select_response_template(self, performance_analysis: Dict[str, float], 
                               cognitive_assessment: Dict[str, Any]) -> str:
        """
        Select appropriate response template based on analysis
        
        Args:
            performance_analysis: Performance analysis results
            cognitive_assessment: Cognitive load assessment
            
        Returns:
            Selected response template
        """
        overall_performance = performance_analysis['overall_performance']
        load_category = cognitive_assessment['load_category']
        
        # Priority order: cognitive load issues, then performance level
        if load_category == 'too_hard':
            template_category = 'cognitive_overload'
        elif load_category == 'too_easy':
            template_category = 'under_challenged'
        elif overall_performance >= self.performance_thresholds['excellent']:
            template_category = 'excellent_performance'
        elif overall_performance >= self.performance_thresholds['good']:
            template_category = 'good_performance'
        elif overall_performance >= self.performance_thresholds['moderate']:
            template_category = 'moderate_performance'
        else:
            template_category = 'needs_support'
            
        # Select random template from appropriate category
        templates = self.response_templates[template_category]
        return np.random.choice(templates)
    
    def generate_trigger_context(self, performance_analysis: Dict[str, float], 
                               cognitive_assessment: Dict[str, Any]) -> str:
        """
        Generate context description for why teacher agent was triggered
        
        Args:
            performance_analysis: Performance analysis
            cognitive_assessment: Cognitive assessment
            
        Returns:
            Human-readable trigger context
        """
        performance = performance_analysis['overall_performance']
        load_category = cognitive_assessment['load_category']
        
        if load_category == 'too_hard':
            return f"Cognitive overload detected (performance: {performance:.1%})"
        elif load_category == 'too_easy':
            return f"Under-challenged state (performance: {performance:.1%})"
        elif performance >= 0.8:
            return f"Excellent performance detected ({performance:.1%}) - advancement opportunity"
        elif performance < 0.5:
            return f"Performance support needed ({performance:.1%})"
        else:
            return f"Steady performance ({performance:.1%}) - reinforcement focus"
    
    def process_input(self, feature_vector: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for teacher agent
        
        Args:
            feature_vector: Input feature vector from perception engine
            context: Context dictionary from context hub
            
        Returns:
            Teacher agent response package
        """
        # Analyze performance patterns
        performance_analysis = self.analyze_performance_pattern(feature_vector, context)
        
        # Assess cognitive load and difficulty appropriateness
        cognitive_assessment = self.assess_cognitive_load(feature_vector, context)
        
        # Compute priority score for this agent
        priority_score = self.compute_teacher_priority(performance_analysis, cognitive_assessment, context)
        
        # Select appropriate response
        response_text = self.select_response_template(performance_analysis, cognitive_assessment)
        
        # Generate trigger context
        trigger_context = self.generate_trigger_context(performance_analysis, cognitive_assessment)
        
        # Compile response package
        return {
            'response': response_text,
            'metadata': {
                'agent': 'Teacher',
                'priority_score': priority_score,
                'performance_analysis': performance_analysis,
                'cognitive_assessment': cognitive_assessment,
                'recommended_difficulty_change': cognitive_assessment['recommended_difficulty_change']
            },
            'trigger': trigger_context
        }
