"""
CoachAgent - Specialized agent for behavioral guidance and activity nudging
Tracks behavior patterns and provides context-aware suggestions based on daily routines
"""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

class CoachAgent:
    """
    Coach persona agent that focuses on behavioral activation through
    pattern recognition and timely nudges for healthy activities.
    """
    
    def __init__(self):
        """Initialize CoachAgent with behavior tracking parameters"""
        
        # Behavior pattern categories
        self.behavior_patterns = {
            'physical_activity': {'threshold': 0.4, 'weight': 0.3},
            'social_engagement': {'threshold': 0.35, 'weight': 0.25},
            'cognitive_tasks': {'threshold': 0.4, 'weight': 0.25},
            'self_care': {'threshold': 0.3, 'weight': 0.2}
        }
        
        # Time-based activity suggestions
        self.activity_nudges = {
            'morning': {
                'low_activity': [
                    "Good morning! How about a gentle walk to start the day?",
                    "Let's begin with some light stretching exercises.",
                    "A morning routine can set a positive tone. Shall we plan your day together?"
                ],
                'moderate_activity': [
                    "Great start to the morning! Ready to tackle today's goals?",
                    "You're off to a good start! Let's build on this momentum.",
                    "Morning energy detected! What would you like to accomplish today?"
                ]
            },
            'afternoon': {
                'low_activity': [
                    "The afternoon is perfect for a social activity. How about reaching out to a friend?",
                    "Let's add some movement to your afternoon. A short walk perhaps?",
                    "Time for a cognitive boost! How about a puzzle or reading session?"
                ],
                'moderate_activity': [
                    "You're doing well today! Keep up the great work.",
                    "Excellent progress! Remember to take breaks when needed.",
                    "Your activity level looks good. Stay hydrated and keep going!"
                ]
            },
            'evening': {
                'low_activity': [
                    "The evening is a good time to reflect. How about journaling or a calming activity?",
                    "Let's wind down with something enjoyable but not too demanding.",
                    "Time to relax. Perhaps some light reading or listening to music?"
                ],
                'moderate_activity': [
                    "You've had an active day! Time to relax and recharge.",
                    "Great job today! Let's transition to some calming activities.",
                    "Well done! Evening is perfect for gentle activities and reflection."
                ]
            }
        }
        
        # Behavioral pattern history (for tracking across sessions)
        self.pattern_history = []
        
    def extract_behavior_features(self, behavioral_features: np.ndarray) -> Dict[str, float]:
        """
        Extract specific behavior indicators from feature vector
        
        Args:
            behavioral_features: Behavioral input features
            
        Returns:
            Dict of behavior category scores
        """
        if len(behavioral_features) >= 7:
            return {
                'physical_activity': float(behavioral_features[0]),
                'social_engagement': float(behavioral_features[1]),
                'cognitive_tasks': float(behavioral_features[2]),
                'self_care': float(behavioral_features[3])
            }
        else:
            # Fallback to neutral values
            return {
                'physical_activity': 0.5,
                'social_engagement': 0.5,
                'cognitive_tasks': 0.5,
                'self_care': 0.5
            }
            
    def analyze_behavior_patterns(self, behavior_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze behavior patterns and identify areas needing attention
        
        Args:
            behavior_scores: Current behavior category scores
            
        Returns:
            Analysis dict with recommendations
        """
        areas_of_concern = []
        areas_of_strength = []
        
        for category, score in behavior_scores.items():
            threshold = self.behavior_patterns[category]['threshold']
            
            if score < threshold:
                areas_of_concern.append(category)
            elif score > 0.6:
                areas_of_strength.append(category)
                
        # Determine overall activity level
        avg_score = np.mean(list(behavior_scores.values()))
        activity_level = 'low_activity' if avg_score < 0.45 else 'moderate_activity'
        
        return {
            'activity_level': activity_level,
            'concerns': areas_of_concern,
            'strengths': areas_of_strength,
            'average_score': float(avg_score)
        }
        
    def get_time_context(self, context: Dict[str, Any]) -> str:
        """
        Determine time of day context for activity suggestions
        
        Args:
            context: Context dictionary with temporal info
            
        Returns:
            Time period label (morning/afternoon/evening)
        """
        # Try to get from context
        time_of_day = context.get('time_of_day', None)
        
        if time_of_day is None:
            # Determine from current time
            current_hour = datetime.now().hour
            if 5 <= current_hour < 12:
                time_of_day = 'morning'
            elif 12 <= current_hour < 18:
                time_of_day = 'afternoon'
            else:
                time_of_day = 'evening'
                
        return time_of_day
        
    def generate_nudge(self, analysis: Dict[str, Any], time_of_day: str, context: Dict[str, Any]) -> str:
        """
        Generate contextual nudge based on behavior analysis and time
        
        Args:
            analysis: Behavior pattern analysis
            time_of_day: Current time period
            context: Additional context
            
        Returns:
            Nudge text
        """
        activity_level = analysis['activity_level']
        
        # Get base nudge from templates
        time_nudges = self.activity_nudges.get(time_of_day, self.activity_nudges['afternoon'])
        base_nudges = time_nudges.get(activity_level, time_nudges['low_activity'])
        base_nudge = np.random.choice(base_nudges)
        
        # Add specific recommendation if there are concerns
        if analysis['concerns']:
            primary_concern = analysis['concerns'][0]
            if primary_concern == 'physical_activity':
                base_nudge += " I notice you could use more movement today."
            elif primary_concern == 'social_engagement':
                base_nudge += " Connecting with others could boost your mood."
            elif primary_concern == 'cognitive_tasks':
                base_nudge += " Some mental stimulation might be refreshing."
            elif primary_concern == 'self_care':
                base_nudge += " Remember to take care of yourself."
                
        # Acknowledge strengths
        if analysis['strengths']:
            primary_strength = analysis['strengths'][0]
            strength_labels = {
                'physical_activity': 'staying active',
                'social_engagement': 'connecting with others',
                'cognitive_tasks': 'keeping mentally engaged',
                'self_care': 'taking care of yourself'
            }
            base_nudge += f" You're doing great with {strength_labels.get(primary_strength, 'your activities')}!"
            
        return base_nudge
        
    def calculate_priority(self, analysis: Dict[str, Any]) -> float:
        """
        Calculate priority score for coach intervention
        Higher priority when multiple behaviors are below threshold
        
        Args:
            analysis: Behavior pattern analysis
            
        Returns:
            Priority score [0, 1]
        """
        # Base priority on number of concerns and overall activity
        num_concerns = len(analysis['concerns'])
        avg_score = analysis['average_score']
        
        # More concerns = higher priority
        concern_factor = min(num_concerns / 4.0, 1.0) * 0.6
        
        # Lower average score = higher priority
        activity_factor = (1.0 - avg_score) * 0.4
        
        priority = concern_factor + activity_factor
        
        return float(priority)
        
    def process_input(self, xt: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main processing method for CoachAgent
        
        Args:
            xt: Input feature vector from perception engine
            context: Optional context dictionary
            
        Returns:
            Dict containing response, metadata, and contextual triggers
        """
        if context is None:
            context = {}
            
        # Extract behavioral features
        behavioral_features = xt[:7] if len(xt) > 7 else np.array([0.5] * 7)
        
        # Get behavior scores
        behavior_scores = self.extract_behavior_features(behavioral_features)
        
        # Analyze patterns
        analysis = self.analyze_behavior_patterns(behavior_scores)
        
        # Get time context
        time_of_day = self.get_time_context(context)
        
        # Generate contextual nudge
        response_text = self.generate_nudge(analysis, time_of_day, context)
        
        # Calculate priority
        priority_score = self.calculate_priority(analysis)
        
        # Track pattern history
        self.pattern_history.append({
            'timestamp': datetime.now().isoformat(),
            'scores': behavior_scores,
            'analysis': analysis
        })
        
        # Keep only recent history
        if len(self.pattern_history) > 20:
            self.pattern_history = self.pattern_history[-20:]
            
        # Compile metadata
        metadata = {
            'agent': 'Coach',
            'behavior_scores': behavior_scores,
            'activity_level': analysis['activity_level'],
            'concerns': analysis['concerns'],
            'strengths': analysis['strengths'],
            'time_of_day': time_of_day,
            'priority_score': priority_score
        }
        
        # Contextual trigger information
        concerns_str = ', '.join(analysis['concerns']) if analysis['concerns'] else 'none'
        trigger = f"Activity={analysis['activity_level']}, Concerns=[{concerns_str}], Time={time_of_day}"
        
        return {
            'response': response_text,
            'metadata': metadata,
            'trigger': trigger
        }
