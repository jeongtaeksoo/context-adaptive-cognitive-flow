"""
Coach Agent - Specialized agent focusing on behavioral nudges and activity suggestions
Emphasizes daily patterns, physical activity, and lifestyle optimization
"""

import numpy as np
from typing import Dict, List, Any, Optional
from utils import sigmoid
from datetime import datetime

class CoachAgent:
    """
    Coach agent specialized in behavioral coaching and activity recommendations.
    Focuses on daily patterns, physical activity, and lifestyle optimization.
    """
    
    def __init__(self):
        """Initialize coach agent with behavioral coaching parameters"""
        
        # Coach-specific priority weights
        self.priority_weights = {
            'activity_level': 0.4,       # Weight on physical/behavioral activity
            'daily_patterns': 0.3,       # Weight on temporal and routine patterns
            'energy_optimization': 0.2,  # Weight on sleep and energy indicators
            'behavioral_consistency': 0.1 # Weight on behavioral pattern consistency
        }
        
        # Activity level thresholds for different coaching approaches
        self.activity_thresholds = {
            'highly_active': 0.8,    # Very active - maintenance/variety focus
            'moderately_active': 0.5, # Good activity - enhancement focus
            'low_active': 0.3,       # Low activity - motivation focus
            'sedentary': 0.1         # Very low activity - gentle encouragement
        }
        
        # Time-based activity recommendations
        self.time_based_activities = {
            'morning': {
                'high_energy': ['dynamic stretching', 'morning walk', 'energizing breathing exercises'],
                'moderate_energy': ['gentle yoga', 'light stretching', 'meditation'],
                'low_energy': ['seated stretches', 'deep breathing', 'mindful moments']
            },
            'afternoon': {
                'high_energy': ['brisk walk', 'standing exercises', 'active hobbies'],
                'moderate_energy': ['chair exercises', 'light movement', 'creative activities'],
                'low_energy': ['gentle stretches', 'relaxation techniques', 'quiet activities']
            },
            'evening': {
                'high_energy': ['evening stroll', 'gentle movement', 'calming activities'],
                'moderate_energy': ['relaxing stretches', 'light activities', 'wind-down routine'],
                'low_energy': ['breathing exercises', 'gentle movements', 'restful preparation']
            }
        }
        
        # Response templates for different activity states
        self.response_templates = {
            'highly_active': [
                "I'm impressed by your excellent activity level! You're really taking charge of your physical well-being. How about we explore some variety to keep things interesting?",
                "Your commitment to staying active is inspiring! Since you're doing so well, let's think about how we can build on this momentum with some new challenges.",
                "Fantastic energy and movement patterns! You're clearly prioritizing your health. What new activities might spark your interest today?"
            ],
            'moderately_active': [
                "You're doing well with staying active! I can see you're building healthy habits. What if we tried adding just one more small movement to your routine?",
                "Great job maintaining good activity levels! You're on a positive path. How would you feel about exploring a slightly more engaging activity today?",
                "I love seeing your consistent movement! You're creating a solid foundation. Ready to take it up just a notch with something enjoyable?"
            ],
            'low_active': [
                "I understand that staying active can be challenging sometimes. What matters most is taking small, manageable steps that feel good to you.",
                "Every bit of movement counts, and I'm here to support you in finding activities that feel achievable and pleasant for you right now.",
                "Let's focus on gentle, enjoyable ways to add a little movement to your day. Even small steps can make a meaningful difference."
            ],
            'sedentary': [
                "Today might be a perfect day to start with something very gentle and comfortable. Even the smallest movement can be a wonderful beginning.",
                "I'm here to help you find the easiest, most comfortable ways to add just a tiny bit of movement that feels manageable right now.",
                "Let's think about the simplest possible way to bring a little gentle activity into your day - something that feels completely doable for you."
            ],
            'energy_mismatch': [
                "I notice there might be a mismatch between your energy and activity levels. Let's find activities that better match how you're feeling right now.",
                "It seems like your current energy might call for a different approach to activity today. Let's adjust to what feels right for your body.",
                "Your energy levels suggest we might want to recalibrate your activity approach. What feels most appropriate for how you're feeling?"
            ],
            'routine_disruption': [
                "I see there might be some changes in your daily patterns. That's completely normal - let's adapt your activity approach to fit your current rhythm.",
                "Changes in routine can affect our activity levels. Let's find flexible ways to maintain movement that work with your current schedule.",
                "Your patterns seem to be shifting, which is natural. How can we adjust your activity approach to support you through these changes?"
            ],
            'sleep_activity_connection': [
                "I notice a connection between your sleep and activity patterns. Quality rest and gentle movement often support each other beautifully.",
                "Your sleep and activity levels seem interrelated. Let's think about activities that might support better rest and overall energy.",
                "The relationship between your rest and movement patterns is interesting. How can we optimize both for your overall well-being?"
            ]
        }
        
        # Specific activity recommendations by energy level and time
        self.activity_recommendations = {
            'chair_based': [
                "Chair-based arm circles and shoulder rolls",
                "Seated marching in place", 
                "Gentle neck and spine stretches",
                "Seated breathing exercises with arm movements"
            ],
            'standing_gentle': [
                "Standing and gentle swaying",
                "Light stretching against a wall or counter",
                "Slow, deliberate walking in place",
                "Standing balance exercises with support"
            ],
            'walking_based': [
                "Short walk around your living space",
                "Walking to different rooms with purpose",
                "Outdoor walk for fresh air (if possible)",
                "Walking while doing light tasks"
            ],
            'engaging_activities': [
                "Dancing to favorite music",
                "Gardening or plant care",
                "Playing with pets",
                "Active hobbies like crafts with movement"
            ]
        }
        
        # Behavioral pattern indicators
        self.pattern_indicators = {
            'morning_person': ['high morning activity', 'declining afternoon energy'],
            'afternoon_peak': ['moderate morning', 'high afternoon activity'],
            'evening_active': ['low morning', 'increasing evening activity'],
            'consistent_moderate': ['steady moderate activity throughout day']
        }
        
    def analyze_activity_patterns(self, feature_vector: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze activity and behavioral patterns from input features
        
        Args:
            feature_vector: Input features from perception engine
            context: Context information
            
        Returns:
            Activity pattern analysis
        """
        # Extract activity-related features (indices 0-2: activity, social, sleep)
        behavioral_features = feature_vector[0:3]
        activity_level = behavioral_features[0]
        social_engagement = behavioral_features[1] 
        sleep_quality = behavioral_features[2]
        
        # Extract temporal features (indices 9-10: time, day)
        temporal_features = feature_vector[9:11]
        time_of_day_score = temporal_features[0]  # Higher = morning
        day_type_score = temporal_features[1]     # Higher = weekday
        
        # Determine activity category
        if activity_level >= self.activity_thresholds['highly_active']:
            activity_category = 'highly_active'
        elif activity_level >= self.activity_thresholds['moderately_active']:
            activity_category = 'moderately_active'
        elif activity_level >= self.activity_thresholds['low_active']:
            activity_category = 'low_active'
        else:
            activity_category = 'sedentary'
            
        # Estimate energy level from multiple indicators
        energy_indicators = [activity_level, sleep_quality, social_engagement]
        estimated_energy = np.mean(energy_indicators)
        
        # Determine time context
        if time_of_day_score > 0.7:
            time_context = 'morning'
        elif time_of_day_score > 0.3:
            time_context = 'afternoon'
        else:
            time_context = 'evening'
            
        # Check for energy-activity alignment
        energy_activity_alignment = 1.0 - abs(estimated_energy - activity_level)
        
        return {
            'activity_level': float(activity_level),
            'activity_category': activity_category,
            'estimated_energy': float(estimated_energy),
            'sleep_quality': float(sleep_quality),
            'social_engagement': float(social_engagement),
            'time_context': time_context,
            'day_type': 'weekday' if day_type_score > 0.5 else 'weekend',
            'energy_activity_alignment': float(energy_activity_alignment)
        }
    
    def assess_behavioral_consistency(self, feature_vector: np.ndarray, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Assess consistency in behavioral patterns
        
        Args:
            feature_vector: Input features
            context: Context information
            
        Returns:
            Behavioral consistency assessment
        """
        # Extract behavioral features
        behavioral_features = feature_vector[0:3]
        
        # Compute consistency as inverse of variance (higher variance = less consistent)
        behavioral_variance = np.var(behavioral_features)
        consistency_score = 1.0 / (1.0 + behavioral_variance)
        
        # Pattern stability (simplified - would use historical data in practice)
        # For now, use feature relationships as proxy
        activity_sleep_correlation = 1.0 - abs(behavioral_features[0] - behavioral_features[2])
        social_activity_correlation = 1.0 - abs(behavioral_features[1] - behavioral_features[0])
        
        pattern_stability = (activity_sleep_correlation + social_activity_correlation) / 2
        
        return {
            'behavioral_consistency': float(consistency_score),
            'pattern_stability': float(pattern_stability),
            'routine_adherence': float((consistency_score + pattern_stability) / 2)
        }
    
    def recommend_activities(self, activity_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate specific activity recommendations based on analysis
        
        Args:
            activity_analysis: Activity pattern analysis results
            
        Returns:
            List of recommended activities
        """
        activity_category = activity_analysis['activity_category']
        time_context = activity_analysis['time_context']
        estimated_energy = activity_analysis['estimated_energy']
        
        # Select energy-appropriate activities for time of day
        if estimated_energy > 0.7:
            energy_level = 'high_energy'
        elif estimated_energy > 0.4:
            energy_level = 'moderate_energy'
        else:
            energy_level = 'low_energy'
            
        # Get time and energy appropriate activities
        time_activities = self.time_based_activities.get(time_context, {})
        recommended_activities = time_activities.get(energy_level, ['gentle movement'])
        
        # Add category-specific recommendations
        if activity_category == 'sedentary':
            recommended_activities.extend(self.activity_recommendations['chair_based'])
        elif activity_category == 'low_active':
            recommended_activities.extend(self.activity_recommendations['standing_gentle'])
        elif activity_category == 'moderately_active':
            recommended_activities.extend(self.activity_recommendations['walking_based'])
        else:  # highly_active
            recommended_activities.extend(self.activity_recommendations['engaging_activities'])
            
        # Return unique recommendations (limit to 3-4 for focus)
        unique_recommendations = list(set(recommended_activities))[:4]
        return unique_recommendations
    
    def compute_coach_priority(self, activity_analysis: Dict[str, Any], 
                             consistency_analysis: Dict[str, float], 
                             context: Dict[str, Any]) -> float:
        """
        Compute priority score for coach agent intervention
        
        Args:
            activity_analysis: Activity pattern analysis
            consistency_analysis: Behavioral consistency analysis
            context: Current context
            
        Returns:
            Priority score [0, 1]
        """
        # High priority for low activity levels
        activity_priority = self.priority_weights['activity_level'] * (
            1.0 - activity_analysis['activity_level']  # Inverted - low activity = high priority
        )
        
        # High priority for poor energy-activity alignment or routine disruption
        pattern_priority = self.priority_weights['daily_patterns'] * (
            1.0 - activity_analysis['energy_activity_alignment']
        )
        
        # High priority for poor sleep quality (affects energy optimization)
        energy_priority = self.priority_weights['energy_optimization'] * (
            1.0 - activity_analysis['sleep_quality']
        )
        
        # High priority for inconsistent behavioral patterns
        consistency_priority = self.priority_weights['behavioral_consistency'] * (
            1.0 - consistency_analysis['behavioral_consistency']
        )
        
        total_priority = activity_priority + pattern_priority + energy_priority + consistency_priority
        
        # Boost priority for very sedentary individuals
        if activity_analysis['activity_category'] == 'sedentary':
            total_priority *= 1.3
            
        # Boost priority for energy-activity mismatches
        if activity_analysis['energy_activity_alignment'] < 0.5:
            total_priority *= 1.2
            
        return max(0.0, min(1.0, total_priority))
    
    def select_response_template(self, activity_analysis: Dict[str, Any], 
                               consistency_analysis: Dict[str, float]) -> str:
        """
        Select appropriate response template based on activity and consistency analysis
        
        Args:
            activity_analysis: Activity analysis results
            consistency_analysis: Consistency analysis results
            
        Returns:
            Selected response template
        """
        activity_category = activity_analysis['activity_category']
        energy_alignment = activity_analysis['energy_activity_alignment']
        consistency = consistency_analysis['behavioral_consistency']
        
        # Priority order: energy mismatch, routine disruption, activity level
        if energy_alignment < 0.4:
            template_category = 'energy_mismatch'
        elif consistency < 0.4:
            template_category = 'routine_disruption'
        elif activity_analysis['sleep_quality'] < 0.4 and activity_analysis['activity_level'] < 0.5:
            template_category = 'sleep_activity_connection'
        else:
            template_category = activity_category
            
        # Select from appropriate template category
        templates = self.response_templates.get(template_category, 
                                              self.response_templates['moderately_active'])
        return np.random.choice(templates)
    
    def generate_trigger_context(self, activity_analysis: Dict[str, Any], 
                               consistency_analysis: Dict[str, float]) -> str:
        """
        Generate context description for why coach agent was triggered
        
        Args:
            activity_analysis: Activity analysis
            consistency_analysis: Consistency analysis
            
        Returns:
            Human-readable trigger context
        """
        activity_level = activity_analysis['activity_level']
        activity_category = activity_analysis['activity_category']
        energy_alignment = activity_analysis['energy_activity_alignment']
        time_context = activity_analysis['time_context']
        
        # Primary trigger identification
        if activity_category == 'sedentary':
            return f"Low activity detected ({activity_level:.1%}) - gentle encouragement needed"
        elif energy_alignment < 0.4:
            return f"Energy-activity mismatch in {time_context} (alignment: {energy_alignment:.1%})"
        elif consistency_analysis['behavioral_consistency'] < 0.4:
            return f"Behavioral pattern disruption (consistency: {consistency_analysis['behavioral_consistency']:.1%})"
        elif activity_category == 'highly_active':
            return f"High activity level ({activity_level:.1%}) - variety and maintenance focus"
        else:
            return f"{activity_category.replace('_', ' ').title()} activity in {time_context} - enhancement opportunity"
    
    def process_input(self, feature_vector: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for coach agent
        
        Args:
            feature_vector: Input feature vector from perception engine
            context: Context dictionary from context hub
            
        Returns:
            Coach agent response package
        """
        # Analyze activity patterns
        activity_analysis = self.analyze_activity_patterns(feature_vector, context)
        
        # Assess behavioral consistency
        consistency_analysis = self.assess_behavioral_consistency(feature_vector, context)
        
        # Generate activity recommendations
        activity_recommendations = self.recommend_activities(activity_analysis)
        
        # Compute priority score
        priority_score = self.compute_coach_priority(activity_analysis, consistency_analysis, context)
        
        # Select appropriate response
        response_text = self.select_response_template(activity_analysis, consistency_analysis)
        
        # Add specific activity suggestions to response
        if len(activity_recommendations) > 0:
            response_text += f"\n\nHere are some activities that might feel good right now: {', '.join(activity_recommendations[:2])}."
        
        # Generate trigger context
        trigger_context = self.generate_trigger_context(activity_analysis, consistency_analysis)
        
        # Compile response package
        return {
            'response': response_text,
            'metadata': {
                'agent': 'Coach',
                'priority_score': priority_score,
                'activity_analysis': activity_analysis,
                'consistency_analysis': consistency_analysis,
                'activity_recommendations': activity_recommendations,
                'coaching_focus': activity_analysis['activity_category']
            },
            'trigger': trigger_context
        }
