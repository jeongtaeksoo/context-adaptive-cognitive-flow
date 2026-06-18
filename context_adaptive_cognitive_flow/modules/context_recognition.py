"""
Stage II: Persona-Specific Context Recognition

To address age-related cognitive variability, the framework proposes a cognitive load
metric based on workload assessment frameworks.

Initial weight parameters (w_1=0.4, w_2=0.35, w_3=0.25) are theory-derived proposals
prioritizing processing speed. The metric is constrained to L_cog ∈ [0, 2]. The
Companion agent employs valence-arousal mapping for emotional state assessment.
"""

from .sensing import SensorData
import numpy as np


class ContextRecognizer:
    """
    Stage II: Persona-Specific Context Recognition
    
    Computes cognitive load index (L_cog) from multimodal sensor data.
    This metric drives persona-based adaptation across Teacher, Companion, and Coach.
    
    Reference: Eq.(1) from paper
    """
    
    def __init__(self):
        """Initialize context recognizer with weight parameters from Eq.(1)."""
        self.w_time = 0.4
        self.w_error = 0.35
        self.w_attention = 0.25
        
    def compute_cognitive_load(self, sensor_data: SensorData) -> float:
        """
        Compute cognitive load index from multimodal data.
        
        Eq.(1): L_cog = w_1*(Δt_resp / t̄_base) + w_2*e_rate + w_3*σ_att²
        Initial weights: w_1=0.4, w_2=0.35, w_3=0.25
        
        Clinical interpretation:
        - L_cog < 0.5: Low cognitive load (task too easy)
        - 0.5 ≤ L_cog ≤ 1.5: Optimal challenge zone
        - L_cog > 1.5: High cognitive load (intervention needed)
        
        Args:
            sensor_data: Multimodal sensor readings
            
        Returns:
            L_cog: Cognitive load index (dimensionless)
        """
        time_component = (sensor_data.response_time / sensor_data.baseline_time)
        
        error_component = sensor_data.error_rate
        
        attention_component = sensor_data.attention_variance
        
        L_cog = (self.w_time * time_component + 
                 self.w_error * error_component + 
                 self.w_attention * attention_component)
        
        return L_cog
    
    def recognize_context(self, sensor_data: SensorData) -> dict:
        """
        Perform full context recognition with cognitive load classification.
        
        Args:
            sensor_data: Multimodal sensor readings
            
        Returns:
            Dictionary with L_cog and contextual interpretation
        """
        L_cog = self.compute_cognitive_load(sensor_data)
        
        if L_cog < 0.5:
            context = "understimulated"
        elif L_cog <= 1.5:
            context = "optimal"
        else:
            context = "overstimulated"
            
        return {
            'L_cog': L_cog,
            'context': context,
            'sensor_data': sensor_data
        }
