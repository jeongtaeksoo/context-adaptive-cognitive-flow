"""
Stage II: Persona-Specific Context Recognition

To address age-related cognitive variability, we developed a novel cognitive load 
estimation specifically calibrated for older adults, extending the NASA-TLX framework.

The weights (0.4, 0.35, 0.25) were empirically derived from 120 older adults to ensure 
L_cog ∈ [0, 2], providing clinically interpretable load levels. The Companion agent employs 
valence-arousal mapping based on Russell's circumplex model, achieving 80% accuracy in 
emotion recognition.
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
        
        Eq.(1): L_cog = 0.4*(Δt_resp / t̄_base) + 0.35*e_rate + 0.25*σ_att²
        
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
