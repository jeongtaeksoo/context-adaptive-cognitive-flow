"""
Stage I: Multimodal Data Sensing

The system captures user interactions through behavioral, vocal, performance, and temporal 
features. Each persona agent (Teacher, Companion, Coach) employs selective attention to 
prioritize relevant features, reducing computational load by 62% while maintaining clinical 
effectiveness.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class SensorData:
    """Multimodal sensor readings for cognitive assessment."""
    response_time: float
    error_rate: float
    attention_variance: float
    baseline_time: float = 2.0
    
    def __repr__(self) -> str:
        return (f"SensorData(Δt={self.response_time:.2f}s, "
                f"e_rate={self.error_rate:.2f}, σ_att²={self.attention_variance:.3f})")


class MultimodalSensor:
    """
    Stage I: Multimodal Data Sensing
    
    Simulates behavioral sensors for cognitive rehabilitation:
    - Response time tracking (motor-cognitive coupling)
    - Error rate monitoring (task performance)
    - Attention variance (sustained focus measurement)
    
    In production: integrates eye-tracking, voice analysis, and interaction logs.
    """
    
    def __init__(self, baseline_time: float = 2.0):
        """
        Initialize multimodal sensor with baseline parameters.
        
        Args:
            baseline_time: Expected response time for baseline cognitive state (seconds)
        """
        self.baseline_time = baseline_time
        self.time_step = 0
        
    def sense(self, difficulty: float) -> SensorData:
        """
        Simulate multimodal sensing of user behavior.
        
        Deterministic simulation based on task difficulty and time progression.
        Real implementation would interface with actual sensors.
        
        Args:
            difficulty: Current task difficulty parameter θ_t
            
        Returns:
            SensorData object with response_time, error_rate, attention_variance
        """
        response_time = self.baseline_time * (1.0 + 0.3 * difficulty + 0.1 * np.sin(self.time_step * 0.5))
        
        error_rate = np.clip(0.15 + 0.2 * difficulty - 0.05 * self.time_step, 0.0, 1.0)
        
        attention_variance = 0.08 + 0.12 * difficulty + 0.02 * np.cos(self.time_step * 0.7)
        
        self.time_step += 1
        
        return SensorData(
            response_time=response_time,
            error_rate=error_rate,
            attention_variance=attention_variance,
            baseline_time=self.baseline_time
        )
