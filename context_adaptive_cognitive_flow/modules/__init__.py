"""
Context-Adaptive Cognitive Flow Modules

This package implements the four-stage framework for cognitive rehabilitation:
Stage I: Multimodal Data Sensing
Stage II: Persona-Specific Context Recognition
Stage III: Emotionally Adaptive Response Strategy  
Stage IV: Feedback and Iterative Adaptation

Reference: Designing a Generative AI Framework for Cognitive Intervention in Older Adults
"""

from .sensing import MultimodalSensor
from .context_recognition import ContextRecognizer
from .response_strategy import ResponseStrategy
from .feedback_loop import FeedbackLoop

__all__ = [
    'MultimodalSensor',
    'ContextRecognizer',
    'ResponseStrategy',
    'FeedbackLoop'
]
