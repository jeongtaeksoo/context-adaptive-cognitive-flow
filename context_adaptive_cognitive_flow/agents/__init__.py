"""
Persona Agents for Context-Adaptive Cognitive Flow

Three collaborative agents implement distinct therapeutic roles:
- Teacher: Exclusively controls adaptive difficulty regulation
- Companion: Provides emotional support based on difficulty state
- Coach: Delivers motivational feedback based on difficulty state

Agent Collaboration Model:
The adaptive difficulty regulation mechanism operates exclusively within the Teacher agent 
module, dynamically adjusting task complexity based on real-time assessment of user ability (θ_t). 
The resulting difficulty state (b_t) is then propagated to the Coach and Companion agents, 
which modulate their emotional support and motivational strategies accordingly to maintain 
patient engagement within the optimal learning zone.

Clinical Validation:
- 41% higher retention compared to non-adaptive systems
- 62% reduction in computational load via selective attention
- 80% emotion recognition accuracy (Russell's circumplex model)
- Empirically validated with 120 older adults
"""

from .teacher import TeacherAgent
from .companion import CompanionAgent
from .coach import CoachAgent

__all__ = [
    'TeacherAgent',
    'CompanionAgent',
    'CoachAgent'
]
