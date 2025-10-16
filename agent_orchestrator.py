"""
Agent Orchestrator - Central controller for multi-agent coordination
Handles parallel invocation, priority-based selection, and response attribution
"""

import numpy as np
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from teacher_agent import TeacherAgent
from companion_agent import CompanionAgent
from coach_agent import CoachAgent

class AgentOrchestrator:
    """
    Central orchestrator that coordinates multiple specialized agents,
    selects optimal responses, and manages agent attribution.
    """
    
    def __init__(self):
        """Initialize orchestrator with all specialized agents"""
        
        # Initialize all agents
        self.agents = {
            'teacher': TeacherAgent(),
            'companion': CompanionAgent(),
            'coach': CoachAgent()
        }
        
        # Response selection strategies
        self.selection_strategies = [
            'highest_priority',    # Select agent with highest priority score
            'weighted_blend',      # Blend responses based on priority weights
            'round_robin'          # Rotate between agents
        ]
        
        self.current_strategy = 'highest_priority'
        
        # Agent invocation history for round-robin
        self.last_selected_agent = None
        
    def invoke_agents_parallel(self, xt: np.ndarray, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Invoke all agents in parallel using ThreadPoolExecutor
        
        Args:
            xt: Input feature vector
            context: Context dictionary
            
        Returns:
            Dict mapping agent names to their outputs
        """
        agent_outputs = {}
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all agent tasks
            future_to_agent = {
                executor.submit(agent.process_input, xt, context): name
                for name, agent in self.agents.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result(timeout=5.0)  # 5 second timeout per agent
                    agent_outputs[agent_name] = result
                except Exception as e:
                    # Handle agent failures gracefully
                    agent_outputs[agent_name] = {
                        'response': f"[{agent_name.capitalize()} agent encountered an error]",
                        'metadata': {'agent': agent_name.capitalize(), 'error': str(e), 'priority_score': 0.0},
                        'trigger': f"Error: {str(e)}"
                    }
                    
        return agent_outputs
        
    def select_highest_priority(self, agent_outputs: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Select response from agent with highest priority score
        
        Args:
            agent_outputs: Dict of agent outputs
            
        Returns:
            Selected agent output with attribution (or None if no agents)
        """
        max_priority = -1
        selected_output = None
        selected_agent = None
        
        for agent_name, output in agent_outputs.items():
            priority = output['metadata'].get('priority_score', 0.0)
            if priority > max_priority:
                max_priority = priority
                selected_output = output
                selected_agent = agent_name
                
        # Add attribution
        if selected_output:
            selected_output['attribution'] = {
                'primary_agent': selected_agent,
                'selection_reason': f'Highest priority ({max_priority:.2f})',
                'strategy': 'highest_priority'
            }
            
        return selected_output
        
    def blend_weighted_responses(self, agent_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create weighted blend of responses based on priority scores
        
        Args:
            agent_outputs: Dict of agent outputs
            
        Returns:
            Blended output with multi-agent attribution
        """
        # Extract priority scores
        priorities = {
            name: output['metadata'].get('priority_score', 0.0)
            for name, output in agent_outputs.items()
        }
        
        # Normalize to get weights
        total_priority = sum(priorities.values())
        if total_priority == 0:
            # Equal weights if all priorities are zero
            weights = {name: 1.0/len(priorities) for name in priorities}
        else:
            weights = {name: p/total_priority for name, p in priorities.items()}
            
        # Sort agents by weight (descending)
        sorted_agents = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        # Build blended response (primary + supporting)
        primary_agent = sorted_agents[0][0]
        primary_output = agent_outputs[primary_agent]
        
        blended_response = primary_output['response']
        
        # Add supporting insights from other agents if their weight is significant
        supporting_agents = []
        for agent_name, weight in sorted_agents[1:]:
            if weight > 0.2:  # Include if weight > 20%
                supporting_agents.append(agent_name)
                support_response = agent_outputs[agent_name]['response']
                # Take first sentence as supporting insight
                first_sentence = support_response.split('.')[0] + '.'
                blended_response += f"\n\n*{agent_name.capitalize()} adds:* {first_sentence}"
                
        # Compile blended metadata
        blended_output = {
            'response': blended_response,
            'metadata': primary_output['metadata'],
            'trigger': primary_output['trigger'],
            'attribution': {
                'primary_agent': primary_agent,
                'supporting_agents': supporting_agents,
                'weights': weights,
                'selection_reason': f'Weighted blend (primary: {weights[primary_agent]:.1%})',
                'strategy': 'weighted_blend'
            }
        }
        
        return blended_output
        
    def select_round_robin(self, agent_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select agents in round-robin fashion for balanced interaction
        
        Args:
            agent_outputs: Dict of agent outputs
            
        Returns:
            Selected agent output with attribution
        """
        agent_names = list(agent_outputs.keys())
        
        # Determine next agent
        if self.last_selected_agent is None or self.last_selected_agent not in agent_names:
            selected_agent = agent_names[0]
        else:
            current_index = agent_names.index(self.last_selected_agent)
            selected_agent = agent_names[(current_index + 1) % len(agent_names)]
            
        self.last_selected_agent = selected_agent
        selected_output = agent_outputs[selected_agent]
        
        # Add attribution
        selected_output['attribution'] = {
            'primary_agent': selected_agent,
            'selection_reason': 'Round-robin rotation',
            'strategy': 'round_robin'
        }
        
        return selected_output
        
    def aggregate_responses(self, agent_outputs: Dict[str, Dict[str, Any]], strategy: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Aggregate agent responses using specified strategy
        
        Args:
            agent_outputs: Dict of all agent outputs
            strategy: Selection strategy (or use current default)
            
        Returns:
            Aggregated response with attribution (or None if no agents available)
        """
        if strategy is None:
            strategy = self.current_strategy
            
        if strategy == 'highest_priority':
            return self.select_highest_priority(agent_outputs)
        elif strategy == 'weighted_blend':
            return self.blend_weighted_responses(agent_outputs)
        elif strategy == 'round_robin':
            return self.select_round_robin(agent_outputs)
        else:
            # Default to highest priority
            return self.select_highest_priority(agent_outputs)
            
    def process_input(
        self, 
        xt: np.ndarray, 
        context: Optional[Dict[str, Any]] = None,
        strategy: Optional[str] = None,
        return_all_agents: bool = False
    ) -> Dict[str, Any]:
        """
        Main orchestrator processing method
        
        Args:
            xt: Input feature vector from perception engine
            context: Context dictionary
            strategy: Response selection strategy
            return_all_agents: If True, return all agent outputs instead of aggregating
            
        Returns:
            Aggregated response or all agent outputs
        """
        if context is None:
            context = {}
            
        # Invoke all agents in parallel
        start_time = time.time()
        agent_outputs = self.invoke_agents_parallel(xt, context)
        invocation_time = time.time() - start_time
        
        # Return all outputs if requested (for visualization)
        if return_all_agents:
            return {
                'all_agents': agent_outputs,
                'invocation_time': invocation_time
            }
            
        # Otherwise, aggregate and select best response
        aggregated_output = self.aggregate_responses(agent_outputs, strategy)
        
        # Handle case where no output was selected
        if aggregated_output is None:
            # Fallback: create default response
            aggregated_output = {
                'response': 'No response generated',
                'metadata': {'priority_score': 0.0},
                'attribution': {'primary_agent': 'none', 'selection_reason': 'No agents available'}
            }
        
        # Add orchestration metadata
        aggregated_output['orchestration'] = {
            'num_agents_invoked': len(agent_outputs),
            'invocation_time_ms': invocation_time * 1000,
            'all_agent_priorities': {
                name: output['metadata'].get('priority_score', 0.0)
                for name, output in agent_outputs.items()
            }
        }
        
        return aggregated_output
        
    def set_strategy(self, strategy: str):
        """
        Update the response selection strategy
        
        Args:
            strategy: One of 'highest_priority', 'weighted_blend', 'round_robin'
        """
        if strategy in self.selection_strategies:
            self.current_strategy = strategy
        else:
            raise ValueError(f"Invalid strategy. Must be one of {self.selection_strategies}")
            
    def get_agent_info(self) -> Dict[str, str]:
        """
        Get information about available agents
        
        Returns:
            Dict mapping agent names to descriptions
        """
        return {
            'teacher': 'Focuses on cognitive activation and difficulty adjustment based on performance',
            'companion': 'Provides emotional support based on valence-arousal emotional state',
            'coach': 'Offers behavioral nudges and activity suggestions based on daily patterns'
        }
