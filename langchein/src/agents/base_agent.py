"""
Base Agent class for the multi-agent system.
All specific agents inherit from this.
"""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    """
    
    def __init__(
        self, 
        name: str,
        model: str,
        llm_client,
        verbose: bool = True
    ):
        """
        Initialize a base agent.
        
        Args:
            name: Agent name (e.g., "Literature Agent")
            model: LLM model to use
            llm_client: Function to call LLM
            verbose: Whether to print agent activity
        """
        self.name = name
        self.model = model
        self.llm_client = llm_client
        self.verbose = verbose
        self.history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def run(self, *args, **kwargs) -> str:
        """
        Execute the agent's main task.
        Must be implemented by subclasses.
        """
        pass
    
    def _call_llm(self, messages: List) -> str:
        """
        Call the LLM with error handling.
        
        Args:
            messages: List of message objects
            
        Returns:
            LLM response content
        """
        try:
            response = self.llm_client(messages, self.model)
            return response.content
        except Exception as e:
            print(f"❌ Error in {self.name}: {e}")
            raise
    
    def _log_interaction(self, input_data: str, output_data: str):
        """
        Log the agent's interaction to history.
        
        Args:
            input_data: Input to the agent
            output_data: Output from the agent
        """
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "input": input_data[:500] + "..." if len(input_data) > 500 else input_data,
            "output": output_data[:500] + "..." if len(output_data) > 500 else output_data,
        })
    
    def _print(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(f"[{self.name}] {message}")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Return the agent's interaction history."""
        return self.history
    
    def clear_history(self):
        """Clear the agent's history."""
        self.history = []