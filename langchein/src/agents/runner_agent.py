"""
Runner Agent - executes Python code and reports results.
"""
from base_agent import BaseAgent
import traceback
from typing import Dict, Any


class RunnerAgent(BaseAgent):
    """
    Agent that executes Python code in a clean environment.
    """
    
    def __init__(self, verbose: bool = True):
        # Runner doesn't need LLM
        super().__init__("Runner Agent", model=None, llm_client=None, verbose=verbose)
    
    def run(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code.
        
        Args:
            code: Python code to execute
            
        Returns:
            Dict with 'success' (bool) and 'output' (str)
        """
        self._print("Executing code...")
        
        local_env: Dict[str, Any] = {}
        
        try:
            exec(code, {}, local_env)
            result = {"success": True, "output": "Execution successful."}
            self._print("✅ Code executed successfully")
        except Exception as e:
            error_trace = traceback.format_exc()
            result = {"success": False, "output": error_trace}
            self._print(f"❌ Execution failed: {str(e)}")
        
        self._log_interaction(
            input_data=code[:200] + "..." if len(code) > 200 else code,
            output_data=result["output"]
        )
        
        return result
    
    def _call_llm(self, messages):
        """Runner doesn't use LLM."""
        raise NotImplementedError("RunnerAgent doesn't use LLM")