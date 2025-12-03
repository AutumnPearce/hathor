"""
Critic Agent - reviews and improves outputs from other agents.
Handles both hypothesis critique and plan critique.
"""
from base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
from typing import Tuple


class CriticAgent(BaseAgent):
    """
    Agent that critiques and improves work from other agents.
    Can operate in different modes: hypotheses critique or plan critique.
    """
    
    def __init__(self, model: str, llm_client, verbose: bool = True):
        super().__init__("Critic Agent", model, llm_client, verbose)
        
        # Different prompt templates for different critique modes
        self.hypotheses_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a critic agent and expert in Galaxy formation. "
             "Analyze hypotheses proposed by the literature review, get rid of impractical ones, "
             "and improve the rest into realistic hypotheses that can be checked by visualization/analysis. "
             "Be specific and practical. Don't just repeat the task and don't give plans for implementation."),
            ("user", "{hypotheses}")
        ])
        
        self.plans_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a critic agent. Review the hypothesis-plan pairs provided. "
             "You MUST eliminate at least 1 less promising pair (you can eliminate more if needed). "
             "Improve the remaining pairs if needed. "
             "If only 1 pair remains and it's good, say 'Plan is OK'."),
            ("user", "{pairs}")
        ])
    
    def critique_hypotheses(self, hypotheses: str) -> str:
        """
        Critique and improve hypotheses.
        
        Args:
            hypotheses: Hypotheses to critique
            
        Returns:
            Improved hypotheses
        """
        self._print("Critiquing hypotheses...")
        
        messages = self.hypotheses_prompt.format_messages(hypotheses=hypotheses)
        critique = self._call_llm(messages)
        
        self._log_interaction(
            input_data=f"Hypotheses: {hypotheses}",
            output_data=critique
        )
        
        self._print("✅ Hypotheses critiqued")
        
        return critique
    
    def critique_plans(self, pairs: str) -> Tuple[str, bool]:
        """
        Critique hypothesis-plan pairs and eliminate at least one.
        
        Args:
            pairs: Hypothesis-plan pairs to critique
            
        Returns:
            Tuple of (critique/improved pairs, is_approved)
        """
        self._print("Critiquing hypothesis-plan pairs...")
        
        messages = self.plans_prompt.format_messages(pairs=pairs)
        critique = self._call_llm(messages)
        
        # Check if only 1 pair remains and is approved
        is_approved = "plan is ok" in critique.lower() or "plan ok" in critique.lower()
        
        self._log_interaction(
            input_data=f"Pairs: {pairs}",
            output_data=critique
        )
        
        if is_approved:
            self._print("✅ Plan approved! Only 1 pair remains.")
        else:
            self._print("🔄 Eliminated at least 1 pair")
        
        return critique, is_approved