# src/agents/reasoner_agent.py

from ..config import reasoner_prompt
from .base_agent import BaseAgent


class ReasonerAgent(BaseAgent):
    def create_pairs(self, hypotheses: str, previous_codes: str = "") -> str:
        """
        Create hypothesis-plan pairs for all hypotheses.
        Optionally uses previous_codes as reference.
        """
        full_task = f"Hypotheses to create plans for:\n{hypotheses}"
        if previous_codes:
            full_task += f"\n\nPrevious codes for reference:\n{previous_codes}"

        messages = reasoner_prompt.format_messages(task=full_task)
        ai = self._call_llm(messages)
        return ai.content
