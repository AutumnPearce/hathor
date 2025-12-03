# src/agents/critic_agent.py

from ..config import critic_literature_prompt, critic_prompt
from .base_agent import BaseAgent


class CriticAgent(BaseAgent):
    """
    Handles both:
    - Literature hypotheses critique
    - Hypothesis-plan pair critique
    """

    def review_literature(self, hypotheses: str) -> str:
        messages = critic_literature_prompt.format_messages(hypotheses=hypotheses)
        ai = self._call_llm(messages)
        return ai.content

    def review_pairs(self, pairs: str) -> str:
        messages = critic_prompt.format_messages(pairs=pairs)
        ai = self._call_llm(messages)
        return ai.content
