# src/agents/literature_agent.py

from ..config import literature_prompt
from .base_agent import BaseAgent


class LiteratureAgent(BaseAgent):
    def generate_hypotheses(self, task: str, num_hypotheses: int = 8) -> str:
        messages = literature_prompt.format_messages(
            task=task, num_hypotheses=num_hypotheses
        )
        ai = self._call_llm(messages)
        return ai.content
