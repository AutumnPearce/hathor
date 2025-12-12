# src/agents/base_agent.py

# from abc import ABC
# from typing import List

# from langchain_core.messages import BaseMessage, AIMessage

# from ..utils import argonne_llm


# class BaseAgent(ABC):
#     """Base class for all LLM-driven agents."""

#     def __init__(self, model: str):
#         self.model = model

#     def _call_llm(self, messages: List[BaseMessage]) -> AIMessage:
#         return argonne_llm(messages, self.model)
from abc import ABC
from typing import List

from langchain_core.messages import BaseMessage, AIMessage
from ..utils import argonne_llm
from ..utils.data_description import load_description


class BaseAgent(ABC):
    """Base class for all LLM-driven agents."""

    def __init__(self, model: str, description_file: str | None = None):
        self.model = model
        self.data_description = None

        if description_file:
            # Load plain text metadata and convert to a prompt block
            lines = load_description(description_file)
            self.data_description = "\n".join(lines)

    def _inject_description(self, user_prompt: str) -> str:
        """Append the dataset description if available."""
        if not self.data_description:
            return user_prompt
        return (
            f"{user_prompt}\n\n"
            "=== DATA DESCRIPTION (DO NOT IGNORE) ===\n"
            f"{self.data_description}\n"
            "=== END DESCRIPTION ===\n"
        )

    def _call_llm(self, messages: List[BaseMessage]) -> AIMessage:
        """
        Injects metadata into the last user message before calling the LLM.
        """
        if (
            self.data_description
            and len(messages) > 0
            and messages[-1].type == "human"
        ):
            messages[-1].content = self._inject_description(messages[-1].content)

        return argonne_llm(messages, self.model)

