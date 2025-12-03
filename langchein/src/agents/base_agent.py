# src/agents/base_agent.py

from abc import ABC
from typing import List

from langchain_core.messages import BaseMessage, AIMessage

from ..utils import argonne_llm


class BaseAgent(ABC):
    """Base class for all LLM-driven agents."""

    def __init__(self, model: str):
        self.model = model

    def _call_llm(self, messages: List[BaseMessage]) -> AIMessage:
        return argonne_llm(messages, self.model)
