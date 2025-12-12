# src/config/__init__.py

from .settings import *
from .prompts import (
    literature_prompt,
    critic_literature_prompt,
    reasoner_prompt,
    critic_prompt,
    coder_prompt,
)

__all__ = [
    "literature_prompt",
    "critic_literature_prompt",
    "reasoner_prompt",
    "critic_prompt",
    "coder_prompt",
]
