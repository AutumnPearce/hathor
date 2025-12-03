# src/utils/llm_client.py

from typing import List

from openai import OpenAI
from inference_auth_token import get_access_token
from langchain_core.messages import BaseMessage, AIMessage


def make_client() -> OpenAI:
    token = get_access_token()
    return OpenAI(
        api_key=token,
        base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    )


# Reuse a single client
_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = make_client()
    return _client


def argonne_llm(messages: List[BaseMessage], model: str) -> AIMessage:
    """
    Simple LangChain-compatible wrapper for the Argonne Sophia inference API.
    Expects a list: [SystemMessage, HumanMessage].
    Returns AIMessage(content="...").
    """
    system_msg = messages[0].content if messages else ""
    user_msg = messages[1].content if len(messages) > 1 else ""

    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return AIMessage(content=resp.choices[0].message.content.strip())
