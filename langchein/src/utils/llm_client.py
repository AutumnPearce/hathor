"""
LLM Client for Argonne Sophia API.
Handles authentication and API calls.
"""
from typing import List, Optional
from openai import OpenAI
from inference_auth_token import get_access_token
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


class ArgonneLLMClient:
    """
    Client for Argonne Sophia inference API.
    Handles token management and API calls.
    """
    
    def __init__(self, base_url: str = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"):
        """
        Initialize the LLM client.
        
        Args:
            base_url: Base URL for the Argonne API
        """
        self.base_url = base_url
        self._client = None
    
    def _get_client(self) -> OpenAI:
        """
        Get or create OpenAI client with fresh token.
        
        Returns:
            OpenAI client instance
        """
        # Get fresh token each time (in case it expires)
        token = get_access_token()
        
        return OpenAI(
            api_key=token,
            base_url=self.base_url,
        )
    
    def call(self, messages: List[SystemMessage | HumanMessage], model: str) -> AIMessage:
        """
        Call the LLM with messages.
        
        Args:
            messages: List of message objects (SystemMessage, HumanMessage)
            model: Model name to use
            
        Returns:
            AIMessage with response
            
        Raises:
            Exception: If API call fails
        """
        # Extract content from messages
        system_msg = messages[0].content if messages and isinstance(messages[0], SystemMessage) else ""
        user_msg = messages[1].content if len(messages) > 1 and isinstance(messages[1], HumanMessage) else ""
        
        # Get client (with fresh token)
        client = self._get_client()
        
        try:
            # Make API call
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            
            # Extract and return response
            content = response.choices[0].message.content.strip()
            return AIMessage(content=content)
        
        except Exception as e:
            raise Exception(f"LLM API call failed: {str(e)}")
    
    def __call__(self, messages: List, model: str) -> AIMessage:
        """
        Make the client callable directly.
        
        Args:
            messages: List of message objects
            model: Model name to use
            
        Returns:
            AIMessage with response
        """
        return self.call(messages, model)


# Singleton instance for easy import
_client_instance = None


def get_llm_client(base_url: Optional[str] = None) -> ArgonneLLMClient:
    """
    Get or create the LLM client singleton.
    
    Args:
        base_url: Optional base URL (only used on first call)
        
    Returns:
        ArgonneLLMClient instance
    """
    global _client_instance
    
    if _client_instance is None:
        if base_url:
            _client_instance = ArgonneLLMClient(base_url)
        else:
            _client_instance = ArgonneLLMClient()
    
    return _client_instance


def argonne_llm(messages: List, model: str) -> AIMessage:
    """
    Convenience function for direct LLM calls.
    Compatible with agent system expectations.
    
    Args:
        messages: List of message objects
        model: Model name to use
        
    Returns:
        AIMessage with response
    """
    client = get_llm_client()
    return client.call(messages, model)


# Alternative: Simple function-based approach (if you prefer)
def make_client() -> OpenAI:
    """
    Create a new OpenAI client with fresh token.
    
    Returns:
        OpenAI client instance
    """
    token = get_access_token()
    return OpenAI(
        api_key=token,
        base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    )


def simple_llm_call(messages: List, model: str) -> AIMessage:
    """
    Simple LLM call without client management.
    
    Args:
        messages: List of message objects
        model: Model name
        
    Returns:
        AIMessage with response
    """
    client = make_client()
    
    system_msg = messages[0].content if messages else ""
    user_msg = messages[1].content if len(messages) > 1 else ""
    
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    
    return AIMessage(content=resp.choices[0].message.content.strip())


if __name__ == "__main__":
    # Example usage
    from langchain_core.messages import SystemMessage, HumanMessage
    
    # Method 1: Using the client class
    client = get_llm_client()
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Say hello!")
    ]
    response = client.call(messages, "openai/gpt-oss-120b")
    print(f"Response: {response.content}")
    
    # Method 2: Using convenience function
    response = argonne_llm(messages, "openai/gpt-oss-120b")
    print(f"Response: {response.content}")
    
    # Method 3: Simple function (your original style)
    response = simple_llm_call(messages, "openai/gpt-oss-120b")
    print(f"Response: {response.content}")