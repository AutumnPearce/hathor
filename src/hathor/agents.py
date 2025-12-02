import autogen

class BrainstormerAgent(autogen.AssistantAgent):
    """Custom agent with private literature knowledge."""
    
    def __init__(self, name, system_message, llm_config):
        super().__init__(name, system_message, llm_config)
        self.literature = []