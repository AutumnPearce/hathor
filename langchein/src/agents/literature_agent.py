"""
Literature Agent - proposes hypotheses based on scientific literature.
"""
from base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate


class LiteratureAgent(BaseAgent):
    """
    Agent that performs literature review and proposes hypotheses.
    """
    
    def __init__(self, model: str, llm_client, verbose: bool = True):
        super().__init__("Literature Agent", model, llm_client, verbose)
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "You are a scientific literature review agent and expert in Galaxy formation. "
             "Analyze the task and propose {num_hypotheses} interesting, realistic hypotheses "
             "that can be checked by visualization/analysis. Be specific and practical. "
             "Don't just repeat the task and don't give plans for implementation those hypotheses."),
            ("user", "{task}")
        ])
    
    def run(self, task: str, num_hypotheses: int = 8) -> str:
        """
        Generate hypotheses based on the task.
        
        Args:
            task: Research task description
            num_hypotheses: Number of hypotheses to generate
            
        Returns:
            Generated hypotheses as string
        """
        self._print(f"Generating {num_hypotheses} hypotheses...")
        
        messages = self.prompt_template.format_messages(
            task=task,
            num_hypotheses=num_hypotheses
        )
        
        hypotheses = self._call_llm(messages)
        
        self._log_interaction(
            input_data=f"Task: {task}, Num: {num_hypotheses}",
            output_data=hypotheses
        )
        
        self._print(f"✅ Generated {num_hypotheses} hypotheses")
        
        return hypotheses