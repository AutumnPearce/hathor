"""
Reasoner Agent - creates detailed plans for hypotheses.
"""
from base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate


class ReasonerAgent(BaseAgent):
    """
    Agent that creates detailed implementation plans for hypotheses.
    """
    
    def __init__(self, model: str, llm_client, verbose: bool = True):
        super().__init__("Reasoner Agent", model, llm_client, verbose)
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "You are a careful reasoning agent. Based on the hypotheses provided, "
             "create a detailed plan for EACH hypothesis to implement it in Python doing visualizations/analysis. "
             "Output hypothesis-plan pairs. Do NOT write code."),
            ("user", "{task}")
        ])
    
    def run(self, hypotheses: str, previous_codes: str = "") -> str:
        """
        Create hypothesis-plan pairs for all hypotheses.
        
        Args:
            hypotheses: Hypotheses to create plans for
            previous_codes: Reference code examples (optional)
            
        Returns:
            Hypothesis-plan pairs as string
        """
        self._print("Creating hypothesis-plan pairs...")
        
        # Build task with hypotheses and optional reference codes
        task = f"Hypotheses to create plans for:\n{hypotheses}"
        
        if previous_codes:
            task += f"\n\nPrevious codes for reference:\n{previous_codes}"
        
        messages = self.prompt_template.format_messages(task=task)
        pairs = self._call_llm(messages)
        
        self._log_interaction(
            input_data=f"Hypotheses: {hypotheses}",
            output_data=pairs
        )
        
        self._print("✅ Created hypothesis-plan pairs")
        
        return pairs
    
    def refine(self, pairs: str, feedback: str, previous_codes: str = "") -> str:
        """
        Refine existing hypothesis-plan pairs based on feedback.
        
        Args:
            pairs: Existing hypothesis-plan pairs
            feedback: Feedback from critic
            previous_codes: Reference code examples (optional)
            
        Returns:
            Refined hypothesis-plan pairs
        """
        self._print("Refining hypothesis-plan pairs based on feedback...")
        
        task = (
            f"Previous hypothesis-plan pairs:\n{pairs}\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Please review and refine these pairs based on the feedback."
        )
        
        if previous_codes:
            task += f"\n\nPrevious codes for reference:\n{previous_codes}"
        
        messages = self.prompt_template.format_messages(task=task)
        refined_pairs = self._call_llm(messages)
        
        self._log_interaction(
            input_data=f"Pairs: {pairs}, Feedback: {feedback}",
            output_data=refined_pairs
        )
        
        self._print("✅ Refined pairs")
        
        return refined_pairs