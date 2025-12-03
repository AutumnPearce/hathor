"""
Multi-Agent Pipeline Orchestrator
Coordinates all agents to complete the full workflow.
"""
import os
from typing import Dict, Any

from ..agents.literature_agent import LiteratureAgent
from ..agents.critic_agent import CriticAgent
from ..agents.reasoner_agent import ReasonerAgent
from ..agents.coder_agent import CoderAgent
from ..agents.runner_agent import RunnerAgent
from ..utils.file_utils import save_answer_to_file, read_codes_from_folder


class MultiAgentPipeline:
    """
    Orchestrates the multi-agent workflow for hypothesis-driven research.
    """
    
    def __init__(
        self,
        llm_client,
        literature_model: str,
        critic_model: str,
        reasoner_model: str,
        coder_model: str,
        output_dir: str = "./outputs",
        reference_codes_dir: str = "./plotting_codes",
        verbose: bool = True
    ):
        """
        Initialize the pipeline with all agents.
        
        Args:
            llm_client: Function to call LLM
            literature_model: Model for literature agent
            critic_model: Model for critic agent
            reasoner_model: Model for reasoner agent
            coder_model: Model for coder agent
            output_dir: Directory for saving outputs
            reference_codes_dir: Directory with reference code examples
            verbose: Whether to print progress
        """
        # Initialize agents
        self.lit_agent = LiteratureAgent(literature_model, llm_client, verbose)
        self.critic_agent = CriticAgent(critic_model, llm_client, verbose)
        self.reasoner_agent = ReasonerAgent(reasoner_model, llm_client, verbose)
        self.coder_agent = CoderAgent(coder_model, llm_client, verbose)
        self.runner_agent = RunnerAgent(verbose)
        
        # Configuration
        self.output_dir = output_dir
        self.reference_codes_dir = reference_codes_dir
        self.verbose = verbose
        
        # Create output directories
        os.makedirs(f"{output_dir}/Ideas", exist_ok=True)
        os.makedirs(f"{output_dir}/figs", exist_ok=True)
        os.makedirs(f"{output_dir}/executed_codes", exist_ok=True)
    
    def run(
        self,
        literature_prompt: str,
        code_prompt: str,
        num_hypotheses: int = 8,
        max_elimination_rounds: int = 10,
        max_debug_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Run the complete multi-agent pipeline.
        
        Args:
            literature_prompt: Prompt for literature review
            code_prompt: Prompt for code generation
            num_hypotheses: Number of initial hypotheses
            max_elimination_rounds: Max rounds for hypothesis-plan elimination
            max_debug_iterations: Max iterations for code debugging
            
        Returns:
            Dict with final results and metadata
        """
        self._print_header("MULTI-AGENT PIPELINE START")
        
        # Step 0: Literature Review → Critic Literature
        hypotheses = self._literature_phase(literature_prompt, num_hypotheses)
        
        # Step 1-2: Reasoner ↔ Critic Loop (Elimination)
        final_pair = self._elimination_phase(hypotheses, max_elimination_rounds)
        
        # Step 3-4: Coder → Runner Loop (Debug)
        code = self._coding_phase(final_pair, code_prompt, max_debug_iterations)
        
        self._print_header("PIPELINE COMPLETE")
        
        return {
            "hypotheses": hypotheses,
            "final_pair": final_pair,
            "code": code,
            "success": True
        }
    
    def _literature_phase(self, prompt: str, num_hypotheses: int) -> str:
        """Phase 0: Generate and refine hypotheses."""
        self._print_header("PHASE 0: LITERATURE REVIEW")
        
        # 0a: Generate hypotheses
        self._print_step("0a", "Literature Agent - Generating hypotheses")
        hypotheses = self.lit_agent.run(prompt, num_hypotheses)
        
        save_answer_to_file(
            hypotheses,
            f"{self.output_dir}/Ideas/initial_hypotheses.txt",
            f"Initial {num_hypotheses} Hypotheses"
        )
        self._print(f"💾 Saved → {self.output_dir}/Ideas/initial_hypotheses.txt")
        
        # 0b: Critique hypotheses
        self._print_step("0b", "Critic Agent - Refining hypotheses")
        refined_hypotheses = self.critic_agent.critique_hypotheses(hypotheses)
        
        save_answer_to_file(
            refined_hypotheses,
            f"{self.output_dir}/Ideas/refined_hypotheses.txt",
            "Refined Hypotheses"
        )
        self._print(f"💾 Saved → {self.output_dir}/Ideas/refined_hypotheses.txt")
        
        return refined_hypotheses
    
    def _elimination_phase(self, hypotheses: str, max_rounds: int) -> str:
        """Phase 1-2: Iteratively eliminate hypothesis-plan pairs."""
        self._print_header("PHASE 1-2: HYPOTHESIS-PLAN ELIMINATION")
        
        # Load reference codes
        previous_codes = read_codes_from_folder(self.reference_codes_dir)
        
        # 1: Create initial pairs
        self._print_step("1", "Reasoner Agent - Creating hypothesis-plan pairs")
        pairs = self.reasoner_agent.run(hypotheses, previous_codes)
        
        # 2: Elimination loop
        for round_num in range(max_rounds):
            self._print_step("2", f"Critic Agent - Elimination Round {round_num + 1}")
            
            critique, is_approved = self.critic_agent.critique_plans(pairs)
            
            if is_approved:
                self._print("✅ Only 1 pair remains and is approved!")
                final_pair = pairs
                break
            
            # Update pairs with critic's output
            pairs = critique
            
            self._print("🔄 Refining remaining pairs...")
            self._print_step("1", f"Reasoner Agent - Refinement {round_num + 1}")
            pairs = self.reasoner_agent.refine(pairs, critique, previous_codes)
        else:
            self._print(f"⚠️ Max rounds ({max_rounds}) reached. Using last pair.")
            final_pair = pairs
        
        save_answer_to_file(
            final_pair,
            f"{self.output_dir}/Ideas/final_hypothesis_plan.txt",
            "Final Hypothesis-Plan Pair"
        )
        self._print(f"💾 Saved → {self.output_dir}/Ideas/final_hypothesis_plan.txt")
        
        return final_pair
    
    def _coding_phase(self, plan: str, code_prompt: str, max_iterations: int) -> str:
        """Phase 3-4: Generate and debug code."""
        self._print_header("PHASE 3-4: CODE GENERATION & EXECUTION")
        
        # Load reference codes
        previous_codes = read_codes_from_folder(self.reference_codes_dir)
        
        # 3: Generate code
        self._print_step("3", "Coder Agent - Generating code")
        
        instructions = (
            f"{code_prompt}\n\n"
            f"Hypothesis and Plan to implement:\n{plan}"
        )
        
        output_path = f"{self.output_dir}/figs/output_plot.png"
        code = self.coder_agent.run(instructions, previous_codes, output_path)
        
        # 4: Debug loop
        for iteration in range(max_iterations):
            self._print_step("4", f"Runner Agent - Iteration {iteration + 1}")
            
            result = self.runner_agent.run(code)
            
            if result["success"]:
                self._print("🎉 Code executed successfully!")
                break
            
            self._print(f"⚠️ Error detected: {result['output'][:100]}...")
            self._print("Sending to Coder for debugging...")
            
            code = self.coder_agent.fix_code(code, result["output"])
        else:
            self._print(f"❌ Max iterations ({max_iterations}) reached.")
        
        # Save final code
        with open(f"{self.output_dir}/executed_codes/plot.py", "w") as f:
            f.write(code)
        self._print(f"💾 Saved → {self.output_dir}/executed_codes/plot.py")
        
        return code
    
    def _print_header(self, text: str):
        """Print section header."""
        if self.verbose:
            print("\n" + "="*50)
            print(text)
            print("="*50 + "\n")
    
    def _print_step(self, step: str, text: str):
        """Print step information."""
        if self.verbose:
            print(f"\n[Step {step}] {text}")
    
    def _print(self, text: str):
        """Print message."""
        if self.verbose:
            print(text)