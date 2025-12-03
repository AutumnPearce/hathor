"""
Coder Agent - generates Python code from plans.
"""
from base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
import re
import ast


class CoderAgent(BaseAgent):
    """
    Agent that generates Python code based on plans.
    """
    
    def __init__(self, model: str, llm_client, verbose: bool = True):
        super().__init__("Coder Agent", model, llm_client, verbose)
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "You are a Python coding agent.\n"
             "STRICT RULES:\n"
             "1. Output ONLY raw Python code (no backticks, no markdown).\n"
             "2. NO explanations or natural language.\n"
             "3. Code must be fully self-contained and runnable.\n"),
            ("user", "{instructions}")
        ])
    
    def run(self, instructions: str, reference_codes: str = "", 
            output_path: str = "./outputs/figs/output_plot.png") -> str:
        """
        Generate Python code based on instructions.
        
        Args:
            instructions: Instructions for code generation
            reference_codes: Reference code examples (optional)
            output_path: Where to save the output plot
            
        Returns:
            Generated Python code
        """
        self._print("Generating code...")
        
        # Build full instructions
        full_instructions = instructions
        
        full_instructions += (
            "\n\nIMPORTANT: Use the make_image() function from the reference codes. "
            "DO NOT manually calculate pixel positions. The make_image() function handles "
            "hierarchical AMR binning correctly. Pass positions, levels, features, dx, "
            "and parameters (view_dir='z', npix=512, lmin=12, lmax=18, redshift=0.5, boxsize=20.0)."
        )
        
        full_instructions += (
            f"\n\nIf you use functions like read_megatron_cutout(), write it in the code, "
            f"don't just import it elsewhere.\n\n"
            f"Save the final plot as '{output_path}'."
        )
        
        if reference_codes:
            full_instructions += f"\n\nReference codes:\n{reference_codes}"
        
        messages = self.prompt_template.format_messages(instructions=full_instructions)
        code_response = self._call_llm(messages)
        
        # Extract clean code
        code = self._extract_code(code_response)
        
        self._log_interaction(
            input_data=instructions,
            output_data=code
        )
        
        if self._is_valid_python(code):
            self._print("✅ Generated valid Python code")
        else:
            self._print("⚠️ Warning: Generated code may have syntax issues")
        
        return code
    
    def fix_code(self, code: str, error: str) -> str:
        """
        Fix code based on error message.
        
        Args:
            code: Code with errors
            error: Error message
            
        Returns:
            Fixed code
        """
        self._print("Fixing code based on error...")
        
        fix_instructions = (
            "Fix the following code so it runs without errors. "
            "Output ONLY raw Python code.\n\n"
            f"ERROR:\n{error}\n\n"
            f"CODE:\n{code}"
        )
        
        messages = self.prompt_template.format_messages(instructions=fix_instructions)
        fixed_code_response = self._call_llm(messages)
        
        fixed_code = self._extract_code(fixed_code_response)
        
        self._log_interaction(
            input_data=f"Error: {error}",
            output_data=fixed_code
        )
        
        self._print("✅ Code fixed")
        
        return fixed_code
    
    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove code fences and markdown."""
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = text.replace("`", "")
        return text.strip()
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find code blocks
        py_blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL)
        
        if not py_blocks:
            py_blocks = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
        
        if py_blocks:
            code = "\n\n".join(block.strip() for block in py_blocks)
        else:
            code = self._strip_markdown(text)
        
        code = self._strip_markdown(code)
        
        return code
    
    @staticmethod
    def _is_valid_python(code: str) -> bool:
        """Check if code is syntactically valid Python."""
        try:
            ast.parse(code)
            return True
        except Exception:
            return False