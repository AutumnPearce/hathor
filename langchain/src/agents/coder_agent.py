# src/agents/coder_agent.py

from ..config import coder_prompt
from ..utils import code_extractor_tool
from .base_agent import BaseAgent


# class CoderAgent(BaseAgent):
#     """
#     Generates Python code given instructions.
#     """

#     def generate_code(self, instructions: str) -> str:
#         """
#         Generate code based on instructions.
#         Automatically appends the make_image() usage hint.
#         """
#         instructions += (
#             "\n\nIMPORTANT: Use the make_image() function from the reference codes. "
#             "DO NOT manually calculate pixel positions. The make_image() function handles "
#             "hierarchical AMR binning correctly. Pass positions, levels, features, dx, "
#             "and parameters (view_dir='z', npix=512, lmin=12, lmax=18, redshift=0.5, boxsize=20.0)."
#         )

#         messages = coder_prompt.format_messages(instructions=instructions)
#         ai = self._call_llm(messages)
#         return code_extractor_tool.invoke(ai.content)


class CoderAgent(BaseAgent):
    """
    Generates Python code given instructions.
    """

    def generate_code(self, instructions: str) -> str:
        """
        Generate code based on instructions.
        Automatically appends the make_image() usage hint.
        """
        # instructions += (
        #     "\n\n Important: consider the information from the data type description file to guide your code generation."
        # )

        messages = coder_prompt.format_messages(instructions=instructions)
        ai = self._call_llm(messages)   # <--- now has metadata injection
        return code_extractor_tool.invoke(ai.content)

