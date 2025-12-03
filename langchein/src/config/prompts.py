"""
Prompt templates for all agents.
Centralized configuration for easy modification.
"""
from langchain_core.prompts import ChatPromptTemplate


class AgentPrompts:
    """
    Collection of all prompt templates used by agents.
    """
    
    # Literature Agent Prompt
    LITERATURE = ChatPromptTemplate.from_messages([
        ("system",
         "You are a scientific literature review agent and expert in Galaxy formation. "
         "Analyze the task and propose {num_hypotheses} interesting, realistic hypotheses "
         "that can be checked by visualization/analysis. Be specific and practical. "
         "Don't just repeat the task and don't give plans for implementation those hypotheses."),
        ("user", "{task}")
    ])
    
    # Critic Agent - Hypotheses Critique
    CRITIC_HYPOTHESES = ChatPromptTemplate.from_messages([
        ("system",
         "You are a critic agent and expert in Galaxy formation. "
         "Analyze hypotheses proposed by the literature review, get rid of impractical ones, "
         "and improve the rest into realistic hypotheses that can be checked by visualization/analysis. "
         "Be specific and practical. Don't just repeat the task and don't give plans for implementation."),
        ("user", "{hypotheses}")
    ])
    
    # Critic Agent - Plans Critique
    CRITIC_PLANS = ChatPromptTemplate.from_messages([
        ("system",
         "You are a critic agent. Review the hypothesis-plan pairs provided. "
         "You MUST eliminate at least 1 less promising pair (you can eliminate more if needed). "
         "Improve the remaining pairs if needed. "
         "If only 1 pair remains and it's good, say 'Plan is OK'."),
        ("user", "{pairs}")
    ])
    
    # Reasoner Agent Prompt
    REASONER = ChatPromptTemplate.from_messages([
        ("system",
         "You are a careful reasoning agent. Based on the hypotheses provided, "
         "create a detailed plan for EACH hypothesis to implement it in Python doing visualizations/analysis. "
         "Output hypothesis-plan pairs. Do NOT write code."),
        ("user", "{task}")
    ])
    
    # Coder Agent Prompt
    CODER = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Python coding agent.\n"
         "STRICT RULES:\n"
         "1. Output ONLY raw Python code (no backticks, no markdown).\n"
         "2. NO explanations or natural language.\n"
         "3. Code must be fully self-contained and runnable.\n"),
        ("user", "{instructions}")
    ])


class TaskPrompts:
    """
    Task-specific prompts for the pipeline.
    """
    
    LITERATURE_TASK = """
I want to do plots for MEGATRON cutout data of gas cells in a halo. It's stored in a binary file. 
The data contains positions (x,y,z), levels, ne (electron number density), dx (cell size), and other 
features for each gas cell.

I want to create 2D images of parameters projected along the z-axis to learn something interesting 
about galaxy clusters.

Please propose interesting hypotheses about galaxy clusters that can be tested through visualization 
and analysis of this data.
"""
    
    CODE_GENERATION_TASK = """
I need to create visualization code for MEGATRON cutout data of gas cells in a halo. 
The binary file is at path: "{data_path}"

The data contains positions (x,y,z), levels, ne (electron number density), dx (cell size), and other 
features for each gas cell.

Create Python code to implement the hypothesis and plan. The image should have a resolution of {npix}x{npix} 
pixels and cover a box size of {boxsize} Mpc at redshift z={redshift}. The levels range from {lmin} to {lmax}.
"""
    
    @staticmethod
    def format_code_task(
        data_path: str = "/Users/yk2047/Documents/GitHub/hathor/dataset_examples/halo_3517_gas.bin",
        npix: int = 512,
        boxsize: float = 20.0,
        redshift: float = 0.5,
        lmin: int = 12,
        lmax: int = 18
    ) -> str:
        """Format the code generation task with parameters."""
        return TaskPrompts.CODE_GENERATION_TASK.format(
            data_path=data_path,
            npix=npix,
            boxsize=boxsize,
            redshift=redshift,
            lmin=lmin,
            lmax=lmax
        )


class CoderInstructions:
    """
    Special instructions for the coder agent.
    """
    
    USE_MAKE_IMAGE = (
        "\n\nIMPORTANT: Use the make_image() function from the reference codes. "
        "DO NOT manually calculate pixel positions. The make_image() function handles "
        "hierarchical AMR binning correctly. Pass positions, levels, features, dx, "
        "and parameters (view_dir='z', npix=512, lmin=12, lmax=18, redshift=0.5, boxsize=20.0)."
    )
    
    INCLUDE_FUNCTIONS = (
        "\n\nIf you use functions like read_megatron_cutout(), write it in the code, "
        "don't just import it elsewhere."
    )
    
    @staticmethod
    def format_save_instruction(output_path: str) -> str:
        """Format the save instruction."""
        return f"\n\nSave the final plot as '{output_path}'."