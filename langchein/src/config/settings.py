"""
Configuration settings for the multi-agent pipeline.
"""
import os
from typing import Dict


class ModelConfig:
    """
    LLM model configurations for different agents.
    """
    
    # Available models
    AVAILABLE_MODELS = [
        "google/gemma-3-27b-it",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistralai/Mistral-Large-Instruct-2407",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
    ]
    
    # Default model assignments per agent
    LITERATURE_MODEL = "openai/gpt-oss-120b"
    CRITIC_MODEL = "openai/gpt-oss-120b"
    REASONER_MODEL = "openai/gpt-oss-120b"
    CODER_MODEL = "openai/gpt-oss-120b"
    
    @classmethod
    def get_all_models(cls) -> Dict[str, str]:
        """Get all model assignments as a dictionary."""
        return {
            "literature": cls.LITERATURE_MODEL,
            "critic": cls.CRITIC_MODEL,
            "reasoner": cls.REASONER_MODEL,
            "coder": cls.CODER_MODEL,
        }
    
    @classmethod
    def set_model(cls, agent_name: str, model: str):
        """
        Set model for a specific agent.
        
        Args:
            agent_name: Name of agent (literature, critic, reasoner, coder)
            model: Model name
        """
        if model not in cls.AVAILABLE_MODELS:
            print(f"⚠️ Warning: {model} not in available models list")
        
        if agent_name == "literature":
            cls.LITERATURE_MODEL = model
        elif agent_name == "critic":
            cls.CRITIC_MODEL = model
        elif agent_name == "reasoner":
            cls.REASONER_MODEL = model
        elif agent_name == "coder":
            cls.CODER_MODEL = model
        else:
            raise ValueError(f"Unknown agent: {agent_name}")


class PathConfig:
    """
    File and directory path configurations.
    """
    
    # Base directories
    OUTPUT_DIR = "./outputs"
    REFERENCE_CODES_DIR = "./plotting_codes"
    
    # Output subdirectories
    IDEAS_DIR = os.path.join(OUTPUT_DIR, "Ideas")
    FIGS_DIR = os.path.join(OUTPUT_DIR, "figs")
    EXECUTED_CODES_DIR = os.path.join(OUTPUT_DIR, "executed_codes")
    
    # Data path
    DATA_PATH = "/Users/yk2047/Documents/GitHub/hathor/dataset_examples/halo_3517_gas.bin"
    
    # Output files
    INITIAL_HYPOTHESES_FILE = os.path.join(IDEAS_DIR, "initial_hypotheses.txt")
    REFINED_HYPOTHESES_FILE = os.path.join(IDEAS_DIR, "refined_hypotheses.txt")
    FINAL_PLAN_FILE = os.path.join(IDEAS_DIR, "final_hypothesis_plan.txt")
    OUTPUT_PLOT_FILE = os.path.join(FIGS_DIR, "output_plot.png")
    FINAL_CODE_FILE = os.path.join(EXECUTED_CODES_DIR, "plot.py")
    
    @classmethod
    def create_directories(cls):
        """Create all necessary output directories."""
        os.makedirs(cls.IDEAS_DIR, exist_ok=True)
        os.makedirs(cls.FIGS_DIR, exist_ok=True)
        os.makedirs(cls.EXECUTED_CODES_DIR, exist_ok=True)


class PipelineConfig:
    """
    Pipeline execution parameters.
    """
    
    # Number of hypotheses to generate
    NUM_HYPOTHESES = 8
    
    # Maximum iterations
    MAX_ELIMINATION_ROUNDS = 10
    MAX_DEBUG_ITERATIONS = 5
    
    # Visualization parameters
    NPIX = 512
    BOXSIZE = 20.0  # Mpc
    REDSHIFT = 0.5
    LEVEL_MIN = 12
    LEVEL_MAX = 18
    
    # Verbosity
    VERBOSE = True
    
    @classmethod
    def get_vis_params(cls) -> Dict:
        """Get visualization parameters as a dictionary."""
        retur