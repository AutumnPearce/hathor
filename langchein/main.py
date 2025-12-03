"""
Main entry point for the multi-agent pipeline.
Example usage of the object-oriented structure.
"""
from openai import OpenAI
from inference_auth_token import get_access_token
from langchain_core.messages import AIMessage

from src.pipeline.multi_agent_pipeline import MultiAgentPipeline


# ============================================
# LLM CLIENT SETUP
# ============================================

def make_client() -> OpenAI:
    """Create OpenAI client for Argonne API."""
    token = get_access_token()
    return OpenAI(
        api_key=token,
        base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    )


def argonne_llm(messages, model: str) -> AIMessage:
    """
    LLM client function compatible with the agent system.
    
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


# ============================================
# CONFIGURATION
# ============================================

MODELS = {
    "literature": "openai/gpt-oss-120b",
    "critic": "openai/gpt-oss-120b",
    "reasoner": "openai/gpt-oss-120b",
    "coder": "openai/gpt-oss-120b",
}

PROMPTS = {
    "literature": """
I want to do plots for MEGATRON cutout data of gas cells in a halo. It's stored in a binary file. 
The data contains positions (x,y,z), levels, ne (electron number density), dx (cell size), and other 
features for each gas cell.

I want to create 2D images of parameters projected along the z-axis to learn something interesting 
about galaxy clusters.

Please propose interesting hypotheses about galaxy clusters that can be tested through visualization 
and analysis of this data.
""",
    
    "code": """
I need to create visualization code for MEGATRON cutout data of gas cells in a halo. 
The binary file is at path: "/Users/yk2047/Documents/GitHub/hathor/dataset_examples/halo_3517_gas.bin"

The data contains positions (x,y,z), levels, ne (electron number density), dx (cell size), and other 
features for each gas cell.

Create Python code to implement the hypothesis and plan. The image should have a resolution of 512x512 
pixels and cover a box size of 20 Mpc at redshift z=0.5. The levels range from 12 to 18.
"""
}


# ============================================
# MAIN
# ============================================

def main():
    """Run the multi-agent pipeline."""
    
    print("🚀 Initializing Multi-Agent Pipeline...")
    
    # Create pipeline
    pipeline = MultiAgentPipeline(
        llm_client=argonne_llm,
        literature_model=MODELS["literature"],
        critic_model=MODELS["critic"],
        reasoner_model=MODELS["reasoner"],
        coder_model=MODELS["coder"],
        output_dir="./outputs",
        reference_codes_dir="./plotting_codes",
        verbose=True
    )
    
    # Run pipeline
    result = pipeline.run(
        literature_prompt=PROMPTS["literature"],
        code_prompt=PROMPTS["code"],
        num_hypotheses=8,
        max_elimination_rounds=10,
        max_debug_iterations=5
    )
    
    if result["success"]:
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Check outputs in: ./outputs/")
    else:
        print("\n❌ Pipeline failed")
    
    return result


if __name__ == "__main__":
    main()