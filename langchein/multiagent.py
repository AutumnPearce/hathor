import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import re
import ast
import traceback
from typing import Any, Dict, List

from openai import OpenAI
from inference_auth_token import get_access_token

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


# ============================================
# 1. ARGONNE LLM WRAPPER (LANGCHAIN-COMPATIBLE)
# ============================================


def make_client() -> OpenAI:
    token = get_access_token()
    return OpenAI(
        api_key=token,
        base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    )

client = make_client()

MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
models = [    "google/gemma-3-27b-it",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-Large-Instruct-2407",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"]

LITERATURE_MODEL = models[-2]
REASONER_MODEL = models[-2]
CRITIC_MODEL   = models[-2]
CODER_MODEL    = models[-2]
RUNNER_MODEL   = None   # Runner is local exec, not LLM


def argonne_llm(messages: List[SystemMessage | HumanMessage], model: str) -> AIMessage:
    """
    Simple LangChain-compatible wrapper for the Argonne Sophia inference API.
    Expects a list: [SystemMessage, HumanMessage].
    Returns AIMessage(content="...").
    """
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
# 2. CODE EXTRACTOR TOOL
# ============================================

def _strip_markdown(text: str) -> str:
    """Remove code fences, stray backticks, and markdown wrappers."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = text.replace("`", "")
    return text.strip()


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


@tool
def code_extractor_tool(text: str) -> str:
    """
    Extract Python code safely from an LLM response.
    Priority:
    1. ```python ... ```
    2. ``` ... ```
    3. fallback: raw text
    """
    py_blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL)

    if not py_blocks:
        py_blocks = re.findall(r"```(.*?)```", text, flags=re.DOTALL)

    if py_blocks:
        code = "\n\n".join(block.strip() for block in py_blocks)
    else:
        code = _strip_markdown(text)

    code = _strip_markdown(code)

    if not _is_valid_python(code):
        print("⚠️ Warning: extracted code may not be valid Python.")

    return code


# ============================================
# 3. SAFE CODE RUNNER TOOL
# ============================================

@tool
def run_code(code: str) -> Dict[str, Any]:
    """Execute Python code in a clean environment."""
    local_env: Dict[str, Any] = {}
    try:
        exec(code, {}, local_env)
        return {"success": True, "output": "Execution successful."}
    except Exception:
        return {"success": False, "output": traceback.format_exc()}


# ============================================
# 4. AGENT PROMPTS
# ============================================

literature_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a scientific literature review agent. Analyze the task and propose 5 interesting, "
     "realistic ideas for visualization/analysis. Be specific and practical."),
    ("user", "{task}")
])

reasoner_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful reasoning agent. Based on the literature review and ideas provided, "
     "choose the BEST idea and create a clear step-by-step plan to implement it. "
     "Do NOT write code."),
    ("user", "{task}")
])

critic_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a critic agent. Improve or fix the provided plan. "
     "If it is already good, say 'Plan OK'."),
    ("user", "{plan}")
])

coder_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Python coding agent.\n"
     "STRICT RULES:\n"
     "1. Output ONLY raw Python code (no backticks, no markdown).\n"
     "2. NO explanations or natural language.\n"
     "3. Code must be fully self-contained and runnable.\n"
    ),
    ("user", "{instructions}")
])


def read_code_from_file(file_path: str) -> str:
    """Utility function to read code from a local file."""
    with open(file_path, 'r') as file:
        return file.read()


def read_codes_from_folder(folder_path: str) -> str:
    """Utility function to read all code files from a local folder."""
    code_collection = ""
    if os.path.exists(folder_path):
        for file_path in os.listdir(folder_path):
            if file_path.endswith(".py"):
                code_collection += read_code_from_file(os.path.join(folder_path, file_path)) + "\n\n"
    return code_collection


# ============================================
# 5. AGENT RUNNERS
# ============================================

def run_literature_review(task: str) -> str:
    """Run literature review and get 5 ideas"""
    messages = literature_prompt.format_messages(task=task)
    ai = argonne_llm(messages, model=LITERATURE_MODEL)
    return ai.content


def run_reasoner(task: str, literature_review: str, previous_codes: str = "") -> str:
    """Choose best idea from literature review and create detailed plan"""
    full_task = f"{task}\n\nLiterature Review and Ideas:\n{literature_review}"
    
    if previous_codes:
        full_task += f"\n\nPrevious codes for reference:\n{previous_codes}"
    
    messages = reasoner_prompt.format_messages(task=full_task)
    ai = argonne_llm(messages, model=REASONER_MODEL)
    return ai.content


def run_critic(plan: str) -> str:
    """Critique and improve the plan"""
    formatted = critic_prompt.format_messages(plan=plan)
    ai = argonne_llm(formatted, model=CRITIC_MODEL)
    return ai.content


def run_coder(instructions: str) -> str:
    """Generate code based on instructions"""
    formatted = coder_prompt.format_messages(instructions=instructions)
    ai = argonne_llm(formatted, model=CODER_MODEL)
    return code_extractor_tool.invoke(ai.content)


# ============================================
# 6. MAIN MULTI-AGENT PIPELINE
# ============================================

def multi_agent(Literature_prompt: str, Idea_choose_prompt: str, 
                Idea_critic_prompt: str, Code_developer_prompt: str):
    
    # 0) Literature Review Agent - Get 5 ideas
    print("\n" + "="*50)
    print("STEP 0: LITERATURE REVIEW AGENT")
    print("="*50 + "\n")
    
    literature_review = run_literature_review(Literature_prompt)
    print("Literature Review & 5 Ideas:\n", literature_review)

    # 1) Reasoner Agent - Choose 1 idea and make detailed plan
    print("\n" + "="*50)
    print("STEP 1: REASONING AGENT")
    print("="*50 + "\n")
    
    # Read previous codes if available
    previous_codes = read_codes_from_folder("./plotting_codes/")
    
    plan = run_reasoner(Idea_choose_prompt, literature_review, previous_codes)
    print("Selected Idea & Detailed Plan:\n", plan)

    # 2) Critic Agent - Review and improve the plan
    print("\n" + "="*50)
    print("STEP 2: CRITIC AGENT")
    print("="*50 + "\n")
    
    critique = run_critic(plan)
    print("Critique:\n", critique)

    # Use improved plan if critic made changes, otherwise use original
    improved_plan = plan if "plan ok" in critique.lower() else critique

    # 3) Coder Agent - Generate code from the plan
    print("\n" + "="*50)
    print("STEP 3: CODER AGENT")
    print("="*50 + "\n")

    # Build full instructions for coder
    coder_instructions = (
        f"{Code_developer_prompt}\n\n"
        f"Plan to implement:\n{improved_plan}\n\n"
    )
    
    if previous_codes:
        coder_instructions += f"Reference codes:\n{previous_codes}"

    code = run_coder(coder_instructions)
    print("Generated Code:\n")
    print(code)

    # 4) Runner Agent - Execute and debug code
    max_iterations = 5
    iteration = 0
    
    while iteration < max_iterations:
        print("\n" + "="*50)
        print(f"STEP 4: RUNNER AGENT (Iteration {iteration + 1})")
        print("="*50 + "\n")

        # Check if code is valid Python
        if not _is_valid_python(code):
            print("❌ Invalid Python syntax detected, asking coder to fix...\n")
            fix_instructions = (
                "The following code has invalid Python syntax. Fix it and output ONLY valid Python code.\n\n"
                f"CODE:\n{code}"
            )
            code = run_coder(fix_instructions)
            print("Fixed Code:\n", code)
            iteration += 1
            continue

        # Try to execute the code
        result = run_code.invoke(code)
        print(result["output"])

        if result["success"]:
            print("\n🎉 SUCCESS: Code executed successfully!")
            break

        print("\n⚠️ ERROR detected → sending to coder for debugging...\n")
        
        # Ask coder to fix the error
        fix_instructions = (
            "Fix the following code so it runs without errors. "
            "Output ONLY raw Python code.\n\n"
            f"ERROR:\n{result['output']}\n\n"
            f"CODE:\n{code}"
        )
        code = run_coder(fix_instructions)
        print("Debugged Code:\n", code)
        iteration += 1

    if iteration >= max_iterations:
        print(f"\n❌ Maximum iterations ({max_iterations}) reached. Saving last version anyway...")

    # Save final code
    os.makedirs("./executed_codes", exist_ok=True)
    with open("./executed_codes/plot.py", "w") as f:
        f.write(code)

    print("\n💾 Saved final code → ./executed_codes/plot.py\n")


# ============================================
# 7. RUN
# ============================================

if __name__ == "__main__":
    Literature_prompt = """
I want to do plots for MEGATRON cutout data of gas cells in a halo. It's stored in a binary file. 
The data contains positions (x,y,z), levels, ne (electron number density), dx (cell size), and other 
features for each gas cell.

I want to create a 2D image of a some parameter projected along the z-axis.

Please do a literature review and propose 5 interesting features I can plot to learn something about 
the clusters. It should be something simple that I can plot. It will be used by other AI agents to 
create code, so propose realistic ideas.

Give me 5 different ideas.
"""

    Idea_choose_prompt = """
I need to do plots for MEGATRON cutout data of gas cells in a halo. It's stored in a binary file. 
The data contains positions (x,y,z), levels, ne (electron number density), dx (cell size), and other 
features for each gas cell.

I want to create a 2D image of some parameter projected along the z-axis.

Based on the literature review and 5 ideas above, choose the 1 idea that is the most interesting and 
realistic to implement, and create a detailed step-by-step plan to accomplish the task.
"""

    Idea_critic_prompt = """
I need to do plots for MEGATRON cutout data of gas cells in a halo. It's stored in a binary file. 
The data contains positions (x,y,z), levels, ne (electron number density), dx (cell size), and other 
features for each gas cell.

I want to create a 2D image of some parameter projected along the z-axis.

Based on the plan from the reasoning agent, improve or fix the provided plan. If it is already good, 
say 'Plan OK'.
"""

    Code_developer_prompt = """
I need to do plots for MEGATRON cutout data of gas cells in a halo. It's stored in a binary file 
at path: "../../dataset_examples/halo_3517_gas.bin"

The data contains positions (x,y,z), levels, ne (electron number density), dx (cell size), and other 
features for each gas cell.

Create a Python code to accomplish the task based on the plan provided. The image should have a 
resolution of 512x512 pixels and cover a box size of 20 Mpc at redshift z=0.5. The levels range 
from 12 to 18.
"""

    multi_agent(Literature_prompt, Idea_choose_prompt, Idea_critic_prompt, Code_developer_prompt)