import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry
from shapely.ops import unary_union
from shapely.geometry import Polygon, Point
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from sklearn.cluster import DBSCAN
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

reasoner_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful reasoning agent. Create a clear step-by-step plan. "
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
     "3. You CAN ONLY import from the following modules:\n"
     "   - geopandas as gpd\n"
     "   - matplotlib.pyplot as plt\n"
     "   - numpy as np\n"
     "   - pandas as pd\n"
     "   - shapely.geometry\n"
     "   - shapely.ops\n"
     "   - sklearn.cluster (DBSCAN)\n"
     "4. You must not import anything not listed above.\n"
     "5. You must use exactly these import names (no alias changes).\n"
     "6. Code must be fully self-contained and runnable.\n"
    ),
    ("user", "{instructions}")
])



# ============================================
# 5. AGENT RUNNERS (FIXED!)
# ============================================

def run_reasoner(task: str) -> str:
    messages = reasoner_prompt.format_messages(task=task)
    ai = argonne_llm(messages, model=REASONER_MODEL)
    return ai.content


def run_critic(plan: str) -> str:
    formatted = critic_prompt.format_messages(plan=plan)
    ai = argonne_llm(formatted, model=CRITIC_MODEL)
    return ai.content


def run_coder(instructions: str) -> str:
    formatted = coder_prompt.format_messages(instructions=instructions)
    ai = argonne_llm(formatted, model=CODER_MODEL)
    return code_extractor_tool.invoke(ai.content)


# ============================================
# 6. MAIN MULTI-AGENT PIPELINE
# ============================================

def multi_agent(task: str):
    # 1) Reasoner
    print("\n==============================")
    print("1) REASONING AGENT")
    print("==============================\n")
    plan = run_reasoner(task)
    print(plan)

    # 2) Critic
    print("\n==============================")
    print("2) CRITIC AGENT")
    print("==============================\n")
    critique = run_critic(plan)
    print(critique)

    improved_plan = plan if "plan ok" in critique.lower() else critique

    # 3) Coder
    print("\n==============================")
    print("3) CODER AGENT")
    print("==============================\n")

    instructions = (
        "Write Python code based on the plan below. "
        "Follow all rules (no backticks, only raw code).\n\n"
        f"Task: {task}\n\nPlan:\n{improved_plan}"
    )

    code = run_coder(instructions)
    print("Initial Code:\n")
    print(code)

    # 4) Runner loop
    while True:
        print("\n==============================")
        print("4) RUNNER AGENT")
        print("==============================\n")

        if not _is_valid_python(code):
            print("❌ Invalid Python code, asking coder to fix...\n")
            instructions = (
                "The following code has invalid Python syntax. Fix it.\n\n"
                f"{code}"
            )
            code = run_coder(instructions)
            print("New Code Draft:\n", code)
            continue

        result = run_code.invoke(code)
        print(result["output"])

        if result["success"]:
            print("\n🎉 SUCCESS: Code executed successfully!")
            break

        print("\n⚠️ ERROR → sending traceback to coder...\n")
        instructions = (
            "Fix the following code so it runs without errors. "
            "Output ONLY raw Python code.\n\n"
            f"ERROR:\n{result['output']}\n\n"
            f"CODE:\n{code}"
        )
        code = run_coder(instructions)
        print("New Code Draft:\n", code)

    # Save working code
    with open("response.py", "w") as f:
        f.write(code)

    print("\n💾 Saved final working code → response.py\n")


# ============================================
# 7. RUN
# ============================================

if __name__ == "__main__":
    Main_prompt = """ \
        "-	Give me a python code to do the next task:
            Analyze and visualize Elk movements in the given dataset. \
            Estimate home ranges and assess habitat preferences using spatial analysis techniques. \
            Identify the spatial clusters of Elk movements. Document the findings with maps and visualizations. Save the figure as \"ELk_multi_agent.png\" \
            using geopandas/geopandas 
        "  
        Domain knowledge:
        -	"Home range" can be defined as the area within which an animal normally lives and finds what it needs for survival. Basically, the home range is the area that an animal travels for its normal daily activities. "Minimum Bounding Geometry" creates a feature class containing polygons which represent a specified minimum bounding geometry enclosing each input feature or each group of input features. "Convex hull" is the smallest convex polygon that can enclose a group of objects, such as a group of points.
        additional information
        -	dataset prewiew:
        [START Preview of ElkMovement/Elk_in_Southwestern_Alberta_2009.geojson] {"type":"FeatureCollection","features":[{"type":"Feature","id":1,"geometry":{"type":"Point","coordinates":[-114.19111179959417,49.536741600111178]},"properties":{"OBJECTID":1,"timestamp":"2009-01-01 01:00:37","long":-114.1911118,"lat":49.536741599999999,"comments":"Carbondale","external_t":-5,"dop":2.3999999999999999,"fix_type_r":"3D","satellite_":0,"height":1375.1900000000001,"crc_status":" ","outlier_ma":0,"sensor_typ":"gps","individual":"Cervus elaphus","tag_ident":"856","ind_ident":"E001","study_name":"Elk in southwestern Alberta","date":1709164800000,"time":" ","timestamp_Converted":1230771637000,"summer_indicator":1}},{"type":"Feature","id":2,"geometry":{"type":"Point","coordinates":[-114.1916239994119,49.536505999952517]},"properties":{"OBJECTID":2,"timestamp":"2009-01-01 03:00:52","long":-114.191624,"lat":49.536506000000003,"comments":"Carbondale","external_t":-6,"dop":2.3999999999999999,"fix_type_r":"3D","satellite_":0,"height":1375.2,"crc_status":" ","outlier_ma":0,"sensor_typ":"gps","individual":"Cervus elaphus","tag_ident":"856","ind_ident":"E001","study_name":"Elk in southwestern Alberta","date":1709164800000,"time":" ","timestamp_Converted":1230778852000,"summer_indicator":1}},{"type":"Feature","id":3,"geometry":{"type":"Point","coordinates":[-114.19169140075056,49.536571800069581]},"properties":{"OBJECTID":3,"timestamp":"2009-01-01 05:00:49","long":-114.1916914,"lat":49.536571799999997,"comments":"Carbondale","external_t":-6,"dop":5.6000000000000014,"fix_type_r":"3D","satellite_":0,"height":1382.0999999999999,"crc_status":" ","outlier_ma":0,"sensor_typ":"gps","individual":"Cervus elaphus","tag_ident":"856","ind_ident":"E001","study_name":"Elk in southwestern Alberta","date":1709164800000,"time":" ","timestamp_Converted":1230786049000,"summer_indicator":1}},...]} [END Preview of ElkMovement/Elk_in_Southwestern_Alberta_2009.geojson]
        "-	Use the following path for the data: \ 
            DATA_PATH = '~/Downloads/benchmark/datasets/ElkMovement/Elk_in_Southwestern_Alberta_2009.geojson' \
            on the fig also plot a histogram of elevation and temperature
        "
        """
    multi_agent(Main_prompt)
