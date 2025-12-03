# src/utils/code_tools.py

import ast
import re
import traceback
from typing import Any, Dict

from langchain_core.tools import tool


def strip_markdown(text: str) -> str:
    """Remove code fences, stray backticks, and markdown wrappers."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = text.replace("`", "")
    return text.strip()


def is_valid_python(code: str) -> bool:
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
        code = strip_markdown(text)

    code = strip_markdown(code)

    if not is_valid_python(code):
        print("⚠️ Warning: extracted code may not be valid Python.")

    return code


@tool
def run_code(code: str) -> Dict[str, Any]:
    """Execute Python code in a clean environment."""
    local_env: Dict[str, Any] = {}
    try:
        exec(code, {}, local_env)
        return {"success": True, "output": "Execution successful."}
    except Exception:
        return {"success": False, "output": traceback.format_exc()}
