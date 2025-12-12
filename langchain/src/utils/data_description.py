import numpy as np

def load_description(path: str):
    """
    Load a plain-text description file.
    Expected structure:
        colname unit min max description
    Or any free text; the caller decides how to parse.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Flexible parsing: ignore comment lines
    content = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    return content

def as_prompt_block(description_lines):
    """Convert description lines into a readable block for the LLM."""
    block = "\n".join(f"- {ln}" for ln in description_lines)
    return f"DATA usage example file:\n{block}\n"
