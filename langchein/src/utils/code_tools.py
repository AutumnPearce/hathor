"""
Code utility tools for extracting, validating, and processing Python code.
"""
import re
import ast
from typing import Optional


def strip_markdown(text: str) -> str:
    """
    Remove code fences, stray backticks, and markdown wrappers.
    
    Args:
        text: Text with potential markdown formatting
        
    Returns:
        Cleaned text without markdown
    """
    # Remove code blocks with language specifier
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove remaining backticks
    text = text.replace("`", "")
    return text.strip()


def is_valid_python(code: str) -> bool:
    """
    Check if code is syntactically valid Python.
    
    Args:
        code: Python code string
        
    Returns:
        True if valid Python syntax, False otherwise
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


def extract_code_blocks(text: str) -> list[str]:
    """
    Extract all code blocks from markdown-formatted text.
    
    Args:
        text: Text potentially containing code blocks
        
    Returns:
        List of extracted code blocks
    """
    # Try to find python-specific code blocks first
    py_blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL)
    
    if not py_blocks:
        # Fall back to generic code blocks
        py_blocks = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    
    return [block.strip() for block in py_blocks]


def extract_python_code(text: str) -> str:
    """
    Extract Python code from LLM response.
    Priority:
    1. ```python ... ```
    2. ``` ... ```
    3. fallback: raw text with markdown stripped
    
    Args:
        text: LLM response text
        
    Returns:
        Extracted Python code
    """
    # Try to extract code blocks
    code_blocks = extract_code_blocks(text)
    
    if code_blocks:
        # Join multiple code blocks with double newline
        code = "\n\n".join(code_blocks)
    else:
        # No code blocks found, strip markdown from entire text
        code = strip_markdown(text)
    
    # Clean up any remaining markdown
    code = strip_markdown(code)
    
    return code


def validate_and_extract(text: str, warn: bool = True) -> tuple[str, bool]:
    """
    Extract and validate Python code from text.
    
    Args:
        text: Text containing Python code
        warn: Whether to print warning if code is invalid
        
    Returns:
        Tuple of (extracted_code, is_valid)
    """
    code = extract_python_code(text)
    is_valid = is_valid_python(code)
    
    if not is_valid and warn:
        print("⚠️ Warning: Extracted code may not be valid Python.")
    
    return code, is_valid


def fix_common_issues(code: str) -> str:
    """
    Attempt to fix common code issues.
    
    Args:
        code: Python code with potential issues
        
    Returns:
        Fixed code
    """
    # Remove any leading/trailing whitespace
    code = code.strip()
    
    # Remove any BOM (Byte Order Mark) characters
    code = code.replace('\ufeff', '')
    
    # Fix common indentation issues (tabs to spaces)
    code = code.replace('\t', '    ')
    
    # Remove any null bytes
    code = code.replace('\x00', '')
    
    return code


def get_imports(code: str) -> list[str]:
    """
    Extract all import statements from code.
    
    Args:
        code: Python code
        
    Returns:
        List of import statements
    """
    imports = []
    
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join([alias.name for alias in node.names])
                imports.append(f"from {module} import {names}")
    except:
        # If parsing fails, use regex fallback
        import_pattern = r"^(?:from\s+[\w.]+\s+)?import\s+.+$"
        imports = re.findall(import_pattern, code, re.MULTILINE)
    
    return imports


def get_function_names(code: str) -> list[str]:
    """
    Extract all function names defined in code.
    
    Args:
        code: Python code
        
    Returns:
        List of function names
    """
    function_names = []
    
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
    except:
        # If parsing fails, use regex fallback
        func_pattern = r"^def\s+(\w+)\s*\("
        function_names = re.findall(func_pattern, code, re.MULTILINE)
    
    return function_names


def get_class_names(code: str) -> list[str]:
    """
    Extract all class names defined in code.
    
    Args:
        code: Python code
        
    Returns:
        List of class names
    """
    class_names = []
    
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_names.append(node.name)
    except:
        # If parsing fails, use regex fallback
        class_pattern = r"^class\s+(\w+)"
        class_names = re.findall(class_pattern, code, re.MULTILINE)
    
    return class_names


def count_lines(code: str) -> dict[str, int]:
    """
    Count different types of lines in code.
    
    Args:
        code: Python code
        
    Returns:
        Dictionary with counts of total, code, comment, and blank lines
    """
    lines = code.split('\n')
    
    total = len(lines)
    blank = sum(1 for line in lines if not line.strip())
    comment = sum(1 for line in lines if line.strip().startswith('#'))
    code_lines = total - blank - comment
    
    return {
        "total": total,
        "code": code_lines,
        "comment": comment,
        "blank": blank
    }


def has_main_block(code: str) -> bool:
    """
    Check if code has a if __name__ == "__main__" block.
    
    Args:
        code: Python code
        
    Returns:
        True if main block exists, False otherwise
    """
    pattern = r'if\s+__name__\s*==\s*["\']__main__["\']\s*:'
    return bool(re.search(pattern, code))


class CodeAnalyzer:
    """
    Utility class for analyzing Python code.
    """
    
    def __init__(self, code: str):
        """
        Initialize analyzer with code.
        
        Args:
            code: Python code to analyze
        """
        self.code = code
        self.is_valid = is_valid_python(code)
    
    def get_summary(self) -> dict:
        """
        Get comprehensive summary of the code.
        
        Returns:
            Dictionary with code statistics
        """
        return {
            "is_valid": self.is_valid,
            "lines": count_lines(self.code),
            "imports": get_imports(self.code),
            "functions": get_function_names(self.code),
            "classes": get_class_names(self.code),
            "has_main": has_main_block(self.code)
        }
    
    def print_summary(self):
        """Print formatted summary of the code."""
        summary = self.get_summary()
        
        print("="*50)
        print("CODE ANALYSIS")
        print("="*50)
        print(f"Valid Python: {'✅ Yes' if summary['is_valid'] else '❌ No'}")
        print(f"Total Lines: {summary['lines']['total']}")
        print(f"  - Code: {summary['lines']['code']}")
        print(f"  - Comments: {summary['lines']['comment']}")
        print(f"  - Blank: {summary['lines']['blank']}")
        print(f"Has main block: {'✅ Yes' if summary['has_main'] else '❌ No'}")
        print(f"Imports: {len(summary['imports'])}")
        print(f"Functions: {len(summary['functions'])}")
        print(f"Classes: {len(summary['classes'])}")
        print("="*50)


# Convenience function for direct use
def analyze_code(code: str) -> CodeAnalyzer:
    """
    Create a CodeAnalyzer instance for the given code.
    
    Args:
        code: Python code to analyze
        
    Returns:
        CodeAnalyzer instance
    """
    return CodeAnalyzer(code)


if __name__ == "__main__":
    # Example usage
    sample_code = """
import numpy as np
import matplotlib.pyplot as plt

def plot_data(x, y):
    '''Plot some data'''
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plot_data(x, y)
    """
    
    analyzer = analyze_code(sample_code)
    analyzer.print_summary()