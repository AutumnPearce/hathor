"""
File utility functions for reading and saving data.
"""
import os
from typing import Optional


def save_answer_to_file(answer: str, file_path: str, title: Optional[str] = None) -> None:
    """
    Save answer to a file with nice formatting.
    
    Args:
        answer: Content to save
        file_path: Path to save file
        title: Optional title header
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "w", encoding='utf-8') as f:
        f.write("="*70 + "\n")
        if title:
            f.write(f"{title}\n")
        f.write("="*70 + "\n\n")
        f.write(answer)
        f.write("\n\n" + "="*70 + "\n")


def read_code_from_file(file_path: str) -> str:
    """
    Read code from a single file.
    
    Args:
        file_path: Path to file
        
    Returns:
        File contents as string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_codes_from_folder(folder_path: str) -> str:
    """
    Read all Python files from a folder.
    
    Args:
        folder_path: Path to folder
        
    Returns:
        Concatenated contents of all .py files
    """
    code_collection = ""
    
    if not os.path.exists(folder_path):
        return code_collection
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".py"):
            file_path = os.path.join(folder_path, file_name)
            code_collection += read_code_from_file(file_path) + "\n\n"
    
    return code_collection