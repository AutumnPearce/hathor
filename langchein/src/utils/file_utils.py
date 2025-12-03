# src/utils/file_utils.py

import os


def read_code_from_file(file_path: str) -> str:
    """Utility function to read code from a local file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_codes_from_folder(folder_path: str) -> str:
    """Utility function to read all .py files from a local folder."""
    code_collection = ""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".py"):
                full_path = os.path.join(folder_path, filename)
                code_collection += read_code_from_file(full_path) + "\n\n"
    return code_collection


def save_answer_to_file(answer: str, file_path: str, title: str | None = None) -> None:
    """Utility function to save answer to a local file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        if title:
            f.write(f"{title}\n")
        f.write("=" * 70 + "\n\n")
        f.write(answer)
        f.write("\n\n" + "=" * 70 + "\n")


def save_code_to_file(code: str, file_path: str) -> None:
    """Save Python code to file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
