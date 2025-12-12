# src/pipeline/test_generation_pipeline.py

import os

from ..agents import CoderAgent, RunnerAgent
from ..config import (
    CODER_MODEL,
    BINARY_FILE_PATH,
    DATA_DESCRIPTION_PATH,
    FINAL_CODE_PATH,
    OUTPUT_FIG_PATH,
    IDEAS_DIR,
    PREVIOUS_CODES_DIR,  # <-- NEW
)
from ..utils import (
    save_code_to_file,
    load_description,      # <-- NEW
    as_prompt_block,       # <-- NEW
    read_codes_from_folder,  # <-- NEW
    read_code_from_file,     # <-- NEW
)


class TestGenerationPipeline:
    """
    Standalone pipeline for testing ONLY:
    - code generation by the CoderAgent
    - execution/debugging by the RunnerAgent

    It loads saved hypothesis/plan instructions from a text file,
    constructs coder instructions (mirroring the main pipeline),
    then runs code generation.
    """

    def __init__(self) -> None:
        self.coder_agent = CoderAgent(CODER_MODEL)
        self.runner_agent = RunnerAgent(self.coder_agent)

    def load_instructions(self, path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Instruction file not found: {path}")

        with open(path, "r") as f:
            return f.read()

    def build_coder_instructions(self, hypothesis_plan: str) -> str:
        """
        Build the coder instructions in a way that is consistent with
        MultiAgentPipeline:
        - inject MEGATRON binary path
        - inject description/example file as a prompt block
        - inject important utils code
        - optionally inject previous codes as references
        """

        # Load description/example file
        description_lines = load_description(DATA_DESCRIPTION_PATH)
        description_block = as_prompt_block(description_lines)

        # Load important utils (e.g., cutout helper functions)
        important_utils = read_code_from_file("./plotting_codes/cutout_utils.py")

        # Previous codes (optional context for the LLM)
        previous_codes = read_codes_from_folder(PREVIOUS_CODES_DIR)

        code_developer_prompt = f"""
I need to create visualization code for MEGATRON cutout data of gas cells in a halo. 
The binary file is at path: "{BINARY_FILE_PATH}"

Create Python code to implement the hypothesis and plan. The image should have a resolution of 512x512 
pixels and cover a box size of 20 Mpc at redshift z=0.5. The levels range from 12 to 18.
""".strip()

        coder_instructions = (
            f"{code_developer_prompt}\n\n"
            f"The MEGATRON gas binary file is located at: {BINARY_FILE_PATH}\n\n"
            "Below is a data description/example explaining one existing analysis workflow.\n"
            "Use it ONLY as inspiration for what kinds of visualizations are possible. "
            "If your task is completely different, no need to use that file.\n\n"
            f"{description_block}\n\n"
            f"Hypothesis and Plan to implement:\n{hypothesis_plan}\n\n"
            "If you use functions like read_megatron_cutout(), write them directly in the code, "
            "don't just import them elsewhere.\n\n"
            f"Save the final plot as '{OUTPUT_FIG_PATH}'.\n\n"
            f"Very important utils that might help you:\n{important_utils}"
        )

        # if previous_codes:
        #     coder_instructions += f"\n\nReference codes:\n{previous_codes}"

        return coder_instructions

    def run(self, saved_instruction_file: str) -> None:
        """
        Executes the test pipeline:
        - load saved instructions (hypothesis/plan)
        - feed them to CoderAgent
        - debug code with RunnerAgent
        - save final code to FINAL_CODE_PATH
        """

        print("\n" + "=" * 50)
        print("TEST PIPELINE: LOAD SAVED INSTRUCTIONS")
        print("=" * 50 + "\n")

        hypothesis_plan = self.load_instructions(saved_instruction_file)
        print("Loaded hypothesis-plan instructions.\n")

        print("\n" + "=" * 50)
        print("BUILDING CODER INSTRUCTIONS")
        print("=" * 50 + "\n")

        coder_instructions = self.build_coder_instructions(hypothesis_plan)
        print("Coder instructions built successfully.\n")

        print("\n" + "=" * 50)
        print("STEP 1: CODE GENERATION")
        print("=" * 50 + "\n")

        generated_code = self.coder_agent.generate_code(coder_instructions)
        print("Generated code (first 500 chars):\n")
        print(generated_code[:500])

        print("\n" + "=" * 50)
        print("STEP 2: RUNNER AGENT (debug+execute)")
        print("=" * 50 + "\n")

        final_code = self.runner_agent.run_with_debug(generated_code)

        print("\n" + "=" * 50)
        print("SAVING FINAL CODE")
        print("=" * 50 + "\n")

        save_code_to_file(final_code, FINAL_CODE_PATH)
        print(f"💾 Final code saved → {FINAL_CODE_PATH}")

        print("\n🎉 Test generation pipeline completed.\n")


def main():
    """
    Default entry point for quick testing.
    This assumes your hypothesis-plan file is located at:
        IDEAS_DIR/final_hypothesis_plan.txt
    """
    default_instruction_file = os.path.join(
        IDEAS_DIR, "final_hypothesis_plan.txt"
    )

    pipeline = TestGenerationPipeline()
    pipeline.run(default_instruction_file)


if __name__ == "__main__":
    main()
