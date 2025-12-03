# src/pipeline/multi_agent_pipeline.py

from __future__ import annotations

import os

from ..agents import (
    LiteratureAgent,
    CriticAgent,
    ReasonerAgent,
    CoderAgent,
    RunnerAgent,
)
from ..config import (
    LITERATURE_MODEL,
    CRITIC_LITERATURE_MODEL,
    REASONER_MODEL,
    CRITIC_MODEL,
    CODER_MODEL,
    BINARY_FILE_PATH,
    IDEAS_DIR,
    PREVIOUS_CODES_DIR,
    OUTPUT_FIG_PATH,
    FINAL_CODE_PATH,
)
from ..utils import (
    read_codes_from_folder,
    save_answer_to_file,
    save_code_to_file,
)


class MultiAgentPipeline:
    def __init__(self) -> None:
        # Instantiate agents with appropriate models
        self.literature_agent = LiteratureAgent(LITERATURE_MODEL)
        self.critic_agent = CriticAgent(CRITIC_LITERATURE_MODEL)
        self.reasoner_agent = ReasonerAgent(REASONER_MODEL)
        self.coder_agent = CoderAgent(CODER_MODEL)
        self.runner_agent = RunnerAgent(self.coder_agent)

        # Ensure output dirs exist
        os.makedirs(IDEAS_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(OUTPUT_FIG_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(FINAL_CODE_PATH), exist_ok=True)

    def run(
        self,
        literature_task: str,
        code_developer_prompt: str,
        num_hypotheses: int = 8,
        max_refinement_iterations: int = 10,
    ) -> None:
        # 0a) Literature Review Agent
        print("\n" + "=" * 50)
        print("STEP 0a: LITERATURE REVIEW AGENT")
        print("=" * 50 + "\n")

        hypotheses = self.literature_agent.generate_hypotheses(
            task=literature_task,
            num_hypotheses=num_hypotheses,
        )
        print(f"Initial {num_hypotheses} Hypotheses:\n", hypotheses)

        save_answer_to_file(
            hypotheses,
            os.path.join(IDEAS_DIR, "initial_hypotheses.txt"),
            title=f"Initial {num_hypotheses} Hypotheses",
        )
        print(
            "\n💾 Saved initial hypotheses → "
            f"{os.path.join(IDEAS_DIR, 'initial_hypotheses.txt')}"
        )

        # 0b) Literature Critic Agent
        print("\n" + "=" * 50)
        print("STEP 0b: LITERATURE CRITIC AGENT")
        print("=" * 50 + "\n")

        refined_hypotheses = self.critic_agent.review_literature(hypotheses)
        print("Refined Hypotheses:\n", refined_hypotheses)

        save_answer_to_file(
            refined_hypotheses,
            os.path.join(IDEAS_DIR, "refined_hypotheses.txt"),
            title="Refined Hypotheses",
        )
        print(
            "\n💾 Saved refined hypotheses → "
            f"{os.path.join(IDEAS_DIR, 'refined_hypotheses.txt')}"
        )

        # 1) Reasoner Agent
        print("\n" + "=" * 50)
        print("STEP 1: REASONING AGENT - Creating hypothesis-plan pairs")
        print("=" * 50 + "\n")

        previous_codes = read_codes_from_folder(PREVIOUS_CODES_DIR)
        pairs = self.reasoner_agent.create_pairs(refined_hypotheses, previous_codes)
        print("Initial Hypothesis-Plan Pairs:\n", pairs)

        # 2) Critic ↔ Reasoner loop
        refinement_iteration = 0
        final_pair = pairs

        while refinement_iteration < max_refinement_iterations:
            print("\n" + "=" * 50)
            print(f"STEP 2: CRITIC AGENT - Elimination Round {refinement_iteration + 1}")
            print("=" * 50 + "\n")

            critique = self.critic_agent.review_pairs(pairs)
            print("Critic feedback:\n", critique)

            if "plan is ok" in critique.lower() or "plan ok" in critique.lower():
                print(
                    "\n✅ Critic approved! Only 1 hypothesis-plan pair remains!"
                )
                final_pair = pairs
                break

            pairs = critique

            print(
                "\n🔄 Critic eliminated at least 1 pair, sending back to Reasoner...\n"
            )

            print("\n" + "=" * 50)
            print(
                f"STEP 1 (cont): REASONING AGENT - Refinement {refinement_iteration + 1}"
            )
            print("=" * 50 + "\n")

            refinement_task = (
                f"Previous hypothesis-plan pairs:\n{pairs}\n\n"
                f"Please review and refine these pairs based on the feedback."
            )

            pairs = self.reasoner_agent.create_pairs(refinement_task, previous_codes)
            print("Refined Pairs:\n", pairs)

            final_pair = pairs
            refinement_iteration += 1

        if refinement_iteration >= max_refinement_iterations:
            print(
                f"\n⚠️ Maximum iterations ({max_refinement_iterations}) reached. "
                "Using last pair."
            )

        save_answer_to_file(
            final_pair,
            os.path.join(IDEAS_DIR, "final_hypothesis_plan.txt"),
            title="Final Hypothesis-Plan Pair",
        )
        print(
            "\n💾 Saved final pair → "
            f"{os.path.join(IDEAS_DIR, 'final_hypothesis_plan.txt')}"
        )

        # 3) Coder Agent
        print("\n" + "=" * 50)
        print("STEP 3: CODER AGENT")
        print("=" * 50 + "\n")

        coder_instructions = (
            f"{code_developer_prompt}\n\n"
            f"The MEGATRON gas binary file is located at: {BINARY_FILE_PATH}\n\n"
            f"Hypothesis and Plan to implement:\n{final_pair}\n\n"
            "If you use functions like read_megatron_cutout(), write it in the code, "
            "don't just import it elsewhere.\n\n"
            f"Save the final plot as '{OUTPUT_FIG_PATH}'."
        )

        if previous_codes:
            coder_instructions += f"\n\nReference codes:\n{previous_codes}"

        code = self.coder_agent.generate_code(coder_instructions)
        print("Generated Code:\n")
        print(code)

        # 4) Runner Agent (exec + debug)
        final_code = self.runner_agent.run_with_debug(code)

        save_code_to_file(final_code, FINAL_CODE_PATH)
        print(f"\n💾 Saved final code → {FINAL_CODE_PATH}\n")


def main() -> None:
    """
    Default entrypoint used when running as a script/module.
    Mirrors your original __main__ block.
    """
    literature_task = """
I want to do plots for MEGATRON cutout data of gas cells in a halo. It's stored in a binary file. 
The data contains positions (x,y,z), levels, ne (electron number density), dx (cell size), and other 
features for each gas cell.

I want to create 2D images of parameters projected along the z-axis to learn something interesting 
about galaxy clusters.

Please propose interesting hypotheses about galaxy clusters that can be tested through visualization 
and analysis of this data.
"""

    code_developer_prompt = f"""
I need to create visualization code for MEGATRON cutout data of gas cells in a halo. 
The binary file is at path: "{BINARY_FILE_PATH}"

The data contains positions (x,y,z), levels, ne (electron number density), dx (cell size), and other 
features for each gas cell.

Create Python code to implement the hypothesis and plan. The image should have a resolution of 512x512 
pixels and cover a box size of 20 Mpc at redshift z=0.5. The levels range from 12 to 18.
"""

    pipeline = MultiAgentPipeline()
    pipeline.run(literature_task, code_developer_prompt, num_hypotheses=8)


if __name__ == "__main__":
    main()
