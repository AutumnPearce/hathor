# src/agents/runner_agent.py

from ..utils import run_code, is_valid_python
from .coder_agent import CoderAgent


class RunnerAgent:
    """
    Executes and debugs generated code, delegating fixes to the CoderAgent.
    """

    def __init__(self, coder_agent: CoderAgent, max_iterations: int = 5):
        self.coder_agent = coder_agent
        self.max_iterations = max_iterations

    def run_with_debug(self, code: str) -> str:
        """
        Try to run code; on failure ask the coder to fix and retry.
        Returns final code (possibly still failing if max_iterations exceeded).
        """
        iteration = 0

        while iteration < self.max_iterations:
            print("\n" + "=" * 50)
            print(f"STEP 4: RUNNER AGENT (Iteration {iteration + 1})")
            print("=" * 50 + "\n")

            # Syntax check
            if not is_valid_python(code):
                print("❌ Invalid Python syntax detected, asking coder to fix...\n")
                fix_instructions = (
                    "The following code has invalid Python syntax. "
                    "Fix it and output ONLY valid Python code.\n\n"
                    f"CODE:\n{code}"
                )
                code = self.coder_agent.generate_code(fix_instructions)
                iteration += 1
                continue

            # Try to run
            result = run_code.invoke(code)

            if result["success"]:
                print("\n🎉 SUCCESS: Code executed successfully!")
                return code

            print("\n⚠️ ERROR detected → sending to coder for debugging...\n")
            print("Error Output:\n", result["output"])

            fix_instructions = (
                "Fix the following code so it runs without errors. "
                "Output ONLY raw Python code.\n\n"
                f"ERROR:\n{result['output']}\n\n"
                f"CODE:\n{code}"
            )
            code = self.coder_agent.generate_code(fix_instructions)
            iteration += 1

        print(
            f"\n❌ Maximum iterations ({self.max_iterations}) reached. "
            "Returning last version of the code..."
        )
        return code
