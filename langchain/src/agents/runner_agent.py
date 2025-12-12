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
        iteration = 0

        while iteration < self.max_iterations:
            print("\n" + "=" * 50)
            print(f"STEP 4: RUNNER AGENT (Iteration {iteration + 1})")
            print("=" * 50 + "\n")
            # NEW: Empty code check (fixes silent failures)
            if not code or not code.strip():
                print("❌ Empty or whitespace-only code detected → asking coder to regenerate.\n")
                fix_instructions = (
                    "The previous code was empty or invalid. "
                    "Generate complete, valid Python code that performs the required task."
                )
                code = self.coder_agent.generate_code(fix_instructions)
                iteration += 1
                continue

            if not is_valid_python(code):
                fix_instructions = (
                    "The following code has invalid Python syntax. "
                    "Fix it and output ONLY valid Python code.\n\n"
                    f"CODE:\n{code}"
                )
                code = self.coder_agent.generate_code(fix_instructions)
                iteration += 1
                continue

            result = run_code.invoke(code)

            if result["success"]:
                print("\n🎉 SUCCESS: Code executed successfully!")
                return code

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

