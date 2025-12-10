from pydantic import BaseModel
from typing import TypedDict, Dict
from langchain_core.output_parsers import PydanticOutputParser
import os
import sys
import numpy as np
import subprocess
TEST_DATA_PATH = "/Users/autumn/Documents/GitHub/hathor/test_data"
EX_CODE = "/Users/autumn/Documents/GitHub/hathor/ex_code/cutout_utils.py"

class HypothesisBrainstormer:
    """
    Generates hypotheses according to user input
    """
    def __init__(self,llm):
        self.llm = llm

    class Output(BaseModel):
        hypotheses: Dict[str, str]

    def __call__(self, state: Dict):
        hyp = state.get("hypotheses", {})
        user_prompt = state.get("user_prompt", "")
        hypotheses_suggestions = state.get("hypotheses_suggestions", {})
        user_prompt = state.get("user_prompt", "")

        parser = PydanticOutputParser(pydantic_object=self.Output)
        prompt = f"""
                    You are the hypothesis brainstormer. You are an expert on galaxy formation and the RAMSES galaxy formation simulations. 
                    Your goal is to create and modify hypotheses that will ultimately be investigated using RAMSES data. In the final output, 
                    your hypotheses must be provided in the form of a statement/paragraph. Please limit each hypothesis to include just one 
                    central idea. 

                    You must take the following user input into account in order to provide the most relevant hypotheses:
                    {user_prompt}

                    Current hypotheses:
                    {hyp}

                    Suggestions for Hypothesis Improvement:
                    {hypotheses_suggestions}

                    Mandatory Workflow:
                    IF NO HYPOTHESES GIVEN:
                        1. Generate 20 hypotheses. Assign each hypothesis a label of "1", "2", "3", ...
                    ELSE:
                        1. Improve hypotheses as advised by the Suggestions for Hypothesis Improvement. If the suggestion is "REMOVE", 
                        DO NOT include that hypothesis OR ITS KEY in your final output.
                    
                    ALWAYS:
                        2. Output ONLY a JSON dictionary of hypotheses according to this format:
                        {parser.get_format_instructions()}
                """

        resp = self.llm.invoke(prompt)
        
        try:
            parsed = parser.parse(resp.content)
        except:
            parsed = self.Output(hypotheses=hyp)
        return {"hypotheses": parsed.hypotheses}

class PlotBrainstormer:
    """
    Generates plot ideas that will use RAMSES data to investigate hiven hypotheses.
    """
    def __init__(self,llm):
        self.llm = llm

    class Output(BaseModel):
        plot_ideas: Dict[str, str]

    def __call__(self, state: Dict):
        plots = state.get("plot_ideas", {})
        hyp = state.get("hypotheses", {})
        user_prompt = state.get("user_prompt", "")
        plot_suggestions = state.get("plot_suggestions", {})

        parser = PydanticOutputParser(pydantic_object=self.Output)
        prompt = f"""
            You are the plot brainstormer. You are an expert on galaxy formation and the RAMSES galaxy formation simulations.
            Your goal is to create and modify plot ideas that are associated with the hypotheses created by the hypothesis brainstormer.
            Each plot idea must be given in the form of a paragraph. DO NOT provide code or pseudo code. Note, however that these
            plots will be created using RAMSES simulation data. More information about this data may be found in the user's prompt.

            User's Prompt:
            {user_prompt}

            Given these hypotheses:
            {hyp}

            Current plot ideas:
            {plots}

            Suggestions for Plot Improvement:
            {plot_suggestions}

            Mandatory Workflow:
            IF NO PLOT IDEAS PROVIDED:
                1. Generate 3 plot ideas for each hypothesis. Assign each plot idea a key of "1a", "1b", "1c", "2a", "2b", ... where
                keys starting with "1" are associated with hypothesis "1", keys starting with "2" are associated with hypothesis 2, etc.
            ELSE:
                1. Improve plot ideas accoring to the Suggestions for Plot Improvement. If the suggestion is "REMOVE", do not include that 
                plot idea's key in your final output.
            
            ALWAYS:
                3. Output ONLY a JSON dictionary of plot ideas according to this format:
                {parser.get_format_instructions()}
        """

        resp = self.llm.invoke(prompt)
        try:
            parsed = parser.parse(resp.content)
        except:
            parsed = self.Output(plot_ideas=plots)
        return {"plot_ideas": parsed.plot_ideas}
    
class Critic:
    """
    Critiques hypotheses and plot ideas.
    """
    def __init__(self,llm):
        self.llm = llm

    class Output(BaseModel):
        next_brainstormer: str
        hypotheses_suggestions: Dict[str, str]  
        plot_suggestions: Dict[str, str]  

    def __call__(self, state: Dict):
        hyp = state.get("hypotheses", {})
        plots = state.get("plot_ideas", {})
        user_prompt = state.get("user_prompt", "")

        parser = PydanticOutputParser(pydantic_object=self.Output)
        prompt = f"""
                    You are the critic agent. You are an expert on galaxy formation and the RAMSES galaxy formation simulations.
                    You are provided with hypotheses and associated plot ideas. It should be possible to create each plot using RAMSES data. 
                    Additionally, each plot should aid in investigating the validity of its associated hypothesis. Your goal is to 
                    remove weak hypotheses/plots, and suggest improvements on the pairs that remain. Note that removing a weak hypothesis
                    requires you to remove all of that hypothesis' associated plot ideas. You MUST ensure that at least one hypothesis/plot
                    pair remains. If there are only a few (~3) pairs left, you must select your favorite and move onto the coding stage. 

                    You MUST always remove at least 30% of the hypothesis/plot pairs provided to you. 
                    (For example, if you are given 6 hypotheses which each have 3 plot ideas, you could remove two hypothesis (along with their associated plot ideas),
                    OR remove 1 plot idea from each hypothesis, OR remove 1 hypothesis (along with its associated plot ideas) and 3 other random plot ideas. )

                    Current Hypotheses:
                    {hyp}

                    Current Plot Ideas:
                    {plots}

                    User Prompt:
                    {user_prompt}

                    Task:
                    IF only a few (~3) hypothesis/plot pairs remain
                        1. select your favorite hypothesis/plot pair and write REMOVE for all other suggestions.
                    ELSE
                        1. Suggest improvements (or write "no improvements necessary") for EVERY hypothesis and EVERY plot present. If ZERO plot ideas are given, you may leave plot suggestions empty. 
                        2. Select the weakest hypotheses/plots for removal. In your final output, replace your suggestions with "REMOVE" or these hypotheses and plots. 
                        3. Decide whether to hypotheses or plots need more urgent improvement. If the hypotheses need more work, the next_brainstormer is "hypothesis". 
                        If the plot ideas need more work, the next_brainstormer is "plot". 

                    ALWAYS
                        Output ONLY a JSON dictionary of your suggestions according to this format:
                        {parser.get_format_instructions()}
                """
        resp = self.llm.invoke(prompt)
        try:
            parsed = parser.parse(resp.content)
        except:
            parsed = self.Output(
                next_brainstormer = "critic",
                hypotheses_suggestions = None,  
                plot_suggestions = None,
            )
        return {
            "next_brainstormer": parsed.next_brainstormer,
            "hypotheses_suggestions": parsed.hypotheses_suggestions,
            "plot_suggestions": parsed.plot_suggestions,
        }

class IdeaCleaner:
    """
    Removes hypotheses and plot ideas that have been suggested for removal
    """
    def __init__(self):
        pass

    def __call__(self, state: Dict):
        hypotheses = state.get("hypotheses", {})
        plots = state.get("plot_ideas", {})
        hypothesis_suggestions = state.get("hypotheses_suggestions", {})  
        plot_suggestions = state.get("plot_suggestions", {})  

        # --- Clean hypotheses ---
        cleaned_hypotheses = {}
        for sug_key, sug_val in hypothesis_suggestions.items():
            if sug_val == "REMOVE":
                continue  # drop it entirely
            else:
                try:
                    cleaned_hypotheses[sug_key] = hypotheses[sug_key]
                except: 
                    continue

        # --- Clean plot ideas ---
        cleaned_plots = {}
        for sug_key, sug_val in plot_suggestions.items():
            if sug_val == "REMOVE":
                continue  # drop plot entirely
            else:
                try:
                    cleaned_plots[sug_key] = plots[sug_key]
                except: 
                    continue

        # Count active hypothesis/plot pairs
        active_pairs = 0
        for hyp_key in cleaned_hypotheses.keys():
            # check if any plots exist for this hypothesis
            for plot_key in cleaned_plots.keys():
                if plot_key.startswith(hyp_key):
                    active_pairs += 1

        print(f"Cleanup counts {active_pairs} active pairs")

        if len(cleaned_hypotheses) == 0:
            cleaned_plots = {}
            state["next_brainstormer"] = "hypothesis"
        elif active_pairs == 0:
            cleaned_plots = {}
            state["next_brainstormer"] = "plot"

        state["hypotheses"] = cleaned_hypotheses
        state["plot_ideas"] = cleaned_plots
        return state

class Coder:
    """
    Generates Python code for the final plot idea.
    """
    def __init__(self, llm):
        self.llm = llm

    class Output(BaseModel): 
        generated_code: str

    def __call__(self, state: Dict):
        print("about to code")
        plots = state.get("plot_ideas", {})
        user_prompt = state.get("user_prompt", "")
        last_attempt = state.get("generated_code","")
        code_error = state.get("latest_error","")

        with open(EX_CODE, 'r') as f:
            ex_code_string = f.read()

        # Build a prompt for LLM to generate Python code
        parser = PydanticOutputParser(pydantic_object=self.Output)
        prompt = f"""
                    You are an expert Python coder for RAMSES simulation data.
                    Generate code to create the following plot based on the chosen hypotheses:

                    Plot to implement:
                    {list(plots.values())[0]}

                    Path to data:
                    {TEST_DATA_PATH}

                    Data Overview:
                    {self._get_data_overview()}

                    User notes:
                    {user_prompt}

                    Your previous attempt (blank if this is your first run):
                    {last_attempt}

                    Error upon running yur previous attempt (blank if this is your first run):
                    {code_error}

                    Example code for importing data:
                    {ex_code_string}

                    Rules:
                    - You must start by explaining your plan for creating/debugging the code. 
                    - This must be possible to run on a research-grade laptop. Please keep computational cemplexity and runtime low when possible. 
                    - You (usually) have access to the following packages: yt, numpy, pandas, matplotlib, scipy
                    - Return code as a string suitable for IMMEDIATE execution with a something like exec().
                    - AT THE END, output a JSON dictionary according to this format:
                    {parser.get_format_instructions()}
                """
        
        structured_llm = self.llm.with_structured_output(self.Output)
        
        # Invoke and get structured response
        try:
            result = structured_llm.invoke(prompt)
            return {"generated_code": result.generated_code}
        except Exception as e:
            print(f"Structured output failed: {e}")
            # Fallback to regular LLM
            resp = llm.invoke(prompt)
            # Extract code from markdown
            if "```python" in resp.content:
                code = resp.content.split("```python")[1].split("```")[0].strip()
            else:
                code = resp.content
            return {"generated_code": code}    
        
    def _get_data_overview(self):
        files = sorted(os.listdir(TEST_DATA_PATH))
        data_str = ""
        for filename in files:
            filepath = os.path.join(TEST_DATA_PATH, filename)
            data = np.fromfile(filepath, count=5)

            # Capture numpy's pretty print as string
            data_str += f"STORED WITHIN {filepath}"
            with np.printoptions(precision=3, suppress=True):
                data_str += np.array2string(data, separator=', ')
            data_str += ("\n" + "="*70)
        return data_str
    
class Executor:
    """
    Runs python code.
    """

    def __call__(self, state: Dict):
        code = state.get("generated_code", "")
        attempts_so_far = state.get("code_iteration", 0)
        print(f"executor is about to run attempt {attempts_so_far+1}")
        
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=240,
            )
            
            if result.returncode == 0:
                print("=" * 70)
                print("EXECUTION SUCCESS")
                print(result.stdout)
                print("=" * 70)
                return {
                    "latest_error": None,
                    "code_iteration": attempts_so_far + 1
                }
            else:
                print("=" * 70)
                print("EXECUTION ERROR")
                print(result.stderr)
                print("=" * 70)
                return {
                    "latest_error": result.stderr,
                    "code_iteration": attempts_so_far + 1
                }
        except subprocess.TimeoutExpired:
            return {
                "latest_error": "Execution timed out after 120 seconds",
                "code_iteration": attempts_so_far + 1
            }
        except Exception as e:
            print(f"Executor exception: {e}")
            return {
                "latest_error": str(e),
                "code_iteration": attempts_so_far + 1
            }