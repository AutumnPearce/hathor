# hathor
Multi-agent system for generating galaxy formation hypotheses and creating relevant plots from RAMSES simulation data.

### Basic Installation (of Scheme 1)
```bash
cd hathor
# cd langhchain  # for accesising the langchein
pip install -e .
```

### Development Installation(of Scheme 1)
```bash
# Navigate to the cloned repository
cd path/to/hathor

# cd langhchain  # for accesising the langchein

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## State of this Project
This code currently functions (see example below), but it is a work in progress. It iss currently optimized for data from the MEGATRON simulations. For more information see the file report.pdf. 

## Example Usage (of Scheme 1)
```python
from hathor import Hathor

from inference_auth_token import get_access_token
from langchain_openai import ChatOpenAI

# create llm
access_token = get_access_token()
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=access_token,
    base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
)

# create your prompt based on your data. 
prompt = """
    I am providing you with ONE halo's gas.bin and stars.bin files at ONE timestep (z = 3.9). 
    This data comes from the MEGATRON simulations, which use RAMSES-RTZ. MEGATRON directly tracks the abundance of MANY ions. Please use this to your advantage.
    RAMSES-RTZ is a specific type of the RAMSES simulation suite. Please use your knowledge of RAMSES to inform your ideas.
"""

# initialize hathor
hathor = Hathor(llm=llm, prompt=prompt)

# run hathor
final_state = hathor.run()

# print output
print(list(final_state['hypotheses'].values())[0])
print(list(final_state['plot_ideas'].values())[0])
print(final_state['generated_code'])
```

## Example Usage (langchain)
```python
from src.pipeline.multi_agent_pipeline import MultiAgentPipeline

# 1. Create your prompts
literature_prompt = """
I want to analyse MEGATRON cutout data from a galaxy cluster.
Please generate scientific hypotheses about interesting cluster properties
that can be tested through 2D projected diagnostics.
"""

coder_prompt = """
Generate Python code that implements the selected hypothesis + analysis plan
and produces a 512x512 plot saved to the output path.
"""

# 2. Instantiate the pipeline
pipeline = MultiAgentPipeline()

# 3. Run the full multi-agent workflow
result = pipeline.run(
    literature_task=literature_prompt,
    code_developer_prompt=coder_prompt,
    num_hypotheses=5
)

# 4. Access outputs
print("=== Hypotheses ===")
print(result["hypotheses"])

print("=== Final Hypothesis–Plan ===")
print(result["final_hypothesis_plan"])

print("=== Generated Code ===")
print(result["generated_code"])

```