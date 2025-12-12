# hathor
Multi-agent system for generating galaxy formation hypotheses and creating relevant plots from RAMSES simulation data.

### Basic Installation (of Scheme 1)
```bash
cd hathor
pip install -e .
```

### Development Installation(of Scheme 1)
```bash
# Navigate to the cloned repository
cd path/to/hathor

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

