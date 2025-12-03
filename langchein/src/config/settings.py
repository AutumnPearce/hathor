# src/config/settings.py

import os

# ==============================
# MODEL CONFIG
# ==============================

MODELS = [
    "google/gemma-3-27b-it",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-Large-Instruct-2407",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

# Keeping same choices you had (second from the end)
LITERATURE_MODEL = MODELS[-2]
CRITIC_LITERATURE_MODEL = MODELS[-2]
REASONER_MODEL = MODELS[-2]
CRITIC_MODEL = MODELS[-2]
CODER_MODEL = MODELS[-2]
RUNNER_MODEL = None  # Runner is local exec, not LLM

# ==============================
# DATA / OUTPUT CONFIG
# ==============================

BINARY_FILE_PATH = "/Users/yk2047/Documents/GitHub/hathor/dataset_examples/halo_3517_gas.bin"

BASE_OUTPUT_DIR = "./outputs"
IDEAS_DIR = os.path.join(BASE_OUTPUT_DIR, "Ideas")
FIGS_DIR = os.path.join(BASE_OUTPUT_DIR, "figs")
EXECUTED_CODES_DIR = os.path.join(BASE_OUTPUT_DIR, "executed_codes")
PREVIOUS_CODES_DIR = "./plotting_codes"

OUTPUT_FIG_PATH = os.path.join(FIGS_DIR, "output_plot.png")
FINAL_CODE_PATH = os.path.join(EXECUTED_CODES_DIR, "plot.py")

# ==============================
# PHYSICAL / VISUALIZATION CONFIG
# ==============================

VIEW_DIR = "z"
NPIX = 512
LMIN = 12
LMAX = 18
REDSHIFT = 0.5
BOXSIZE = 20.0
