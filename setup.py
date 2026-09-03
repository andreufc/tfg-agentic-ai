"""
setup.py — Kaggle secrets bootstrap script.

After cloning the repository and installing its dependencies in a Kaggle
notebook, run:

    import setup  # loads secrets
"""

import os

from utils.config import load_env

# Load Kaggle Secrets in a kernel, or a local .env during development.
load_env("kaggle" if "KAGGLE_KERNEL_RUN_TYPE" in os.environ else "local")

print("🚀 Setup complete — secrets loaded, ready to go!")

