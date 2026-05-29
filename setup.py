"""
setup.py — Colab / Kaggle bootstrap script.

Run this cell at the top of every Colab or Kaggle notebook to install
dependencies and mount the repository:

    !git clone https://github.com/<user>/tfg-agentic-ai.git
    %cd tfg-agentic-ai
    !pip install -q -r requirements.txt

    import setup  # loads secrets + logs into W&B
"""

from utils.config import load_env

# Load secrets from Colab userdata, Kaggle Secrets, or .env
load_env()

print("🚀 Setup complete — secrets loaded, ready to go!")

