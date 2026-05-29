# 🤖 tfg-agentic-ai

> **Final Thesis — GCED (Grau en Ciència i Enginyeria de Dades)**

This thesis explores using **multi-agent collaboration** (a *Society of Minds* approach) to enhance the reasoning capabilities of **Small Language Models** (SLMs) like Phi-3 and TinyLlama, which often struggle with complex tasks.

The goal is to develop an optimised pipeline where specialised agents collaboratively solve tasks in **GSM8K** and **MMLU**, aiming for accuracy gains over single-agent inference with minimal resources.

**Tech stack:** LangChain (LCEL) for compositional agent calls · LangGraph for stateful collaboration graphs · W&B for experiment tracking · Google Colab / Kaggle T4 GPU (16 GB VRAM).

---

## 📁 Project Structure

```
tfg-agentic-ai/
├── notebooks/          # Colab / Kaggle notebooks (core experiments)
│   └── ...
├── utils/              # Shared Python utilities
│   ├── __init__.py
│   ├── config.py       # Environment & secrets loader (Colab + Kaggle + local)
│   ├── data.py         # Dataset loading (GSM8K, MMLU)
│   ├── models.py       # HF model loading with 4-bit quantisation
│   ├── evaluation.py   # Accuracy metrics & result display
│   ├── agents.py       # LangChain/LangGraph agent factories
│   └── logging_utils.py# W&B experiment tracking helpers
├── .env.example        # Template for required secrets
├── .gitignore          # Files excluded from version control
├── requirements.txt    # Python dependencies
├── setup.py            # Colab / Kaggle bootstrap script
├── LICENSE             # MIT License
├── CODE_OF_CONDUCT.md  # Contributor Covenant
└── README.md           # ← You are here
```

---

## 🚀 Quick Start

### 1. Clone & configure secrets

```bash
git clone https://github.com/<your-user>/tfg-agentic-ai.git
cd tfg-agentic-ai
cp .env.example .env
# Edit .env and fill in your HF_TOKEN and WANDB_API_KEY
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run in Google Colab

Open any notebook from `notebooks/` in Colab, then add a first cell:

```python
!git clone https://github.com/<your-user>/tfg-agentic-ai.git
%cd tfg-agentic-ai
!pip install -q -r requirements.txt

import setup  # auto-loads secrets from Colab userdata
```

> **Note:** Store your `HF_TOKEN`, `WANDB_API_KEY`, and `GITHUB_TOKEN` in
> [Colab Secrets](https://colab.research.google.com/) (🔑 icon in the sidebar)
> so `setup.py` can pick them up automatically.

### 4. Run on Kaggle

1. Upload a notebook from `notebooks/` to a Kaggle kernel.
2. Add the following secrets via **Add-ons → Secrets**:
   - `GITHUB_TOKEN` — GitHub PAT with `repo` read scope
   - `HF_TOKEN` — Hugging Face read-access token
   - `WANDB_API_KEY` — Weights & Biases API key
3. Enable each secret for the notebook.
4. Enable **GPU T4 × 2** (or the accelerator of your choice) in notebook settings.

The first code cell will automatically detect Kaggle, clone the repo, and load secrets.

---

## 🧪 Utility Modules

| Module | Purpose |
|---|---|
| `utils.config` | Load secrets from `.env` (local), Colab userdata, or Kaggle Secrets |
| `utils.data` | Load & subsample GSM8K / MMLU datasets |
| `utils.models` | Load HF models with NF4 quantisation for T4 GPUs |
| `utils.evaluation` | Compute accuracy (numeric match / letter match) |
| `utils.agents` | Wrap HF models as LangChain LLMs |
| `utils.logging_utils` | Initialise & log to W&B runs |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
