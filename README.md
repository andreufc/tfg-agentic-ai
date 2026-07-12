# 🤖 tfg-agentic-ai

> **Final Thesis — GCED (Grau en Ciència i Enginyeria de Dades)**

This thesis explores using **multi-agent collaboration** (a *Society of Minds* approach) to enhance the reasoning capabilities of **Small Language Models** (SLMs) like Mistral/Ministral-3B and Phi-3, which often struggle with complex tasks under tight resource constraints.

The goal is to develop an optimised pipeline where specialised agents collaboratively solve tasks in **GSM8K** (mathematical reasoning) and **MMLU** (knowledge retrieval), measuring not just accuracy gains over single-agent inference but also the associated costs in latency, token consumption, and GPU memory — using rigorous cluster-corrected statistical methods throughout.

**Tech stack:** LangChain (LCEL) for compositional agent calls · LangGraph for stateful collaboration graphs · W&B for experiment tracking · Kaggle T4 GPU (16 GB VRAM, `batch_size=1`).

---

## 📁 Project Structure

```
tfg-agentic-ai/
├── notebooks/                    # Kaggle notebooks (core experiments)
│   ├── v1.1.0 — Automated GSM8K Evaluation (LangChain)
│   ├── v1.1.1 — Automated MMLU Evaluation (LangChain)
│   ├── v1.2.0 — Automated GSM8K Evaluation (LangChain, revised)
│   ├── v1.2.1 — Automated MMLU Evaluation (LangChain, revised)
│   ├── v2.1.0 — GSM8K Single-Agent Baseline
│   └── v2.1.1 — MMLU Single-Agent Baseline
├── utils/                        # Shared Python utilities
│   ├── __init__.py
│   ├── config.py                 # Secrets loader (local .env, Colab, Kaggle)
│   ├── data.py                   # Dataset loading (GSM8K, MMLU)
│   ├── models.py                 # HF model loading with NF4 quantisation
│   ├── evaluation.py             # Accuracy metrics (numeric / letter match)
│   ├── agents.py                 # LangChain LLM factory + seeded generation
│   ├── instrumentation.py        # Token counting, GPU VRAM & utilisation tracking
│   ├── warmup.py                 # Pre-measurement model warmup
│   ├── statistics.py             # Cluster-corrected statistical estimators
│   └── logging_utils.py          # W&B experiment tracking helpers
├── .env.example                  # Template for required secrets
├── .gitignore
├── requirements.txt              # Python dependencies
├── setup.py                      # Colab / Kaggle bootstrap script
├── LICENSE                       # MIT License
├── CODE_OF_CONDUCT.md
└── README.md                     # ← You are here
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
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
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
4. Enable **GPU T4 × 2** in notebook settings (the model is pinned to one GPU; the second is unused by the model but available).

The first code cell will automatically detect Kaggle, clone the repo, and load secrets.

---

## 🧪 Utility Modules

| Module | Purpose |
|---|---|
| `utils.config` | Load secrets from `.env` (local), Colab userdata, or Kaggle Secrets |
| `utils.data` | Load & subsample GSM8K / MMLU datasets |
| `utils.models` | Load HF models with NF4 quantisation (bitsandbytes) |
| `utils.evaluation` | Compute accuracy — numeric match (GSM8K) / letter match (MMLU) |
| `utils.agents` | Wrap HF models as LangChain `HuggingFacePipeline` LLMs with optional seeded generation |
| `utils.instrumentation` | Count input/output tokens; measure GPU peak VRAM and utilisation; compute throughput (tokens/s) |
| `utils.warmup` | Run throwaway generations to trigger CUDA/cuBLAS kernel compilation before timed measurements |
| `utils.statistics` | Cluster-corrected estimators: accuracy (Student-t CI), latency/tokens (BCa bootstrap), overhead ratios (log-space t), VRAM (Student-t); between-system Welch t-test and Mann-Whitney U |
| `utils.logging_utils` | Initialise & log metrics to W&B runs |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
