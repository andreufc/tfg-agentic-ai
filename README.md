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
| `utils.models` | Load HF models — NF4 4-bit (bitsandbytes) for most families, or the native compressed-tensors w4a16 path for the pre-quantized Gemma-4 QAT checkpoints |
| `utils.evaluation` | Compute accuracy — numeric match (GSM8K) / letter match (MMLU) |
| `utils.agents` | Wrap HF models as LangChain `HuggingFacePipeline` LLMs with optional seeded generation |
| `utils.instrumentation` | Count input/output tokens; measure GPU peak VRAM and utilisation; compute throughput (tokens/s) |
| `utils.warmup` | Run throwaway generations to trigger CUDA/cuBLAS kernel compilation before timed measurements |
| `utils.statistics` | Cluster-corrected estimators: accuracy (Student-t CI), latency/tokens (BCa bootstrap), overhead ratios (log-space t), VRAM (Student-t); between-system Welch t-test and Mann-Whitney U |
| `utils.logging_utils` | Initialise & log metrics to W&B runs |

---

## 🗜️ Model Quantization

All models are loaded 4-bit within the T4's 16 GB budget via `utils.models.load_model_and_tokenizer`, but the quantization mechanism differs by family:

| Family | Checkpoint | Quantization |
|---|---|---|
| Mistral/Ministral-3, Phi-4, Qwen3 | full-precision HF checkpoint | **bitsandbytes NF4**, applied on the fly at load time |
| Gemma-4 (E2B/E4B) | `google/gemma-4-{E2B,E4B}-it-qat-w4a16-ct` | **compressed-tensors w4a16** (QAT) — already 4-bit quantized on disk, no bitsandbytes step |

The Gemma-4 checkpoints are Google's official pre-quantized QAT release, serialized in the `compressed-tensors` format (4-bit weights / 16-bit activations). They are loaded by calling `load_model_and_tokenizer(model_id, quantize_4bit=False)`: `quantize_4bit=False` skips bitsandbytes and lets `transformers`' compressed-tensors `HFQuantizer` integration restore the checkpoint's own `quantization_config` on load — forcing `dtype=torch.float16` rather than the checkpoint's declared `bfloat16`, since the T4 (compute capability 7.5) has no native bf16 support and float16 is also compressed-tensors' own recommended compute dtype. (Passing `quantize_4bit=True` for such a `-qat-w4a16-ct` checkpoint raises, since it cannot be re-quantized.) This requires the `compressed-tensors` PyPI package (pinned in `requirements.txt`) in addition to `transformers`' built-in quantizer support.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
