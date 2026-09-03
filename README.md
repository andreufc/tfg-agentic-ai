# 🤖 tfg-agentic-ai

> **Final Thesis — GCED (Grau en Ciència i Enginyeria de Dades), ETSETB/UPC**
> *Optimizing Agentic AI Collaboration with SLMs*

This thesis explores whether **multi-agent collaboration** (a *Society of Minds* approach) can improve the reasoning accuracy and reliability of **Small Language Models** (SLMs, ≤ 8B parameters), which often struggle with complex tasks under tight resource constraints.

Specialised agents collaboratively solve tasks in **GSM8K**, the level 1–4 subset of **MATH-500** (mathematical reasoning), and **MMLU** (knowledge retrieval). The evaluation records both accuracy and the associated latency, token consumption, and GPU memory, using cluster-corrected statistical methods throughout.

**Tech stack:** LangChain (LCEL) for compositional agent calls · LangGraph for stateful collaboration graphs · W&B for experiment tracking · Kaggle T4 GPU (16 GB VRAM, `batch_size=1`).

---

## 📁 Project Structure

```
tfg-agentic-ai/
├── notebooks/                    # Kaggle notebooks (core experiments)
│   ├── setup/                    # v1.x — LangChain (LCEL) evaluation harness
│   ├── baselines/                # v2.x — single-agent zero-shot / CoT baselines
│   ├── self-consistency/         # v3.x — Self-Consistency (the L1 generation pass)
│   ├── majority-voting/          # v4.x — L1 Majority Voting (replay over SC samples)
│   ├── bon-mav/                  # v5.x — L1 BoN-MAV (multi-aspect verification)
│   └── archive/                  # Never-executed or superseded notebook drafts
├── utils/                        # Shared Python library used by all notebooks
│   ├── config.py                 # Secrets loader (local .env, Colab, Kaggle)
│   ├── data.py                   # Dataset loading (GSM8K, MMLU, MATH-500)
│   ├── models.py                 # HF model loading (NF4 / compressed-tensors w4a16)
│   ├── registry.py               # Single source of truth for model identity & naming
│   ├── ensembles.py              # Ensemble rosters (E1–E7) and slot sampling
│   ├── agents.py                 # LangChain LLM factory + seeded generation
│   ├── graphs.py                 # LangGraph templates (Majority Voting, BoN-MAV)
│   ├── voting.py                 # Plurality voting and the Self-Consistency ladder
│   ├── mav.py                    # Aspect verifiers, verdict parsing, BoN selection
│   ├── evaluation.py             # Accuracy metrics (numeric / letter / symbolic match)
│   ├── instrumentation.py        # Token counting, GPU peak-VRAM tracking, throughput
│   ├── warmup.py                 # Pre-measurement model warmup
│   ├── persistence.py            # Full-text generation persistence + W&B artifacts
│   ├── statistics.py             # Cluster-corrected statistical estimators
│   └── logging_utils.py          # W&B experiment tracking helpers
├── .gitignore
├── requirements.txt              # Python dependencies
├── setup.py                      # Colab / Kaggle bootstrap script
├── LICENSE                       # MIT License
├── CODE_OF_CONDUCT.md
└── README.md                     # ← You are here
```

Notebook naming convention:

```
[Agentic] [GCED] v<major>.<minor>.<patch> [LangChain] [pub] - <Description> <Dataset>.ipynb
```

### Notebook collection

`notebooks/` contains the complete notebook record of the experiments run for this thesis: environment and evaluation setup, single-agent baselines, and the evaluated multi-agent systems. Executed notebooks retain their committed outputs when available, while `notebooks/archive/` keeps never-executed or superseded drafts separate from the experimental evidence.

See [`notebooks/README.md`](notebooks/README.md) for the detailed folder guide, status conventions, and run instructions.

- **`setup/`** — early v1.x end-to-end templates used to validate the LangChain evaluation harness, instrumentation, W&B logging, and benchmark workflow. The v1.1.x notebooks are retained as historical runs and may no longer be compatible with the current utility API.
- **`baselines/`** — v2.x single-agent zero-shot and Chain-of-Thought baselines across GSM8K, MATH-500, and MMLU, including the model-family comparisons used as references for the collaborative protocols.
- **`self-consistency/`** — v3.x Self-Consistency experiments and parameter sweeps. These runs generate and persist the candidate responses later replayed by the ensemble protocols.
- **`majority-voting/`** — v4.x L1 Majority Voting experiments over homogeneous and heterogeneous ensembles, evaluated on MATH-500 and MMLU.
- **`bon-mav/`** — v5.x Best-of-N with Model-as-Verifier experiments, where independent aspect verifiers score persisted candidates before final selection.
- **`archive/`** — notebook drafts that were never executed or were superseded before becoming part of the thesis evidence. See [`notebooks/archive/README.md`](notebooks/archive/README.md) for the inventory and reasons.

The original dual-benchmark Gemma-4 notebooks `v2.4.0` and `v2.4.1` were executed on Kaggle: GSM8K completed, but the session reached Kaggle's 12-hour limit during MMLU, so the executed notebook outputs were never committed to GitHub. The corresponding `[FIX v2.4.0]` and `[FIX v2.4.1]` notebooks rerun only the unfinished MMLU evaluation.

---

## 🧠 Models

All models are ≤ 8B parameters and run 4-bit on a single T4. Model identity (HF repo id, `model_key`, run and artifact naming) is recorded exactly once, in `utils/registry.py`.

| Family | Checkpoints evaluated |
|---|---|
| Ministral-3 | 8B-Instruct, 3B-Instruct, 3B-Reasoning |
| Qwen3 | 8B, 4B, 1.7B (hybrid thinking / non-thinking) |
| Phi-4 | Phi-4-mini-instruct |
| Gemma-4 | E4B-it, E2B-it (pre-quantized QAT checkpoints) |

The multi-agent (L1) campaign runs on the four registered models: `qwen3-1.7b`, `qwen3-4b`, `ministral-3-3b`, `phi-4-mini`.

---

## 🤝 Agent Protocols

Each protocol is a LangGraph graph supporting homogeneous (same model in every slot) and heterogeneous (different models per slot) configurations:

| Level | Protocol | Notebooks | Idea |
|---|---|---|---|
| — | Single-agent baseline | `baselines/` (v2.x) | Zero-shot / CoT reference point |
| L1 | Self-Consistency | `self-consistency/` (v3.x) | One model, *k* sampled generations, plurality vote |
| L1 | Majority Voting | `majority-voting/` (v4.x) | Heterogeneous ensembles (E1–E5) vote across models |
| L1 | BoN-MAV | `bon-mav/` (v5.x) | Best-of-N candidates scored by multi-aspect verifiers (E2, E6, E7) |
| L2 | Multi-round Debate | *(planned)* | Agents exchange critiques across rounds |

Self-Consistency is the only pass that spends fresh generation compute: Majority Voting and BoN-MAV **replay** its persisted per-generation text and metrics (`utils/persistence.py`) instead of re-generating, so protocol comparisons are paired and cheap.

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/<your-user>/tfg-agentic-ai.git
cd tfg-agentic-ai
```

### 2. Configure secrets

Three secrets are needed: `HF_TOKEN` (Hugging Face read-access token), `WANDB_API_KEY` (Weights & Biases API key), and `GITHUB_TOKEN` (GitHub PAT with `repo` read scope, used to clone the repo from a notebook).

- **Local:** create a `.env` file in the repo root (gitignored) with `HF_TOKEN=...`, `WANDB_API_KEY=...`, `GITHUB_TOKEN=...`.
- **Kaggle:** add the three as kernel secrets via **Add-ons → Secrets** and enable each for the notebook.
- **Colab:** add the three as Colab Secrets (🔑 icon in the sidebar).

Each notebook detects its runtime and calls `utils.config.load_env` with the corresponding secrets backend.

### 3. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The venv is for editing `utils/` and local scripting — experiments need a GPU and run on Kaggle.

### 4. Run an experiment on Kaggle

Experiments run on **Kaggle** (free T4 tier), chosen because sessions keep running offline for hours unattended.

1. Upload a notebook from `notebooks/` to a Kaggle kernel.
2. Add the secrets as described above and enable **GPU T4 × 2** in notebook settings. Experiments are pinned to `cuda:0` (16 GB budget, `batch_size=1`); the second GPU is unused.
3. Run. The first code cell auto-detects Kaggle, clones the repo, installs `requirements.txt` and loads the secrets.

Google Colab is also supported but is not the platform the experiments were executed.

---

## 🧪 Utility Modules

| Module | Purpose |
|---|---|
| `utils.config` | Load secrets from `.env` locally, Colab userdata or Kaggle Secrets in experiments |
| `utils.data` | Load & subsample GSM8K, MMLU and the level 1–4 MATH-500 subset with standardised columns |
| `utils.models` | Load HF models — NF4 4-bit (bitsandbytes) for most families, or the native compressed-tensors w4a16 path for the pre-quantized Gemma-4 QAT checkpoints |
| `utils.registry` | Single source of truth for `model_key` → HF repo id, run names and artifact names (stdlib only, importable without the GPU stack) |
| `utils.ensembles` | Ensemble rosters E1–E7 and seeded slot sampling over persisted SC generations |
| `utils.agents` | Wrap HF models as LangChain `HuggingFacePipeline` LLMs with optional seeded generation |
| `utils.graphs` | LangGraph templates: one builder for Majority Voting / Self-Consistency (live and replay modes), one for BoN-MAV |
| `utils.voting` | Plurality voting and the Self-Consistency ladder (vote once, re-derive every *k*) |
| `utils.mav` | Aspect-verifier prompts, verdict parsing and BoN-MAV candidate selection |
| `utils.evaluation` | Accuracy — numeric match (GSM8K), letter match (MMLU), symbolic equivalence via `math-verify` (MATH-500) |
| `utils.instrumentation` | Count input/output tokens; measure GPU peak VRAM; compute throughput (tokens/s) |
| `utils.warmup` | Run throwaway generations to trigger CUDA/cuBLAS kernel compilation before timed measurements |
| `utils.persistence` | Persist full generation text + metrics as a JSONL W&B artifact and retrieve it downstream with real producer→consumer lineage |
| `utils.statistics` | Cluster-corrected estimators: accuracy (Student-t CI), latency/tokens (BCa bootstrap), overhead ratios (log-space t), VRAM (Student-t); paired Student-t and Wilcoxon comparisons, plus independent-sample tests when needed |
| `utils.logging_utils` | Initialise & log metrics to W&B runs |

---

## 🗜️ Model Quantization

All models are loaded 4-bit within the T4's 16 GB budget via `utils.models.load_model_and_tokenizer`, but the quantization mechanism differs by family:

| Family | Checkpoint | Quantization |
|---|---|---|
| Ministral-3, Phi-4, Qwen3 | full-precision HF checkpoint | **bitsandbytes NF4**, applied on the fly at load time |
| Gemma-4 (E2B/E4B) | `google/gemma-4-{E2B,E4B}-it-qat-w4a16-ct` | **compressed-tensors w4a16** (QAT) — already 4-bit quantized on disk, no bitsandbytes step |

The Gemma-4 checkpoints are Google's official pre-quantized QAT release, serialized in the `compressed-tensors` format (4-bit weights / 16-bit activations). They are loaded by calling `load_model_and_tokenizer(model_id, quantize_4bit=False)`: `quantize_4bit=False` skips bitsandbytes and lets `transformers`' compressed-tensors `HFQuantizer` integration restore the checkpoint's own `quantization_config` on load — forcing `dtype=torch.float16` rather than the checkpoint's declared `bfloat16`, since the T4 (compute capability 7.5) has no native bf16 support and float16 is also compressed-tensors' own recommended compute dtype. (Passing `quantize_4bit=True` for such a `-qat-w4a16-ct` checkpoint raises, since it cannot be re-quantized.) This requires the `compressed-tensors` PyPI package (pinned in `requirements.txt`) in addition to `transformers`' built-in quantizer support.

---

## 📊 Experiment Tracking

Every run is logged to Weights & Biases with per-sample accuracy, aggregate accuracy, per-sample latency, total token consumption, peak VRAM and GPU utilisation. Confidence intervals use the cluster-corrected estimators in `utils/statistics.py`, and every thesis claim maps to a specific W&B run ID and the exact figure or table supporting it (Claim-to-Evidence).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
