# Thesis notebooks

This directory is the complete notebook record for the TFG *Optimizing Agentic AI Collaboration with SLMs*. It contains the setup runs, single-agent baselines, self-consistency experiments, multi-agent protocols, and the supporting W&B run directories used during development and evaluation.

Experiments were run on Kaggle with one T4 GPU (`cuda:0`), a 16 GB VRAM budget, and `batch_size=1`. Results that were committed with the notebook remain in its outputs; the canonical experiment tracking is in W&B. The two dual-benchmark Gemma notebooks `v2.4.0` and `v2.4.1` completed GSM8K but timed out during MMLU at Kaggle's 12-hour limit, so their unfinished MMLU stage is represented by the matching `FIX v2.4.0` and `FIX v2.4.1` notebooks.

## Subfolders

| Folder | Contents |
|---|---|
| [`setup/`](setup/) | v1.x end-to-end templates for dataset loading, LCEL generation, evaluation, instrumentation, and W&B logging. The v1.1.x notebooks are historical runs; v1.2.x are the later setup templates. |
| [`baselines/`](baselines/) | v2.x single-agent zero-shot and Chain-of-Thought reference runs over GSM8K, MATH-500, and MMLU. The `FIX` notebooks rerun only the MMLU portion missing after the Gemma timeout. |
| [`self-consistency/`](self-consistency/) | v3.x Self-Consistency generation passes and the L1 smoke test. Generation text and metrics are persisted for downstream replay. |
| [`majority-voting/`](majority-voting/) | v4.x L1 Majority Voting over homogeneous and heterogeneous model ensembles, replaying persisted Self-Consistency generations. . |
| [`bon-mav/`](bon-mav/) | v5.x Best-of-N with Model-as-Verifier runs. Verifiers score persisted candidates across benchmark-specific aspects. |
| [`archive/`](archive/) | Discarded notebooks retained for provenance but excluded from the current experiment set. The archive README documents the four never-executed drafts. |

## Running a notebook

Upload the selected notebook to a Kaggle kernel, enable GPU T4 x2, add `GITHUB_TOKEN`, `HF_TOKEN`, and `WANDB_API_KEY` under **Add-ons → Secrets**, and run all cells. The bootstrap cell pins the experiment to `cuda:0`, clones the private repository, and loads the shared utilities. The local virtual environment is intended for utility tests and notebook inspection; full inference requires Kaggle GPU access.

## Status conventions

- A notebook in an active protocol folder is part of the thesis experiment record unless its opening cell says otherwise.
- Historical setup notebooks are kept to preserve the evolution of the evaluation harness, even when their old utility API is no longer current.
- Archived notebooks are discarded, never-executed drafts retained for provenance and must not be cited as completed evidence.
