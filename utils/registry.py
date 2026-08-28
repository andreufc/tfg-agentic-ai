"""
registry.py: the single source of truth for model identity and artifact
naming across the L1 campaign (Self-Consistency, Majority Voting, BoN-MAV).

Stdlib only: no torch, transformers, wandb or datasets import here, so any
CPU-only consumer notebook (e.g. the planned Majority Voting notebook) can
import this module without dragging in the GPU stack.

`model_key` (e.g. `"qwen3-1.7b"`) is the primary-key form of a model used
throughout the persisted record schema (see `utils.persistence`), W&B
tables, run names and artifact names. The HF repo id (`model_id`) is
recorded exactly once, here, and recovered everywhere else via
:func:`get_model`, not duplicated into every persisted row.

Connection to the project WPs
------------------------------
* WP2-T4 (L1 multi-agent protocols): `utils/ensembles.py`
  (Majority Voting / BoN-MAV) imports `MODELS` from here rather than
  redefining it, so run names, artifact names and thesis tables cannot
  drift between protocols.
"""

from __future__ import annotations

from typing import Optional

GENERATIONS_ARTIFACT_PREFIX = "sc-generations"
MATRICES_ARTIFACT_PREFIX = "evaluation-matrices"

DATASETS = ("math500", "mmlu", "gsm8k")

# model_key -> {model_id, run_name, display_name, no_think}
#   model_id     HF repo id, the only place this is recorded.
#   run_name     W&B run-name fragment, mirrors each notebook's RUN_NAME
#                literal so run names on Kaggle don't change.
#   display_name matches the v2.x baseline artifact naming convention, kept
#                for a future cross-era join. New artifacts use model_key.
#   no_think     whether the model needs a "/no_think" directive to suppress
#                reasoning traces (Qwen3 family only).
MODELS: dict[str, dict] = {
    "qwen3-1.7b": {
        "model_id": "Qwen/Qwen3-1.7B",
        "run_name": "SC-Qwen3-1.7B",
        "display_name": "Qwen3-1.7B",
        "no_think": True,
    },
    "qwen3-4b": {
        "model_id": "Qwen/Qwen3-4B",
        "run_name": "SC-Qwen3-4B",
        "display_name": "Qwen3-4B",
        "no_think": True,
    },
    "ministral-3-3b": {
        "model_id": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
        "run_name": "SC-Ministral-3-3B-Instruct",
        "display_name": "Ministral-3-3B-Instruct",
        "no_think": False,
    },
    "phi-4-mini": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "run_name": "SC-Phi-4-mini-instruct",
        "display_name": "Phi-4-mini-instruct",
        "no_think": False,
    },
}


def get_model(model_key: str) -> dict:
    """Look up a model's registry entry by its short key.

    Raises
    ------
    ValueError
        If `model_key` is not registered, naming the valid set.
    """
    try:
        return MODELS[model_key]
    except KeyError:
        raise ValueError(
            f"Unknown model_key {model_key!r}; expected one of {sorted(MODELS)}"
        ) from None


def _check_dataset(dataset: str) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")


def generations_artifact_name(model_key: str, dataset: str) -> str:
    """Canonical artifact name for a model's SC generations on a benchmark."""
    get_model(model_key)  # validates model_key
    _check_dataset(dataset)
    return f"{GENERATIONS_ARTIFACT_PREFIX}-{model_key}-{dataset}"


def matrices_artifact_name(model_key: str, protocol_tag: str, dataset: str) -> str:
    """Canonical name for an evaluation-matrices `.npz` artifact.

    `protocol_tag` is a short free-form tag distinguishing the protocol and
    its configuration within that protocol, e.g. `"sc3"` for SC@3 or
    `"maj5-E4"` for a 5-slot Majority Voting ensemble.
    """
    get_model(model_key)
    _check_dataset(dataset)
    return f"{MATRICES_ARTIFACT_PREFIX}-{model_key}-{protocol_tag}-{dataset}"


def generation_run_name(model_key: str, dataset: str, *, suffix: str = "") -> str:
    """W&B run name for an SC generation pass."""
    run_name = get_model(model_key)["run_name"]
    _check_dataset(dataset)
    name = f"l1-sc-gen-{run_name}-{dataset}"
    return f"{name}-{suffix}" if suffix else name


def sc_run_name(model_key: str, dataset: str, k: int, *, suffix: str = "") -> str:
    """W&B run name for an SC@k aggregation run."""
    run_name = get_model(model_key)["run_name"]
    _check_dataset(dataset)
    name = f"l1-sc@{k}-{run_name}-{dataset}"
    return f"{name}-{suffix}" if suffix else name


def problem_ref(dataset: str, problem_id: int) -> str:
    """Human-citable label for a problem, e.g. `"math500#042"`.

    Used in thesis prose and on the SC results table to refer to an
    individual problem. Never persisted in the JSONL record itself: it is a
    pure function of `(dataset, problem_id)`, so storing it per-row would be
    redundant with the primary key.
    """
    _check_dataset(dataset)
    return f"{dataset}#{problem_id:03d}"
