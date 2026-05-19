"""
logging_utils.py — Weights & Biases experiment tracking wrappers.

Thin helpers to initialise W&B runs and log metrics consistently
across all notebooks in the project.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import wandb  # type: ignore


# Detect Google Colab environment
_IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ

_DEFAULT_PROJECT = "agentic-collab" if _IN_COLAB else "dev-agentic-collab"


def init_wandb(
    run_name: str,
    config: Optional[dict[str, Any]] = None,
    project: Optional[str] = None,
    tags: Optional[list[str]] = None,
    notes: Optional[str] = None,
    **kwargs: Any,
) -> wandb.sdk.wandb_run.Run:
    """Initialise a W&B run with project defaults.

    Parameters
    ----------
    run_name : str
        Human-readable name for this experiment run.
    config : dict | None
        Hyperparameters / metadata to log.
    project : str | None
        W&B project name. Defaults to ``WANDB_PROJECT`` env var
        or ``"tfg-agentic-ai"``.
    tags : list[str] | None
        Tags to attach to the run.
    notes : str | None
        Free-text notes.

    Returns
    -------
    wandb.Run
    """
    project = project or os.environ.get("WANDB_PROJECT", _DEFAULT_PROJECT)
    return wandb.init(
        project=project,
        name=run_name,
        config=config or {},
        tags=tags or [],
        notes=notes,
        **kwargs,
    )


def log_metrics(metrics: dict[str, Any], step: Optional[int] = None) -> None:
    """Log a dictionary of metrics to the active W&B run."""
    wandb.log(metrics, step=step)


def finish_wandb() -> None:
    """Gracefully finish the current W&B run."""
    wandb.finish()
