"""
logging_utils.py — Weights & Biases experiment tracking wrappers.

Thin helpers to initialise W&B runs and log metrics consistently
across all notebooks in the project.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import wandb  # type: ignore


# Detect cloud runtime environment
_IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ
_IN_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

_DEFAULT_PROJECT = "agentic-collab" if (_IN_COLAB or _IN_KAGGLE) else "dev-agentic-collab"


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
        W&B project name. Defaults to the ``WANDB_PROJECT`` env var, or
        else ``"agentic-collab"`` on Colab/Kaggle and ``"dev-agentic-collab"``
        locally — keeping ad-hoc local runs out of the real project.
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


def define_wandb_chart_only_metrics(
    prefixes: Optional[list[str]] = None,
) -> None:
    """Mark per-sample metric prefixes as chart-only (no summary).

    By default, W&B automatically promotes the **last** logged value
    of every metric key into run.summary.  For per-sample metrics
    (e.g. gsm8k_sample/*, mmlu_sample/*) this clutters the summary with a
    single, non-representative 0/1 value.

    Calling this function right after :func:`init_wandb` tells W&B
    to keep those keys in the **history** (charts) only.

    Parameters
    ----------
    prefixes : list[str] | None
        Glob-style metric prefixes to exclude from summary.
        Defaults to ``["gsm8k_sample/*", "mmlu_sample/*"]``.
    """
    if prefixes is None:
        prefixes = ["gsm8k_sample/*", "mmlu_sample/*"]
    for prefix in prefixes:
        wandb.define_metric(prefix, summary="none")


def log_metrics(metrics: dict[str, Any], step: Optional[int] = None) -> None:
    """Log a dictionary of metrics to the active W&B run."""
    wandb.log(metrics, step=step)


def finish_wandb() -> None:
    """Gracefully finish the current W&B run."""
    wandb.finish()


def log_stat_summary(
    prefix: str,
    stat: dict,
    keys: list[str] | None = None,
) -> None:
    """Push a point estimate + CI (and ci_method) into ``wandb.summary``.

    Parameters
    ----------
    prefix : str
        Namespace prefix, e.g. ``"acc"``, ``"latency"``,
        ``"overhead_latency"``, ``"vram"``.  Each key is logged as
        ``"{prefix}/{key}"``.
    stat : dict
        Result dict from any ``*_statistics`` function.
    keys : list[str] | None
        Subset of keys to log.  Defaults to the common set:
        ``mean`` (or ``p_hat``/``geometric_mean``/``mean_diff``/
        ``mean_of_ratios``), ``ci_lower``, ``ci_upper``, ``ci_method``.
        ``mean_peak`` (from ``vram_statistics``) is a special case,
        defaulting to ``peak_ci_lower``/``peak_ci_upper`` instead.
    """
    if keys is None:
        # Pick the primary point-estimate key present in the dict.
        # mean_peak (vram_statistics) uses peak_ci_lower/peak_ci_upper
        # instead of ci_lower/ci_upper, to disambiguate from mean_avg.
        if "mean_peak" in stat:
            keys = ["mean_peak", "peak_ci_lower", "peak_ci_upper"]
        else:
            for candidate in ("mean", "p_hat", "geometric_mean",
                              "mean_diff", "mean_of_ratios"):
                if candidate in stat:
                    keys = [candidate, "ci_lower", "ci_upper"]
                    break
            else:
                keys = ["ci_lower", "ci_upper"]
        if "ci_method" in stat:
            keys = list(keys) + ["ci_method"]

    for key in keys:
        if key in stat:
            value = stat[key]
            # Skip non-scalar values (arrays, nested dicts)
            if not isinstance(value, (int, float, str, bool)):
                continue
            wandb.run.summary[f"{prefix}/{key}"] = value  # type: ignore[union-attr]
