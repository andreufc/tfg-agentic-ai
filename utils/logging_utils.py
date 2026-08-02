"""
logging_utils.py - Weights & Biases experiment tracking wrappers.

Thin helpers to initialise W&B runs and log metrics consistently
across all notebooks in the project.

Connection to the project WPs
-----------------------------
* WP1-T1/T4 & WP2-T3/T4/T6: opens a named run and logs metrics + result
  matrices in one consistent schema.
* WP3-T1: makes each thesis claim traceable to a specific W&B run ID
  (Claim-to-Evidence).
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

import numpy as np
import wandb  # type: ignore

from utils.statistics import per_problem_means


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
        W&B project name. Defaults to the WANDB_PROJECT env var, or else
        "agentic-collab" on Colab/Kaggle and "dev-agentic-collab" locally,
        keeping ad-hoc local runs out of the real project.
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

    W&B by default promotes the last logged value of every metric key into
    run.summary. For per-sample metrics this clutters the summary with a
    single, non-representative 0/1 value. Call this right after init_wandb
    to keep those keys in the history (charts) only.

    Parameters
    ----------
    prefixes : list[str] | None
        Glob-style metric prefixes to exclude from summary. Defaults to
        ["gsm8k_sample/*", "mmlu_sample/*", "math_sample/*"].
    """
    if prefixes is None:
        prefixes = ["gsm8k_sample/*", "mmlu_sample/*", "math_sample/*"]
    for prefix in prefixes:
        wandb.define_metric(prefix, summary="none")


def log_metrics(metrics: dict[str, Any], step: Optional[int] = None) -> None:
    """Log a dictionary of metrics to the active W&B run."""
    wandb.log(metrics, step=step)


# Per-sample chart precision, one place for every protocol notebook. Token
# counts stay floats at 2 dp because they are rep-means, not counts: rounding
# them to int would drift from the *_tokens_stats CIs, which are computed on
# exactly these per-problem means.
PER_SAMPLE_ROUNDING: dict[str, int] = {
    "accuracy": 4,
    "latency_s": 2,
    "output_tokens": 2,
    "total_tokens": 2,
    "gpu_peak_mb": 2,
    "throughput_tok_s": 4,
    "pct_votes_top": 4,
}


def log_per_problem_means(
    prefix: str,
    matrices: Mapping[str, np.ndarray],
    rounding: Mapping[str, int] | None = None,
) -> None:
    """Log one per-sample history point per problem, averaged over reps.

    Protocols that repeat the evaluation (MAJ runs N_REPS reps per problem)
    must still emit a single per-sample series indexed by problem, so their
    charts line up on the implicit _step axis with single-rep protocols
    such as BoN-MAV. Each (N, m) matrix is collapsed to its N per-problem
    means via utils.statistics.per_problem_means and emitted as one
    wandb.log at step=problem_id.

    Call once, after the evaluation loops and before any other wandb.log
    in the run: an earlier history write pushes _step past problem_id and
    W&B then rejects the points.

    Parameters
    ----------
    prefix : str
        Metric namespace without trailing slash, e.g. "math500_sample".
    matrices : Mapping[str, np.ndarray]
        Metric name -> (n_problems, n_reps) matrix. All matrices must
        share the same n_problems.
    rounding : Mapping[str, int] | None
        Decimal places per metric name. Defaults to PER_SAMPLE_ROUNDING;
        names absent from the mapping are logged unrounded.
    """
    if not matrices:
        raise ValueError("matrices must contain at least one metric.")

    rounding = PER_SAMPLE_ROUNDING if rounding is None else rounding
    means = {
        name: per_problem_means(np.asarray(matrix, dtype=float))
        for name, matrix in matrices.items()
    }

    sizes = {name: arr.shape[0] for name, arr in means.items()}
    if len(set(sizes.values())) > 1:
        raise ValueError(
            f"All matrices must share the same n_problems, got {sizes}"
        )

    for problem_id in range(next(iter(sizes.values()))):
        log_metrics(
            {
                f"{prefix}/{name}": (
                    round(float(arr[problem_id]), rounding[name])
                    if name in rounding
                    else float(arr[problem_id])
                )
                for name, arr in means.items()
            },
            step=problem_id,
        )


def finish_wandb() -> None:
    """Gracefully finish the current W&B run."""
    wandb.finish()


def log_stat_summary(
    prefix: str,
    stat: dict,
    keys: list[str] | None = None,
) -> None:
    """Push a point estimate + CI (and ci_method) into wandb.summary.

    Parameters
    ----------
    prefix : str
        Namespace prefix, e.g. "acc", "latency", "overhead_latency",
        "vram". Each key is logged as "{prefix}/{key}".
    stat : dict
        Result dict from any *_statistics function.
    keys : list[str] | None
        Subset of keys to log. Defaults to the common set: mean (or
        p_hat/geometric_mean/mean_diff/mean_of_ratios), ci_lower,
        ci_upper, ci_method. mean_peak (from vram_statistics) is a
        special case, defaulting to peak_ci_lower/peak_ci_upper instead.
    """
    if keys is None:
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
