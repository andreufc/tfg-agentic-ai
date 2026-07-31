"""
statistics.py — Statistically consistent, unbiased estimators for
clustered experimental data (repeated seeds per problem).

Connection to the project WPs
-----------------------------
* WP2-T5/T7 & WP3-T1: turns the seed×problem matrices into cluster-corrected
  accuracy CIs, BCa-bootstrap/overhead intervals and between-system tests.
* WP1-T1 & WP2-T3: also called inline to report per-run accuracy/cost CIs.

Provides:
    Shared reduction:
    - per_problem_means:                Validate (N, m) and reduce to the N
                                        per-problem means — the unit of analysis
                                        for every estimator here and for the
                                        per-sample W&B series

    Per-system estimation (primary scope):
    - pooled_accuracy:                  Unbiased point estimate of accuracy
    - wilson_ci_clustered:              Wilson CI adjusted for within-problem
                                        correlation via the Design Effect (DEFF)
    - accuracy_statistics:              Cluster-t CI for accuracy on per-problem
                                        means; Wilson kept as supplementary
    - pct_votes_top_statistics:         Cluster-t CI for the winning option's
                                        vote/approval share, on per-problem means
    - time_statistics / token_statistics:
                                        Nonparametric summary for any (N, m)
                                        time or token-count matrix, built on the
                                        shared ``_metric_statistics`` helper;
                                        CI on per-problem means (BCa bootstrap)
    - overhead_ratio_statistics:        Geometric-mean CI via log-space Student-t
                                        for multiplicative overhead (latency, tokens)
    - overhead_difference_statistics:   Cluster-t CI for additive overhead (VRAM)
    - vram_statistics:                  Cluster-t CI on per-problem peak-VRAM means

    Between-system comparison tests (deferred — not part of per-system scope):
    - welch_t_test_metric:              Welch t-test on per-problem means
    - mann_whitney_u_test_metric:       Rank-based distribution-free comparison

All functions expect data as NumPy arrays of shape ``(N, m)`` where
``N`` = number of problems and ``m`` = number of seeds (repetitions).

Unit of analysis
----------------
Every CI and inferential variance is computed on the ``N`` per-problem means,
never on the flattened ``N×m`` array. The flattened statistics (mean, std,
percentiles, skewness, …) are retained as *descriptive* diagnostics only.

Note
----
``print_throughput_report`` is implemented for a forthcoming
``throughput_statistics`` estimator (tokens/s with mean-of-ratios as the
canonical value); until that estimator lands, callers must assemble its
input dict by hand — see the function's docstring for the expected shape.

References
----------
- Wilson (1927), JASA 22(158), 209–212.
- Rao & Scott (1992), Biometrics 48(2), 577–585.
- Korn & Graubard (1998), Survey Methodology 24, 193–201.
- Efron & Tibshirani (1993), An Introduction to the Bootstrap, Ch. 14.
- Mann & Whitney (1947), Ann. Math. Statist. 18(1), 50–60.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Helpers — per-problem reduction (public) and CI machinery (private)
# ---------------------------------------------------------------------------

def per_problem_means(arr: np.ndarray) -> np.ndarray:
    """Validate (N, m) shape and return the N per-problem means.

    Public because the unit of analysis is shared: every estimator here and
    the per-sample W&B series in :mod:`utils.logging_utils` must collapse a
    problems x reps matrix the same way.
    """
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D array of shape (N, m), got shape {arr.shape}"
        )
    return np.mean(arr, axis=1)


def _cluster_t_ci(
    values: np.ndarray,
    alpha: float = 0.05,
    clip: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Two-sided Student-t CI on a 1-D array of per-problem values.

    Computes ``mean ± t_{N-1, alpha/2} * std(ddof=1) / sqrt(N)``.
    Degenerates to ``(mean, mean)`` when N == 1 or all values are identical.
    Optionally clips both endpoints into ``clip = (lo, hi)``.
    """
    v = np.asarray(values, dtype=float)

    if v.ndim != 1:
        raise ValueError(f"Expected 1-D array of per-problem values, got shape {v.shape}")

    n = len(v)
    mean = float(np.mean(v))

    if n <= 1 or float(np.ptp(v)) == 0.0:
        lo, hi = mean, mean
    else:
        se = float(np.std(v, ddof=1)) / np.sqrt(n)
        t_crit = float(sp_stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        lo = mean - t_crit * se
        hi = mean + t_crit * se

    if clip is not None:
        lo = float(np.clip(lo, clip[0], clip[1]))
        hi = float(np.clip(hi, clip[0], clip[1]))

    return float(lo), float(hi)


def _cluster_bca_ci(
    values: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 9999,
    seed: int = 0,
    clip: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """BCa bootstrap CI on a 1-D array of per-problem values.

    Zero-variance guard: constant input returns (mean, mean) without
    calling ``sp_stats.bootstrap`` (which would raise on zero-variance data).
    """
    v = np.asarray(values, dtype=float)
    mean = float(np.mean(v))

    if float(np.ptp(v)) == 0.0:
        return mean, mean

    boot = sp_stats.bootstrap(
        (v,),
        statistic=np.mean,
        n_resamples=n_boot,
        confidence_level=1.0 - alpha,
        method="BCa",
        random_state=seed,
    )

    lo = boot.confidence_interval.low
    hi = boot.confidence_interval.high

    if clip is not None:
        lo = float(np.clip(lo, clip[0], clip[1]))
        hi = float(np.clip(hi, clip[0], clip[1]))

    return (float(lo), float(hi))


def _log_ratio_t_ci(
    ratios: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Geometric-mean CI via t-interval in log space, endpoints exponentiated.

    Requires ratios > 0.
    """
    log_r = np.log(np.asarray(ratios, dtype=float))
    lo_log, hi_log = _cluster_t_ci(log_r, alpha)
    return float(np.exp(lo_log)), float(np.exp(hi_log))


# ---------------------------------------------------------------------------
# Accuracy estimation
# ---------------------------------------------------------------------------

def pooled_accuracy(correctness: ArrayLike) -> float:
    """Unbiased point estimate of accuracy from clustered binary data.

    Parameters
    ----------
    correctness : array-like of shape (N, m)
        Binary correctness matrix.

    Returns
    -------
    float
        Pooled accuracy :math:`\\hat{p} = \\frac{1}{Nm} \\sum_{i,j} X_{i,j}`.
    """
    arr = np.asarray(correctness, dtype=float)
    return float(np.mean(arr))


def wilson_ci_clustered(
    correctness: ArrayLike,
    alpha: float = 0.05,
) -> dict[str, float]:
    """DEFF-adjusted Wilson confidence interval for clustered binary data.

    Accounts for within-problem correlation across seeds by computing the
    design effect (DEFF), which reduces the effective sample size, and
    then applies the Wilson score interval with that effective *n*.

    Parameters
    ----------
    correctness : array-like of shape (N, m)
        Binary correctness matrix.
    alpha : float, default 0.05
        Significance level (0.05 -> 95 % CI).

    Returns
    -------
    dict
        Keys: p_hat, ci_lower, ci_upper, deff, n_eff, icc, se,
        n_problems, n_seeds, n_total.
    """
    arr = np.asarray(correctness, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D array of shape (N, m), got shape {arr.shape}"
        )
    n_problems, n_seeds = arr.shape
    n_total = n_problems * n_seeds

    p_hat = pooled_accuracy(arr)
    x_bar_i = per_problem_means(arr)
    sigma2_c = float(np.var(x_bar_i, ddof=1))

    if p_hat == 0.0 or p_hat == 1.0:
        return {
            "p_hat": p_hat,
            "ci_lower": p_hat,
            "ci_upper": p_hat,
            "deff": float(n_seeds),
            "n_eff": float(n_problems),
            "icc": 1.0,
            "se": 0.0,
            "n_problems": n_problems,
            "n_seeds": n_seeds,
            "n_total": n_total,
        }

    deff = n_seeds * sigma2_c / (p_hat * (1.0 - p_hat))
    deff = max(deff, 1.0)
    n_eff = n_total / deff
    icc = (deff - 1.0) / (n_seeds - 1.0) if n_seeds > 1 else 0.0
    se = float(np.sqrt(sigma2_c / n_problems))

    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)
    denom = 1.0 + z ** 2 / n_eff
    centre = (p_hat + z ** 2 / (2.0 * n_eff)) / denom
    margin = (
        z
        * np.sqrt(p_hat * (1.0 - p_hat) / n_eff + z ** 2 / (4.0 * n_eff ** 2))
        / denom
    )
    ci_lower = max(centre - margin, 0.0)
    ci_upper = min(centre + margin, 1.0)

    return {
        "p_hat": p_hat,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "deff": float(deff),
        "n_eff": float(n_eff),
        "icc": float(icc),
        "se": se,
        "n_problems": n_problems,
        "n_seeds": n_seeds,
        "n_total": n_total,
    }


def accuracy_statistics(
    correctness: ArrayLike,
    alpha: float = 0.05,
) -> dict:
    """Cluster-t CI for accuracy, with Wilson kept as supplementary.

    The primary CI is a Student-t interval on the N per-problem
    accuracy means, clipped to [0, 1]. This correctly treats the
    problem as the unit of analysis.  The DEFF-adjusted Wilson CI from
    :func:`wilson_ci_clustered` is flattened in under the ``wilson_*``
    keys for comparison.

    Parameters
    ----------
    correctness : array-like of shape (N, m)
        Binary correctness matrix (0 or 1).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Keys: p_hat, ci_lower, ci_upper, ci_method, alpha,
        between_problem_std, n_problems, n_seeds, n_total,
        wilson_ci_lower, wilson_ci_upper, wilson_deff, wilson_n_eff,
        wilson_icc, wilson_se.
    """
    arr = np.asarray(correctness, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D array of shape (N, m), got shape {arr.shape}"
        )
    n_problems, n_seeds = arr.shape
    n_total = n_problems * n_seeds

    p_i = per_problem_means(arr)          # per-problem accuracies, shape (N,)
    p_hat = pooled_accuracy(arr)
    ci_lower, ci_upper = _cluster_t_ci(p_i, alpha, clip=(0.0, 1.0))
    between_problem_std = float(np.std(p_i, ddof=1)) if n_problems > 1 else 0.0

    wilson = wilson_ci_clustered(arr, alpha)
    return {
        "p_hat": p_hat,
        "between_problem_std": between_problem_std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_method": "cluster_t",
        "alpha": alpha,
        "n_problems": n_problems,
        "n_seeds": n_seeds,
        "n_total": n_total,
        "wilson_ci_lower": wilson["ci_lower"],
        "wilson_ci_upper": wilson["ci_upper"],
        "wilson_deff": wilson["deff"],
        "wilson_n_eff": wilson["n_eff"],
        "wilson_icc": wilson["icc"],
        "wilson_se": wilson["se"],
    }


# ---------------------------------------------------------------------------
# Latency and token counts estimation (nonparametric and right-skewed)
# ---------------------------------------------------------------------------

def _metric_statistics(
    values: ArrayLike,
    kind: str = "continuous",
    alpha: float = 0.05,
    n_bootstrap: int = 9999,
    bounds: tuple[float, float]|None = None,
) -> dict:
    """Distribution-free summary for any (N, m) resource metric.

    The CI is computed on the N per-problem means (cluster-corrected).
    Flattened descriptive statistics are retained unchanged.

    Parameters
    ----------
    values : array-like of shape (N, m)
        Resource metric matrix (e.g. output tokens, latency).
    kind : str, default "continuous"
        One of "count", "continuous".
    alpha : float, default 0.05
        Significance level for the BCa bootstrap CI.
    n_bootstrap : int, default 9999
        Number of bootstrap replications.
    bounds : tuple[float, float]|None, default None
        Used to clamp the CI and compute floor/ceiling fractions if the metric has natural bounds (e.g. [0, 1] for a proportion).

    Returns
    -------
    dict
        Common keys: mean, percentiles, flat_std, flat_cv,
            between_problem_std, ci_lower, ci_upper, ci_method,
            alpha, n_problems, n_seeds, n_total, kind.
        Additional keys: dispersion_index when kind="count"; frac_at_ci_lower /
            frac_at_ci_upper when bounds is given and some values sit at it.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D array of shape (N, m), got shape {arr.shape}"
        )
    n_problems, n_seeds = arr.shape
    n_total = n_problems * n_seeds
    flat = arr.ravel()

    # ── Descriptive (flattened) ──
    mean = float(np.mean(flat))
    variance = float(np.var(flat, ddof=1))
    std = float(np.std(flat, ddof=1))
    cv = std / mean if mean != 0.0 else float("nan")

    p25, p50, p75, p95 = np.percentile(flat, [25, 50, 75, 95])
    percentiles = {
        "p25": float(p25), "p50": float(p50),
        "p75": float(p75), "p95": float(p95),
    }

    # ── Inferential (per-problem means) ──
    pp_means = per_problem_means(arr)
    ci_lower, ci_upper = _cluster_bca_ci(pp_means, alpha, n_bootstrap)
    between_problem_std = float(np.std(pp_means, ddof=1)) if n_problems > 1 else 0.0

    result = {
        "mean": mean,
        "percentiles": percentiles,
        "flat_std": std,
        "flat_cv": cv,
        "between_problem_std": between_problem_std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_method": "cluster_bca",
        "alpha": alpha,
        "n_problems": n_problems,
        "n_seeds": n_seeds,
        "n_total": n_total,
        "kind": kind,
    }

    if kind == "count":
        di = variance / mean if mean != 0.0 else float("nan")
        result["dispersion_index"] = di
        result["percentiles"] = {k: int(v) for k, v in percentiles.items()}

    if bounds is not None:
        lo, hi = bounds
        result["ci_lower"] = float(np.clip(ci_lower, lo, hi))
        result["ci_upper"] = float(np.clip(ci_upper, lo, hi))
        if np.min(flat) < lo:
            result["frac_at_ci_lower"] = float(np.mean(flat == lo))
        if np.max(flat) > hi:
            result["frac_at_ci_upper"] = float(np.mean(flat == hi))

    return result


def time_statistics(
    times: ArrayLike,
    alpha: float = 0.05,
    n_bootstrap: int = 9999,
) -> dict:
    """Distribution-free execution time summary with BCa bootstrap CI.

    The CI is computed on the N per-problem means (cluster-corrected),
    not on the flattened N*m array.  Flattened descriptive statistics
    (mean, std, percentiles) are retained unchanged.

    Parameters
    ----------
    times : array-like of shape (N, m)
        Execution time matrix (positive reals, in seconds).
    alpha : float, default 0.05
        Significance level (0.05 -> 95 % CI).
    n_bootstrap : int, default 9999
        Bootstrap replications for the BCa CI.

    Returns
    -------
    dict
        Keys: mean, percentiles, flat_std, flat_cv,
            between_problem_std, ci_lower, ci_upper, ci_method,
            alpha, n_problems, n_seeds, n_total, kind="continuous".
    """

    return _metric_statistics(
        times,
        kind="continuous",
        alpha=alpha,
        n_bootstrap=n_bootstrap
    )


def token_statistics(
    values: ArrayLike,
    alpha: float = 0.05,
    n_bootstrap: int = 9999,
    max_tokens: int | None = None,
) -> dict:
    """Nonparametric summary for an (N, m) token-count matrix.

    Thin wrapper over :func:`_metric_statistics` with ``kind="count"``,
    ensuring the CI is computed on per-problem means.  Call twice —
    once for ``output_tokens`` and once for ``total_tokens``.

    Parameters
    ----------
    values : array-like of shape ``(N, m)``
        Token count matrix (non-negative integers).
    alpha : float, default 0.05
        Significance level.
    n_bootstrap : int, default 9999
        Bootstrap replications.

    Returns
    -------
    dict
        Keys: mean, percentiles, flat_std, flat_cv,
            between_problem_std, ci_lower, ci_upper, ci_method,
            alpha, n_problems, n_seeds, n_total, kind="count".
        Optional keys if max_tokens is finite and some values are at the bound.
    """
    bounds = (0.0, float(max_tokens)) if max_tokens is not None else None
    return _metric_statistics(
        values,
        kind="count",
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        bounds=bounds
    )


def vram_statistics(
    peak_matrix: ArrayLike,
    avg_matrix: ArrayLike | None = None,
    alpha: float = 0.05,
) -> dict:
    """Cluster-t CI on per-problem peak-VRAM means.

    ``torch.cuda.max_memory_allocated`` (the authoritative per-run peak)
    is largely deterministic given the model and sequence length, so a
    Student-t interval on the per-problem means is appropriate.

    Parameters
    ----------
    peak_matrix : array-like of shape ``(N, m)``
        Per-run peak VRAM in MB (from ``gpu_peak_mem_mb``).
    avg_matrix : array-like of shape ``(N, m)`` or None
        Per-run average VRAM utilisation in MB (optional).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Keys: mean_peak, peak_ci_lower, peak_ci_upper, ci_method, alpha,
        peak_max, peak_percentiles, between_problem_peak_std, n_problems,
        n_seeds, n_total.  Optional: mean_avg (when avg_matrix is given).
    """
    arr = np.asarray(peak_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D array of shape (N, m), got shape {arr.shape}"
        )
    n_problems, n_seeds = arr.shape
    n_total = n_problems * n_seeds
    flat = arr.ravel()

    pp_means = per_problem_means(arr)
    mean_peak = float(np.mean(pp_means))
    ci_lower, ci_upper = _cluster_t_ci(pp_means, alpha)
    peak_max = float(arr.max())
    between_problem_std = float(np.std(pp_means, ddof=1)) if n_problems > 1 else 0.0

    p25, p50, p75, p95 = np.percentile(flat, [25, 50, 75, 95])
    percentiles = {
        "p25": float(p25), "p50": float(p50),
        "p75": float(p75), "p95": float(p95),
    }

    result: dict = {
        "mean_peak": mean_peak,
        "peak_max": peak_max,
        "peak_percentiles": percentiles,
        "between_problem_peak_std": between_problem_std,
        "peak_ci_lower": ci_lower,
        "peak_ci_upper": ci_upper,
        "ci_method": "cluster_t",
        "alpha": alpha,
        "n_problems": n_problems,
        "n_seeds": n_seeds,
        "n_total": n_total
    }

    if avg_matrix is not None:
        avg = np.asarray(avg_matrix, dtype=float)
        result["mean_avg"] = float(np.mean(avg))

    return result


# ---------------------------------------------------------------------------
# Winning-option vote/approval share estimation
# ---------------------------------------------------------------------------

def pct_votes_top_statistics(
    pct_matrix: ArrayLike,
    alpha: float = 0.05,
) -> dict:
    """Cluster-t CI for the winning option's vote/approval share.

    Same cluster-t-over-per-problem-means approach as
    :func:`accuracy_statistics`, treating the problem as the unit of
    analysis. No Wilson supplementary CI here: the input is a continuous
    share in [0, 1] (fraction of votes/approvals the winner received),
    not a binary correctness indicator. Works identically whether ``m``
    (seeds/reps per problem) is 1 (BoN-MAV) or 2 (MAJ).

    Parameters
    ----------
    pct_matrix : array-like of shape (N, m)
        Per-run winning-option vote/approval share, in [0, 1].
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Keys: mean_pct_votes_top, between_problem_std, ci_lower, ci_upper,
        ci_method, alpha, n_problems, n_seeds, n_total.
    """
    arr = np.asarray(pct_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D array of shape (N, m), got shape {arr.shape}"
        )
    n_problems, n_seeds = arr.shape
    n_total = n_problems * n_seeds

    p_i = per_problem_means(arr)          # per-problem vote-share means, shape (N,)
    mean_pct_votes_top = float(np.mean(p_i))
    ci_lower, ci_upper = _cluster_t_ci(p_i, alpha, clip=(0.0, 1.0))
    between_problem_std = float(np.std(p_i, ddof=1)) if n_problems > 1 else 0.0

    return {
        "mean_pct_votes_top": mean_pct_votes_top,
        "between_problem_std": between_problem_std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_method": "cluster_t",
        "alpha": alpha,
        "n_problems": n_problems,
        "n_seeds": n_seeds,
        "n_total": n_total,
    }


# ---------------------------------------------------------------------------
# Ratios statistics: overhead ratios (latency, tokens) and throughput (tokens/s)
# ---------------------------------------------------------------------------

def overhead_ratio_statistics(
    values_mas: ArrayLike,
    values_base: ArrayLike,
    alpha: float = 0.05,
    bca_check: bool = False,
    n_bootstrap: int = 9999,
) -> dict:
    """Geometric-mean CI for multiplicative overhead (latency, tokens).

    Computes the per-problem ratio r_i = mean_mas_i / mean_base_i and
    derives a geometric-mean CI via a Student-t interval in log space.
    This is the correct approach for ratios: symmetric on the log scale,
    always positive endpoints.

    Parameters
    ----------
    values_mas : array-like of shape N, m
        MAS metric matrix.
    values_base : array-like of shape N, m
        Baseline metric matrix, matched problem order.
    alpha : float, default 0.05
        Significance level.
    bca_check : bool, default False
        If True, also return bca_ci (BCa on log-ratios, exponentiated)
        as an optional cross-check.
    n_bootstrap : int, default 9999
        Bootstrap replications (used only when bca_check=True).

    Returns
    -------
    dict
        Keys: geometric_mean, arithmetic_mean, between_problem_std,
        ci_lower, ci_upper, ci_method, alpha, n_problems, n_seeds,
        n_total, per_problem_ratios.  Optional: bca_ci (when bca_check=True).
    """
    mas = np.asarray(values_mas, dtype=float)
    base = np.asarray(values_base, dtype=float)
    if mas.ndim != 2 or base.ndim != 2:
        raise ValueError(
            "Both inputs must be 2-D arrays of shape (N, m)."
        )
    if mas.shape != base.shape:
        raise ValueError(
            f"Shape mismatch: values_mas {mas.shape} vs values_base {base.shape}."
        )

    n_problems, n_seeds = mas.shape
    base_means = per_problem_means(base)      # shape (N,)
    if np.any(base_means <= 0.0):
        raise ValueError(
            "All per-problem baseline means must be > 0 to compute ratios."
        )

    r_i = per_problem_means(mas) / base_means   # per-problem ratios

    geometric_mean = float(np.exp(np.mean(np.log(r_i))))
    arithmetic_mean = float(np.mean(r_i))
    ci_lower, ci_upper = _log_ratio_t_ci(r_i, alpha)
    between_problem_std = float(np.std(r_i, ddof=1)) if n_problems > 1 else 0.0

    result: dict = {
        "geometric_mean": geometric_mean,
        "arithmetic_mean": arithmetic_mean,
        "between_problem_std": between_problem_std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_method": "log_t",
        "alpha": alpha,
        "n_problems": n_problems,
        "n_seeds": n_seeds,
        "n_total": n_problems * n_seeds,
        "per_problem_ratios": r_i
    }

    if bca_check:
        lo_log, hi_log = _cluster_bca_ci(np.log(r_i), alpha, n_bootstrap)
        result["bca_ci"] = (float(np.exp(lo_log)), float(np.exp(hi_log)))

    return result


def overhead_difference_statistics(
    values_mas: ArrayLike,
    values_base: ArrayLike,
    alpha: float = 0.05,
) -> dict:
    """Cluster-t CI for additive overhead (e.g. VRAM delta).

    Computes per-problem differences d_i = mean_mas_i - mean_base_i
    and derives a Student-t CI.  Used for metrics where differences (not
    ratios) are the natural scale, such as peak VRAM.

    Parameters
    ----------
    values_mas : array-like of shape (N, m)
        MAS metric matrix.
    values_base : array-like of shape (N, m)
        Baseline metric matrix, matched problem order.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Keys: mean_diff, ci_lower, ci_upper, ci_method,
        between_problem_std, per_problem_diffs, n_problems, n_seeds.
    """
    mas = np.asarray(values_mas, dtype=float)
    base = np.asarray(values_base, dtype=float)
    if mas.ndim != 2 or base.ndim != 2:
        raise ValueError("Both inputs must be 2-D arrays of shape (N, m).")
    if mas.shape != base.shape:
        raise ValueError(
            f"Shape mismatch: values_mas {mas.shape} vs values_base {base.shape}."
        )

    n_problems, n_seeds = mas.shape
    d_i = per_problem_means(mas) - per_problem_means(base)

    mean_diff = float(np.mean(d_i))
    ci_lower, ci_upper = _cluster_t_ci(d_i, alpha)
    between_problem_std = float(np.std(d_i, ddof=1)) if n_problems > 1 else 0.0

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_method": "cluster_t",
        "between_problem_std": between_problem_std,
        "per_problem_diffs": d_i,
        "n_problems": n_problems,
        "n_seeds": n_seeds,
    }


# ---------------------------------------------------------------------------
# Between-system comparison tests  (deferred — not per-system scope)
# ---------------------------------------------------------------------------

def welch_t_test_metric(
    per_problem_means_a: ArrayLike,
    per_problem_means_b: ArrayLike,
) -> dict:
    """Welch t-test comparing per-problem means between two systems.

    .. note::
        Between-system scope: compares two already-computed per-problem mean
        vectors (e.g. MAS vs. baseline), unlike the per-system estimators
        above which take a raw (N, m) matrix.

    Parameters
    ----------
    per_problem_means_a : array-like, 1-D
        Per-problem means for system A, shape (N,) — e.g. the output of
        ``per_problem_means`` on a raw (N, m) metric matrix.
    per_problem_means_b : array-like, 1-D
        Per-problem means for system B.

    Returns
    -------
    dict
        Keys: t_statistic, p_value, df, mean_a, mean_b, mean_diff,
        significant_at_05.
    """
    a = np.asarray(per_problem_means_a, dtype=float)
    b = np.asarray(per_problem_means_b, dtype=float)

    t_stat, p_value = sp_stats.ttest_ind(a, b, equal_var=False)

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))

    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    na, nb = len(a), len(b)
    numerator = (var_a / na + var_b / nb) ** 2
    denominator = (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
    df = numerator / denominator if denominator != 0.0 else float("nan")

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "df": float(df),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_a - mean_b,
        "significant_at_05": bool(p_value < 0.05),
    }


def mann_whitney_u_test_metric(
    per_problem_means_a: ArrayLike,
    per_problem_means_b: ArrayLike,
    alternative: str = "two-sided",
) -> dict:
    """Rank-based, distribution-free comparison between two systems.

    .. note::
        **Deferred scope.** This is a *between-system* comparison test and
        is not part of the per-system estimation scope.  It is kept
        unchanged for backwards compatibility.

    Parameters
    ----------
    per_problem_means_a : array-like, 1-D
        Per-problem means for system A.
    per_problem_means_b : array-like, 1-D
        Per-problem means for system B.
    alternative : str, default ``"two-sided"``
        Passed directly to ``scipy.stats.mannwhitneyu``.

    Returns
    -------
    dict
        Keys: u_statistic, p_value, alternative, n_a, n_b, median_a,
        median_b, median_diff, rank_biserial, significant_at_05.

    References
    ----------
    Mann & Whitney (1947), Ann. Math. Statist. 18(1), 50–60.
    """
    a = np.asarray(per_problem_means_a, dtype=float)
    b = np.asarray(per_problem_means_b, dtype=float)

    res = sp_stats.mannwhitneyu(a, b, alternative=alternative)
    u_stat = float(res.statistic)
    p_value = float(res.pvalue)

    n_a, n_b = len(a), len(b)
    median_a = float(np.median(a))
    median_b = float(np.median(b))
    rank_biserial = 1.0 - 2.0 * u_stat / (n_a * n_b) if (n_a * n_b) > 0 else float("nan")

    return {
        "u_statistic": u_stat,
        "p_value": p_value,
        "alternative": alternative,
        "n_a": n_a,
        "n_b": n_b,
        "median_a": median_a,
        "median_b": median_b,
        "median_diff": median_a - median_b,
        "rank_biserial": float(rank_biserial),
        "significant_at_05": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def print_accuracy_report(
    label: str,
    result: dict,
    *,
    as_percentage: bool = True,
) -> None:
    """Print a formatted accuracy report from :func:`wilson_ci_clustered`."""
    scale = 100.0 if as_percentage else 1.0
    unit = "%" if as_percentage else ""
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  Accuracy (pooled)          {result['p_hat'] * scale:>7.2f}{unit}")
    print(f"  95% Wilson CI              [{result['ci_lower'] * scale:.2f}, {result['ci_upper'] * scale:.2f}]{unit}")
    print(f"  Cluster-robust SE          {result['se'] * scale:>7.4f}{unit}")
    print(f"  Design Effect (DEFF)       {result['deff']:>7.3f}")
    print(f"  ICC                        {result['icc']:>7.3f}")
    print(f"  Effective sample size      {result['n_eff']:>7.1f}  (of {result['n_total']} total)")
    print(f"  Problems x Seeds           {result['n_problems']} x {result['n_seeds']}")
    print(f"{'=' * 55}\n")


def print_accuracy_stats_report(
    label: str,
    result: dict,
    *,
    as_percentage: bool = True,
) -> None:
    """Print a formatted report from :func:`accuracy_statistics`."""
    scale = 100.0 if as_percentage else 1.0
    unit = "%" if as_percentage else ""
    ci_pct = int(round((1.0 - result.get("alpha", 0.05)) * 100))
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  Accuracy p̂                {result['p_hat'] * scale:>7.2f}{unit}")
    print(f"  {ci_pct}% Cluster-t CI         [{result['ci_lower'] * scale:.2f}, {result['ci_upper'] * scale:.2f}]{unit}")
    print(f"  Between-problem std        {result['between_problem_std'] * scale:>7.4f}{unit}")
    print(f"  ── Supplementary: Wilson ──")
    print(f"  95% Wilson CI              [{result['wilson_ci_lower'] * scale:.2f}, {result['wilson_ci_upper'] * scale:.2f}]{unit}")
    print(f"  DEFF                       {result['wilson_n_eff']:>7.3f}")
    print(f"  ICC                        {result['wilson_icc']:>7.3f}")
    print(f"  ── Sample ──")
    print(f"  Problems x Seeds           {result['n_problems']} x {result['n_seeds']}  (n={result['n_total']})")
    print(f"{'=' * 55}\n")


def print_pct_votes_top_report(
    label: str,
    result: dict,
    *,
    as_percentage: bool = True,
) -> None:
    """Print a formatted report from :func:`pct_votes_top_statistics`."""
    scale = 100.0 if as_percentage else 1.0
    unit = "%" if as_percentage else ""
    ci_pct = int(round((1.0 - result.get("alpha", 0.05)) * 100))
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  Mean winner vote share     {result['mean_pct_votes_top'] * scale:>7.2f}{unit}")
    print(f"  {ci_pct}% Cluster-t CI         [{result['ci_lower'] * scale:.2f}, {result['ci_upper'] * scale:.2f}]{unit}")
    print(f"  Between-problem std        {result['between_problem_std'] * scale:>7.4f}{unit}")
    print(f"  Problems x Seeds           {result['n_problems']} x {result['n_seeds']}  (n={result['n_total']})")
    print(f"{'=' * 55}\n")


def print_time_report(
    label: str,
    result: dict,
) -> None:
    """Print a formatted execution time report from :func:`time_statistics`."""
    ci_pct = int(round((1.0 - result["alpha"]) * 100))
    pct = result["percentiles"]
    iqr = pct["p75"] - pct["p25"]
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  Mean (s)                   {result['mean']:>9.4f}")
    print(f"  Std (s)                    {result['flat_std']:>9.4f}")
    print(f"  CV                         {result['flat_cv']:>9.4f}")
    print(f"  {ci_pct}% BCa CI (s)             [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]  [{result['ci_method']}]")
    print(f"  Between-problem std (s)    {result['between_problem_std']:>9.4f}")
    print(f"  ── Percentiles (descriptive) ──")
    print(f"  Median / P50 (s)           {pct['p50']:>9.4f}")
    print(f"  P25 (s)                    {pct['p25']:>9.4f}")
    print(f"  P75 (s)                    {pct['p75']:>9.4f}")
    print(f"  IQR (s)                    {iqr:>9.4f}")
    print(f"  P95 (s)                    {pct['p95']:>9.4f}")
    print(f"  ── Sample ──")
    print(f"  Problems x Seeds           {result['n_problems']} x {result['n_seeds']}  (n={result['n_total']})")
    print(f"{'=' * 55}\n")


def print_welch_report(
    label: str,
    result: dict,
) -> None:
    """Print a formatted Welch t-test report from :func:`welch_t_test_metric`."""
    sig = "YES" if result["significant_at_05"] else "NO"
    print(f"\n{'=' * 55}")
    print(f"  {label} — Welch t-test")
    print(f"{'=' * 55}")
    print(f"  Mean A (s)                 {result['mean_a']:>9.4f}")
    print(f"  Mean B (s)                 {result['mean_b']:>9.4f}")
    print(f"  Difference A - B (s)       {result['mean_diff']:>+9.4f}")
    print(f"  t-statistic                {result['t_statistic']:>9.4f}")
    print(f"  Degrees of freedom         {result['df']:>9.2f}")
    print(f"  p-value                    {result['p_value']:>9.4f}")
    print(f"  Significant at alpha=0.05      {sig:>9}")
    print(f"{'=' * 55}\n")


def print_metric_report(
    label: str,
    result: dict,
    unit: str = "",
) -> None:
    """Print a formatted report from :func:`_metric_statistics`."""
    ci_pct = int(round((1.0 - result["alpha"]) * 100))
    pct = result["percentiles"]
    p25 = pct["p25"]
    p75 = pct["p75"]
    iqr = p75 - p25
    u = f" {unit}" if unit else ""
    kind = result.get("kind", "continuous")
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  Mean{u:<20}       {result['mean']:>9.4f}")
    print(f"  Std{u:<21}       {result['flat_std']:>9.4f}")
    print(f"  CV                         {result['flat_cv']:>9.4f}")
    print(f"  {ci_pct}% BCa CI{u:<16}   [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]  [{result['ci_method']}]")
    print(f"  Between-problem std        {result['between_problem_std']:>9.4f}")
    print(f"  ── Percentiles ──")
    print(f"  Median / P50{u:<13}      {pct['p50']}")
    print(f"  P25{u:<21}      {pct['p25']}")
    print(f"  P75{u:<21}      {pct['p75']}")
    print(f"  IQR{u:<21}       {iqr:.4f}")
    print(f"  P95{u:<21}      {pct['p95']}")
    print(f"  ── Sample ──")
    print(f"  Problems x Seeds           {result['n_problems']} x {result['n_seeds']}  (n={result['n_total']})")
    if kind == "count":
        print(f"  ── Count-specific ──")
        print(f"  Dispersion index (Fano)    {result['dispersion_index']:>9.4f}")
    print(f"{'=' * 55}\n")


def print_token_report(
    label: str,
    result: dict,
    unit: str = "tok",
) -> None:
    """Print a formatted token-count report from :func:`token_statistics`."""
    print_metric_report(label, result, unit=unit)


def print_throughput_report(
    label: str,
    result: dict,
) -> None:
    """Print a formatted throughput (tok/s) report.

    Expects ``result`` with keys: mean, std, cv, ci_lower, ci_upper, ci_method,
    alpha, aggregate, aggregate_ci, percentiles, n_problems, n_seeds, n_total,
    n_bootstrap — the mean-of-ratios canonical estimate plus the ratio-of-means
    aggregate as a supplementary cross-check (see module docstring note on the
    forthcoming ``throughput_statistics`` estimator).
    """
    ci_pct = int(round((1.0 - result["alpha"]) * 100))
    agg_lo, agg_hi = result["aggregate_ci"]
    print(f"\n{'=' * 55}")
    print(f"  {label} — Throughput (tok/s)")
    print(f"{'=' * 55}")
    print(f"  ── Canonical: mean-of-ratios [batch=1, per-run] ──")
    print(f"  Mean tok/s                 {result['mean']:>9.4f}")
    print(f"  {ci_pct}% BCa CI (tok/s)          [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]  [{result['ci_method']}]")
    print(f"  Std tok/s                  {result['std']:>9.4f}")
    print(f"  CV                         {result['cv']:>9.4f}")
    print(f"  ── Supplementary: ratio-of-means [system aggregate] ──")
    print(f"  Aggregate tok/s            {result['aggregate']:>9.4f}")
    print(f"  Aggregate {ci_pct}% CI (tok/s)   [{agg_lo:.4f}, {agg_hi:.4f}]")
    print(f"  ── Percentiles ──")
    pct = result["percentiles"]
    print(f"  Median / P50 (tok/s)       {pct['p50']:>9.4f}")
    print(f"  P25 (tok/s)                {pct['p25']:>9.4f}")
    print(f"  P75 (tok/s)                {pct['p75']:>9.4f}")
    print(f"  P95 (tok/s)                {pct['p95']:>9.4f}")
    print(f"  ── Sample ──")
    print(f"  Problems x Seeds           {result['n_problems']} x {result['n_seeds']}  (n={result['n_total']})")
    print(f"  Bootstrap replications     {result['n_bootstrap']}")
    print(f"{'=' * 55}\n")


def print_overhead_ratio_report(
    label: str,
    result: dict,
) -> None:
    """Print a formatted overhead ratio report from :func:`overhead_ratio_statistics`."""
    ci_pct = int(round((1.0 - result["alpha"]) * 100))
    print(f"\n{'=' * 55}")
    print(f"  {label} — Overhead ratio (MAS / baseline)")
    print(f"{'=' * 55}")
    print(f"  Geometric mean ratio       {result['geometric_mean']:>9.4f}")
    print(f"  Arithmetic mean ratio      {result['arithmetic_mean']:>9.4f}")
    print(f"  {ci_pct}% log-t CI               [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]  [{result['ci_method']}]")
    print(f"  Between-problem std        {result['between_problem_std']:>9.4f}")
    print(f"  N problems                 {result['n_problems']:>9}")
    if "bca_ci" in result:
        lo, hi = result["bca_ci"]
        print(f"  BCa CI (cross-check)       [{lo:.4f}, {hi:.4f}]")
    print(f"{'=' * 55}\n")


def print_overhead_difference_report(
    label: str,
    result: dict,
    unit: str = "MB",
) -> None:
    """Print a formatted overhead difference report from :func:`overhead_difference_statistics`."""
    u = f" {unit}" if unit else ""
    print(f"\n{'=' * 55}")
    print(f"  {label} — Overhead difference (MAS − baseline)")
    print(f"{'=' * 55}")
    print(f"  Mean difference{u:<10}    {result['mean_diff']:>9.4f}")
    print(f"  95% Cluster-t CI{u:<9}   [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]  [{result['ci_method']}]")
    print(f"  Between-problem std        {result['between_problem_std']:>9.4f}")
    print(f"  N problems                 {result['n_problems']:>9}")
    print(f"{'=' * 55}\n")


def print_vram_report(
    label: str,
    result: dict,
) -> None:
    """Print a formatted VRAM report from :func:`vram_statistics`."""
    ci_pct = int(round((1.0 - result["alpha"]) * 100))
    pct = result["peak_percentiles"]
    print(f"\n{'=' * 55}")
    print(f"  {label} — Peak VRAM (MB)")
    print(f"{'=' * 55}")
    print(f"  Mean peak (MB)             {result['mean_peak']:>9.2f}")
    print(f"  {ci_pct}% Cluster-t CI (MB)      [{result['peak_ci_lower']:.2f}, {result['peak_ci_upper']:.2f}]  [{result['ci_method']}]")
    print(f"  Global max (MB)            {result['peak_max']:>9.2f}")
    print(f"  Between-problem std (MB)   {result['between_problem_peak_std']:>9.2f}")
    print(f"  ── Percentiles ──")
    print(f"  Median (MB)                {pct['p50']:>9.2f}")
    print(f"  P25 (MB)                   {pct['p25']:>9.2f}")
    print(f"  P75 (MB)                   {pct['p75']:>9.2f}")
    print(f"  P95 (MB)                   {pct['p95']:>9.2f}")
    print(f"  ── Sample ──")
    print(f"  Problems x Seeds           {result['n_problems']} x {result['n_seeds']}  (n={result['n_total']})")
    if "mean_avg" in result:
        print(f"  Mean avg VRAM (MB)         {result['mean_avg']:>9.2f}")
    print(f"{'=' * 55}\n")


def print_mannwhitney_report(
    label: str,
    result: dict,
) -> None:
    """Print a formatted Mann-Whitney U report from :func:`mann_whitney_u_test_metric`."""
    sig = "YES" if result["significant_at_05"] else "NO"
    print(f"\n{'=' * 55}")
    print(f"  {label} — Mann-Whitney U test")
    print(f"{'=' * 55}")
    print(f"  Median A                   {result['median_a']:>9.4f}")
    print(f"  Median B                   {result['median_b']:>9.4f}")
    print(f"  Median diff (A - B)        {result['median_diff']:>+9.4f}")
    print(f"  U statistic                {result['u_statistic']:>9.1f}")
    print(f"  p-value                    {result['p_value']:>9.4f}")
    print(f"  Alternative                {result['alternative']:>9}")
    print(f"  Rank-biserial r            {result['rank_biserial']:>9.4f}")
    print(f"  n_a / n_b                  {result['n_a']:>4} / {result['n_b']:<4}")
    print(f"  Significant at alpha=0.05      {sig:>9}")
    print(f"{'=' * 55}\n")
