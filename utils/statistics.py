"""
statistics.py — Statistically consistent, unbiased estimators for
clustered experimental data (repeated seeds per problem).

Provides:
    - pooled_accuracy:              Unbiased point estimate of accuracy
    - wilson_ci_clustered:          Wilson CI adjusted for within-problem
                                    correlation via the Design Effect (DEFF)
    - time_statistics:              Distribution-free execution time summary
                                    with BCa bootstrap CI (no parametric
                                    assumptions)
    - welch_t_test_times:           Welch t-test comparing per-problem mean
                                    times between two systems
    - metric_statistics:            Nonparametric summary for any (N, m)
                                    resource metric (tokens, GPU util, etc.)
    - throughput_statistics:        Tailored estimator for tokens/s with
                                    mean-of-ratios as canonical value
    - welch_t_test_metric:          Alias of welch_t_test_times; primary
                                    comparison test for any metric
    - mann_whitney_u_test_metric:   Rank-based, distribution-free comparison
                                    test for any metric

All functions expect data as NumPy arrays of shape ``(N, m)`` where
``N`` = number of problems and ``m`` = number of seeds (repetitions).

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


# ───────────────────────────────────────────────────────────────────────
# Accuracy estimation
# ───────────────────────────────────────────────────────────────────────

def pooled_accuracy(correctness: ArrayLike) -> float:
    """Unbiased point estimate of accuracy from clustered binary data.

    Parameters
    ----------
    correctness : array-like of shape ``(N, m)``
        Binary correctness matrix.  ``correctness[i, j] = 1`` if
        problem *i* was answered correctly with seed *j*, else 0.

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
    correctness : array-like of shape ``(N, m)``
        Binary correctness matrix (values 0 or 1).
    alpha : float, default 0.05
        Significance level (0.05 → 95 % CI).

    Returns
    -------
    dict
        Keys:

        - p_hat    - pooled accuracy (unbiased point estimate)
        - ci_lower - lower bound of Wilson CI
        - ci_upper - upper bound of Wilson CI
        - deff     - estimated design effect
        - n_eff    - effective sample size
        - icc      - intra-cluster correlation coefficient
        - se       - cluster-robust standard error of p_hat
        - n_total  - total number of observations (N × m)
        - n_problems - number of problems (N)
        - n_seeds  - number of seeds (m)
    """
    arr = np.asarray(correctness, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D array of shape (N, m), got shape {arr.shape}"
        )
    n_problems, n_seeds = arr.shape
    n_total = n_problems * n_seeds

    p_hat = float(np.mean(arr))

    # Per-problem averages  X̄_i = (1/m) Σ_j X_{i,j}
    x_bar_i = np.mean(arr, axis=1)                     # shape (N,)

    # Cluster-robust variance  (unbiased, Bessel-corrected)
    sigma2_c = float(np.var(x_bar_i, ddof=1))

    # ── Degenerate cases ──
    if p_hat == 0.0 or p_hat == 1.0:
        return {
            "p_hat": p_hat,
            "ci_lower": p_hat,
            "ci_upper": p_hat,
            "deff": float(n_seeds),
            "n_eff": float(n_problems),
            "icc": 1.0,
            "se": 0.0,
            "n_total": n_total,
            "n_problems": n_problems,
            "n_seeds": n_seeds,
        }

    # Design effect  DEFF = m · σ²_c / [ p̂(1 - p̂) ]
    deff = n_seeds * sigma2_c / (p_hat * (1.0 - p_hat))
    deff = max(deff, 1.0)                               # floor at 1-
    # Effective sample size
    n_eff = n_total / deff

    # ICC  ρ = (DEFF - 1) / (m - 1)
    icc = (deff - 1.0) / (n_seeds - 1.0) if n_seeds > 1 else 0.0

    # Cluster-robust SE
    se = float(np.sqrt(sigma2_c / n_problems))

    # Wilson interval with effective n
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
        "n_total": n_total,
        "n_problems": n_problems,
        "n_seeds": n_seeds,
    }


# ───────────────────────────────────────────────────────────────────────
# Execution-time estimation (nonparametric)
# ───────────────────────────────────────────────────────────────────────

def time_statistics(
    times: ArrayLike,
    alpha: float = 0.05,
    n_bootstrap: int = 9999,
) -> dict:
    """Distribution-free execution time summary with BCa bootstrap CI.

    Makes no parametric assumptions about the underlying distribution.
    The BCa bootstrap CI corrects for both bias and skewness, making it
    well-suited for the right-skewed, heavy-tailed nature of latency data.

    Parameters
    ----------
    times : array-like of shape ``(N, m)``
        Execution time matrix (positive reals, in seconds).
        N = number of problems, m = number of seeds.
    alpha : float, default 0.05
        Significance level (0.05 → 95 % CI).
    n_bootstrap : int, default 9999
        Number of bootstrap replications for the BCa CI.

    Returns
    -------
    dict
        Keys:

        - mean              - sample mean of all Nxm observations
        - variance          - sample variance (Bessel-corrected, ddof=1)
        - std               - sample standard deviation
        - cv                - coefficient of variation (std / mean)
        - median            - median; convenience alias for percentiles['p50']
        - percentiles       - dict with keys p25, p50, p75, p95
        - ci_lower          - BCa bootstrap CI lower bound for the mean
        - ci_upper          - BCa bootstrap CI upper bound for the mean
        - alpha             - significance level used
        - n_bootstrap       - number of bootstrap replications
        - n_problems        - N
        - n_seeds           - m
        - n_total           - N x m
        - per_problem_means - 1-D array of shape (N,), for downstream t-tests
    """
    arr = np.asarray(times, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D array of shape (N, m), got shape {arr.shape}"
        )
    n_problems, n_seeds = arr.shape
    n_total = n_problems * n_seeds
    arr_flat = arr.ravel()

    mean = float(np.mean(arr_flat))
    variance = float(np.var(arr_flat, ddof=1))
    std = float(np.std(arr_flat, ddof=1))
    cv = std / mean if mean != 0.0 else float("nan")

    p25, p50, p75, p95 = np.percentile(arr_flat, [25, 50, 75, 95])
    percentiles = {
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
    }
    median = percentiles["p50"]

    # BCa bootstrap CI for the mean — corrects for bias and skewness
    boot_result = sp_stats.bootstrap(
        (arr_flat,),
        statistic=np.mean,
        n_resamples=n_bootstrap,
        confidence_level=1.0 - alpha,
        method="BCa",
        random_state=0
        )
    ci_lower = float(boot_result.confidence_interval.low)
    ci_upper = float(boot_result.confidence_interval.high)

    per_problem_means = np.mean(arr, axis=1)

    return {
        "mean": mean,
        "variance": variance,
        "std": std,
        "cv": cv,
        "median": median,
        "percentiles": percentiles,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "n_problems": n_problems,
        "n_seeds": n_seeds,
        "n_total": n_total,
        "per_problem_means": per_problem_means,
    }


def welch_t_test_metric(
    per_problem_means_a: ArrayLike,
    per_problem_means_b: ArrayLike,
) -> dict:
    """Welch t-test comparing execution times between two systems.

    By the CLT, per-problem means (each averaged over m seeds) are
    approximately normal even when individual times are heavily skewed,
    provided m is not tiny.  Welch's variant does not assume equal
    variances — essential when comparing systems with different call
    patterns.

    Parameters
    ----------
    per_problem_means_a : array-like, 1-D
        Per-problem mean times for system A
        (``time_statistics(...)['per_problem_means']``).
    per_problem_means_b : array-like, 1-D
        Per-problem mean times for system B.

    Returns
    -------
    dict
        Keys:

        - t_statistic      - Welch t-statistic
        - p_value          - two-sided p-value
        - df               - Welch-Satterthwaite degrees of freedom
        - mean_a           - mean of system A
        - mean_b           - mean of system B
        - mean_diff        - difference in means (A - B)
        - significant_at_05 - bool: p < 0.05
    """
    a = np.asarray(per_problem_means_a, dtype=float)
    b = np.asarray(per_problem_means_b, dtype=float)

    t_stat, p_value = sp_stats.ttest_ind(a, b, equal_var=False)

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))

    # Welch–Satterthwaite degrees of freedom
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


# ───────────────────────────────────────────────────────────────────────
# Resource-metric estimation (nonparametric)
# ───────────────────────────────────────────────────────────────────────

def metric_statistics(
    values: ArrayLike,
    kind: str = "continuous",
    alpha: float = 0.05,
    n_bootstrap: int = 9999,
    bounds: tuple[float, float] = (0.0, 100.0),
) -> dict:
    """Nonparametric summary for any (N, m) resource metric.

    Parameters
    ----------
    values : array-like of shape ``(N, m)``
        Resource metric matrix (e.g. output tokens, GPU utilisation).
    kind : str, default ``"continuous"``
        One of ``"count"``, ``"bounded"``, ``"ratio"``, ``"continuous"``.
    alpha : float, default 0.05
        Significance level for the BCa bootstrap CI.
    n_bootstrap : int, default 9999
        Number of bootstrap replications.
    bounds : tuple[float, float], default ``(0.0, 100.0)``
        Used only when ``kind == "bounded"`` to clamp the CI and compute
        floor/ceiling fractions.

    Returns
    -------
    dict
        Common keys: mean, variance, std, cv, median, percentiles,
        ci_lower, ci_upper, alpha, n_bootstrap,
        n_problems, n_seeds, n_total, per_problem_means, kind.
        Additional kind-specific keys as documented in the spec.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D array of shape (N, m), got shape {arr.shape}"
        )
    n_problems, n_seeds = arr.shape
    n_total = n_problems * n_seeds
    flat = arr.ravel()

    mean = float(np.mean(flat))
    variance = float(np.var(flat, ddof=1))
    std = float(np.std(flat, ddof=1))
    cv = std / mean if mean != 0.0 else float("nan")

    p25, p50, p75, p95 = np.percentile(flat, [25, 50, 75, 95])
    percentiles = {
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
    }
    median = percentiles["p50"]

    per_problem_means = np.mean(arr, axis=1)

    # Zero-variance guard: sp_stats.bootstrap errors on constant input
    if float(flat.max()) == float(flat.min()):
        ci_lower = mean
        ci_upper = mean
        cv = 0.0

    else:
        boot_result = sp_stats.bootstrap(
            (flat,),
            statistic=np.mean,
            n_resamples=n_bootstrap,
            confidence_level=1.0 - alpha,
            method="BCa",
            random_state=0,
        )
        ci_lower = float(boot_result.confidence_interval.low)
        ci_upper = float(boot_result.confidence_interval.high)


    result = {
        "mean": mean,
        "variance": variance,
        "std": std,
        "cv": cv,
        "median": median,
        "percentiles": percentiles,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "n_problems": n_problems,
        "n_seeds": n_seeds,
        "n_total": n_total,
        "per_problem_means": per_problem_means,
        "kind": kind,
    }

    if kind == "count":
        di = variance / mean if mean != 0.0 else float("nan")
        result["dispersion_index"] = di
        result["overdispersed"] = bool(di > 1.0) if not np.isnan(di) else False
        result["percentiles"] = {k: int(v) for k, v in percentiles.items()}
        result["median"] = int(p50)

    elif kind == "bounded":
        lo, hi = bounds
        result["ci_lower"] = float(np.clip(ci_lower, lo, hi))
        result["ci_upper"] = float(np.clip(ci_upper, lo, hi))
        result["frac_at_lower"] = float(np.mean(flat == lo))
        result["frac_at_upper"] = float(np.mean(flat == hi))
        result["cv_reliable"] = bool(min(mean - lo, hi - mean) > std)

    elif kind == "ratio":
        result["mean_kind"] = "mean_of_ratios"

    return result


def throughput_statistics(
    output_tokens: ArrayLike,
    elapsed_s: ArrayLike,
    alpha: float = 0.05,
    n_bootstrap: int = 9999,
) -> dict:
    """Tailored estimator for tokens/second throughput.

    The canonical ``mean`` is the mean-of-ratios (average of per-run tps).
    The ratio-of-means aggregate is computed as a supplementary value only.

    Parameters
    ----------
    output_tokens : array-like of shape ``(N, m)``
        Output token counts per run.
    elapsed_s : array-like of shape ``(N, m)``
        Wall-clock elapsed time per run, in seconds.
    alpha : float, default 0.05
        Significance level.
    n_bootstrap : int, default 9999
        Bootstrap replications.

    Returns
    -------
    dict
        Keys: mean (canonical, mean-of-ratios), ci_lower, ci_upper, std, cv,
        median, percentiles, aggregate (supplementary), aggregate_ci, skewness,
        excess_kurtosis, per_problem_means, alpha, n_bootstrap, n_problems,
        n_seeds, n_total.
    """
    tok = np.asarray(output_tokens, dtype=float)
    ela = np.asarray(elapsed_s, dtype=float)
    if tok.ndim != 2 or ela.ndim != 2:
        raise ValueError(
            "Both output_tokens and elapsed_s must be 2-D arrays of shape (N, m)."
        )
    if tok.shape != ela.shape:
        raise ValueError(
            f"Shape mismatch: output_tokens {tok.shape} vs elapsed_s {ela.shape}."
        )

    tps = tok / ela

    base = metric_statistics(tps, kind="ratio", alpha=alpha, n_bootstrap=n_bootstrap)

    aggregate = float(tok.sum() / ela.sum())

    # Paired bootstrap for aggregate CI: resample run indices jointly
    tok_flat = tok.ravel()
    ela_flat = ela.ravel()
    n_runs = len(tok_flat)
    rng = np.random.default_rng(0)
    agg_resamples = np.empty(n_bootstrap)
    for k in range(n_bootstrap):
        idx = rng.integers(0, n_runs, size=n_runs)
        agg_resamples[k] = tok_flat[idx].sum() / ela_flat[idx].sum()
    agg_ci = (
        float(np.percentile(agg_resamples, 100.0 * alpha / 2.0)),
        float(np.percentile(agg_resamples, 100.0 * (1.0 - alpha / 2.0))),
    )

    base["aggregate"] = aggregate
    base["aggregate_ci"] = agg_ci
    return base


def mann_whitney_u_test_metric(
    per_problem_means_a: ArrayLike,
    per_problem_means_b: ArrayLike,
    alternative: str = "two-sided",
) -> dict:
    """Rank-based, distribution-free comparison between two systems.

    Operates on per-problem means (the ``per_problem_means`` field from
    ``metric_statistics`` or ``time_statistics``), mirroring the interface
    of ``welch_t_test_metric``.

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


# ───────────────────────────────────────────────────────────────────────
# Pretty-printing helpers
# ───────────────────────────────────────────────────────────────────────

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
    print(f"  Problems × Seeds           {result['n_problems']} × {result['n_seeds']}")
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
    print(f"  Std (s)                    {result['std']:>9.4f}")
    print(f"  Variance (s²)              {result['variance']:>9.4f}")
    print(f"  CV                         {result['cv']:>9.4f}")
    print(f"  {ci_pct}% BCa CI (s)             [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"  ── Percentiles ──")
    print(f"  Median / P50 (s)           {pct['p50']:>9.4f}")
    print(f"  P25 (s)                    {pct['p25']:>9.4f}")
    print(f"  P75 (s)                    {pct['p75']:>9.4f}")
    print(f"  IQR (s)                    {iqr:>9.4f}")
    print(f"  P95 (s)                    {pct['p95']:>9.4f}")
    print(f"  ── Sample ──")
    print(f"  Problems × Seeds           {result['n_problems']} × {result['n_seeds']}  (n={result['n_total']})")
    print(f"  Bootstrap replications     {result['n_bootstrap']}")
    print(f"{'=' * 55}\n")


def print_welch_report(
    label: str,
    result: dict,
) -> None:
    """Print a formatted Welch t-test report from :func:`welch_t_test_times`."""
    sig = "YES" if result["significant_at_05"] else "NO"
    print(f"\n{'=' * 55}")
    print(f"  {label} — Welch t-test")
    print(f"{'=' - 55}")
    print(f"  Mean A (s)                 {result['mean_a']:>9.4f}")
    print(f"  Mean B (s)                 {result['mean_b']:>9.4f}")
    print(f"  Difference A - B (s)       {result['mean_diff']:>+9.4f}")
    print(f"  t-statistic                {result['t_statistic']:>9.4f}")
    print(f"  Degrees of freedom         {result['df']:>9.2f}")
    print(f"  p-value                    {result['p_value']:>9.4f}")
    print(f"  Significant at α=0.05      {sig:>9}")
    print(f"{'=' * 55}\n")


def print_metric_report(
    label: str,
    result: dict,
    unit: str = "",
) -> None:
    """Print a formatted report from :func:`metric_statistics`."""
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
    print(f"  Std{u:<21}       {result['std']:>9.4f}")
    print(f"  Variance                   {result['variance']:>9.4f}")
    print(f"  CV                         {result['cv']:>9.4f}")
    print(f"  {ci_pct}% BCa CI{u:<16}   [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"  ── Percentiles ──")
    print(f"  Median / P50{u:<13}      {pct['p50']}")
    print(f"  P25{u:<21}      {pct['p25']}")
    print(f"  P75{u:<21}      {pct['p75']}")
    print(f"  IQR{u:<21}       {iqr:.4f}")
    print(f"  P95{u:<21}      {pct['p95']}")
    print(f"  ── Sample ──")
    print(f"  Problems × Seeds           {result['n_problems']} × {result['n_seeds']}  (n={result['n_total']})")
    print(f"  Bootstrap replications     {result['n_bootstrap']}")
    if kind == "count":
        print(f"  ── Count-specific ──")
        print(f"  Dispersion index (Fano)    {result['dispersion_index']:>9.4f}")
        print(f"  Overdispersed              {'YES' if result['overdispersed'] else 'NO':>9}")
    elif kind == "bounded":
        print(f"  ── Bounded-specific ──")
        print(f"  Frac at lower bound        {result['frac_at_lower']:>9.4f}")
        print(f"  Frac at upper bound        {result['frac_at_upper']:>9.4f}")
        print(f"  CV reliable                {'YES' if result['cv_reliable'] else 'NO':>9}")
    elif kind == "ratio":
        print(f"  Note: mean is mean-of-ratios (per-run {unit or 'ratio'} averaged)")
    print(f"{'=' * 55}\n")


def print_throughput_report(
    label: str,
    result: dict,
) -> None:
    """Print a formatted throughput report from :func:`throughput_statistics`."""
    ci_pct = int(round((1.0 - result["alpha"]) * 100))
    agg_lo, agg_hi = result["aggregate_ci"]
    print(f"\n{'=' * 55}")
    print(f"  {label} — Throughput (tok/s)")
    print(f"{'=' * 55}")
    print(f"  ── Canonical: mean-of-ratios ──")
    print(f"  Mean tok/s                 {result['mean']:>9.4f}")
    print(f"  {ci_pct}% BCa CI (tok/s)          [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"  Std tok/s                  {result['std']:>9.4f}")
    print(f"  CV                         {result['cv']:>9.4f}")
    print(f"  ── Supplementary: ratio-of-means ──")
    print(f"  Aggregate tok/s            {result['aggregate']:>9.4f}")
    print(f"  Aggregate {ci_pct}% CI (tok/s)   [{agg_lo:.4f}, {agg_hi:.4f}]")
    print(f"  ── Percentiles ──")
    pct = result["percentiles"]
    print(f"  Median / P50 (tok/s)       {pct['p50']:>9.4f}")
    print(f"  P25 (tok/s)                {pct['p25']:>9.4f}")
    print(f"  P75 (tok/s)                {pct['p75']:>9.4f}")
    print(f"  P95 (tok/s)                {pct['p95']:>9.4f}")
    print(f"  ── Sample ──")
    print(f"  Problems × Seeds           {result['n_problems']} × {result['n_seeds']}  (n={result['n_total']})")
    print(f"  Bootstrap replications     {result['n_bootstrap']}")
    print(f"{'=' * 55}\n")


def print_mannwhitney_report(
    label: str,
    result: dict,
) -> None:
    """Print a formatted Mann–Whitney U report from :func:`mann_whitney_u_test_metric`."""
    sig = "YES" if result["significant_at_05"] else "NO"
    print(f"\n{'=' * 55}")
    print(f"  {label} — Mann–Whitney U test")
    print(f"{'=' * -5}")
    print(f"  Median A                   {result['median_a']:>9.4f}")
    print(f"  Median B                   {result['median_b']:>9.4f}")
    print(f"  Median diff (A - B)        {result['median_diff']:>+9.4f}")
    print(f"  U statistic                {result['u_statistic']:>9.1f}")
    print(f"  p-value                    {result['p_value']:>9.4f}")
    print(f"  Alternative                {result['alternative']:>9}")
    print(f"  Rank-biserial r            {result['rank_biserial']:>9.4f}")
    print(f"  n_a / n_b                  {result['n_a']:>4} / {result['n_b']:<4}")
    print(f"  Significant at α=0.05      {sig:>9}")
    print(f"{'=' * 55}\n")
