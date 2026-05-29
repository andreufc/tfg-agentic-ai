"""
statistics.py — Statistically consistent, unbiased estimators for
clustered experimental data (repeated seeds per problem).

Provides:
    - pooled_accuracy:         Unbiased point estimate of accuracy
    - wilson_ci_clustered:     Wilson CI adjusted for within-problem
                               correlation via the Design Effect (DEFF)
    - mean_time_clustered:     Cluster-robust mean and t-based CI for
                               execution times
    - fit_time_distribution:   MLE fitting of Gamma and Log-Normal
                               distributions with AIC comparison

All functions expect data as NumPy arrays of shape ``(N, m)`` where
``N`` = number of problems and ``m`` = number of seeds (repetitions).

References
----------
- Wilson (1927), JASA 22(158), 209–212.
- Rao & Scott (1992), Biometrics 48(2), 577–585.
- Korn & Graubard (1998), Survey Methodology 24, 193–201.
"""

from __future__ import annotations

from typing import Any

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

        - ``p_hat``    – pooled accuracy (unbiased point estimate)
        - ``ci_lower`` – lower bound of Wilson CI
        - ``ci_upper`` – upper bound of Wilson CI
        - ``deff``     – estimated design effect
        - ``n_eff``    – effective sample size
        - ``icc``      – intra-cluster correlation coefficient
        - ``se``       – cluster-robust standard error of ``p_hat``
        - ``n_total``  – total number of observations (N × m)
        - ``n_problems`` – number of problems (N)
        - ``n_seeds``  – number of seeds (m)
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

    # Design effect  DEFF = m · σ²_c / [ p̂(1 − p̂) ]
    deff = n_seeds * sigma2_c / (p_hat * (1.0 - p_hat))
    deff = max(deff, 1.0)                               # floor at 1

    # Effective sample size
    n_eff = n_total / deff

    # ICC  ρ = (DEFF − 1) / (m − 1)
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
# Execution-time estimation
# ───────────────────────────────────────────────────────────────────────

def mean_time_clustered(
    times: ArrayLike,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Cluster-robust mean and *t*-based CI for execution times.

    Computes the unbiased pooled mean, standard error accounting for
    within-problem correlation, and a confidence interval using the
    *t*-distribution with ``N − 1`` degrees of freedom.

    Parameters
    ----------
    times : array-like of shape ``(N, m)``
        Execution time matrix (positive reals, in seconds).
    alpha : float, default 0.05
        Significance level (0.05 → 95 % CI).

    Returns
    -------
    dict
        Keys:

        - ``mean``       – pooled mean time (unbiased)
        - ``se``         – cluster-robust standard error
        - ``ci_lower``   – lower bound of *t*-based CI
        - ``ci_upper``   – upper bound of *t*-based CI
        - ``var_cluster``– sample variance of per-problem means
        - ``var_total``  – total (marginal) sample variance
        - ``n_problems`` – number of problems
        - ``n_seeds``    – number of seeds
        - ``df``         – degrees of freedom (N − 1)
    """
    arr = np.asarray(times, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D array of shape (N, m), got shape {arr.shape}"
        )
    n_problems, n_seeds = arr.shape

    # Per-problem means  T̄_i = (1/m) Σ_j T_{i,j}
    t_bar_i = np.mean(arr, axis=1)

    mu_hat = float(np.mean(t_bar_i))

    # Cluster-robust variance  (unbiased)
    var_cluster = float(np.var(t_bar_i, ddof=1))

    # Standard error of μ̂
    se = float(np.sqrt(var_cluster / n_problems))

    # Total marginal variance (unbiased)
    var_total = float(np.var(arr, ddof=1))

    # t-based CI
    df = n_problems - 1
    t_crit = sp_stats.t.ppf(1.0 - alpha / 2.0, df=df)
    ci_lower = mu_hat - t_crit * se
    ci_upper = mu_hat + t_crit * se

    return {
        "mean": mu_hat,
        "se": se,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "var_cluster": var_cluster,
        "var_total": var_total,
        "n_problems": n_problems,
        "n_seeds": n_seeds,
        "df": df,
    }


# ───────────────────────────────────────────────────────────────────────
# Distribution fitting
# ───────────────────────────────────────────────────────────────────────

def fit_time_distribution(
    times_flat: ArrayLike,
) -> dict[str, Any]:
    """Fit Gamma and Log-Normal distributions via MLE and compare via AIC.

    Parameters
    ----------
    times_flat : array-like, 1-D
        Pooled execution times (all problems × all seeds).  All values
        must be strictly positive.

    Returns
    -------
    dict
        Keys:

        - ``preferred``  – ``'gamma'`` or ``'lognorm'`` (lower AIC)
        - ``gamma``      – sub-dict with ``'params'``, ``'aic'``,
          ``'log_likelihood'``, ``'mean'``, ``'var'``, ``'ks_statistic'``,
          ``'ks_pvalue'``
        - ``'lognorm'``  – same structure as ``gamma``

    Notes
    -----
    The KS p-value can be used for an absolute goodness-of-fit check:
    if *both* p-values are small the data may follow neither distribution
    well.
    """
    arr = np.asarray(times_flat, dtype=float).ravel()
    if np.any(arr <= 0):
        raise ValueError("All execution times must be strictly positive.")
    n = len(arr)

    results: dict[str, Any] = {}

    # ── Gamma ──
    gamma_a, gamma_loc, gamma_scale = sp_stats.gamma.fit(arr, floc=0)
    gamma_ll = float(np.sum(sp_stats.gamma.logpdf(arr, gamma_a, loc=0, scale=gamma_scale)))
    gamma_aic = -2.0 * gamma_ll + 2.0 * 2  # 2 free params (shape, scale)
    gamma_ks, gamma_ksp = sp_stats.kstest(
        arr, "gamma", args=(gamma_a, 0, gamma_scale)
    )
    gamma_mean = gamma_a * gamma_scale
    gamma_var = gamma_a * gamma_scale ** 2

    results["gamma"] = {
        "params": {"shape": float(gamma_a), "scale": float(gamma_scale)},
        "aic": float(gamma_aic),
        "log_likelihood": gamma_ll,
        "mean": float(gamma_mean),
        "var": float(gamma_var),
        "ks_statistic": float(gamma_ks),
        "ks_pvalue": float(gamma_ksp),
    }

    # ── Log-Normal ──
    ln_s, ln_loc, ln_scale = sp_stats.lognorm.fit(arr, floc=0)
    ln_mu = float(np.log(ln_scale))   # scipy parameterisation: scale = exp(μ)
    ln_sigma = float(ln_s)
    ln_ll = float(np.sum(sp_stats.lognorm.logpdf(arr, ln_s, loc=0, scale=ln_scale)))
    ln_aic = -2.0 * ln_ll + 2.0 * 2
    ln_ks, ln_ksp = sp_stats.kstest(
        arr, "lognorm", args=(ln_s, 0, ln_scale)
    )
    ln_mean = float(np.exp(ln_mu + ln_sigma ** 2 / 2.0))
    ln_var = float((np.exp(ln_sigma ** 2) - 1.0) * np.exp(2.0 * ln_mu + ln_sigma ** 2))

    results["lognorm"] = {
        "params": {"mu": ln_mu, "sigma": ln_sigma},
        "aic": float(ln_aic),
        "log_likelihood": ln_ll,
        "mean": ln_mean,
        "var": ln_var,
        "ks_statistic": float(ln_ks),
        "ks_pvalue": float(ln_ksp),
    }

    # ── Comparison ──
    results["preferred"] = (
        "gamma" if results["gamma"]["aic"] < results["lognorm"]["aic"] else "lognorm"
    )
    results["aic_difference"] = abs(
        results["gamma"]["aic"] - results["lognorm"]["aic"]
    )

    return results


# ───────────────────────────────────────────────────────────────────────
# Pretty-printing helper
# ───────────────────────────────────────────────────────────────────────

def print_accuracy_report(
    label: str,
    result: dict[str, float],
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
    result: dict[str, float],
) -> None:
    """Print a formatted execution time report from :func:`mean_time_clustered`."""
    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  Mean time (s)              {result['mean']:>9.4f}")
    print(f"  Cluster-robust SE (s)      {result['se']:>9.4f}")
    print(f"  95% CI (s)                 [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"  Cluster variance           {result['var_cluster']:>9.4f}")
    print(f"  Total variance             {result['var_total']:>9.4f}")
    print(f"  d.f.                       {result['df']}")
    print(f"{'=' * 55}\n")


def print_distribution_report(
    label: str,
    result: dict[str, Any],
) -> None:
    """Print a formatted distribution comparison from :func:`fit_time_distribution`."""
    print(f"\n{'=' * 55}")
    print(f"  {label} — Distribution Fit")
    print(f"{'=' * 55}")
    for dist_name in ("gamma", "lognorm"):
        d = result[dist_name]
        pretty = "Gamma" if dist_name == "gamma" else "Log-Normal"
        winner = " <-- preferred" if result["preferred"] == dist_name else ""
        print(f"\n  {pretty}{winner}")
        print(f"    AIC               {d['aic']:>12.2f}")
        print(f"    Log-likelihood    {d['log_likelihood']:>12.2f}")
        print(f"    Fitted mean       {d['mean']:>12.4f} s")
        print(f"    Fitted variance   {d['var']:>12.4f} s²")
        print(f"    KS statistic      {d['ks_statistic']:>12.4f}")
        print(f"    KS p-value        {d['ks_pvalue']:>12.4f}")
        params = d["params"]
        print(f"    Parameters        {params}")
    print(f"\n  Delta-AIC = {result['aic_difference']:.2f}")
    print(f"{'=' * 55}\n")
