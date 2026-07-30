"""
voting.py — Plurality voting and the Self-Consistency (SC) ladder.

Pure aggregation logic shared by Self-Consistency, Majority Voting and
BoN-MAV, unified across benchmarks via :class:`BenchmarkSpec`: MATH-500's
LaTeX-equivalence grouping and MMLU's exact-letter grouping are two
configurations of the *same* grouping/voting path (:func:`group_generations`
+ :func:`majority_vote`), not two separate implementations. Answer parsing
and comparison stay in ``utils.evaluation`` — this module only groups and
votes.

Self-Consistency's "vote once, re-derive every k" property — grouping a
problem's generations costs the same whether you read off SC@1 or the whole
SC@1/3/4/5 ladder — now rests on ``utils.evaluation``'s memoized
``math_equiv``/``mmlu_match`` rather than on a precomputed grouping passed
between callers: every subset groups its own members independently (see
:func:`self_consistency`), and the cache absorbs the repeated pairwise
comparisons.

Connection to the project WPs
-----------------------------
* WP2-T4 (L1 multi-agent protocols): the voting rule shared by Self-
  Consistency, heterogeneous Majority Voting, and BoN-MAV's tie-break.
* WP3-T1: tie rate and vote entropy are logged as reliability diagnostics.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, TypedDict

from utils.data import extract_gsm8k_answer, format_mmlu_prompt
from utils.evaluation import (
    extract_gsm8k_prediction,
    extract_math_prediction,
    get_ground_truth_letter,
    gsm8k_match,
    math_equiv,
    math_match,
    mmlu_match,
    normalize_mmlu_answer,
)
from utils.instrumentation import tokens_per_second


class VoteResult(TypedDict):
    winner_idx: Optional[int]
    winner_pred: Optional[str]
    counts: list[int]
    pct_votes_top: float
    tie: bool
    tie_break_used: bool
    n_abstain: int
    vote_entropy: float


class SubsetResult(TypedDict):
    subset_id: str
    indices: tuple[int, ...]
    winner_gen_id: Optional[int]
    winner_output: Optional[str]
    correct: bool
    tie: bool
    tie_break_used: bool
    n_abstain: int
    pct_votes_top: float


class ProblemSCResult(TypedDict):
    predictions: list[Optional[str]]
    correct_by_gen: list[bool]
    by_k: dict[int, list[SubsetResult]]


# ---------------------------------------------------------------------------
# BenchmarkSpec — the one place a benchmark's answer handling is defined
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkSpec:
    """Binds a benchmark's answer extraction, equivalence, and scoring rules.

    ``compare_on`` is the only axis of variation between benchmarks in the
    grouping/voting path: MATH-500 groups on the *full* candidate text
    (``math_verify`` is a more robust extractor than any pre-cut substring —
    see ``evaluation.extract_math_prediction``'s docstring), MMLU and GSM8K
    group on the already-extracted prediction. Keeping "what you group on"
    (``compare_on`` / ``equiv``) separate from "what you score against the
    ground truth" (``score`` / ``get_truth``) means they can never be
    silently coupled, even where they happen to use the same function.

    ``render_task(item)`` returns exactly the string substituted into the
    notebook prompt template's task placeholder — all problem content
    (including, for MMLU, the rendered answer choices), no instruction
    wording. The generation loop stores this same string as the record's
    ``task``; ``validate_alignment`` re-runs the identical callable against
    the locally loaded dataset and compares the result directly against
    that stored ``task``. Using one shared function for "what got shown to
    the model", "what got persisted" and "what gets re-checked for
    validation" makes the three provably agree — there is no second,
    divergent place to describe a benchmark's prompt.
    """

    name: str
    extract: Callable[[str], Optional[str]]
    equiv: Callable[[str, str], bool]
    score: Callable[[str, Any], bool]
    compare_on: Literal["answer", "prediction"]
    get_truth: Callable[[Any], Any]
    render_task: Callable[[Any], str]


MATH500_SPEC = BenchmarkSpec(
    name="math500",
    extract=extract_math_prediction,
    equiv=math_equiv,
    score=math_match,
    compare_on="answer",
    get_truth=lambda item: item["answer"],
    render_task=lambda item: item["problem"],
)

MMLU_SPEC = BenchmarkSpec(
    name="mmlu",
    extract=normalize_mmlu_answer,
    equiv=mmlu_match,
    score=mmlu_match,
    compare_on="prediction",
    get_truth=lambda item: get_ground_truth_letter(item["answer"]),
    render_task=lambda item: format_mmlu_prompt(item["question"], item["choices"]),
)

GSM8K_SPEC = BenchmarkSpec(
    name="gsm8k",
    extract=extract_gsm8k_prediction,
    equiv=gsm8k_match,
    score=gsm8k_match,
    compare_on="prediction",
    get_truth=lambda item: extract_gsm8k_answer(item["answer"]),
    render_task=lambda item: item["question"],
)

BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {
    "math500": MATH500_SPEC,
    "mmlu": MMLU_SPEC,
    "gsm8k": GSM8K_SPEC,
}


def get_benchmark_spec(name: str) -> BenchmarkSpec:
    """Look up a :class:`BenchmarkSpec` by dataset name (e.g. ``DATASET``)."""
    try:
        return BENCHMARK_SPECS[name]
    except KeyError:
        raise ValueError(
            f"Unknown benchmark {name!r}; expected one of {sorted(BENCHMARK_SPECS)}"
        ) from None


# ---------------------------------------------------------------------------
# Grouping — one implementation for every benchmark
# ---------------------------------------------------------------------------

def group_equivalent(
    items: Sequence[Optional[str]],
    equiv: Callable[[str, str], bool],
) -> list[int]:
    """Union-find grouping of *items* under a pairwise equivalence predicate.

    ``None`` entries are abstentions and always get label ``-1`` (a shared
    placeholder, not a genuine equivalence class — it must never win a vote,
    so it is never merged into any group). ``equiv`` need only be symmetric;
    transitivity (``a≡b``, ``b≡c`` ⇒ one class with ``c``) is imposed by the
    union-find merge, which is a deliberate approximation when ``equiv``
    itself is not perfectly transitive (e.g. ``math_verify`` on pathological
    LaTeX).
    """
    n = len(items)
    labels = [-1] * n
    live = [i for i, item in enumerate(items) if item is not None]

    parent = {i: i for i in live}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for a in range(len(live)):
        for b in range(a + 1, len(live)):
            i, j = live[a], live[b]
            if find(i) != find(j) and equiv(items[i], items[j]):
                union(i, j)

    root_to_label: dict[int, int] = {}
    for i in live:
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels[i] = root_to_label[root]

    return labels


def group_generations(
    answers: Sequence[str],
    spec: BenchmarkSpec,
) -> tuple[list[int], list[Optional[str]]]:
    """Group a set of full model outputs into equivalence classes, per *spec*.

    The single grouping path for every benchmark: predictions are extracted
    via ``spec.extract`` (``None`` = abstention — no extractable answer —
    which always gets label ``-1``, uniformly across benchmarks); the values
    actually compared for equivalence — full text or extracted prediction,
    per ``spec.compare_on`` — are grouped via :func:`group_equivalent` using
    ``spec.equiv``.

    Returns ``(labels, predictions)`` — ``predictions`` is exposed so
    callers can build display output (e.g. a vote's ``winner_pred``) without
    re-extracting.
    """
    predictions = [spec.extract(a) if a else None for a in answers]
    if spec.compare_on == "answer":
        values: list[Optional[str]] = [
            answer if prediction is not None else None
            for answer, prediction in zip(answers, predictions)
        ]
    else:
        values = predictions
    labels = group_equivalent(values, spec.equiv)
    return labels, predictions


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

def majority_vote(labels: Sequence[int], *, rng) -> VoteResult:
    """Plurality vote over integer equivalence-class labels.

    ``-1`` is reserved for abstentions and is excluded from the count — an
    abstention-only input yields ``winner_idx=None``. Ties among the
    top-voted labels are broken by a uniform random choice drawn from *rng*
    — callers share one ``random.Random`` instance across an entire run
    (see :func:`run_self_consistency`), so re-running from the same seed
    reproduces every tie-break exactly, in the order the votes are taken.
    """
    n = len(labels)
    n_abstain = sum(1 for label in labels if label == -1)
    valid_labels = [label for label in labels if label != -1]

    if not valid_labels:
        return VoteResult(
            winner_idx=None, winner_pred=None, counts=[], pct_votes_top=0.0,
            tie=False, tie_break_used=False, n_abstain=n_abstain, vote_entropy=0.0,
        )

    counter = Counter(valid_labels)
    counts_sorted = sorted(counter.values(), reverse=True)
    max_count = counts_sorted[0]
    top_labels = sorted(label for label, count in counter.items() if count == max_count)
    tie = len(top_labels) > 1
    winner_label = rng.choice(top_labels) if tie else top_labels[0]

    winner_idx = next(i for i, label in enumerate(labels) if label == winner_label)
    pct_votes_top = max_count / n

    probs = [count / len(valid_labels) for count in counter.values()]
    vote_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    return VoteResult(
        winner_idx=winner_idx, winner_pred=None, counts=counts_sorted,
        pct_votes_top=pct_votes_top, tie=tie, tie_break_used=tie,
        n_abstain=n_abstain, vote_entropy=vote_entropy,
    )


# ---------------------------------------------------------------------------
# Self-Consistency — one entry point per problem / per run
# ---------------------------------------------------------------------------

def self_consistency(
    records: Sequence[Mapping],
    ground_truth: Any,
    spec: BenchmarkSpec,
    *,
    graphs_by_k: Mapping[int, Sequence[tuple[tuple[int, ...], Any]]],
    problem_id: int,
) -> ProblemSCResult:
    """Run the SC ladder for one problem's generations.

    ``records`` holds one generation record per ``gen_id`` (index = gen_id),
    each with an ``"answer"`` field (the full model output). Every
    generation is scored against *ground_truth* exactly once via
    ``spec.score`` (shares ``utils.evaluation``'s parse cache). For each
    ``k`` in ``graphs_by_k`` (built once by :func:`run_self_consistency`,
    reused across every problem), every size-``k`` subset's *own* compiled
    replay graph is invoked — it groups only that subset's members, so
    transitivity from generations outside the subset never leaks in — and
    the vote's winner is mapped back through the subset's ``combo`` to a
    ``gen_id``.
    """
    predictions = [spec.extract(record["answer"]) for record in records]
    compare_values = (
        [record["answer"] for record in records]
        if spec.compare_on == "answer" else predictions
    )
    correct_by_gen = [spec.score(value, ground_truth) for value in compare_values]

    by_k: dict[int, list[SubsetResult]] = {}
    for k, subset_graphs in graphs_by_k.items():
        subset_results: list[SubsetResult] = []
        for combo, graph in subset_graphs:
            out = graph.invoke({
                "task": "", "problem_id": problem_id,
                "dataset": spec.name, "samples": [], "vote": None, "metrics": None,
            })
            vote = out["vote"]

            winner_gen_id: Optional[int] = None
            winner_output: Optional[str] = None
            correct = False
            if vote["winner_idx"] is not None:
                winner_gen_id = combo[vote["winner_idx"]]
                winner_output = predictions[winner_gen_id]
                correct = correct_by_gen[winner_gen_id]

            subset_results.append(SubsetResult(
                subset_id="+".join(f"g{gen_id}" for gen_id in combo),
                indices=combo,
                winner_gen_id=winner_gen_id,
                winner_output=winner_output,
                correct=correct,
                tie=vote["tie"],
                tie_break_used=vote["tie_break_used"],
                n_abstain=vote["n_abstain"],
                pct_votes_top=vote["pct_votes_top"],
            ))
        by_k[k] = subset_results

    return ProblemSCResult(
        predictions=predictions,
        correct_by_gen=correct_by_gen,
        by_k=by_k,
    )


def run_self_consistency(
    store: Mapping[tuple[int, int], Mapping],
    dataset_sample: Sequence[Any],
    spec: BenchmarkSpec,
    k_values: Sequence[int],
    *,
    n_generators: int,
    rng,
    model_key: str,
    model_id: str,
) -> dict[int, ProblemSCResult]:
    """Run the full SC ladder over a fixed problem set.

    Compiles **one replay graph per distinct subset** of the ladder — e.g.
    21 graphs for the full ``k ∈ {1,3,4,5}`` ladder over 5 generators
    (``C(5,1)+C(5,3)+C(5,4)+C(5,5) = 5+10+5+1``) — once, up front,
    independent of problem count: each replay node indexes *store* by
    ``(problem_id, gen_id)`` at invoke time, so the same compiled graph
    serves every problem. *rng* is a single seeded ``random.Random`` shared
    across every vote in the run (see :func:`majority_vote`).

    ``model_key``/``model_id`` identify the single model this store's
    generations came from (an SC store holds exactly one model — see
    ``utils.persistence.build_generation_store``'s invariant) and are
    required explicitly rather than defaulted, since a meaningless default
    would silently leak into every replayed ``AgentSample``.
    """
    from utils.graphs import AgentSpec, build_majority_voting_graph

    graphs_by_k: dict[int, list[tuple[tuple[int, ...], Any]]] = {}
    for k in k_values:
        if not (1 <= k <= n_generators):
            raise ValueError(f"k={k} must satisfy 1 <= k <= n_generators={n_generators}")
        subset_graphs = []
        for combo in itertools.combinations(range(n_generators), k):
            agent_specs = [
                AgentSpec(agent_id=f"g{gen_id}", model_key=model_key, model_id=model_id,
                          mode="replay", store=store, gen_id=gen_id)
                for gen_id in combo
            ]
            graph = build_majority_voting_graph(agent_specs, spec, rng=rng)
            subset_graphs.append((combo, graph))
        graphs_by_k[k] = subset_graphs

    results: dict[int, ProblemSCResult] = {}
    for problem_id, item in enumerate(dataset_sample):
        ground_truth = spec.get_truth(item)
        records = [store[(problem_id, gen_id)] for gen_id in range(n_generators)]
        results[problem_id] = self_consistency(
            records, ground_truth, spec,
            graphs_by_k=graphs_by_k, problem_id=problem_id,
        )
    return results


# ---------------------------------------------------------------------------
# Metric composition
# ---------------------------------------------------------------------------

def compose_generation_metrics(
    calls: Sequence[Mapping[str, Any]],
    *,
    tokens_per_s_rule: Literal["recomputed", "mean"] = "recomputed",
    vram_rule: Literal["sum_per_model", "coresident_peak"] = "sum_per_model",
) -> dict:
    """Compute composition shared by every L1 protocol: SC, MAJ, BoN-MAV.

    Each call is a persisted generation record, an ``AgentSample`` (MAJ
    candidates), or an ``Approval`` (BoN-MAV verifier calls) — all three
    carry ``model_key`` plus the same metric fields, so one function serves
    every protocol. Latency and token counts (input/output/total) always sum
    across every call — the honest total wall time / token spend, whether
    the calls ran sequentially (SC, one resident model) or as a heterogeneous
    ensemble (MAJ/BoN-MAV, several models resident at once).

    ``vram_rule`` selects how the per-call ``gpu_peak_mb`` values combine
    into one ensemble-level peak — picking the wrong one silently
    mis-measures VRAM:

    * ``"sum_per_model"`` (default) — sum, over distinct ``model_key``, of
      that model's max per-call peak. Valid only when each call was measured
      with a single model resident (replayed records, or one-at-a-time
      sequential runs); reduces to a plain max when every call shares one
      ``model_key``.
    * ``"coresident_peak"`` — a single ``max()`` over every call's
      ``gpu_peak_mb``. Use for calls from models genuinely loaded together
      (a live ensemble): each such peak already reflects the whole
      co-resident footprint, so summing them (the other rule) over-counts.

    ``tokens_per_s_rule`` selects the one axis that genuinely differs by
    protocol, since the two aggregation stories mean different things:

    * ``"recomputed"`` (default) — ``Σoutput_tokens / Σlatency_s``, the
      *sustained* rate of one resident model working through several calls
      in sequence (SC's own story: same model, one generator after another).
    * ``"mean"`` — the mean of each call's own rate. For a heterogeneous
      ensemble (MAJ/BoN-MAV) there is no single shared per-token cost to
      average over, so summing totals would let one slow model drag down a
      fast one's own rate; the calls' rates are averaged instead.

    Raises
    ------
    KeyError
        If any call is missing ``model_key`` or one of the metric fields.
    ValueError
        If ``calls`` is empty, or ``vram_rule``/``tokens_per_s_rule`` is
        unrecognized.
    """
    if not calls:
        raise ValueError("compose_generation_metrics requires at least one call")

    latency_s = sum(call["latency_s"] for call in calls)
    input_tokens = sum(call["input_tokens"] for call in calls)
    output_tokens = sum(call["output_tokens"] for call in calls)
    total_tokens = sum(call["total_tokens"] for call in calls)

    if vram_rule == "sum_per_model":
        peak_by_model: dict[str, float] = {}
        for call in calls:
            model_key = call["model_key"]
            peak_by_model[model_key] = max(peak_by_model.get(model_key, 0.0), call["gpu_peak_mb"])
        gpu_peak_mb = sum(peak_by_model.values())
    elif vram_rule == "coresident_peak":
        gpu_peak_mb = max(call["gpu_peak_mb"] for call in calls)
    else:
        raise ValueError(f"Unknown vram_rule {vram_rule!r}; expected 'sum_per_model' or 'coresident_peak'")

    if tokens_per_s_rule == "recomputed":
        # Unrounded — this is a composed value from summed totals, not a
        # single logged sample, so it should not carry tokens_per_second's
        # own display rounding (see utils.instrumentation.tokens_per_second).
        tokens_per_s = tokens_per_second(output_tokens, latency_s, round_dp=None)
    elif tokens_per_s_rule == "mean":
        rates = [tokens_per_second(call["output_tokens"], call["latency_s"], round_dp=None) for call in calls]
        tokens_per_s = sum(rates) / len(rates)
    else:
        raise ValueError(f"Unknown tokens_per_s_rule {tokens_per_s_rule!r}; expected 'recomputed' or 'mean'")

    return {
        "latency_s": latency_s,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "gpu_peak_mb": gpu_peak_mb,
        "tokens_per_s": tokens_per_s,
    }
