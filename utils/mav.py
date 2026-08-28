"""
mav.py - aspect verifiers (AVs), verdict parsing, and BoN-MAV candidate
selection.

Aspect content is transcribed verbatim from the original verifier
specification, including its typos, inconsistent spacing, and curly '
quote glyph, so the transcription stays diffable against that source
rather than a paraphrase. <Question>/<Solution> placeholders become the
Python format-string placeholders {task}/{candidate}, filled in via
prompt.format(task=..., candidate=...).

Every AV falls into one of three strategies by its instruction text, not
always its title (see the MMLU-AV1 note below):

    direct        max_new_tokens=128   "Respond with ONLY ... Do not provide
                                         any explanation or additional text."
    reasoning     max_new_tokens=512   "Think out loud" (reasons freely)
    step_by_step  max_new_tokens=1024  "Go over each step ..." (iterates the
                                         candidate's own steps)

Every subset has exactly 3 AVs (reduced from the original 4 to cut verifier
compute): AV1 is always direct. MATH-S1/MMLU-S1 are otherwise 2 reasoning;
MATH-S2 is otherwise 1 reasoning + 1 step_by_step; MMLU-S2 is otherwise 2
step_by_step. MATH-S2 and MMLU-S2 are therefore no longer structurally
parallel. The verdict marker every prompt itself specifies is
'FINAL VERIFICATION ANSWER: True' / '...: False'.

Transcription trap: both MMLU subsets' AV1 is titled "General correctness,
Step-by-step" in the original spec, but its instruction text is
byte-identical to the direct prompt every other subset's AV1 uses. This
module classifies by instruction text, not title, so it is correctly
strategy="direct" here; see test_mav.py's explicit pin on this.

Connection to the project WPs
------------------------------
* WP2-T4 (L1 multi-agent protocols): the aspect-verifier content and
  selection rule for BoN-MAV.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping, Optional, Sequence

from utils.voting import BenchmarkSpec, group_equivalent

STRATEGY_MAX_NEW_TOKENS: dict[str, int] = {
    "direct": 128,
    "reasoning": 512,
    "step_by_step": 1024,
}


# ---------------------------------------------------------------------------
# Preambles (per-dataset, shared across every AV in that dataset)
# ---------------------------------------------------------------------------

_MATH_PREAMBLE = (
    "You are a critical verifier tasked with evaluating mathematical "
    "problem-solving. You will be presented with a question and a proposed "
    "solution. Your job is to carefully go over and analyze the solution. "
    "Follow the instructions."
)

_MMLU_PREAMBLE = (
    "You are a critical verifier tasked with evaluating multiple-choice "
    "question-answering. You will be presented with a question, the "
    "multiple-choice options, and a proposed solution. Your job is to "
    "carefully go over and analyze the solution. Follow the instructions."
)


def _prompt(preamble: str, instructions: str) -> str:
    return (
        preamble + "\n\n"
        "QUESTION: {task}\n"
        "PROPOSED SOLUTION: {candidate}\n"
        "INSTRUCTIONS: " + instructions
    )


# ---------------------------------------------------------------------------
# Instruction text, transcribed verbatim (including source typos) from
# the original verifier specification
# ---------------------------------------------------------------------------

_DIRECT_INSTRUCTIONS = (
    "Is this solution correct for the given question? Respond with ONLY "
    "'FINAL VERIFICATION ANSWER: True' or ONLY 'FINAL VERIFICATION ANSWER: False'. "
    "Do not provide any explanation or additional text."
)

_SUMMARIZE_INSTRUCTIONS = (
    "Summarize the solution in your own words, explore anything you think "
    "may be incorrect. Think out loud. If you find something that's "
    "incorrect, stop and reply 'FINAL VERIFICATION ANSWER: False'. If "
    "you've gone over the solution and everything seems correct, reply "
    "'FINAL VERIFICATION ANSWER: True'."
)
_EDGE_CASES_INSTRUCTIONS = (
    "Check if the solution handles edge cases and boundary conditions, test "
    "extreme values or special cases. Think out loud. If any boundary "
    "conditions or edge cases fail, stop and reply 'FINAL VERIFICATION "
    "ANSWER: False'. If all boundary conditions and edge cases are handled "
    "correctly, reply 'FINAL VERIFICATION ANSWER: True'."
)
_COMMON_MISTAKES_INSTRUCTIONS = (
    "Check if the solution has any common mistakes, calculation errors, or "
    "misconceptions that typically found in this type of problem. Think out "
    "loud. If you find any common mistakes, stop and reply 'FINAL "
    "VERIFICATION ANSWER: False'. If no common mistakes are found, reply "
    "'FINAL VERIFICATION ANSWER: True'."
)
_MATH_CORRECTNESS_INSTRUCTIONS = (
    "Go over each step in the proposed solution and check whether it is "
    "mathematically correct. Think out loud. If you reach a step that is "
    "incorrect, stop and reply 'FINAL VERIFICATION ANSWER: False'. If "
    "you get to the end of all the steps and each step was correct, reply "
    "'FINAL VERIFICATION ANSWER: True'."
)
_EXPLAIN_DIFFERENTLY_INSTRUCTIONS = (
    "Explain the solution in a different way than it was presented. Try to "
    "find any flaws in the solution. Think out loud. If you find something "
    "that's incorrect, stop and reply 'FINAL VERIFICATION ANSWER: False'. "
    "If you've gone over the solution and everything seems correct, reply "
    "'FINAL VERIFICATION ANSWER: True'."
)
_LOGICAL_SOUNDNESS_INSTRUCTIONS = (
    "Go over each step in the proposed solution and check whether it is "
    "logically sound. Think out loud. If you reach a step that is not "
    "logically sound, stop and reply 'FINAL VERIFICATION ANSWER: False'. "
    "If you get to the end of all the steps and each step was logically "
    "sound, reply 'FINAL VERIFICATION ANSWER: True'."
)
_FACTUAL_CORRECTNESS_INSTRUCTIONS = (
    "Go over each step in the proposed solution and check whether the facts "
    "presented are correct. Think out loud. If you reach a step with "
    "incorrect facts, stop and reply 'FINAL VERIFICATION ANSWER: False'. "
    "If you get to the end of all the steps and each step had correct "
    "facts, reply 'FINAL VERIFICATION ANSWER: True'."
)


# ---------------------------------------------------------------------------
# MAV_ASPECTS: {dataset: {aspect_id: {prompt, strategy, max_new_tokens}}}
# ---------------------------------------------------------------------------

def _aspect(preamble: str, instructions: str, strategy: str, intention: str) -> dict:
    return {
        "prompt": _prompt(preamble, instructions),
        "strategy": strategy,
        "max_new_tokens": STRATEGY_MAX_NEW_TOKENS[strategy],
        "intention": intention,
    }


MAV_ASPECTS: dict[str, dict[str, dict]] = {
    "math500": {
        "math-s1-direct": _aspect(_MATH_PREAMBLE, _DIRECT_INSTRUCTIONS, "direct", "direct"),
        "math-s1-summarize": _aspect(_MATH_PREAMBLE, _SUMMARIZE_INSTRUCTIONS, "reasoning", "summarize"),
        "math-s1-edge-cases": _aspect(_MATH_PREAMBLE, _EDGE_CASES_INSTRUCTIONS, "reasoning", "edge_cases"),
        "math-s2-direct": _aspect(_MATH_PREAMBLE, _DIRECT_INSTRUCTIONS, "direct", "direct"),
        "math-s2-common-mistakes": _aspect(_MATH_PREAMBLE, _COMMON_MISTAKES_INSTRUCTIONS, "reasoning", "common_mistakes"),
        "math-s2-math-correctness": _aspect(_MATH_PREAMBLE, _MATH_CORRECTNESS_INSTRUCTIONS, "step_by_step", "math_correctness"),
    },
    "mmlu": {
        # direct is titled "General correctness, Step-by-step" in both MMLU
        # subsets, but its text is byte-identical to the direct prompt
        # (classified by text, not title; see module docstring).
        "mmlu-s1-direct": _aspect(_MMLU_PREAMBLE, _DIRECT_INSTRUCTIONS, "direct", "direct"),
        "mmlu-s1-summarize": _aspect(_MMLU_PREAMBLE, _SUMMARIZE_INSTRUCTIONS, "reasoning", "summarize"),
        "mmlu-s1-explain-differently": _aspect(_MMLU_PREAMBLE, _EXPLAIN_DIFFERENTLY_INSTRUCTIONS, "reasoning", "explain_differently"),
        "mmlu-s2-direct": _aspect(_MMLU_PREAMBLE, _DIRECT_INSTRUCTIONS, "direct", "direct"),
        "mmlu-s2-logical-soundness": _aspect(_MMLU_PREAMBLE, _LOGICAL_SOUNDNESS_INSTRUCTIONS, "step_by_step", "logical_soundness"),
        "mmlu-s2-factual-correctness": _aspect(_MMLU_PREAMBLE, _FACTUAL_CORRECTNESS_INSTRUCTIONS, "step_by_step", "factual_correctness"),
    },
}

MAV_ASPECT_SUBSETS: dict[str, dict[str, list[str]]] = {
    "math500": {
        "S1": ["math-s1-direct", "math-s1-summarize", "math-s1-edge-cases"],
        "S2": ["math-s2-direct", "math-s2-common-mistakes", "math-s2-math-correctness"],
    },
    "mmlu": {
        "S1": ["mmlu-s1-direct", "mmlu-s1-summarize", "mmlu-s1-explain-differently"],
        "S2": ["mmlu-s2-direct", "mmlu-s2-logical-soundness", "mmlu-s2-factual-correctness"],
    },
}


def av_position_labels(aspect_ids: Sequence[str]) -> dict[str, str]:
    """Map each aspect_id to its generic AV1/AV2/AV3 position label.

    Position, not aspect_id, is comparable across subsets: S1 and S2 use
    different aspect_ids for AV2/AV3, but both have a "position 1" AV.
    Built once per notebook run from MAV_ASPECT_SUBSETS[dataset][subset_id]
    and reused for both per-problem metrics and run-level summary stats.
    """
    return {aspect_id: f"AV{i + 1}" for i, aspect_id in enumerate(aspect_ids)}


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_VERDICT_RE = re.compile(
    r"FINAL VERIFICATION ANSWER:\s*[\"''‘]?\s*(True|False)\s*[\"''‘]?",
    re.IGNORECASE,
)
_FALLBACK_RE = re.compile(r"\b(True|False|Yes|No)\b", re.IGNORECASE)


def parse_verdict(text: str) -> Optional[bool]:
    """Parse a verifier's raw output into True/False/None.

    Strips a leading <think>...</think> block, then takes the last "FINAL
    VERIFICATION ANSWER: (True|False)" marker (case-insensitive, tolerant
    of straight or curly quoting; the source prompts use curly quotes but
    a smaller verifier may echo straight quotes). Falls back to the last
    standalone True/False/Yes/No if the exact marker is missing (a
    smaller verifier may drift off-format). Anything else, including
    empty text, is None: scored as no approval, never as approval.
    """
    if not text:
        return None
    cleaned = _THINK_RE.sub("", text)

    matches = list(_VERDICT_RE.finditer(cleaned))
    if matches:
        return matches[-1].group(1).lower() == "true"

    fallback_matches = list(_FALLBACK_RE.finditer(cleaned))
    if fallback_matches:
        return fallback_matches[-1].group(1).lower() in ("true", "yes")

    return None


# ---------------------------------------------------------------------------
# BoN-MAV candidate selection
# ---------------------------------------------------------------------------

def select_bon_candidate(
    n_candidates: int,
    approvals: Sequence[Mapping[str, Any]],
    candidate_answers: Sequence[str],
    *,
    spec: BenchmarkSpec,
    rng,
) -> dict:
    """Score candidates by approval count; ties broken by SC-among-tied.

    Score is the number of approving verdicts (verdict is True) per
    candidate, over every approval whose candidate_idx matches.
    Unparseable verdicts and disapprovals both score zero, never approval.

    A tie among top-scoring candidates is broken by grouping only the
    tied candidates' answers via spec.equiv (utils.voting.group_equivalent,
    the same union-find grouper SC uses): the largest resulting
    equivalence class wins (the tied candidates that most agree with each
    other), with any further tie broken by a uniform random draw from the
    shared rng.
    """
    scores = [0] * n_candidates
    for approval in approvals:
        if approval["verdict"] is True:
            scores[approval["candidate_idx"]] += 1

    max_score = max(scores)
    top_indices = [i for i, s in enumerate(scores) if s == max_score]
    tie = len(top_indices) > 1

    if not tie:
        selected_idx = top_indices[0]
        tie_break_used = False
    else:
        tied_answers = [candidate_answers[i] for i in top_indices]
        labels = group_equivalent(tied_answers, spec.equiv)
        label_counts = Counter(label for label in labels if label != -1)

        if label_counts:
            max_count = max(label_counts.values())
            winning_labels = [label for label, count in label_counts.items() if count == max_count]
            winning_label = (
                rng.choice(winning_labels) if len(winning_labels) > 1 else winning_labels[0]
            )
            candidates_in_label = [
                top_indices[i] for i, label in enumerate(labels) if label == winning_label
            ]
            selected_idx = (
                rng.choice(candidates_in_label) if len(candidates_in_label) > 1 else candidates_in_label[0]
            )
        else:
            # Every tied candidate abstained (no extractable answer) -> pure
            # uniform random draw among the tied candidates.
            selected_idx = rng.choice(top_indices)
        tie_break_used = True

    return {
        "selected_idx": selected_idx,
        "scores": scores,
        "tie": tie,
        "tie_break_used": tie_break_used,
    }


def approval_diagnostics(approvals: Sequence[Mapping[str, Any]], n_candidates: int) -> dict:
    """Per-aspect approval rate and verdict parse rate.

    approval_rate is the fraction of an aspect's approvals with verdict is
    True (unparseable verdicts count as non-approvals, same rule as
    select_bon_candidate); aspects approving nearly every candidate
    (>95%) are a sycophancy flag. parse_rate is the fraction with
    verdict is not None.
    """
    by_aspect: dict[str, list[Mapping[str, Any]]] = {}
    for approval in approvals:
        by_aspect.setdefault(approval["aspect_id"], []).append(approval)

    result: dict[str, dict] = {}
    for aspect_id, aspect_approvals in by_aspect.items():
        n = len(aspect_approvals)
        n_approved = sum(1 for a in aspect_approvals if a["verdict"] is True)
        n_parsed = sum(1 for a in aspect_approvals if a["verdict"] is not None)
        result[aspect_id] = {
            "approval_rate": n_approved / n if n else 0.0,
            "parse_rate": n_parsed / n if n else 0.0,
            "n": n,
        }
    return result


def candidate_correctness(
    samples: Sequence[Mapping[str, Any]], ground_truth: Any, spec: BenchmarkSpec
) -> list[bool]:
    """Score every candidate against ground truth, not just the selected one.

    Same compare_on/score rule the run loop already applies to the selected
    candidate (`utils.voting.BenchmarkSpec`), applied to all N candidates.
    Pure CPU comparison since every candidate is already generated/replayed,
    so this is the basis for the oracle/recovery and verifier-quality
    metrics below.
    """
    compare_values = [
        sample["answer"] if spec.compare_on == "answer" else sample["parsed_output"]
        for sample in samples
    ]
    return [spec.score(value, ground_truth) for value in compare_values]


def verifier_quality_diagnostics(approvals: Sequence[Mapping[str, Any]]) -> dict:
    """Per-aspect verifier accuracy against ground truth, not just approval rate.

    `approval_diagnostics` only measures how often an aspect approves; this
    measures whether it is right to approve, treating `verdict is True` as a
    prediction that the candidate is actually correct
    (`approval["candidate_correct"]`, attached per-approval by the caller via
    `candidate_correctness`). Distinguishes a genuinely strong verifier from
    a rubber stamp when approval_rate alone can't.

    Returns
    -------
    dict
        Per aspect_id: precision, recall, f1, accuracy, n. precision/recall
        (and therefore f1) are None when undefined (no approvals, or no
        correct candidates seen).
    """
    by_aspect: dict[str, list[Mapping[str, Any]]] = {}
    for approval in approvals:
        by_aspect.setdefault(approval["aspect_id"], []).append(approval)

    result: dict[str, dict] = {}
    for aspect_id, aspect_approvals in by_aspect.items():
        tp = fp = fn = tn = 0
        for a in aspect_approvals:
            predicted_correct = a["verdict"] is True
            actual_correct = a["candidate_correct"]
            if predicted_correct and actual_correct:
                tp += 1
            elif predicted_correct and not actual_correct:
                fp += 1
            elif not predicted_correct and actual_correct:
                fn += 1
            else:
                tn += 1
        n = len(aspect_approvals)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        result[aspect_id] = {
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and (precision + recall) > 0
                else None
            ),
            "accuracy": (tp + tn) / n if n else 0.0,
            "n": n,
        }
    return result


def per_problem_av_metrics(
    approvals: Sequence[Mapping[str, Any]],
    av_labels: Mapping[str, str],
    candidate_correct: Sequence[bool],
) -> dict[str, dict[str, float]]:
    """Per-AV average latency, output tokens and accuracy, for one problem.

    Reduces one problem's flat (candidate x aspect) approvals to one row
    per AV label, averaged -- never summed -- over the candidates that AV
    verified in this problem. "accuracy" reuses verifier_quality_diagnostics'
    point-estimate rule scoped to a single problem instead of a whole run.
    """
    by_label: dict[str, list[Mapping[str, Any]]] = {}
    for approval in approvals:
        by_label.setdefault(av_labels[approval["aspect_id"]], []).append(approval)

    result: dict[str, dict[str, float]] = {}
    for label, items in by_label.items():
        n = len(items)
        result[label] = {
            "latency_s": sum(a["latency_s"] for a in items) / n,
            "output_tokens": sum(a["output_tokens"] for a in items) / n,
            "accuracy": sum(
                1 for a in items if (a["verdict"] is True) == candidate_correct[a["candidate_idx"]]
            ) / n,
        }
    return result


