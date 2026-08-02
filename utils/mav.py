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

Every subset has exactly 4 AVs: AV1 is always direct; MATH-S1/MMLU-S1 are
otherwise 2 reasoning + 1 step_by_step; MATH-S2/MMLU-S2 are otherwise 1
reasoning + 2 step_by_step. The verdict marker every prompt itself
specifies is 'FINAL VERIFICATION ANSWER: True' / '...: False'.

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

# MATH - S1
_MATH_S1_AV2_INSTRUCTIONS = (
    "Summarize the solution in your own words, explore anything you think "
    "may be incorrect. Think out loud. If you find something that's "
    "incorrect, stop and reply 'FINAL VERIFICATION ANSWER: False'. If "
    "you've gone over the solution and everything seems correct, reply "
    "'FINAL VERIFICATION ANSWER: True'."
)
_MATH_S1_AV3_INSTRUCTIONS = (
    "Check if the solution handles edge cases and boundary conditions, test "
    "extreme values or special cases. Think out loud. If any boundary "
    "conditions or edge cases fail, stop and reply 'FINAL VERIFICATION "
    "ANSWER: False'. If all boundary conditions and edge cases are handled "
    "correctly, reply 'FINAL VERIFICATION ANSWER: True'."
)
_MATH_STEP_BY_STEP_MATH_CORRECTNESS_INSTRUCTIONS = (
    "Go over each step in the proposed solution and check whether it is "
    "mathematically correct. Think out loud. If you reach a step that is "
    "incorrect, stop and reply 'FINAL VERIFICATION ANSWER: False'. If "
    "you get to the end of all the steps and each step was correct, reply "
    "'FINAL VERIFICATION ANSWER: True'."
)

# MATH - S2
_MATH_S2_AV2_INSTRUCTIONS = (
    "Check if the solution has any common mistakes, calculation errors, or "
    "misconceptions that typically found in this type of problem. Think out "
    "loud. If you find any common mistakes, stop and reply 'FINAL "
    "VERIFICATION ANSWER: False'. If no common mistakes are found, reply "
    "'FINAL VERIFICATION ANSWER: True'."
)
_MATH_S2_AV3_INSTRUCTIONS = (
    "Go over each step in the proposed solution and check whether it is "
    "logically sound. Think out loud. If you reach a step that is not "
    "logically sound, stop and reply 'FINAL VERIFICATION ANSWER: False'. "
    "If you get to the end of all the steps and each step was logically "
    "sound, reply 'FINAL VERIFICATION ANSWER: True'."
)

# MMLU - S1
_MMLU_S1_AV2_INSTRUCTIONS = _MATH_S1_AV2_INSTRUCTIONS
_MMLU_S1_AV3_INSTRUCTIONS = (
    "Explain the solution in a different way than it was presented. Try to "
    "find any flaws in the solution. Think out loud. If you find something "
    "that's incorrect, stop and reply 'FINAL VERIFICATION ANSWER: False'. "
    "If you've gone over the solution and everything seems correct, reply "
    "'FINAL VERIFICATION ANSWER: True'."
)
_MMLU_STEP_BY_STEP_FACTUAL_CORRECTNESS_INSTRUCTIONS = (
    "Go over each step in the proposed solution and check whether the facts "
    "presented are correct. Think out loud. If you reach a step with "
    "incorrect facts, stop and reply 'FINAL VERIFICATION ANSWER: False'. "
    "If you get to the end of all the steps and each step had correct "
    "facts, reply 'FINAL VERIFICATION ANSWER: True'."
)

# MMLU - S2
_MMLU_S2_AV2_INSTRUCTIONS = (
    "Check if the solution correctly applies relevant domain knowledge, "
    "established theories, and standard practices for this type of "
    "problem. Think out loud. If any domain knowledge is misapplied or "
    "violated, stop and reply 'FINAL VERIFICATION ANSWER: False'. If all "
    "domain-specific knowledge is correctly applied, reply 'FINAL "
    "VERIFICATION ANSWER: True'."
)
_MMLU_S2_AV3_INSTRUCTIONS = (
    "Go over each step in the proposed solution and check whether it is "
    "logically sound. Think out loud. If you reach a step that is not "
    "logically sound, stop and reply 'FINAL VERIFICATION ANSWER: False'. "
    "If you get to the end of all the steps and each step was logically "
    "sound, reply 'FINAL VERIFICATION ANSWER: True'."
)


# ---------------------------------------------------------------------------
# MAV_ASPECTS: {dataset: {aspect_id: {prompt, strategy, max_new_tokens}}}
# ---------------------------------------------------------------------------

def _aspect(preamble: str, instructions: str, strategy: str) -> dict:
    return {
        "prompt": _prompt(preamble, instructions),
        "strategy": strategy,
        "max_new_tokens": STRATEGY_MAX_NEW_TOKENS[strategy],
    }


MAV_ASPECTS: dict[str, dict[str, dict]] = {
    "math500": {
        "math-s1-av1": _aspect(_MATH_PREAMBLE, _DIRECT_INSTRUCTIONS, "direct"),
        "math-s1-av2": _aspect(_MATH_PREAMBLE, _MATH_S1_AV2_INSTRUCTIONS, "reasoning"),
        "math-s1-av3": _aspect(_MATH_PREAMBLE, _MATH_S1_AV3_INSTRUCTIONS, "reasoning"),
        "math-s1-av4": _aspect(_MATH_PREAMBLE, _MATH_STEP_BY_STEP_MATH_CORRECTNESS_INSTRUCTIONS, "step_by_step"),
        "math-s2-av1": _aspect(_MATH_PREAMBLE, _DIRECT_INSTRUCTIONS, "direct"),
        "math-s2-av2": _aspect(_MATH_PREAMBLE, _MATH_S2_AV2_INSTRUCTIONS, "reasoning"),
        "math-s2-av3": _aspect(_MATH_PREAMBLE, _MATH_S2_AV3_INSTRUCTIONS, "step_by_step"),
        "math-s2-av4": _aspect(_MATH_PREAMBLE, _MATH_STEP_BY_STEP_MATH_CORRECTNESS_INSTRUCTIONS, "step_by_step"),
    },
    "mmlu": {
        # AV1 is titled "General correctness, Step-by-step" in both MMLU
        # subsets, but its text is byte-identical to the direct prompt
        # (classified by text, not title; see module docstring).
        "mmlu-s1-av1": _aspect(_MMLU_PREAMBLE, _DIRECT_INSTRUCTIONS, "direct"),
        "mmlu-s1-av2": _aspect(_MMLU_PREAMBLE, _MMLU_S1_AV2_INSTRUCTIONS, "reasoning"),
        "mmlu-s1-av3": _aspect(_MMLU_PREAMBLE, _MMLU_S1_AV3_INSTRUCTIONS, "reasoning"),
        "mmlu-s1-av4": _aspect(_MMLU_PREAMBLE, _MMLU_STEP_BY_STEP_FACTUAL_CORRECTNESS_INSTRUCTIONS, "step_by_step"),
        "mmlu-s2-av1": _aspect(_MMLU_PREAMBLE, _DIRECT_INSTRUCTIONS, "direct"),
        "mmlu-s2-av2": _aspect(_MMLU_PREAMBLE, _MMLU_S2_AV2_INSTRUCTIONS, "reasoning"),
        "mmlu-s2-av3": _aspect(_MMLU_PREAMBLE, _MMLU_S2_AV3_INSTRUCTIONS, "step_by_step"),
        "mmlu-s2-av4": _aspect(_MMLU_PREAMBLE, _MMLU_STEP_BY_STEP_FACTUAL_CORRECTNESS_INSTRUCTIONS, "step_by_step"),
    },
}

MAV_ASPECT_SUBSETS: dict[str, dict[str, list[str]]] = {
    "math500": {
        "S1": ["math-s1-av1", "math-s1-av2", "math-s1-av3", "math-s1-av4"],
        "S2": ["math-s2-av1", "math-s2-av2", "math-s2-av3", "math-s2-av4"],
    },
    "mmlu": {
        "S1": ["mmlu-s1-av1", "mmlu-s1-av2", "mmlu-s1-av3", "mmlu-s1-av4"],
        "S2": ["mmlu-s2-av1", "mmlu-s2-av2", "mmlu-s2-av3", "mmlu-s2-av4"],
    },
}


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
        Per aspect_id: precision, recall, accuracy, n. precision/recall are
        None when undefined (no approvals, or no correct candidates seen).
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
        result[aspect_id] = {
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
            "accuracy": (tp + tn) / n if n else 0.0,
            "n": n,
        }
    return result


def oracle_and_recovery(records: Sequence[Mapping[str, Any]]) -> dict:
    """Oracle upper bound and recovery rate for a completed BoN-MAV run.

    oracle_accuracy is the fraction of problems where at least one of the N
    replayed candidates is correct, the best any verifier could achieve
    since BoN-MAV only selects among existing candidates. recovery_rate
    isolates selection quality from candidate quality: of the problems where
    a correct candidate existed, what fraction did the verifier actually
    select? Each record must carry candidate_correct (from
    `candidate_correctness`) and correct (the selected candidate's own
    correctness, already computed by every BoN-MAV notebook).
    """
    n_recoverable = 0
    n_recovered = 0
    for r in records:
        if any(r["candidate_correct"]):
            n_recoverable += 1
            if r["correct"]:
                n_recovered += 1
    n = len(records)
    return {
        "oracle_accuracy": n_recoverable / n if n else 0.0,
        "n_recoverable": n_recoverable,
        "recovery_rate": n_recovered / n_recoverable if n_recoverable else None,
    }
