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
    reasoning     max_new_tokens=768   "Think out loud" (reasons freely)
    step_by_step  max_new_tokens=1280  "Go over each step ..." (iterates the
                                         candidate's own steps)

Every subset has exactly 3 AVs (reduced from the original 4 to cut verifier
compute): AV1 is always direct. MATH-S1/MMLU-S1 are otherwise 2 reasoning;
MATH-S2 is otherwise 1 reasoning + 1 step_by_step; MMLU-S2 is otherwise 2
step_by_step. MATH-S2 and MMLU-S2 are therefore no longer structurally
parallel. The verdict marker every prompt itself specifies is
'FINAL VERIFICATION ANSWER: True' / '...: False'.

To address a measured parse-rate failure on the BoN-MAV artifacts (sub-4B
verifiers truncating before reaching verdict markers), each non-direct
instruction has one short sentence appended: "Keep your analysis under N words."
with N sized close to STRATEGY_MAX_NEW_TOKENS. See _WORD_BUDGET_SUFFIX and _aspect().

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
    "reasoning": 768,
    "step_by_step": 1280,
}

# Empty for "direct" (already forbids explanation); others sized at roughly
# 0.75 words/token to leave reasoning room without hitting the token cap.
_WORD_BUDGET_SUFFIX: dict[str, str] = {
    "direct": "",
    "reasoning": " Keep your analysis under 500 words.",
    "step_by_step": " Keep your analysis under 900 words.",
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
# Instruction text, reused from the 'Multi-Agent Verification: Scaling Test-Time 
# Compute with Multiple Verifiers' paper (Lifshitz et al., 2024)
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
    instructions = instructions + _WORD_BUDGET_SUFFIX[strategy]
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


def aspect_prompt(dataset: str, aspect_id: str, *, no_think: bool = False) -> str:
    """
    Retrieves the prompt for a given dataset and aspect_id, 
    optionally adding /no_think flag (convention for all Qwen3 agents in this project).
    """
    prompt = MAV_ASPECTS[dataset][aspect_id]["prompt"]
    return f"/no_think\n\n{prompt}" if no_think else prompt


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

# Markdown emphasis run (bold/italic/code/strikethrough markers), tolerated
# anywhere around the marker text since verifiers routinely wrap it, e.g.
# "**FINAL VERIFICATION ANSWER:** True" or "ANSWER: **True**".
_EMPH = r"[*_`~]*"
_QUOTE = r"[\"''‘]?"

_VERDICT_RE = re.compile(
    rf"FINAL\s+VERIFICATION\s+ANSWER{_EMPH}\s*:\s*{_EMPH}\s*{_QUOTE}\s*{_EMPH}(True|False)\b",
    re.IGNORECASE,
)

# Anchored fallback patterns tried only over text tail to avoid stray
# True/False matches in reasoning prose. Yes/No dropped (never requested).
_FALLBACK_TAIL_CHARS = 500
_FALLBACK_PATTERNS = (
    re.compile(rf"VERIFICATION\s+ANSWER{_EMPH}\s*:\s*{_EMPH}\s*{_QUOTE}\s*{_EMPH}(True|False)\b", re.IGNORECASE),
    re.compile(rf"\bANSWER{_EMPH}\s*:\s*{_EMPH}\s*{_QUOTE}\s*{_EMPH}(True|False)\b", re.IGNORECASE),
    re.compile(rf"^\s*{_EMPH}(True|False){_EMPH}\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(rf"\\boxed\{{\s*{_EMPH}(True|False){_EMPH}\s*\}}", re.IGNORECASE),
)


def _fallback_verdict(text: str) -> Optional[bool]:
    tail = text[-_FALLBACK_TAIL_CHARS:]
    for pattern in _FALLBACK_PATTERNS:
        matches = list(pattern.finditer(tail))
        if matches:
            return matches[-1].group(1).lower() == "true"
    return None


def parse_verdict_detail(
    text: str, *, output_tokens: Optional[int] = None, max_new_tokens: Optional[int] = None
) -> dict:
    """Parse a verifier's raw output into {"verdict": bool|None, "source": str}.

    source is one of:
      "marker"     -- a real "FINAL VERIFICATION ANSWER: True/False", tolerant
                      of markdown emphasis (bold/italic/code) around it.
      "fallback"   -- no exact marker, but an anchored fallback form matched
                      in the tail (see _FALLBACK_PATTERNS).
      "truncated"  -- no verdict found and output_tokens >= max_new_tokens,
                      i.e. generation was cut off by the token budget rather
                      than the model choosing not to answer. Diagnostic only:
                      requires both output_tokens and max_new_tokens to be
                      passed, otherwise indistinguishable from "no_marker".
      "no_marker"  -- finished generating but never produced anything
                      verdict-shaped (or truncation info wasn't supplied).
      "empty"      -- empty text.
    """
    if not text:
        return {"verdict": None, "source": "empty"}

    matches = list(_VERDICT_RE.finditer(text))
    if matches:
        return {"verdict": matches[-1].group(1).lower() == "true", "source": "marker"}

    verdict = _fallback_verdict(text)
    if verdict is not None:
        return {"verdict": verdict, "source": "fallback"}

    hit_cap = (
        output_tokens is not None
        and max_new_tokens is not None
        and output_tokens >= max_new_tokens
    )
    return {"verdict": None, "source": "truncated" if hit_cap else "no_marker"}


def parse_verdict(text: str) -> Optional[bool]:
    """Parse a verifier's raw output into True/False/None.

    Thin wrapper around parse_verdict_detail for call sites that only need
    the verdict, not why it was (or wasn't) extracted.
    """
    return parse_verdict_detail(text)["verdict"]


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
    """Score candidates by effective approvals; ties broken by fewest
    abstentions, then by SC-among-tied.

    Every aspect prompt instructs the verifier to reply False only on
    finding a concrete flaw, and True otherwise,so a verifier call that
    never produces a parseable verdict (verdict is None) has produced no
    rejection evidence, not a rejection. Each candidate's primary score is
    therefore its count of effective approvals, verdict is True or
    verdict is None, so an unparseable call does not count against the
    candidate it was verifying. Candidates tied on that primary score are
    then ranked by fewest None verdicts, before running the 
    SC-among-tied rule: grouping only the still-tied candidates via
    spec.equiv, applied over the value spec.compare_on names (full answer
    text or spec.extract's prediction, depending on the benchmark); the
    largest resulting equivalence class wins, with any remaining tie broken
    by a uniform random draw from the shared rng.
    """
    n_true = [0] * n_candidates
    abstentions = [0] * n_candidates
    for approval in approvals:
        idx = approval["candidate_idx"]
        verdict = approval["verdict"]
        if verdict is True:
            n_true[idx] += 1
        elif verdict is None:
            abstentions[idx] += 1

    scores = [t + a for t, a in zip(n_true, abstentions)]

    max_score = max(scores)
    top_indices = [i for i, s in enumerate(scores) if s == max_score]
    tie = len(top_indices) > 1

    if not tie:
        selected_idx = top_indices[0]
        tie_break_used = False
        tie_break_stage = "none"
    else:
        min_abstentions = min(abstentions[i] for i in top_indices)
        top_indices = [i for i in top_indices if abstentions[i] == min_abstentions]
        tie_break_used = True

        if len(top_indices) == 1:
            selected_idx = top_indices[0]
            tie_break_stage = "abstentions"
        else:
            tied_answers = [candidate_answers[i] for i in top_indices]
            tied_values = (
                tied_answers if spec.compare_on == "answer"
                else [spec.extract(a) if a else None for a in tied_answers]
            )
            labels = group_equivalent(tied_values, spec.equiv)
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
                if len(candidates_in_label) > 1:
                    selected_idx = rng.choice(candidates_in_label)
                    tie_break_stage = "random"
                else:
                    selected_idx = candidates_in_label[0]
                    tie_break_stage = "sc"
            else:
                # Every tied candidate abstained (no extractable answer) -> pure
                # uniform random draw among the tied candidates.
                selected_idx = rng.choice(top_indices)
                tie_break_stage = "random"

    return {
        "selected_idx": selected_idx,
        "scores": scores,
        "tie": tie,
        "tie_break_used": tie_break_used,
        "abstentions": abstentions,
        "tie_break_stage": tie_break_stage,
    }


def approval_diagnostics(approvals: Sequence[Mapping[str, Any]], n_candidates: int) -> dict:
    """Per-aspect approval rate and verdict parse rate.

    approval_rate is the fraction of an aspect's approvals with verdict is
    True (unparseable verdicts count as non-approvals); this measures
    verifier behaviour, so aspects approving nearly every candidate (>95%)
    are a sycophancy flag regardless of how selection treats abstentions.
    effective_approval_rate additionally counts verdict is None as an
    approval, matching the rule select_bon_candidate actually uses. parse_rate
    is the fraction with verdict is not None.
    """
    by_aspect: dict[str, list[Mapping[str, Any]]] = {}
    for approval in approvals:
        by_aspect.setdefault(approval["aspect_id"], []).append(approval)

    result: dict[str, dict] = {}
    for aspect_id, aspect_approvals in by_aspect.items():
        n = len(aspect_approvals)
        n_approved = sum(1 for a in aspect_approvals if a["verdict"] is True)
        n_effective = sum(1 for a in aspect_approvals if a["verdict"] is not False)
        n_parsed = sum(1 for a in aspect_approvals if a["verdict"] is not None)
        result[aspect_id] = {
            "approval_rate": n_approved / n if n else 0.0,
            "effective_approval_rate": n_effective / n if n else 0.0,
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


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[Optional[float], Optional[float], Optional[float]]:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return precision, recall, f1


def verifier_quality_diagnostics(approvals: Sequence[Mapping[str, Any]]) -> dict:
    """Per-aspect verifier accuracy against ground truth, not just approval rate.

    `approval_diagnostics` only measures how often an aspect approves; this
    measures whether it is right to approve, treating a prediction of
    "correct" as `approval["candidate_correct"]` (attached per-approval by
    the caller via `candidate_correctness`). 

    Two conventions for what counts as a "correct" prediction are reported,
    matching the two rules used elsewhere in this module:
      effective (primary) -- `verdict is not False`, i.e. True or None counts
                              as predicting correct. Mirrors the selection
                              rule in `select_bon_candidate`: an abstention is
                              not rejection evidence, so it should not be
                              scored as one here either.
      strict (*_strict)    -- `verdict is True` only; an abstention is scored
                              as predicting incorrect. Kept so a caller that wants to
                              ignore abstentions entirely still can.
                              
    Precision/recall/f1 are None when undefined (no positive predictions, or
    no correct candidates seen).

    Returns
    -------
    dict
        Per aspect_id: precision, recall, f1, accuracy (effective rule);
        precision_strict, recall_strict, f1_strict, accuracy_strict (strict
        rule); n_abstain, abstain_rate; n.
    """
    by_aspect: dict[str, list[Mapping[str, Any]]] = {}
    for approval in approvals:
        by_aspect.setdefault(approval["aspect_id"], []).append(approval)

    result: dict[str, dict] = {}
    for aspect_id, aspect_approvals in by_aspect.items():
        tp = fp = fn = tn = 0
        tp_s = fp_s = fn_s = tn_s = 0
        n_abstain = 0
        for a in aspect_approvals:
            actual_correct = a["candidate_correct"]
            verdict = a["verdict"]
            if verdict is None:
                n_abstain += 1

            predicted_correct = verdict is not False
            if predicted_correct and actual_correct:
                tp += 1
            elif predicted_correct and not actual_correct:
                fp += 1
            elif not predicted_correct and actual_correct:
                fn += 1
            else:
                tn += 1

            predicted_correct_strict = verdict is True
            if predicted_correct_strict and actual_correct:
                tp_s += 1
            elif predicted_correct_strict and not actual_correct:
                fp_s += 1
            elif not predicted_correct_strict and actual_correct:
                fn_s += 1
            else:
                tn_s += 1

        n = len(aspect_approvals)
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        precision_s, recall_s, f1_s = _precision_recall_f1(tp_s, fp_s, fn_s)
        result[aspect_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": (tp + tn) / n if n else 0.0,
            "precision_strict": precision_s,
            "recall_strict": recall_s,
            "f1_strict": f1_s,
            "accuracy_strict": (tp_s + tn_s) / n if n else 0.0,
            "n_abstain": n_abstain,
            "abstain_rate": n_abstain / n if n else 0.0,
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


