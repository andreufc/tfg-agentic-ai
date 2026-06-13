"""
evaluation.py — Evaluation metrics and result aggregation.

Provides functions to compute accuracy for GSM8K (exact numeric match)
and MMLU (multiple-choice letter match), plus helpers to aggregate
and display results.
"""

from __future__ import annotations

import math
import re
from typing import Sequence


# ---------------------------------------------------------------------------
# GSM8K evaluation
# ---------------------------------------------------------------------------

def normalize_numeric(text: str) -> str | None:
    """Extract the last number from a string, stripping commas and
    trailing dots.

    The returned string is the canonical representation of the
    number — trailing dots (42. → 42) and unnecessary
    trailing zeros (42.00 → 42) are removed so that
    downstream comparisons are reliable.

    Returns None if no number is found.
    """
    # Remove commas from numbers like 1,234
    text = text.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if not numbers:
        return None
    raw = numbers[-1].rstrip(".")
    # Canonicalize via float round-trip to drop trailing zeros
    try:
        return _canonicalize_number(raw)
    except (ValueError, OverflowError):
        return raw


def _canonicalize_number(s: str) -> str:
    """Return a canonical string form of a numeric string.

    42.0 → 42, 3.10 → 3.1, 7 → 7.
    """
    f = float(s)
    if math.isfinite(f) and f == int(f):
        return str(int(f))
    return str(f)


def gsm8k_match(prediction: str | None, reference: str | None) -> bool:
    """Check whether a predicted GSM8K answer matches the reference.

    Both values may be None (extraction failure) — in that case
    the result is False.

    The comparison strategy is:

    1. Try to convert both sides to float and compare with an
       epsilon margin (rel_tol=1e-9) to absorb floating-point
       rounding.  This handles 42.0 vs 42
       3.10 vs 3.1 etc.
    2. If either side cannot be converted to float, fall back to
       a plain string comparison of the canonicalized forms.
    """
    if prediction is None or reference is None:
        return False
    pred_norm = normalize_numeric(prediction)
    ref_norm = normalize_numeric(reference)
    if pred_norm is None or ref_norm is None:
        return False
    # Primary: float comparison with epsilon
    try:
        return math.isclose(
            float(pred_norm), float(ref_norm), rel_tol=1e-9, abs_tol=1e-9,
        )
    except (ValueError, OverflowError):
        pass
    # Fallback: plain string comparison (for non-numeric edge-cases)
    return pred_norm == ref_norm


def gsm8k_accuracy(
    predictions: Sequence[str],
    references: Sequence[str],
) -> float:
    """Compute exact-match accuracy for GSM8K.

    Both predictions and references can contain free-form text;
    the last number in each string is compared numerically, so
    formatting differences (trailing dots, trailing zeros, different
    decimal precision) do not cause false negatives.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    correct = sum(
        1 for pred, ref in zip(predictions, references)
        if gsm8k_match(pred, ref)
    )
    return correct / len(predictions) if predictions else 0.0

def _number_from_boxed_content(content: str) -> str | None:
    """Return the numeric value from a \\boxed{...} content string, or None.

    Handles pure numbers (``750``), currency-prefixed values (``\\$3220``),
    and numbers with trailing unit words (``1 hour``, ``60 minutes``).
    Returns None for algebraic expressions (``2A+S+\\$720``).
    """
    cleaned = content.replace("\\$", "").replace("$", "").replace(",", "").strip()
    # Accept: optional leading sign, digits, optional decimal — then only letters/spaces
    m = re.match(r"^(-?\d+\.?\d*)\s*([A-Za-z\s]*)$", cleaned)
    if m and not re.search(r"\d", m.group(2)):
        return m.group(1).rstrip(".")
    return None


def extract_gsm8k_prediction(agent_output: str) -> str | None:
    """Extract a numeric prediction from an agent's GSM8K output.

    Search order:
    1. Last ``\\boxed{...}`` whose content resolves to a number.
       Handles currency prefixes (``\\$``) and unit suffixes (``hours``,
       ``minutes``, etc.); skips algebraic expressions.
    2. ``#### <number>`` marker (legacy GSM8K format).
    3. Last number in the full text (fallback).

    The returned value is canonicalized so it can be compared directly
    with the ground truth.  Returns ``None`` if no number is found.
    """
    # Collect all \boxed{} contents and scan from the end so the last
    # numeric answer wins over intermediate boxed intermediate values.
    for content in reversed(re.findall(r"\\boxed\{([^}]*)\}", agent_output)):
        raw = _number_from_boxed_content(content)
        if raw is not None:
            try:
                return _canonicalize_number(raw)
            except (ValueError, OverflowError):
                return raw

    # Try the #### marker (legacy format)
    marker = re.search(r"####\s*(-?\d+\.?\d*)", agent_output)
    if marker:
        raw = marker.group(1).rstrip(".")
        try:
            return _canonicalize_number(raw)
        except (ValueError, OverflowError):
            return raw

    # Fallback: last number in text (strip commas first)
    cleaned = agent_output.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", cleaned)
    if not numbers:
        return None
    raw = numbers[-1].rstrip(".")
    try:
        return _canonicalize_number(raw)
    except (ValueError, OverflowError):
        return raw


# ---------------------------------------------------------------------------
# MMLU evaluation
# ---------------------------------------------------------------------------

_LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def normalize_mmlu_answer(text: str) -> str | None:
    """Extract a single A/B/C/D letter from a model response.

    Extraction priority:

    1. EvalScope-style ``ANSWER: X`` — matched case-sensitively on the
       word ``ANSWER`` so it cannot accidentally fire on ``Final Answer: X``.
    2. ``Final Answer: X`` — legacy / fallback for older prompt formats.
    3. Direct single-letter response after stripping punctuation.
    4. First standalone A–D letter found anywhere in the text.
    """
    # Priority 1: EvalScope format — uppercase 'ANSWER: X' (letter A–D any case)
    ans_match = re.search(r"ANSWER\s*:\s*([A-Da-d])", text)
    if ans_match:
        return ans_match.group(1).upper()

    # Priority 2: case-insensitive 'Final Answer: X' (legacy)
    fa_match = re.search(
        r"final\s+answer\s*:\s*([A-Da-d])[.)\s]?",
        text,
        re.IGNORECASE,
    )
    if fa_match:
        return fa_match.group(1).upper()

    # Strip whitespace and common trailing/leading punctuation
    cleaned = text.strip().strip(".").strip(")").strip("(").strip().upper()

    # Direct single-letter answer (possibly with stripped punctuation)
    if cleaned in ("A", "B", "C", "D"):
        return cleaned
    # Pattern like "The answer is B" or "The answer is B."
    match = re.search(r"\b([A-D])\b[.)]?", cleaned)
    return match.group(1) if match else None


def mmlu_match(prediction: str | None, reference: str) -> bool:
    """Check whether a predicted MMLU letter matches the reference.

    Both values are uppercased before comparison. None predictions
    always return False.
    """
    if prediction is None:
        return False
    return prediction.strip().upper() == str(reference).strip().upper()


def mmlu_accuracy(
    predictions: Sequence[str],
    references: Sequence[int | str],
) -> float:
    """Compute accuracy for MMLU.

    Parameters
    ----------
    predictions : sequence of str
        Raw model outputs.
    references : sequence of int or str
        Gold labels — either letter strings (A–D) or
        integer indices (0–3).
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    correct = 0
    for pred, ref in zip(predictions, references):
        pred_letter = normalize_mmlu_answer(pred)
        ref_letter = get_ground_truth_letter(ref) if isinstance(ref, int) else str(ref).upper()
        if pred_letter == ref_letter:
            correct += 1
    return correct / len(predictions) if predictions else 0.0


def get_ground_truth_letter(answer_idx: int) -> str:
    """Map a 0-based answer index to the corresponding letter.

    Parameters
    ----------
    answer_idx : int
        Index (0–3) of the correct choice.

    Returns
    -------
    str
        The letter ``"A"``–``"D"``.
    """
    return _LETTER_MAP[answer_idx]

# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

def print_results(results: dict[str, float], title: str = "Results") -> None:
    """Pretty-print a dictionary of metric → value pairs."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric:<30s} {value:>8.2%}")
        else:
            print(f"  {metric:<30s} {value!s:>8s}")
    print(f"{'=' * 50}\n")
