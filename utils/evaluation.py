"""
evaluation.py — Evaluation metrics and result aggregation.

Provides functions to compute accuracy for GSM8K (exact numeric match)
and MMLU (multiple-choice letter match), plus helpers to aggregate
and display results.
"""

from __future__ import annotations

import re
from typing import Sequence


# ---------------------------------------------------------------------------
# GSM8K evaluation
# ---------------------------------------------------------------------------

def normalize_numeric(text: str) -> str | None:
    """Extract the last number from a string, stripping commas.

    Returns ``None`` if no number is found.
    """
    # Remove commas from numbers like 1,234
    text = text.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else None


def gsm8k_accuracy(
    predictions: Sequence[str],
    references: Sequence[str],
) -> float:
    """Compute exact-match accuracy for GSM8K.

    Both *predictions* and *references* can contain free-form text;
    the last number in each string is compared.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    correct = 0
    for pred, ref in zip(predictions, references):
        pred_num = normalize_numeric(pred)
        ref_num = normalize_numeric(ref)
        if pred_num is not None and pred_num == ref_num:
            correct += 1
    return correct / len(predictions) if predictions else 0.0


# ---------------------------------------------------------------------------
# MMLU evaluation
# ---------------------------------------------------------------------------

_LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def normalize_mmlu_answer(text: str) -> str | None:
    """Extract a single A/B/C/D letter from a model response.

    Looks first for an explicit ``Final Answer: X`` pattern
    (case-insensitive), then falls back to generic letter extraction.
    """
    # Priority: explicit "Final Answer: X" pattern
    fa_match = re.search(r"final\s+answer\s*:\s*([A-Da-d])", text, re.IGNORECASE)
    if fa_match:
        return fa_match.group(1).upper()

    text_upper = text.strip().upper()
    # Direct single-letter answer
    if text_upper in ("A", "B", "C", "D"):
        return text_upper
    # Pattern like "The answer is B"
    match = re.search(r"\b([A-D])\b", text_upper)
    return match.group(1) if match else None


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
        Gold labels — either letter strings (``"A"``–``"D"``) or
        integer indices (0–3).
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    correct = 0
    for pred, ref in zip(predictions, references):
        pred_letter = normalize_mmlu_answer(pred)
        ref_letter = _LETTER_MAP.get(ref, ref) if isinstance(ref, int) else str(ref).upper()
        if pred_letter == ref_letter:
            correct += 1
    return correct / len(predictions) if predictions else 0.0


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


def extract_gsm8k_prediction(agent_output: str) -> str | None:
    """Extract a numeric prediction from an agent's GSM8K output.

    Tries to match the ``#### <number>`` pattern first; if not found,
    falls back to the last number in the text (after stripping commas).

    Returns ``None`` if no number is found.
    """
    # Try the #### marker first
    marker = re.search(r"####\s*(-?\d+)", agent_output)
    if marker:
        return marker.group(1)

    # Fallback: last number in text (strip commas first)
    cleaned = agent_output.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", cleaned)
    return numbers[-1] if numbers else None


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
