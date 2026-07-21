"""
evaluation.py — Evaluation metrics and result aggregation.

Provides functions to compute accuracy for GSM8K (exact numeric match),
MMLU (multiple-choice letter match) and MATH-500 (LaTeX/symbolic
equivalence via math-verify), plus helpers to aggregate and display
results.

Connection to the project WPs
-----------------------------
* WP1-T1 & WP2-T3/T4/T6: scores outputs (GSM8K number / MMLU letter /
  MATH-500 expression) against ground truth into the 0/1 correctness
  matrices.
* WP2-T5/T7 & WP3-T1: those matrices are the input to the results and
  trade-off analyses.
"""

from __future__ import annotations

import math
import re
import sys
from typing import Sequence


# ---------------------------------------------------------------------------
# GSM8K evaluation
# ---------------------------------------------------------------------------

def normalize_numeric(text: str) -> str | None:
    """Extract the last number from a string and canonicalize it.

    Strips commas, then canonicalizes via a float round-trip so
    formatting differences don't cause false negatives in downstream
    comparisons (e.g. ``42.`` -> ``42``, ``42.00`` -> ``42``).
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

    1. Try to convert both sides to float and compare with an epsilon
       margin (rel_tol=1e-9) to absorb floating-point rounding —
       e.g. 42.0 vs 42, or 3.10 vs 3.1.
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


def _iter_boxed_contents(text: str) -> list[str]:
    """Return every brace-balanced ``\\boxed{...}`` content, in document order.

    Walks each ``\\boxed{`` with a depth counter so nested LaTeX
    (``\\boxed{\\frac{1}{2}}``) is captured in full instead of being
    truncated at the first ``}``. A ``\\boxed{`` whose braces never
    balance (e.g. a truncated generation) is skipped. Shared by GSM8K
    (which scans all occurrences in reverse) and MATH-500 (which wants
    only the last one).
    """
    contents = []
    pos = 0
    marker = "\\boxed{"
    while True:
        start = text.find(marker, pos)
        if start == -1:
            break
        i = start + len(marker)
        depth = 1
        content_start = i
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            contents.append(text[content_start:i - 1])
            pos = i
        else:
            break
    return contents


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
    # Collect all balanced \boxed{} contents and scan from the end so the
    # last numeric answer wins over intermediate boxed values.
    for content in reversed(_iter_boxed_contents(agent_output)):
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
# MATH-500 evaluation
# ---------------------------------------------------------------------------

def _last_boxed_content(text: str) -> str | None:
    """Return the content of the last ``\\boxed{...}`` in text, brace-balanced.

    Thin wrapper over :func:`_iter_boxed_contents` — see there for the
    balanced-scan semantics. Returns None if no balanced ``\\boxed{}`` exists.
    """
    contents = _iter_boxed_contents(text)
    return contents[-1] if contents else None


def extract_math_prediction(agent_output: str) -> str | None:
    """Extract a display-only MATH-500 prediction from an agent's output.

    This is used solely to populate the ``final_output`` column of the
    results table for human inspection — correctness is computed by
    :func:`math_match` on the *full* agent output, since ``math_verify``
    is itself a robust boxed/LaTeX extractor and re-cutting the string
    here first would only throw away information (and could disagree
    with what actually got scored).

    Search order:
    1. Last ``\\boxed{...}`` content (brace-balanced).
    2. Last ``$...$`` inline-math span.
    3. None.
    """
    boxed = _last_boxed_content(agent_output)
    if boxed is not None:
        return boxed.strip()

    inline = re.findall(r"\$([^$]+)\$", agent_output)
    if inline:
        return inline[-1].strip()

    return None


def math_match(prediction: str | None, reference: str | None) -> bool:
    """Check whether a full MATH-500 model output matches the reference answer.

    Both values may be None/empty — in that case the result is False.
    Scoring uses ``math_verify`` (sympy-based) for LaTeX/symbolic
    equivalence (e.g. ``1/2`` == ``\\frac{1}{2}`` == ``0.5``), since a
    plain numeric or string comparison cannot handle MATH-500's
    LaTeX-formatted answers (fractions, roots, intervals, matrices, ...).

    ``prediction`` should be the *full* model output (not a pre-extracted
    substring) — ``math_verify.parse`` already performs boxed/LaTeX
    extraction with its own fallback chain, so handing it the raw text is
    more robust than re-extracting first.

    Note the argument order: ``verify(gold, pred)`` takes the gold
    reference first. This is intentionally asymmetric (e.g. for
    inequalities vs. their solution sets) — swapping the arguments
    would silently change results.

    Both ``math_verify.parse`` and ``verify`` apply a per-call timeout
    using ``signal.SIGALRM`` on Unix (cheap) but fall back to spawning a
    subprocess on Windows, which silently breaks under a test runner
    (the subprocess can't be spawned, and the failure is swallowed as an
    empty parse). Kaggle (the actual execution environment) is Linux, so
    both timeouts stay enabled there; on Windows (local dev/tests only)
    they're disabled.
    """
    if not prediction or not reference:
        return False

    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise ImportError(
            "math-verify is required for MATH-500 scoring — "
            "pip install 'math-verify[antlr4_13_2]'"
        ) from exc

    on_windows = sys.platform == "win32"
    parsing_timeout = None if on_windows else 5
    verify_timeout = None if on_windows else 5

    try:
        gold = parse(f"${reference}$", parsing_timeout=parsing_timeout)
        if not gold:
            gold = parse(reference, parsing_timeout=parsing_timeout)
        if not gold:
            return False

        pred = parse(prediction, parsing_timeout=parsing_timeout)
        if not pred:
            return False

        return bool(verify(gold, pred, timeout_seconds=verify_timeout))
    except Exception:
        # A single pathological LaTeX sample must not kill a 384-invocation
        # evaluation run — treat any parser/sympy failure as "not a match".
        return False


def math_accuracy(
    predictions: Sequence[str],
    references: Sequence[str],
) -> float:
    """Compute MATH-500 accuracy via ``math_match`` over full model outputs."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    correct = sum(
        1 for pred, ref in zip(predictions, references)
        if math_match(pred, ref)
    )
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
