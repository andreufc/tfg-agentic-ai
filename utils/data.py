"""
data.py — Dataset loading and preprocessing utilities.

Provides thin wrappers around Hugging Face Datasets for GSM8K, MMLU and
MATH-500, with standardised column names and optional subsampling.

Connection to the project WPs
-----------------------------
* WP2-T1: implements GSM8K/MMLU/MATH-500 ingestion — standardised columns +
  seeded subsampling.
* WP1-T1 & WP2-T3/T4/T6: serves that same split as the evaluation set for the
  template, baselines and protocols.
"""

from __future__ import annotations

from typing import Optional, Sequence

from datasets import load_dataset, Dataset  # type: ignore


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def load_gsm8k(
    split: str = "test",
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    """Load the GSM8K dataset.

    Parameters
    ----------
    split : str
        Dataset split (train or test).
    n_samples : int | None
        If given, randomly subsample to this many examples.
    seed : int
        Random seed for reproducible subsampling.

    Returns
    -------
    Dataset
        A HF Dataset with columns question and answer.
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if n_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    return ds


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numeric answer from a GSM8K answer string.

    GSM8K answers contain step-by-step reasoning followed by
    ``#### <number>``. This function returns the canonicalized number
    (trailing dots and redundant zeros stripped).
    """
    if "####" in answer_text:
        raw = answer_text.split("####")[-1].replace(',', '').strip()
    else:
        raw = answer_text.strip()
    # Canonicalize for consistent comparison with predictions
    raw = raw.rstrip(".")
    try:
        from utils.evaluation import _canonicalize_number
        return _canonicalize_number(raw)
    except (ValueError, OverflowError, ImportError):
        return raw


def format_gsm8k_prompt(question: str) -> str:
    """Format a GSM8K question into a prompt string with answer-format instructions.

    Appends a standardised instruction telling the model to provide
    its final numerical answer in the ``#### <number>`` format used
    by the GSM8K dataset.

    Parameters
    ----------
    question : str
        The raw GSM8K question text.

    Returns
    -------
    str
        A formatted prompt ready to be sent to a model.
    """
    return (
        f"{question}\n\n"
        "IMPORTANT: Provide the final numerical answer at the very end "
        "formatted exactly as '#### [number]'."
    )


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

def load_mmlu(
    subjects: Optional[list[str]] = None,
    split: str = "test",
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    """Load the MMLU dataset (cais/mmlu variant).

    Parameters
    ----------
    subjects : list[str] | None
        If given, filter to these MMLU subjects.
    split : str
        Dataset split.
    n_samples : int | None
        If given, randomly subsample to this many examples.
    seed : int
        Random seed for reproducible subsampling.

    Returns
    -------
    Dataset
        A HF Dataset with columns ``question``, ``choices``, ``answer``,
        and ``subject``.
    """
    ds = load_dataset("cais/mmlu", "all", split=split)
    if subjects:
        ds = ds.filter(lambda x: x["subject"] in subjects)
    if n_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    return ds


def format_mmlu_prompt(question: str, choices: Sequence[str]) -> str:
    """Format an MMLU question and its choices into a prompt string.

    Parameters
    ----------
    question : str
        The question text.
    choices : sequence of str
        The four answer choices (A through D).

    Returns
    -------
    str
        A formatted prompt ready to be sent to a model.
    """
    labels = ["A", "B", "C", "D"]
    choices_block = "\n".join(
        f"{label}) {choice}" for label, choice in zip(labels, choices)
    )
    return (
        f"{question}\n\n{choices_block}"
    )


# ---------------------------------------------------------------------------
# MATH-500
# ---------------------------------------------------------------------------

def load_math500(
    split: str = "test",
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    """Load the MATH-500 dataset (HuggingFaceH4/MATH-500).

    MATH-500 is the standard 500-problem subset of the hendrycks MATH
    test set (the original ``hendrycks/competition_math`` repo was
    removed from the Hub for DMCA reasons). Only a ``test`` split
    exists.

    Parameters
    ----------
    split : str
        Dataset split (only "test" exists for this dataset).
    n_samples : int | None
        If given, randomly subsample to this many examples.
    seed : int
        Random seed for reproducible subsampling.

    Returns
    -------
    Dataset
        A HF Dataset with columns ``problem``, ``solution``, ``answer``,
        ``subject``, ``level`` (1-4 — level 5 is excluded, see below) and
        ``unique_id``. Unlike GSM8K, ``answer`` is already the final answer
        (raw LaTeX, no delimiters) — there is no ``####`` marker to parse
        out, so no ``extract_math500_answer`` counterpart is needed.
    """
    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    # Level-5 (hardest) problems produce long reasoning chains that frequently
    # exceed the fixed generation budget and get truncated before `\boxed{}` —
    # excluded so accuracy reflects capability, not truncation.
    ds = ds.filter(lambda x: x["level"] <= 4)
    if n_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    return ds


def format_math_prompt(problem: str) -> str:
    """Format a MATH-500 problem using EvalScope's competition-math template.

    Parameters
    ----------
    problem : str
        The raw MATH-500 problem statement.

    Returns
    -------
    str
        A formatted prompt ready to be sent to a model. Notebooks build
        their own equivalent ``PromptTemplate`` (as with GSM8K); this
        function documents the canonical wording in one importable place.
    """
    return (
        f"Problem:\n{problem}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
