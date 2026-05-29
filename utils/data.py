"""
data.py — Dataset loading and preprocessing utilities.

Provides thin wrappers around Hugging Face Datasets for GSM8K and MMLU,
with standardised column names and optional subsampling.
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
        Dataset split (``"train"`` or ``"test"``).
    n_samples : int | None
        If given, randomly subsample to this many examples.
    seed : int
        Random seed for reproducible subsampling.

    Returns
    -------
    Dataset
        A HF Dataset with columns ``question`` and ``answer``.
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if n_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    return ds


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numeric answer from a GSM8K answer string.

    GSM8K answers contain step-by-step reasoning followed by
    ``#### <number>``. This function returns only the number.
    """
    if "####" in answer_text:
        return answer_text.split("####")[-1].replace(',', '').strip()
    return answer_text.strip()


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
        f"Question: {question}\n\n"
        f"Choices:\n{choices_block}\n\n"
        "IMPORTANT: Provide your final answer at the very end formatted "
        "exactly as 'Final Answer: [Letter]' (e.g., Final Answer: B)."
    )
