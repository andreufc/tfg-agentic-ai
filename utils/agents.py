"""
agents.py — LangChain / LangGraph agent construction helpers.

Provides factory functions to build the Society-of-Minds agents
and collaboration graphs used in the thesis experiments.
"""

from __future__ import annotations

import random
from typing import Any, Optional

import torch
from langchain_huggingface import HuggingFacePipeline  # type: ignore
from transformers import pipeline as hf_pipeline  # type: ignore


def _seed_all(seed: int) -> None:
    """Seed every relevant RNG source for fully reproducible generation.

    Sets the seed on Python's ``random`` module, NumPy (if available),
    PyTorch (CPU and all CUDA devices), and the HuggingFace/CUDA
    deterministic-algorithm flag so that the complete sequence of
    stochastic generation calls within a run is deterministic.

    Parameters
    ----------
    seed : int
        The integer seed value to apply everywhere.
    """
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_hf_langchain_llm(
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20,
    return_full_text: bool = False,
    seed: Optional[int] = None,
    batch_size: int = 1,
    **kwargs: Any,
) -> HuggingFacePipeline:
    """Wrap a loaded HF model as a LangChain-compatible LLM.

    Parameters
    ----------
    model : PreTrainedModel
        A loaded HF causal-LM model.
    tokenizer : PreTrainedTokenizer
        The associated tokenizer.
    max_new_tokens : int
        Maximum tokens to generate.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling parameter.
    top_k : int
        Top-k sampling parameter.
    return_full_text : bool
        If ``False`` (default), only the generated continuation is
        returned; if ``True``, the prompt is prepended to the output.
    seed : int | None
        If provided, all RNG sources (Python random, NumPy, PyTorch CPU
        and CUDA) are seeded before the pipeline is created, making the
        entire generation sequence within a run fully reproducible.
        Pass ``None`` (default) to keep stochastic behaviour.
    batch_size : int
        Number of sequences processed per forward pass (default ``1``).
        Must stay at 1 for all experiments so that latency, throughput,
        and GPU-utilization metrics remain attributable to individual
        problems and both MAS and baseline are measured under identical
        hardware load. Only raise this for a dedicated sustained-
        throughput ceiling experiment, never for the main comparison.

    Returns
    -------
    HuggingFacePipeline
        A LangChain LLM wrapping the HF pipeline.
    """
    if seed is not None:
        _seed_all(seed)

    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        return_full_text=return_full_text,
        do_sample=temperature > 0,
        batch_size=batch_size,
        **kwargs,
    )
    return HuggingFacePipeline(pipeline=pipe)
