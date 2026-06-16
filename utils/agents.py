"""
agents.py — LangChain / LangGraph agent construction helpers.

Provides factory functions to build the Society-of-Minds agents
and collaboration graphs used in the thesis experiments.
"""

from __future__ import annotations

import random
from typing import Any, Optional

import torch
from langchain_core.runnables import Runnable, RunnableLambda  # type: ignore
from langchain_huggingface import HuggingFacePipeline  # type: ignore
from transformers import pipeline as hf_pipeline  # type: ignore


def _seed_all(seed: int) -> None:
    """Seed Python's ``random``, NumPy (if available), and PyTorch (CPU and all
    CUDA devices) so a run's full sequence of stochastic generation calls is
    reproducible.
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
    chat_template: bool = True,
    **kwargs: Any,
) -> Runnable:
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
        If given, seeds all RNG sources via ``_seed_all`` before the pipeline
        is created, for reproducible generation. ``None`` (default) keeps
        stochastic behaviour.
    batch_size : int
        Number of sequences processed per forward pass (default ``1``).
        Must stay at 1 for all experiments so that latency, throughput,
        and GPU-utilization metrics remain attributable to individual
        problems and both MAS and baseline are measured under identical
        hardware load. Only raise this for a dedicated sustained-
        throughput ceiling experiment, never for the main comparison.
    chat_template : bool
        If ``True`` (default), the rendered prompt is wrapped as a single user
        message and the HF pipeline applies the model's instruct chat template
        internally (``apply_chat_template(tokenize=True)``) before generating.
        Instruct models expect this; without it, weaker models (e.g.
        Ministral-3-3B) frequently emit an immediate EOS and return empty
        outputs. Requires the tokenizer to expose a chat template — base models
        without one must use ``chat_template=False``.

    Returns
    -------
    Runnable
        A LangChain Runnable. With ``chat_template=True`` this is a
        ``RunnableLambda`` that feeds the prompt to the pipeline as a chat
        message (template applied internally). With ``chat_template=False`` it
        is a ``HuggingFacePipeline`` fed the raw prompt.
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

    if not chat_template:
        # Base/non-instruct path: feed the raw rendered prompt straight to the
        # pipeline (no chat template applied).
        return HuggingFacePipeline(pipeline=pipe)

    # Instruct path: pass a one-message chat list so the pipeline applies the
    # chat template internally (apply_chat_template(tokenize=True,
    # return_dict=True)), getting the instruct control tokens to the model
    # without a string round-trip. Correct decoding requires the tokenizer to
    # be the mistral-common backend (see load_model_and_tokenizer) — a plain
    # LlamaTokenizer garbles spaces / emits byte markers.
    def _generate(prompt: Any) -> str:
        # The upstream PromptTemplate yields a StringPromptValue, not a str.
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        generated_text = pipe([{"role": "user", "content": text}])[0]["generated_text"]
        # return_full_text=True on a chat yields the full message list rather
        # than a string — normalise to the assistant turn's text.
        if isinstance(generated_text, list):
            return generated_text[-1]["content"]
        return generated_text

    return RunnableLambda(_generate)
