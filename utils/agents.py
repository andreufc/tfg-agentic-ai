"""
agents.py: LangChain / LangGraph agent construction helpers.

Provides factory functions to build the Society-of-Minds agents
and collaboration graphs used in the thesis experiments.

Connection to the project WPs
-----------------------------
All agents are seeded LangChain LLMs wrapping the loaded HF model, so runs stay
reproducible.
* WP1-T1: builds the proposer/critic/refiner template chain + baseline.
* WP2-T3: builds the single-agent baseline chain.
* WP2-T4/T6: builds the L1 (voting, proposer-verifier) and L2 (debate)
  LangGraph graphs, homogeneous or heterogeneous.
"""

from __future__ import annotations

import random
from typing import Any, Optional

import torch
from langchain_core.runnables import Runnable, RunnableLambda  # type: ignore
from langchain_huggingface import HuggingFacePipeline  # type: ignore
from transformers import pipeline as hf_pipeline  # type: ignore


def _seed_all(seed: int) -> None:
    """Seed Python's `random`, NumPy (if available), and PyTorch (CPU and all
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
    system_prompt: Optional[str] = None,
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
        If `False` (default), only the generated continuation is returned;
        if `True`, the prompt is prepended to the output.
    seed : int | None
        If given, seeds all RNG sources via `_seed_all` before the pipeline
        is created, for reproducible generation.
    batch_size : int
        Sequences per forward pass. Must stay at 1 for all experiments so
        latency, throughput, and GPU-utilization metrics stay attributable
        to individual problems under identical hardware load; only raise
        it for a dedicated throughput-ceiling experiment.
    chat_template : bool
        If `True` (default), the prompt is sent as a chat message and the
        pipeline applies the model's instruct chat template internally.
        Instruct models expect this; without it, weaker models (e.g.
        Ministral-3-3B) often emit an immediate EOS and return empty output.
        Base models without a chat template must use `chat_template=False`.
    system_prompt : str | None
        Chat path only. Prepended as a system message before the user turn,
        for reasoning models that ship a recommended system prompt (e.g.
        Ministral-3 Reasoning's `SYSTEM_PROMPT.txt`, the `[THINK]` protocol).

    Returns
    -------
    Runnable
        With `chat_template=True`, a `RunnableLambda` feeding the pipeline a
        chat message. With `chat_template=False`, a `HuggingFacePipeline`
        fed the raw prompt.
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
        # Base/non-instruct path: raw rendered prompt straight to the pipeline.
        return HuggingFacePipeline(pipeline=pipe)

    # Instruct path: use the chat template by passing messages to the pipeline.
    # The tokenizer handles instruct control tokens and formatting.
    base_messages = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )

    def _generate(prompt: Any) -> str:
        # PromptTemplate yields a StringPromptValue, not a str.
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        messages = base_messages + [{"role": "user", "content": text}]
        generated_text = pipe(messages)[0]["generated_text"]
        # return_full_text=True on a chat yields the full message list;
        # normalize to the assistant turn's text.
        if isinstance(generated_text, list):
            return generated_text[-1]["content"]
        return generated_text

    return RunnableLambda(_generate)
