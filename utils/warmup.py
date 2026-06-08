"""
warmup.py — Model warmup to remove first-run latency/throughput artifacts.

Runs a few throwaway generations through the same HuggingFacePipeline path
the evaluation uses, forcing CUDA/bitsandbytes kernel compilation and cuBLAS
autotuning to happen before any timed measurement. Degrades gracefully when
CUDA is unavailable (local CPU dev) — the generation still runs, just slower,
and no GPU-only calls are forced.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from utils.agents import build_hf_langchain_llm


def warmup_model(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 128,
    n_iterations: int = 2,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    prompt: Optional[str] = None,
    verbose: bool = True,
) -> float:
    """Run throwaway generations to trigger one-time kernel compilation.

    Builds a temporary LangChain LLM via ``build_hf_langchain_llm`` (the same
    factory the evaluation loop uses) and invokes it ``n_iterations`` times on
    a short dummy prompt. The generated text is discarded.

    Parameters mirror the notebook generation config so the warmed kernels
    match the measured ones. Returns the total warmup wall-time in seconds.

    Parameters
    ----------
    model : PreTrainedModel
        A loaded HF causal-LM model.
    tokenizer : PreTrainedTokenizer
        The associated tokenizer.
    max_new_tokens : int
        Tokens to generate per warmup iteration (default ``64`` — enough to
        exercise the decode loop without wasting time).
    n_iterations : int
        Number of throwaway generations (default ``2``).
    temperature, top_p, top_k : float / int
        Sampling parameters; pass the same values as the evaluation so the
        warmed sampling path matches the measured one.
    prompt : str | None
        Dummy prompt. A short default question is used when ``None``.
    verbose : bool
        Print progress messages.

    Returns
    -------
    float
        Total warmup wall-time in seconds.

    Notes
    -----
    - Does NOT pass a ``seed`` so it never perturbs the RNG state used by the
      seeded evaluation runs that follow — reproducibility is preserved.
    - Peak-VRAM stats are reset per step inside the eval loop
      (``reset_gpu_peak`` in ``run_agent_step``), so warmup allocations do not
      contaminate any measurement.
    """
    if prompt is None:
        prompt = "What is 2 + 2 * 2? Reason briefly and give the answer."

    llm = build_hf_langchain_llm(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        return_full_text=False,
    )
    chain = PromptTemplate.from_template("{task}") | llm | StrOutputParser()

    if verbose:
        print(
            f"🔥 Warming up model ({n_iterations} iterations, "
            f"max_new_tokens={max_new_tokens})…"
        )

    t0 = time.perf_counter()
    for _ in range(n_iterations):
        _ = chain.invoke({"task": prompt})
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if verbose:
        print(
            f"✅ Warmup complete in {elapsed:.1f}s — "
            f"first evaluated sample will no longer pay cold-start cost."
        )
    return elapsed
