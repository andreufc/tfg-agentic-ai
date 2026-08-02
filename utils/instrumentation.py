"""
instrumentation.py - Token counting, GPU peak-VRAM tracking, and throughput.

Per-step helpers: token counting (via the model's own tokenizer), GPU peak
VRAM measurement (torch.cuda, deterministic, near-zero overhead), generation
throughput, and timed_invoke, which composes the above around one
chain.invoke call. All GPU helpers degrade to 0.0 when CUDA is unavailable
(e.g. local CPU development).

Connection to the project WPs
-----------------------------
* WP1-T1/T4 & WP2-T3/T4/T6: measures tokens, peak VRAM and throughput per
  generation, outside the timed region so latency stays clean.
* WP3-T1: feeds those cost metrics into the accuracy-to-compute analysis (OBJ-3).

Design notes
------------
- Token counting runs outside the timed chain.invoke region so latency
  stays uncontaminated (see timed_invoke below; the v1.x/v2.x notebooks'
  run_agent_step inlines the same sequence for reproducibility).
- count_tokens uses add_special_tokens=False for consistency; chat-template
  wrapping may add 1-2 tokens vs what the pipeline feeds the model, but MAS
  and baseline are measured identically so relative comparisons hold.
- Peak VRAM must be reset with reset_gpu_peak() immediately before each
  measured scope, or the counter reflects the global max since the last
  manual reset (including model load).
- tokens_per_second measures end-to-end throughput (prefill + decode +
  wrapper overhead), not pure decode rate. MAS and baseline are measured
  identically so the ratio is a sound cross-protocol efficiency comparison
  for the same model. A prefill-free "pure decode rate" metric was
  prototyped but dropped: it's a per-model hardware constant that doesn't
  vary with protocol, so it can't inform that tradeoff, and it adds real
  per-token GPU-sync overhead for a number no experiment consumes.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Union

import torch


# ---------------------------------------------------------------------------
# Device resolution: measure only the GPU the model actually uses.
# On multi-GPU kernels (e.g. Kaggle's 2x T4) the model is typically pinned to
# a single device, so measurements must target that device, not index 0.
# ---------------------------------------------------------------------------

DeviceLike = Union[int, str, "torch.device", None]


def _resolve_torch_index(device: DeviceLike = None) -> Optional[int]:
    """Return the torch CUDA device index to measure.

    None resolves to the current active device (where the model likely is).
    Returns None when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return None
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, torch.device):
        return device.index if device.index is not None else torch.cuda.current_device()
    if isinstance(device, str):
        idx = torch.device(device).index
        return idx if idx is not None else torch.cuda.current_device()
    return int(device)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(tokenizer: Any, text: str) -> int:
    """Count the number of tokens in *text* using *tokenizer*.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        The model's tokenizer (any HuggingFace fast tokenizer).
    text : str
        Any string: a rendered prompt or a model completion.

    Returns
    -------
    int
        Number of tokens (add_special_tokens=False).
    """
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def render_prompt(prompt_template: Any, inputs: dict) -> str:
    """Render a LangChain PromptTemplate to a plain string.

    Uses PromptTemplate.format(**inputs) which returns the fully
    rendered prompt as a plain string, suitable for tokenization.

    Parameters
    ----------
    prompt_template : PromptTemplate
        A LangChain PromptTemplate instance.
    inputs : dict
        Variable bindings to fill the template.

    Returns
    -------
    str
        The fully rendered prompt string.
    """
    return prompt_template.format(**inputs)


# ---------------------------------------------------------------------------
# GPU peak VRAM (torch.cuda: deterministic, microsecond overhead)
# ---------------------------------------------------------------------------

def reset_gpu_peak(device: DeviceLike = None) -> None:
    """Reset the CUDA peak-memory counter for the measured device.

    Call this immediately before the code block whose peak VRAM you want
    to measure. device defaults to the active device (where the model
    lives); pass an explicit index/str/torch.device to override. No-op
    when CUDA is unavailable.
    """
    idx = _resolve_torch_index(device)
    if idx is not None:
        torch.cuda.reset_peak_memory_stats(idx)


def gpu_peak_mem_mb(device: DeviceLike = None) -> float:
    """Return the peak GPU memory allocated since the last reset.

    Parameters
    ----------
    device : int | str | torch.device | None
        The CUDA device to read. None (default) reads the active device,
        i.e. the one the model is loaded on, important on multi-GPU
        kernels where only one device is in use.

    Returns
    -------
    float
        Peak allocated VRAM in megabytes, or 0.0 if CUDA is unavailable.
    """
    idx = _resolve_torch_index(device)
    if idx is not None:
        return torch.cuda.max_memory_allocated(idx) / 1e6
    return 0.0


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------

def tokens_per_second(
    output_tokens: int, elapsed_s: float, *, round_dp: Optional[int] = 4
) -> float:
    """Compute generation throughput in output tokens per second.

    Measures end-to-end generation throughput (prefill + decode).
    Both MAS (sum of all agent output tokens / total MAS time) and
    baseline are measured identically, making their ratio a sound
    efficiency comparison.

    Parameters
    ----------
    output_tokens : int
        Number of tokens generated.
    elapsed_s : float
        Wall-clock seconds of the full chain.invoke call.
    round_dp : int | None
        Decimal places to round to (default 4). Pass None for the
        unrounded ratio, used by utils.voting.compose_generation_metrics,
        which recomputes throughput from summed or per-call totals rather
        than a single sample and should not carry this module's own rounding.

    Returns
    -------
    float
        Tokens per second (0.0 if elapsed_s is ~0).
    """
    if elapsed_s < 1e-9:
        return 0.0
    rate = output_tokens / elapsed_s
    return round(rate, round_dp) if round_dp is not None else rate


# ---------------------------------------------------------------------------
# Instrumented invoke (composes the primitives above around one chain.invoke)
# ---------------------------------------------------------------------------

def timed_invoke(chain: Any, prompt_template: Any, tokenizer: Any, inputs: dict) -> dict:
    """Render, time, and measure a single chain.invoke call.

    Centralizes the render, count input tokens, reset peak VRAM, time,
    count output tokens, read peak VRAM sequence every notebook's
    generation step needs. v1.x/v2.x notebooks keep their own
    hand-inlined copy of this sequence (run_agent_step) for
    reproducibility; v3.x+ notebooks call this helper instead.

    Parameters
    ----------
    chain : Runnable
        A LangChain PromptTemplate | llm | OutputParser chain.
    prompt_template : PromptTemplate
        Used only to render the prompt for token counting; chain itself
        is still invoked with inputs directly.
    tokenizer : PreTrainedTokenizer
        The model's own tokenizer, for consistent token counting.
    inputs : dict
        Variable bindings passed to both render_prompt and chain.invoke.

    Returns
    -------
    dict with keys:
        output_text    (str)   the chain's output
        latency_s      (float) wall-clock seconds of the invoke call
        input_tokens   (int)
        output_tokens  (int)
        total_tokens   (int)
        gpu_peak_mb    (float)
    """
    prompt_str = render_prompt(prompt_template, inputs)
    input_tokens = count_tokens(tokenizer, prompt_str)

    reset_gpu_peak()
    t0 = time.perf_counter()
    output_text = chain.invoke(inputs)
    elapsed = time.perf_counter() - t0

    output_tokens = count_tokens(tokenizer, output_text)
    peak_vram_mb = gpu_peak_mem_mb()

    return {
        "output_text": output_text,
        "latency_s": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "gpu_peak_mb": peak_vram_mb,
    }
