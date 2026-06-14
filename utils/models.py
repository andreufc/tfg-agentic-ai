"""
models.py — Model loading utilities.

Helpers to load HF causal-LM (and text-capable multimodal) models with optional
NF4 4-bit quantisation via bitsandbytes, targeting a single 16 GB T4 GPU.

Loading is dispatched per model family (see ``load_model_and_tokenizer``):
Ministral-3 ships as a multimodal ``Mistral3`` checkpoint and needs a bespoke
loader; every other in-scope family (Phi-4, Qwen3, Gemma-4 text) is a plain
causal LM. New families get their own branch as the need arises.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def _strip_vision_modules(model: PreTrainedModel) -> None:
    """Drop the vision tower + multimodal projector of a Ministral-3 model.

    Ministral-3 ships as a multimodal ``Mistral3`` checkpoint, but the project
    uses it for text only. ``Mistral3Model.forward`` gates every vision op
    behind ``pixel_values is not None`` (verified on transformers 5.x), so for
    text generation the vision tower and projector are never invoked — freeing
    them reclaims their VRAM and leaves just the language model + ``lm_head``.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        return
    freed = False
    for attr in ("vision_tower", "multi_modal_projector"):
        if getattr(inner, attr, None) is not None:
            setattr(inner, attr, None)
            freed = True
    if freed and torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_and_tokenizer(
    model_name: str,
    quantize_4bit: bool = True,
    device_map: str = "auto",
    token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a model and its tokenizer from HuggingFace, dispatched by family.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    quantize_4bit : bool
        Load with NF4 4-bit quantisation via bitsandbytes (~5 GB for 8B models).
    device_map : str
        Device placement strategy passed to ``from_pretrained``.
    token : str | None
        HF access token; falls back to the ``HF_TOKEN`` env var.
    trust_remote_code : bool
        Whether to trust remote code in the model repo.

    Returns
    -------
    tuple[PreTrainedModel, PreTrainedTokenizer]
    """
    token = token or os.environ.get("HF_TOKEN")

    quant_config = None
    if quantize_4bit:
        from transformers.utils import is_bitsandbytes_available  # type: ignore

        # is_bitsandbytes_available is lru_cached. On Kaggle it is often called
        # at kernel startup (before CUDA is live), which freezes False for the
        # session. If that happens, transformers silently skips quantisation and
        # loads full BF16 (~16 GB) — OOMing the T4. Clearing the cache forces a
        # fresh evaluation here and inside the subsequent from_pretrained call.
        try:
            is_bitsandbytes_available.cache_clear()  # type: ignore[attr-defined]
        except AttributeError:
            pass

        if not is_bitsandbytes_available():
            raise RuntimeError(
                "quantize_4bit=True but bitsandbytes is not available — "
                "transformers would silently fall back to full BF16 and OOM the T4. "
                "Run `pip install -U bitsandbytes` and restart the kernel, or pass "
                "quantize_4bit=False to load in full precision intentionally."
            )

        # Force synchronous weight loading. transformers >= 5.8 auto-disables its
        # async loader during on-the-fly quantisation, but earlier 5.x (e.g. the
        # 5.0.0 on Kaggle's image) race full-precision shards onto the GPU faster
        # than NF4 can shrink them, spiking VRAM to the full BF16 size and OOMing
        # the T4. This env var (honoured at load time) keeps loading sequential.
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    load_kwargs = dict(
        quantization_config=quant_config,
        device_map=device_map,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    # Dispatch by model family. Add an `elif` here when another in-scope family
    # needs a bespoke loader (e.g. the multimodal Gemma-4 E2B/E4B checkpoints).
    if "ministral" in model_name.lower():
        # Multimodal Mistral3 checkpoint: only loads via the image-text class.
        # Keep the language model, discard the vision tower (text-only use).
        model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        _strip_vision_modules(model)
    else:
        # Plain causal LM (Phi-4, Qwen3, Gemma-4 text, ...).
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Verify NF4 actually engaged by scanning for bitsandbytes' own Linear4bit
    # layers (more reliable than transformers-internal flags across major versions).
    if quantize_4bit:
        from bitsandbytes.nn import Linear4bit  # type: ignore

        if not any(isinstance(m, Linear4bit) for m in model.modules()):
            raise RuntimeError(
                f"Model loaded in full precision (dtype={model.dtype}) despite "
                "quantize_4bit=True — no Linear4bit layers found. "
                "This will OOM the T4. Check bitsandbytes install / restart kernel."
            )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
