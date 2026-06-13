"""
models.py — Model loading utilities.

Helpers to load HF causal-LM and vision-language models with optional
NF4 4-bit quantisation via bitsandbytes, targeting a single 16 GB T4 GPU.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import (  # type: ignore
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def load_model_and_tokenizer(
    model_name: str,
    quantize_4bit: bool = True,
    device_map: str = "auto",
    token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a causal-LM (or text-capable VL) model and its tokenizer from HuggingFace.

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

    # Some models (Ministral-3, Gemma-4 E2B/E4B) ship as multimodal archs
    # registered only under AutoModelForImageTextToText, not AutoModelForCausalLM.
    # We pick the right class from the config, then fall back to the other if needed.
    # The fallback is cheap: the arch-mismatch check raises before any weights load.
    config = AutoConfig.from_pretrained(
        model_name, token=token, trust_remote_code=trust_remote_code
    )
    if type(config) in AutoModelForCausalLM._model_mapping:
        auto_classes = (AutoModelForCausalLM, AutoModelForImageTextToText)
    else:
        auto_classes = (AutoModelForImageTextToText, AutoModelForCausalLM)

    model = None
    last_error: Exception | None = None
    for auto_cls in auto_classes:
        try:
            model = auto_cls.from_pretrained(model_name, **load_kwargs)
            break
        except ValueError as exc:
            if "Unrecognized configuration class" not in str(exc):
                raise
            last_error = exc
    if model is None:
        raise last_error  # type: ignore[misc]

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
