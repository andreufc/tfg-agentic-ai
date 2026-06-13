"""
models.py — Model loading utilities.

Provides helpers to load Hugging Face causal-LM models with optional
4-bit quantisation via bitsandbytes (designed for Colab T4 GPUs).
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
    """Load a causal-LM model and its tokenizer from Hugging Face.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier (e.g. ``"microsoft/Phi-3-mini-4k-instruct"``).
    quantize_4bit : bool
        If ``True``, load with NF4 quantisation via bitsandbytes.
    device_map : str
        Device placement strategy (default ``"auto"``).
    token : str | None
        HF token. If ``None``, falls back to ``HF_TOKEN`` env var.
    trust_remote_code : bool
        Whether to trust remote code in the model repo.

    Returns
    -------
    tuple[PreTrainedModel, PreTrainedTokenizer]
    """
    token = token or os.environ.get("HF_TOKEN")

    quant_config = None
    if quantize_4bit:
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

    # Pick the right auto class from the config. Most project models
    # (Phi-4, Qwen3, plain Mistral, Gemma-4 text) are plain causal LMs, but a
    # few targets ship as vision/multimodal architectures registered only under
    # AutoModelForImageTextToText — e.g. Ministral-3 (``Mistral3Config``) and
    # the Gemma-4 E2B/E4B checkpoints (Gemma-3n-style conditional-generation
    # configs). Forcing those through the causal-LM loader raises an
    # "Unrecognized configuration class ... for AutoModelForCausalLM" error, so
    # dispatch on the config class instead. The resulting VL model still
    # generates text-only and is accepted by the downstream ``text-generation``
    # pipeline (a benign warning aside): ``model.can_generate()`` is True and
    # text-only input needs no pixels.
    config = AutoConfig.from_pretrained(
        model_name, token=token, trust_remote_code=trust_remote_code
    )
    # Try the most likely class first, then fall back to the other, so a new
    # model family loads regardless of which auto class it happens to register
    # under. The config-class check raises *before* any weights are read, so
    # the fallback attempt is cheap.
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
            # Only retry the other class on the arch-mismatch error; re-raise
            # anything else (e.g. a genuine quantisation/config failure).
            if "Unrecognized configuration class" not in str(exc):
                raise
            last_error = exc
    if model is None:
        raise last_error  # type: ignore[misc]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    # Ensure pad token is set (common issue with decoder-only models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
