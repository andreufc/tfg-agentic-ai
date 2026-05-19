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
    AutoModelForCausalLM,
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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map=device_map,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    # Ensure pad token is set (common issue with decoder-only models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
