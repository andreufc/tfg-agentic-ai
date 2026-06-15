"""
models.py — Model loading utilities.

Helpers to load HF causal-LM (and text-capable multimodal) models with optional
NF4 4-bit quantisation via bitsandbytes, targeting a single 16 GB T4 GPU.

Loading is dispatched per model family (see ``load_model_and_tokenizer``):
Ministral-3 and Gemma-4 (E2B/E4B) ship as multimodal checkpoints and need a
bespoke loader (load via the image-text class, then drop the vision/audio
encoders for text-only use); every other in-scope family (Phi-4, Qwen3) is a
plain causal LM. New families get their own branch as the need arises.
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
    """Drop the vision/audio encoders of a multimodal text-capable checkpoint.

    Both Ministral-3 (``Mistral3``) and Gemma-4 E2B/E4B ship as multimodal
    checkpoints, but the project uses them for text only.

    Only modules actually present on the checkpoint are dropped; absent ones are
    skipped, so this is safe to call on any of the multimodal families (e.g.
    Ministral-3 has no ``audio_tower``).
    """
    inner = getattr(model, "model", None)
    if inner is None:
        return
    freed = False
    for attr in (
        # Mistral 3 projector name
        "multi_modal_projector",
        # Common to both Mistral 3 and Gemma 4
        "vision_tower",
        "audio_tower",
        # Gemma 4 multimodal embedding projectors (replace multi_modal_projector)
        "embed_vision",
        "embed_audio",
    ):
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

        # Force synchronous weight loading. 
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

    name_lower = model_name.lower()
    if "ministral" in name_lower:
        model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        _strip_vision_modules(model)
    elif "gemma-4" in name_lower or "gemma4" in name_lower:
        model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        _strip_vision_modules(model)
    else:
        # Plain causal LM (Phi-4, Qwen3, ...).
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

    tokenizer_kwargs: dict = dict(token=token, trust_remote_code=trust_remote_code)
    if "ministral" in model_name.lower():
        tokenizer_kwargs["tokenizer_type"] = "mistral"
    elif "gemma-4" in model_name.lower() or "gemma4" in model_name.lower():
        tokenizer_kwargs["padding_side"] = "left"

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
