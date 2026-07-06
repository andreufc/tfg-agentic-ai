"""
models.py — Model loading utilities.

Loads HF causal-LM (and text-capable multimodal) models with optional NF4 4-bit
quantisation via bitsandbytes, targeting a single 16 GB T4 GPU.

Loading is dispatched per family: Ministral-3 and Gemma-4 (E2B/E4B) ship as
multimodal checkpoints (loaded via the image-text class, then stripped of their
vision/audio encoders for text-only use); Phi-4 and Qwen3 are plain causal LMs.

Connection to the project WPs
-----------------------------
* WP1-T1/T4 & WP2-T3/T4/T6: loads each SLM per-family in NF4 4-bit within the
  16 GB T4 budget, incl. per-agent models in heterogeneous protocol configs.
"""

from __future__ import annotations

import gc
import os
import warnings
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

    Ministral-3 and Gemma-4 E2B/E4B ship as multimodal checkpoints but are used
    here for text only. Only modules actually present are dropped, so this is safe
    across families (e.g. Ministral-3 has no ``audio_tower``).
    """
    inner = getattr(model, "model", None)
    if inner is None:
        return
    freed = False
    for attr in (
        "multi_modal_projector",  # Mistral 3 projector
        "vision_tower",           # Mistral 3 + Gemma 4
        "audio_tower",            # Mistral 3 + Gemma 4
        "embed_vision",           # Gemma 4 multimodal embedding projectors
        "embed_audio",
    ):
        if getattr(inner, attr, None) is not None:
            setattr(inner, attr, None)
            freed = True
    if freed:
        # Collect the dropped modules before reclaiming their VRAM.
        gc.collect()
        if torch.cuda.is_available():
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
        Placement passed to ``from_pretrained``. The default ``"auto"`` is forced
        to ``{"": 0}`` (single visible GPU) for the multimodal families when CUDA
        is available; pass an explicit value to opt out.
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

        # is_bitsandbytes_available is lru_cached and on Kaggle is often first
        # called before CUDA is live, freezing False for the session — which makes
        # transformers silently skip quantisation and OOM the T4 in full BF16.
        # Clear the cache to force a fresh evaluation here and in from_pretrained.
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

        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")  # synchronous load

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    name_lower = model_name.lower()
    multimodal = (
        "ministral" in name_lower or "gemma-4" in name_lower or "gemma4" in name_lower
    )

    # Pin multimodal checkpoints to the single GPU. They are loaded whole (vision
    # + audio towers included) and stripped afterwards; under "auto" accelerate
    # budgets for those towers and can offload language layers to CPU, which then
    # stay there and cripple throughput with per-token PCIe transfers.
    effective_device_map = device_map
    if multimodal and device_map == "auto" and torch.cuda.is_available():
        effective_device_map = {"": 0}

    load_kwargs = dict(
        quantization_config=quant_config,
        device_map=effective_device_map,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    if quantize_4bit:
        # Keep non-quantized modules (norms, token + Gemma-4 per-layer embeddings,
        # which bitsandbytes never touches) in fp16 to match the compute dtype and
        # avoid a per-version fp32 fallback. ``dtype`` is the transformers>=5 name.
        load_kwargs["dtype"] = torch.float16

    if multimodal:
        model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        _strip_vision_modules(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Warn if any layer was offloaded off-GPU — generation would slow to single
    # digits from per-token PCIe transfers rather than the model being "slow".
    hf_map = getattr(model, "hf_device_map", None)
    if hf_map:
        offloaded = sorted(
            name
            for name, dev in hf_map.items()
            if str(dev) in ("cpu", "disk") or dev == torch.device("cpu")
        )
        if offloaded:
            placements = {n: str(hf_map[n]) for n in offloaded}
            warnings.warn(
                f"Model has layers offloaded off-GPU ({placements}) — generation "
                "will be drastically slower. Pass device_map={'': 0} to pin the "
                "model to the GPU, or reduce memory pressure.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Confirm NF4 engaged by scanning for bitsandbytes' Linear4bit layers (more
    # reliable than transformers flags across versions).
    if quantize_4bit:
        from bitsandbytes.nn import Linear4bit  # type: ignore

        if not any(isinstance(m, Linear4bit) for m in model.modules()):
            raise RuntimeError(
                f"Model loaded in full precision (dtype={model.dtype}) despite "
                "quantize_4bit=True — no Linear4bit layers found. "
                "This will OOM the T4. Check bitsandbytes install / restart kernel."
            )

    tokenizer_kwargs: dict = dict(token=token, trust_remote_code=trust_remote_code)
    if "ministral" in name_lower:
        tokenizer_kwargs["tokenizer_type"] = "mistral"
    elif "gemma-4" in name_lower or "gemma4" in name_lower:
        tokenizer_kwargs["padding_side"] = "left"

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
