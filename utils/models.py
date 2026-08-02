"""
models.py: Model loading utilities.

Loads HF causal-LM (and text-capable multimodal) models within a single 16 GB
T4 GPU budget. The loaded model's linear weights are always 4-bit; where those
4-bit weights come from is selected by the `quantize_4bit` flag:

* `quantize_4bit=True` (default): the checkpoint ships in full precision and
  is quantised to NF4 on the fly via bitsandbytes. Used by every family except
  the pre-quantised Gemma-4 QAT checkpoints below.
* `quantize_4bit=False`: the checkpoint is already 4-bit on disk in the
  compressed-tensors w4a16 format (`google/gemma-4-{E2B,E4B}-it-qat-w4a16-ct`),
  restored by transformers via its compressed-tensors quantizer, no bitsandbytes.

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
    here for text only. Only modules actually present are dropped, so this is
    safe across families (e.g. Ministral-3 has no `audio_tower`).
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
        # Collect dropped modules before reclaiming VRAM.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _assert_4bit_weights(
    model: PreTrainedModel, model_name: str, quantize_4bit: bool
) -> None:
    """Fail loudly if the model did not load with 4-bit linear weights.

    A silent fallback to full precision (missing bitsandbytes, or an
    unrecognised compressed-tensors config) would OOM the T4. bitsandbytes
    wraps linear layers as `Linear4bit`; compressed-tensors as `CompressedLinear`.
    """
    if quantize_4bit:
        from bitsandbytes.nn import Linear4bit  # type: ignore

        quant_layer: type = Linear4bit
        hint = "check bitsandbytes install / restart kernel"
    else:
        from compressed_tensors.linear.compressed_linear import CompressedLinear  # type: ignore

        quant_layer = CompressedLinear
        hint = (
            "check that `compressed-tensors` is installed and the transformers "
            "version recognises this checkpoint's quantization_config"
        )

    if not any(isinstance(m, quant_layer) for m in model.modules()):
        raise RuntimeError(
            f"{model_name!r} loaded in full precision (dtype={model.dtype}), no "
            f"{quant_layer.__name__} layers found. This will OOM the T4; {hint}."
        )


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
        `True` NF4-quantises a full-precision checkpoint on load; `False`
        loads an already-4-bit compressed-tensors checkpoint as-is. Passing
        `True` for a pre-quantised checkpoint errors (cannot re-quantise).
    device_map : str
        Passed to `from_pretrained`. `"auto"` is forced to `{"": 0}` for
        multimodal families when CUDA is available; pass an explicit value
        to opt out.
    token : str | None
        HF access token; falls back to the `HF_TOKEN` env var.
    trust_remote_code : bool
        Whether to trust remote code in the model repo.

    Returns
    -------
    tuple[PreTrainedModel, PreTrainedTokenizer]
    """
    token = token or os.environ.get("HF_TOKEN")
    name_lower = model_name.lower()

    # QAT checkpoints are already 4-bit on disk; bitsandbytes would try to
    # re-quantise them, so require quantize_4bit=False for these.
    if quantize_4bit and name_lower.endswith("-qat-w4a16-ct"):
        raise ValueError(
            f"{model_name!r} is already 4-bit quantised (compressed-tensors w4a16) "
            "and cannot be re-quantised — call with quantize_4bit=False."
        )

    quant_config = None
    if quantize_4bit:
        # bitsandbytes NF4 path: quantise a full-precision checkpoint on load.
        from transformers.utils import is_bitsandbytes_available  # type: ignore

        # is_bitsandbytes_available is lru_cached and often first evaluated on
        # Kaggle before CUDA is live, freezing False and OOMing the T4 in BF16;
        # clear the cache to force a fresh evaluation here and in from_pretrained.
        try:
            is_bitsandbytes_available.cache_clear()  # type: ignore[attr-defined]
        except AttributeError:
            pass

        if not is_bitsandbytes_available():
            raise RuntimeError(
                "quantize_4bit=True but bitsandbytes is not available — "
                "transformers would silently fall back to full BF16 and OOM the T4. "
                "Run `pip install -U bitsandbytes` and restart the kernel, or pass "
                "quantize_4bit=False to load an already-quantised checkpoint."
            )

        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")  # synchronous load

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        # compressed-tensors path: the checkpoint carries its own quantization_config,
        # which transformers reads to restore the 4-bit weights; no config passed here.
        from transformers.utils import is_compressed_tensors_available  # type: ignore

        if not is_compressed_tensors_available():
            raise RuntimeError(
                "quantize_4bit=False expects an already-quantised checkpoint, but the "
                "`compressed-tensors` package is not installed. Run "
                "`pip install -U compressed-tensors` and restart the kernel."
            )

    multimodal = (
        "ministral" in name_lower or "gemma-4" in name_lower or "gemma4" in name_lower
    )

    # Pin multimodal checkpoints to one GPU: they load whole (vision/audio
    # towers included) then get stripped; "auto" can offload layers to CPU
    # permanently, crippling throughput with per-token PCIe transfers.
    effective_device_map = device_map
    if multimodal and device_map == "auto" and torch.cuda.is_available():
        effective_device_map = {"": 0}

    # float16 for both paths: the T4 has no native bf16. Matches the bitsandbytes
    # compute dtype and overrides compressed-tensors' declared bf16.
    # `dtype` is the transformers>=5 kwarg name.
    load_kwargs = dict(
        quantization_config=quant_config,
        device_map=effective_device_map,
        dtype=torch.float16,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    if multimodal:
        model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        _strip_vision_modules(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Warn if any layer was offloaded off-GPU: generation would slow to single
    # digits from per-token PCIe transfers, not because the model itself is slow.
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

    # Guarantee the weights loaded in 4-bit (scanning for the quantised
    # layer type is more reliable than transformers flags across versions).
    _assert_4bit_weights(model, model_name, quantize_4bit)

    tokenizer_kwargs: dict = dict(token=token, trust_remote_code=trust_remote_code)
    if "ministral" in name_lower:
        tokenizer_kwargs["tokenizer_type"] = "mistral"
    elif "gemma-4" in name_lower or "gemma4" in name_lower:
        tokenizer_kwargs["padding_side"] = "left"

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
