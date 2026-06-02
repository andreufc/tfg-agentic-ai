"""
instrumentation.py — Token counting and GPU resource tracking.

Provides per-step helpers for:
- Token counting (input and output tokens via the model's own tokenizer)
- GPU peak VRAM measurement (torch.cuda, deterministic, zero overhead)
- GPU utilization % sampling (pynvml background thread, ~100 ms polling)
- Generation throughput (output tokens / generation latency)

All GPU helpers degrade gracefully to 0.0 when CUDA is unavailable
(e.g. local CPU development). The pynvml sampler likewise degrades
if nvidia-ml-py is not installed — a RuntimeWarning is issued only
when a CUDA GPU is present but pynvml is missing.

Design notes
------------
- Token counting is intentionally performed *outside* the timed
  ``chain.invoke`` region so latency measurements remain
  uncontaminated (see ``run_agent_step`` in the notebooks).
- ``count_tokens`` uses ``add_special_tokens=False`` for consistency
  across all agents — chat-template wrapping may add 1–2 tokens vs
  what the pipeline actually feeds the model, but both MAS and
  baseline are measured identically so relative comparisons are sound.
- Peak VRAM must be reset with ``reset_gpu_peak()`` immediately before
  each measured scope; otherwise the counter reflects the global max
  since the last manual reset (which includes model load).
- The pynvml sampler polls at ≥100 ms intervals; the daemon thread
  holds the GIL only briefly per sample (NVML query ≈ 0.1–1 ms), so
  GIL contention during GPU-bound generation is negligible.
- ``tokens_per_second`` measures end-to-end throughput (prefill +
  decode), not pure decode rate. Both MAS and baseline are measured
  identically so the ratio is a sound efficiency comparison.
"""

from __future__ import annotations

import os
import threading
import time
import warnings
from typing import Any, Optional, Union

import torch


# ---------------------------------------------------------------------------
# pynvml: optional import — graceful degradation
# ---------------------------------------------------------------------------

try:
    import pynvml  # type: ignore
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Device resolution — measure only the GPU the model actually uses
# ---------------------------------------------------------------------------
#
# On multi-GPU kernels (e.g. Kaggle's 2× T4) the model is typically pinned to
# a single device, so all measurements must target *that* device rather than
# a hardcoded index 0. Two distinct index spaces are involved:
#
#   * the *torch* index   — what torch.cuda / a model's parameters live on,
#                           filtered and renumbered by CUDA_VISIBLE_DEVICES;
#   * the *NVML* index    — the physical device order pynvml addresses.
#
# These coincide only when CUDA_VISIBLE_DEVICES is unset and the CUDA/NVML
# orderings match. We therefore resolve the torch device first, then map it to
# the matching NVML handle by UUID (robust to any reordering), falling back to
# a CUDA_VISIBLE_DEVICES translation and finally to the raw index.

DeviceLike = Union[int, str, "torch.device", None]


def _resolve_torch_index(device: DeviceLike = None) -> Optional[int]:
    """Return the torch CUDA device index to measure.

    ``None`` -> the current active device (where the model most likely is).
    Returns ``None`` when CUDA is unavailable.
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


def _nvml_handle_for_torch_index(torch_index: int):
    """Map a torch device index to the matching NVML device handle.

    Prefers exact UUID matching (correct regardless of CUDA/NVML ordering or
    CUDA_VISIBLE_DEVICES). Falls back to translating through
    CUDA_VISIBLE_DEVICES, then to the raw index. Returns ``None`` on failure.
    """
    # 1) UUID match — authoritative.
    try:
        torch_uuid = str(torch.cuda.get_device_properties(torch_index).uuid).replace("GPU-", "")
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            nvml_uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(nvml_uuid, bytes):
                nvml_uuid = nvml_uuid.decode()
            if nvml_uuid.replace("GPU-", "") == torch_uuid:
                return handle
    except Exception:
        pass

    # 2) Translate the torch index through CUDA_VISIBLE_DEVICES.
    physical_index = torch_index
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        visible = [x.strip() for x in cvd.split(",") if x.strip() != ""]
        try:
            physical_index = int(visible[torch_index])
        except (IndexError, ValueError):
            physical_index = torch_index

    # 3) Raw index lookup.
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(physical_index)
    except Exception:
        return None


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
        Any string — a rendered prompt or a model completion.

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
# GPU peak VRAM  (torch.cuda — deterministic, microsecond overhead)
# ---------------------------------------------------------------------------

def reset_gpu_peak(device: DeviceLike = None) -> None:
    """Reset the CUDA peak-memory counter for the measured device.

    Call this immediately before the code block whose peak VRAM you
    want to measure. ``device`` defaults to the active device (where the
    model lives); pass an explicit index/str/torch.device to override.
    No-op when CUDA is unavailable.
    """
    idx = _resolve_torch_index(device)
    if idx is not None:
        torch.cuda.reset_peak_memory_stats(idx)


def gpu_peak_mem_mb(device: DeviceLike = None) -> float:
    """Return the peak GPU memory allocated since the last reset.

    Parameters
    ----------
    device : int | str | torch.device | None
        The CUDA device to read. ``None`` (default) reads the active
        device, i.e. the one the model is loaded on — important on
        multi-GPU kernels where only one device is in use.

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
# GPU utilization sampler  (pynvml background thread)
# ---------------------------------------------------------------------------

class GpuUtilSampler:
    """Background daemon thread that polls GPU utilization via pynvml.

    Polls nvmlDeviceGetUtilizationRates every *interval_s* seconds
    (floored at 100 ms to keep GIL contention negligible).
    Call start() before the code block to measure and stop()
    after it to retrieve aggregated stats.

    Graceful degradation
    --------------------
    - No CUDA: sampler is a no-op; stop() returns all-zero dict.
    - CUDA present but pynvml missing: a one-time RuntimeWarning is
      issued and the sampler is a no-op (peak VRAM is still tracked).
    - pynvml error mid-run: sampling stops silently; samples collected
      so far are used for aggregation.

    Notes
    -----
    - Utilization is **whole-device** — not process-isolated. On shared
      Colab / Kaggle GPUs another tenant could theoretically inflate
      the reading, though in practice you are alone on the GPU.
    - On multi-GPU kernels (e.g. Kaggle's 2× T4) only the device the
      model is loaded on is sampled. ``device`` defaults to the active
      torch device and is mapped to the correct NVML handle by UUID, so
      the idle second GPU is never measured by mistake.
    - Very short execution blocks (e.g. a fast baseline) yield fewer
      samples, making gpu_util_mean noisier for that block.
    - Absolute latencies in notebooks using this sampler include a
      sub-percent overhead.
    """

    def __init__(self, interval_s: float = 0.1, device: DeviceLike = None) -> None:
        self._interval_s = max(interval_s, 0.1)   # floor at 100 ms
        self._device = device                     # resolved at start()
        self._samples: list[tuple[float, float]] = []  # (gpu_util%, mem_util%)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- public interface ----

    def start(self) -> "GpuUtilSampler":
        """Start the background polling thread."""
        self._samples = []
        self._stop_event.clear()
        if torch.cuda.is_available():
            if _NVML_AVAILABLE:
                self._thread = threading.Thread(target=self._poll, daemon=True)
                self._thread.start()
            else:
                warnings.warn(
                    "CUDA GPU detected but pynvml is not available. "
                    "Install nvidia-ml-py for per-step GPU utilization sampling. "
                    "Peak VRAM is still tracked via torch.cuda.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return self

    def stop(self) -> dict[str, float]:
        """Stop the sampler and return aggregated utilization stats.

        Returns
        -------
        dict with keys:
            gpu_util_mean  — mean GPU compute utilization (%)
            gpu_util_max   — peak GPU compute utilization (%)
            mem_util_mean  — mean GPU memory bandwidth utilization (%)
            n_samples      — number of samples collected
        All numeric values are 0.0 / 0 when pynvml is unavailable.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if not self._samples:
            return {
                "gpu_util_mean": 0.0,
                "gpu_util_max":  0.0,
                "mem_util_mean": 0.0,
                "n_samples":     0,
            }

        gpu_vals = [s[0] for s in self._samples]
        mem_vals = [s[1] for s in self._samples]
        return {
            "gpu_util_mean": round(sum(gpu_vals) / len(gpu_vals), 2),
            "gpu_util_max":  round(max(gpu_vals), 2),
            "mem_util_mean": round(sum(mem_vals) / len(mem_vals), 2),
            "n_samples":     len(self._samples),
        }

    # ---- context manager support ----

    def __enter__(self) -> "GpuUtilSampler":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ---- internal ----

    def _poll(self) -> None:
        try:
            torch_index = _resolve_torch_index(self._device)
            handle = _nvml_handle_for_torch_index(torch_index)
            if handle is None:
                return  # could not resolve the device — degrade to no samples
            while not self._stop_event.is_set():
                rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self._samples.append((float(rates.gpu), float(rates.memory)))
                time.sleep(self._interval_s)
        except Exception:
            pass  # degrade silently; samples collected so far are preserved


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------

def tokens_per_second(output_tokens: int, elapsed_s: float) -> float:
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
        Wall-clock seconds of the full ``chain.invoke`` call.

    Returns
    -------
    float
        Tokens per second (0.0 if elapsed_s ≈ 0).
    """
    if elapsed_s < 1e-9:
        return 0.0
    return round(output_tokens / elapsed_s, 4)
