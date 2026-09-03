"""Shared utilities for the TFG Agentic AI experiments."""

__all__ = ["load_env", "warmup_model"]


def __getattr__(name: str):
    """Load optional convenience exports only when they are requested."""
    if name == "load_env":
        from utils.config import load_env

        return load_env
    if name == "warmup_model":
        from utils.warmup import warmup_model

        return warmup_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
