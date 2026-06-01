"""
utils — Shared utilities for the TFG Agentic AI project.

This package provides helper modules for:
- config:           Environment & secrets loading
- data:             Dataset loading and preprocessing (GSM8K, MMLU)
- models:           LLM / SLM loading with quantisation support
- evaluation:       Metric computation and result aggregation
- agents:           LangChain / LangGraph agent construction helpers
- logging_utils:    W&B experiment tracking wrappers
- statistics:       Cluster-robust statistical estimators for experiments
- instrumentation:  Token counting, GPU peak VRAM / utilization sampling,
                    and generation throughput helpers (v1.2.x+)
"""

from utils.config import load_env
