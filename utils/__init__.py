"""
utils — Shared utilities for the TFG Agentic AI project.

Helper modules and the project tasks each is used in (see each module's
"Connection to the project WPs" header for how):
- config:           secrets loading — WP1-T1/T4, WP2-T1/T3/T4/T6
- data:             GSM8K/MMLU loading & preprocessing — WP2-T1, WP1-T1, WP2-T3/T4/T6
- models:           SLM loading with NF4 quantisation — WP1-T1/T4, WP2-T3/T4/T6
- agents:           LangChain/LangGraph agent construction — WP1-T1, WP2-T3/T4/T6
- evaluation:       metrics & correctness matrices — WP1-T1, WP2-T3/T4/T6 -> T5/T7, WP3-T1
- instrumentation:  token/VRAM/util/throughput capture — WP1-T1/T4, WP2-T3/T4/T6 -> WP3-T1
- warmup:           pre-measurement kernel warmup — WP1-T1/T4, WP2-T3/T4/T6
- statistics:       cluster-robust estimators & tests — WP2-T5/T7, WP3-T1 (+ WP1-T1, WP2-T3)
- logging_utils:    W&B experiment tracking — WP1-T1/T4, WP2-T3/T4/T6 -> WP3-T1
"""

from utils.config import load_env
from utils.warmup import warmup_model
