"""
ensembles.py: ensemble roster and slot sampling for Majority Voting (MAJ)
and BoN-MAV, shared by both protocols.

Imports MODELS/get_model from utils.registry rather than redefining model
identity: this module only combines registered models into ensembles and
samples which SC generation each slot replays. Also provides the
ensemble-scoped run/artifact naming helpers utils.registry deliberately
doesn't have, since its matrices_artifact_name takes a single model_key,
which doesn't fit a 3-5-model ensemble.

Connection to the project WPs
------------------------------
* WP2-T4 (L1 multi-agent protocols): the ensemble roster (E1-E7) and the
  keyed, randomly-addressable slot sampler that both MAJ and BoN-MAV draw
  candidates from.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from utils.registry import MATRICES_ARTIFACT_PREFIX, _check_dataset, get_model

SLOT_SAMPLE_SEED = 3407


@dataclass(frozen=True)
class Ensemble:
    ensemble_id: str
    slots: tuple[str, ...]  # model_key per slot, in ensemble order
    label: str

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError(f"Ensemble {self.ensemble_id!r} must have at least one slot")
        for model_key in self.slots:
            get_model(model_key)  # validates every slot is a real MODELS key


# E1-E5 are the MAJ ensembles; E2/E6/E7 are BoN-MAV's candidate ensembles
# (E2 shared with MAJ for the paired protocol comparison; E6/E7 are the
# self-improvement same-model-x3 ensembles).
E1 = Ensemble("E1", ("qwen3-1.7b", "ministral-3-3b", "phi-4-mini"), "E1 (Qwen3-1.7B + Ministral-3-3B + Phi-4-mini)")
E2 = Ensemble("E2", ("qwen3-4b", "ministral-3-3b", "phi-4-mini"), "E2 (Qwen3-4B + Ministral-3-3B + Phi-4-mini)")
E3 = Ensemble("E3", ("qwen3-1.7b", "qwen3-4b", "phi-4-mini"), "E3 (Qwen3-1.7B + Qwen3-4B + Phi-4-mini)")
E4 = Ensemble(
    "E4", ("qwen3-1.7b", "qwen3-4b", "ministral-3-3b", "phi-4-mini", "qwen3-1.7b"),
    "E4 (all four + 2nd Qwen3-1.7B)",
)
E5 = Ensemble(
    "E5", ("qwen3-1.7b", "qwen3-4b", "ministral-3-3b", "phi-4-mini", "qwen3-4b"),
    "E5 (all four + 2nd Qwen3-4B)",
)
E6 = Ensemble("E6", ("ministral-3-3b",) * 3, "E6 (Ministral-3-3B x3, self-improvement)")
E7 = Ensemble("E7", ("phi-4-mini",) * 3, "E7 (Phi-4-mini x3, self-improvement)")

MAJ_ENSEMBLES: tuple[Ensemble, ...] = (E1, E2, E3, E4, E5)
BONMAV_ENSEMBLES: tuple[Ensemble, ...] = (E2, E6, E7)

_ALL_ENSEMBLES: dict[str, Ensemble] = {ens.ensemble_id: ens for ens in (E1, E2, E3, E4, E5, E6, E7)}


def get_ensemble(ensemble_id: str) -> Ensemble:
    """Look up an :class:`Ensemble` by its id (e.g. "E2")."""
    try:
        return _ALL_ENSEMBLES[ensemble_id]
    except KeyError:
        raise ValueError(
            f"Unknown ensemble_id {ensemble_id!r}; expected one of {sorted(_ALL_ENSEMBLES)}"
        ) from None


def slot_agent_ids(ens: Ensemble) -> tuple[str, ...]:
    """One unique agent_id per slot, e.g. ("s0-qwen3-1.7b", "s1-ministral-3-3b",
    ..., "s3-qwen3-1.7b") for E4's duplicated model. Required because
    build_majority_voting_graph/build_bon_mav_graph raise on a duplicate
    agent_id.
    """
    return tuple(f"s{i}-{model_key}" for i, model_key in enumerate(ens.slots))


def _distinct_model_keys_in_order(ens: Ensemble) -> tuple[str, ...]:
    seen: list[str] = []
    for model_key in ens.slots:
        if model_key not in seen:
            seen.append(model_key)
    return tuple(seen)


def sample_slots(
    ens: Ensemble,
    *,
    dataset: str,
    problem_id: int,
    rep: int,
    n_gens_by_model: int = 5,
    sample_seed: int = SLOT_SAMPLE_SEED,
) -> tuple[tuple[str, int], ...]:
    """Return one (model_key, gen_id) per slot, in ens.slots order.

    Fixed a priori, no cherry-picking: the draw key is
    sha256(f"{sample_seed}:{dataset}:{ensemble_id}:{problem_id}:{rep}"),
    truncated to a 64-bit int seed for random.Random(...). The key
    deliberately excludes the verifier, so BoN-MAV reusing this function
    with rep=0 gets candidate sets identical across verifier configurations
    and identical to MAJ's own rep-0 draw for the same ensemble_id.

    For each distinct model_key in the ensemble, its slot count of gen_ids
    is drawn without replacement from range(n_gens_by_model); distinct
    models are iterated in the order they first appear in ens.slots, so the
    draw never depends on dict/set ordering. A model occupying two slots
    (E4, E5) gets two different generations. Reps are independent draws
    (they may coincide by chance) by design: the 2 MAJ reps estimate the
    ensemble's own sampling variance.
    """
    key = f"{sample_seed}:{dataset}:{ens.ensemble_id}:{problem_id}:{rep}"
    seed_int = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**64)
    rng = random.Random(seed_int)

    draws: dict[str, list[int]] = {
        model_key: rng.sample(range(n_gens_by_model), ens.slots.count(model_key))
        for model_key in _distinct_model_keys_in_order(ens)
    }

    cursor = {model_key: 0 for model_key in draws}
    result: list[tuple[str, int]] = []
    for model_key in ens.slots:
        gen_id = draws[model_key][cursor[model_key]]
        cursor[model_key] += 1
        result.append((model_key, gen_id))
    return tuple(result)


# ---------------------------------------------------------------------------
# Naming helpers: ensemble-scoped, mirroring utils.registry's own style
# ---------------------------------------------------------------------------

def maj_run_name(ensemble_id: str, dataset: str) -> str:
    """W&B run name for a Majority Voting run, e.g. "l1-maj3-E1-math500"."""
    ens = get_ensemble(ensemble_id)
    _check_dataset(dataset)
    return f"l1-maj{len(ens.slots)}-{ensemble_id}-{dataset}"


def maj_matrices_artifact_name(ensemble_id: str, dataset: str) -> str:
    """Canonical name for a MAJ evaluation-matrices .npz artifact."""
    ens = get_ensemble(ensemble_id)
    _check_dataset(dataset)
    return f"{MATRICES_ARTIFACT_PREFIX}-maj{len(ens.slots)}-{ensemble_id}-{dataset}"


def bonmav_run_name(verifier_key: str, ensemble_id: str, subset_id: str, dataset: str) -> str:
    """W&B run name for a BoN-MAV run, e.g. "l1-bonmav-qwen3-1.7b-E2-MATH-S1-math500"."""
    get_model(verifier_key)
    get_ensemble(ensemble_id)
    _check_dataset(dataset)
    return f"l1-bonmav-{verifier_key}-{ensemble_id}-{subset_id}-{dataset}"


def bonmav_matrices_artifact_name(verifier_key: str, ensemble_id: str, subset_id: str, dataset: str) -> str:
    """Canonical name for a BoN-MAV evaluation-matrices .npz artifact."""
    get_model(verifier_key)
    get_ensemble(ensemble_id)
    _check_dataset(dataset)
    return f"{MATRICES_ARTIFACT_PREFIX}-bonmav-{verifier_key}-{ensemble_id}-{subset_id}-{dataset}"
