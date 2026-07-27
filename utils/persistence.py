"""
persistence.py — Full-text generation persistence and W&B artifact retrieval.

Self-Consistency (SC) is the only fresh generator compute in the L1 campaign:
Majority Voting and BoN-MAV reuse its per-generation text and metrics instead
of re-generating. The v2.x baselines only log a truncated results table and a
scalar-only ``.npz`` artifact — neither carries full response text — so this
module is the reuse vehicle: a JSONL artifact (one record per
``(problem_id, gen_id)``) plus the W&B API glue to retrieve it in a
downstream run with real producer -> consumer lineage.

Every persisted record's primary key is ``(dataset, problem_id, model_key,
gen_id)`` — all four human-readable, all four citable in the thesis via
``utils.registry.problem_ref``. Neither the dataset row's native id nor the
model's HF path is persisted per-record: both are recoverable, the first from
``(dataset, data_seed, eval_size)`` (already in the manifest — ``problem_id``
is a position in that seeded subset), the second from ``model_key`` via
``utils.registry.get_model``.

Connection to the project WPs
-----------------------------
* WP2-T4 (L1 multi-agent protocols): persists SC's full generations so MAJ and
  BoN-MAV can consume them without regenerating.
* WP3-T1: ``run.use_artifact`` records the exact producer version a consumer
  run used (Claim-to-Evidence).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import wandb  # type: ignore

GENERATIONS_ARTIFACT_TYPE = "sc-generations"
GENERATIONS_FILENAME = "generations.jsonl"
MANIFEST_FILENAME = "manifest.json"

# The full field set of a generation record. `_TABLE_COLUMNS` (see
# `generation_table_rows`) is a reordering of this set (PK columns first) —
# a test asserts the two can never drift apart.
RECORD_FIELDS = (
    "dataset", "problem_id", "model_key", "gen_id",
    "seed", "protocol",
    "task", "answer", "parsed_output", "correct",
    "latency_s", "input_tokens", "output_tokens", "total_tokens",
    "gpu_peak_mb", "tokens_per_s",
)

_TABLE_COLUMNS = [
    "dataset", "problem_id", "model_key", "gen_id",
    "seed", "task", "answer", "parsed_output", "correct",
    "latency_s", "output_tokens", "input_tokens", "total_tokens",
    "gpu_peak_mb", "tokens_per_s",
    "protocol",
]


class AlignmentError(RuntimeError):
    """Raised when a downloaded generation store doesn't match the locally
    loaded dataset subset (wrong dataset/model_key/data_seed/eval_size, or a
    task-text mismatch at some problem index)."""


def _normalize_task(text: str) -> str:
    """Collapse all whitespace runs to a single space.

    Used by :func:`validate_alignment` so a stored ``task`` and a freshly
    rendered one compare equal despite formatting-only differences.
    """
    return " ".join(text.split())


def make_generation_record(
    *,
    dataset: str,
    problem_id: int,
    model_key: str,
    gen_id: int,
    seed: Optional[int],
    protocol: str,
    task: str,
    sample: Mapping[str, Any],
    correct: bool,
    tokens_per_s: float,
    extra: Optional[dict] = None,
) -> dict:
    """Build one generation record with a consistent primary key and checksum.

    ``sample`` is an ``AgentSample`` (see ``utils.graphs``) — the generator's
    own output — from which ``answer``, ``parsed_output`` and the metric
    fields are pulled. ``validate_alignment`` compares the stored ``task``
    text directly against a fresh render, so producer and validator can
    never diverge. ``extra`` carries per-benchmark fields (``subject``,
    ``level``) not shared across datasets.
    """
    record = {
        "dataset": dataset,
        "problem_id": problem_id,
        "model_key": model_key,
        "gen_id": gen_id,
        "seed": seed,
        "protocol": protocol,
        "task": task,
        "answer": sample["answer"],
        "parsed_output": sample.get("parsed_output"),
        "correct": correct,
        "latency_s": sample["latency_s"],
        "input_tokens": sample["input_tokens"],
        "output_tokens": sample["output_tokens"],
        "total_tokens": sample["total_tokens"],
        "gpu_peak_mb": sample["gpu_peak_mb"],
        "tokens_per_s": tokens_per_s,
    }
    if extra:
        record.update(extra)
    return record


def assert_unique_pk(records: Sequence[Mapping[str, Any]]) -> None:
    """Raise ``ValueError`` if any two records share a primary key.

    The primary key is ``(dataset, problem_id, model_key, gen_id)``. Intended
    as an on-Kaggle invariant check after merging generation records across
    models/artifacts (e.g. the smoke test's merged record set), where a
    silent duplicate would otherwise corrupt a downstream store.
    """
    seen: set[tuple] = set()
    for record in records:
        pk = (record["dataset"], record["problem_id"], record["model_key"], record["gen_id"])
        if pk in seen:
            raise ValueError(f"Duplicate primary key {pk!r} in generation records")
        seen.add(pk)


def write_generations_jsonl(path: str | Path, records: list[dict]) -> None:
    """Write generation records as JSON Lines (one record per line)."""
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def read_generations_jsonl(path: str | Path) -> list[dict]:
    """Read generation records back from a JSON Lines file."""
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def log_generations_artifact(
    records: list[dict],
    manifest: dict[str, Any],
    *,
    name: str,
    run: Optional["wandb.sdk.wandb_run.Run"] = None,
) -> "wandb.Artifact":
    """Log a generations JSONL + manifest as a W&B artifact.

    Intended to be called once per completed generation seed pass — logging
    under the same ``name`` each time creates a new artifact *version*, so a
    Kaggle session killed partway through resumes from the last version
    instead of losing every prior seed's generations.

    Blocks on ``artifact.wait()`` before returning, so the caller can read
    the *actual* committed version off the returned artifact (via
    :func:`artifact_ref`) rather than computing an assumed ``:vN`` — the
    computed form silently repoints at the wrong version after any resume or
    partial rerun.

    Parameters
    ----------
    records : list[dict]
        Generation records (see the module docstring for the schema).
    manifest : dict
        Run/config metadata — see the SC plan's manifest schema.
    name : str
        Artifact name, typically ``utils.registry.generations_artifact_name``.
    run : wandb.Run | None
        The active run to log to. Defaults to ``wandb.run``.

    Returns
    -------
    wandb.Artifact
        The committed artifact, with ``.version`` populated.
    """
    run = run or wandb.run
    if run is None:
        raise RuntimeError(
            "log_generations_artifact requires an active W&B run "
            "(pass run=... or call this inside an active wandb.init session)"
        )

    artifact = wandb.Artifact(
        name,
        type=GENERATIONS_ARTIFACT_TYPE,
        description=(
            f"Full-text Self-Consistency generations for "
            f"{manifest.get('model_key', '?')} on {manifest.get('dataset', '?')}"
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gen_path = tmp_path / GENERATIONS_FILENAME
        manifest_path = tmp_path / MANIFEST_FILENAME
        write_generations_jsonl(gen_path, records)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        artifact.add_file(str(gen_path), name=GENERATIONS_FILENAME)
        artifact.add_file(str(manifest_path), name=MANIFEST_FILENAME)

    run.log_artifact(artifact)
    artifact.wait()  # blocks until committed — .version is only assigned after this
    assert artifact.version, f"artifact {name!r} did not commit a version"
    return artifact


def artifact_ref(artifact: "wandb.Artifact", *, entity: str, project: str) -> str:
    """The pinned ``entity/project/name:version`` reference for a committed artifact.

    Call only after :func:`log_generations_artifact` (i.e. after
    ``artifact.wait()``), so ``artifact.version`` is populated. Consumers must
    pin an explicit version (never ``:latest``) so the run config records
    exactly which producer version was used.
    """
    return f"{entity}/{project}/{artifact.name}:{artifact.version}"


def load_generations_artifact(
    artifact_ref: str,
    *,
    run: Optional["wandb.sdk.wandb_run.Run"] = None,
    api: Optional["wandb.Api"] = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Download and parse a generations artifact.

    Parameters
    ----------
    artifact_ref : str
        Explicit artifact reference, e.g.
        ``"entity/project/sc-generations-qwen3-1.7b-math500:v4"``. Pin an
        explicit version (not ``:latest``) in every real consumer run so the
        run config records exactly which producer version was used.
    run : wandb.Run | None
        If given, uses ``run.use_artifact(artifact_ref)`` so the download is
        recorded as W&B lineage on the consumer run (Claim-to-Evidence).
    api : wandb.Api | None
        If ``run`` is not given, used for a read-only download (no lineage
        recorded). Defaults to a fresh ``wandb.Api()``.

    Returns
    -------
    (records, manifest)
    """
    if run is not None:
        artifact = run.use_artifact(artifact_ref)
    else:
        api = api or wandb.Api()
        artifact = api.artifact(artifact_ref)

    download_dir = Path(artifact.download())
    records = read_generations_jsonl(download_dir / GENERATIONS_FILENAME)
    manifest = json.loads((download_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    return records, manifest


def build_generation_store(records: list[dict]) -> dict[tuple[int, int], dict]:
    """Index generation records by ``(problem_id, gen_id)``.

    Sound only within a single ``(dataset, model_key)`` — the 2-tuple key
    drops both, so mixing records from more than one dataset or model would
    silently overwrite unrelated problems' generations.

    Raises
    ------
    ValueError
        If records span more than one ``dataset`` or ``model_key``, or if the
        same ``(problem_id, gen_id)`` key appears more than once — a
        corrupted or double-logged artifact must not silently drop data.
    """
    store: dict[tuple[int, int], dict] = {}
    seen_dataset: Any = None
    seen_model_key: Any = None
    have_seen = False
    for record in records:
        dataset = record.get("dataset")
        model_key = record.get("model_key")
        if not have_seen:
            seen_dataset, seen_model_key = dataset, model_key
            have_seen = True
        elif dataset != seen_dataset or model_key != seen_model_key:
            raise ValueError(
                f"build_generation_store received records spanning multiple "
                f"(dataset, model_key) pairs — ({seen_dataset!r}, {seen_model_key!r}) "
                f"and ({dataset!r}, {model_key!r}); the (problem_id, gen_id) key is "
                "only sound within a single (dataset, model_key)"
            )
        key = (record["problem_id"], record["gen_id"])
        if key in store:
            raise ValueError(
                f"Duplicate generation record for "
                f"problem_id={key[0]}, gen_id={key[1]}"
            )
        store[key] = record
    return store


def validate_alignment(
    manifest: dict[str, Any],
    store: dict[tuple[int, int], dict],
    dataset: Any,
    *,
    render_task: Callable[[Any], str],
    expected_dataset: str,
    expected_model_key: str,
    expected_data_seed: int,
    expected_eval_size: int,
    n_generators: int,
) -> None:
    """Validate a downloaded generation store against the locally loaded subset.

    Checks (in order, raising on the first failure):

    1. The manifest's ``dataset`` / ``model_key`` match what the caller
       expects.
    2. The manifest's ``data_seed`` / ``eval_size`` match what the caller
       expects (i.e. the producer and consumer used the same seeded subset).
    3. Every ``(problem_id, gen_id)`` for ``problem_id`` in
       ``range(expected_eval_size)`` and ``gen_id`` in
       ``range(n_generators)`` is present in the store.
    4. Each stored record's ``task`` text (whitespace-normalized) matches
       ``render_task(dataset[problem_id])`` — guards against the producer
       and consumer silently disagreeing on the question ordering or
       rendering, and against dataset content drift.

    All checks raise :class:`AlignmentError`.

    Parameters
    ----------
    manifest : dict
        The producer's manifest (see :func:`load_generations_artifact`).
    store : dict
        Output of :func:`build_generation_store`.
    dataset : Sequence
        The locally loaded, seeded dataset subset (e.g. from
        ``load_math500`` / ``load_mmlu`` + the fixed ``random.sample``).
    render_task : Callable[[Any], str]
        Renders a dataset item into the exact task string the producer
        stored — typically ``BenchmarkSpec.render_task`` for the same
        benchmark, so producer and validator provably render identically.
    expected_dataset : str
        The dataset name the consumer expects the producer to have used.
    expected_model_key : str
        The model_key the consumer expects the producer to have used.
    expected_data_seed : int
        The data seed the consumer expects the producer to have used.
    expected_eval_size : int
        Number of problems expected to be covered.
    n_generators : int
        Number of generators (``gen_id`` values) expected per problem.
    """
    if manifest.get("dataset") != expected_dataset:
        raise AlignmentError(
            f"manifest dataset {manifest.get('dataset')!r} != "
            f"expected {expected_dataset!r}"
        )
    if manifest.get("model_key") != expected_model_key:
        raise AlignmentError(
            f"manifest model_key {manifest.get('model_key')!r} != "
            f"expected {expected_model_key!r}"
        )
    if manifest["data_seed"] != expected_data_seed:
        raise AlignmentError(
            f"manifest data_seed {manifest['data_seed']!r} != "
            f"expected {expected_data_seed!r}"
        )
    if manifest["eval_size"] != expected_eval_size:
        raise AlignmentError(
            f"manifest eval_size {manifest['eval_size']!r} != "
            f"expected {expected_eval_size!r}"
        )

    for problem_id in range(expected_eval_size):
        local_task = _normalize_task(render_task(dataset[problem_id]))
        for gen_id in range(n_generators):
            key = (problem_id, gen_id)
            if key not in store:
                raise AlignmentError(
                    f"missing generation record for problem_id={problem_id}, gen_id={gen_id}"
                )
            record = store[key]
            stored_task = _normalize_task(record["task"])
            if stored_task != local_task:
                raise AlignmentError(
                    f"task text mismatch at problem_id={problem_id}: "
                    f"local={local_task!r} stored={stored_task!r}"
                )


def generation_table_rows(
    records: list[dict], *, extra_columns: Sequence[str] = ()
) -> tuple[list[str], list[list[Any]]]:
    """Build ``(columns, rows)`` for the W&B generator table.

    Column order leads with the primary key (``dataset, problem_id,
    model_key, gen_id``) so the table can be joined back to
    ``build_generation_store``'s key by inspection.

    ``extra_columns`` appends per-benchmark fields not shared across
    datasets (``subject``, ``level``) — passed explicitly rather than baked
    into ``_TABLE_COLUMNS`` so, e.g., MMLU rows don't get an all-``None``
    ``level`` column.
    """
    columns = list(_TABLE_COLUMNS) + list(extra_columns)
    rows = [[record.get(col) for col in columns] for record in records]
    return columns, rows
