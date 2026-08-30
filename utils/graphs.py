"""
graphs.py - LangGraph orchestration templates for the L1 protocols.

One graph template per protocol. build_majority_voting_graph serves
Self-Consistency in two modes: a live generation pass (one AgentSpec per
seed, k=1) and a replay aggregation pass (one compiled graph per SC-ladder
subset, voting over that subset's own fanned-in samples). Majority Voting
later calls the same builder with replay specs sourced from downloaded
artifacts, so it is the same code path, not a parallel implementation.
build_bon_mav_graph is a separate template since BoN-MAV's topology
(candidates, verifiers, selection) differs from a plain vote.

The aggregate node groups and votes over its own fanned-in samples via
utils.voting, so every reported number is one the graph itself computed.
The ground-truth answer never enters graph state; correctness is always
computed by the caller after invoke() returns.

Connection to the project WPs
-----------------------------
* WP2-T4: the LangGraph implementation of the L1 majority-voting protocol,
  homogeneous (SC) or heterogeneous (MAJ) via the same AgentSpec list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Callable, Literal, Mapping, Optional, Sequence, TypedDict
import operator

from langgraph.graph import END, START, StateGraph  # type: ignore
from langgraph.graph.state import CompiledStateGraph  # type: ignore

from utils.voting import (
    BenchmarkSpec, VoteResult, compose_generation_metrics, group_generations, majority_vote,
)


class AgentSample(TypedDict):
    agent_id: str
    model_key: str
    model_id: str
    source: Literal["live", "replay"]
    gen_id: int
    seed: Optional[int]
    answer: str
    parsed_output: Optional[str]
    latency_s: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    gpu_peak_mb: float


class VotingState(TypedDict):
    task: str
    problem_id: int
    dataset: str
    samples: Annotated[list[AgentSample], operator.add]
    vote: Optional[VoteResult]
    metrics: Optional[dict]


@dataclass(frozen=True)
class AgentSpec:
    """One graph node's identity and data source.

    mode="live" calls generate(task, seed), an instrumented closure built
    in the notebook (wraps build_hf_langchain_llm plus the measurement
    helpers in utils.instrumentation). mode="replay" looks up
    store[(problem_id, gen_id)], a utils.persistence.build_generation_store
    output: either the in-session dict (SC's own aggregation pass) or a
    downloaded artifact (Majority Voting).

    model_key is the registry slug carried onto every emitted AgentSample
    (the model's persisted identity). model_id is the HF repo id, needed
    operationally by "live" mode to load the model; it is not itself
    persisted per-record, only recovered from model_key via the registry
    at notebook config time.
    """

    agent_id: str
    model_key: str
    model_id: str
    mode: Literal["live", "replay"]
    generate: Optional[Callable[[str, Optional[int]], AgentSample]] = None
    seed: Optional[int] = None
    store: Optional[Mapping[tuple[int, int], dict]] = None
    gen_id: int = 0

    def __post_init__(self) -> None:
        if self.mode == "live" and self.generate is None:
            raise ValueError(f"AgentSpec {self.agent_id!r}: mode='live' requires `generate`")
        if self.mode == "replay" and self.store is None:
            raise ValueError(f"AgentSpec {self.agent_id!r}: mode='replay' requires `store`")


def _make_gen_node(spec: AgentSpec) -> Callable[[VotingState], dict]:
    if spec.mode == "live":
        generate = spec.generate
        seed = spec.seed

        def live_node(state: VotingState) -> dict:
            # generate() is the notebook's own closure, already bound to this
            # spec's identity, and returns a complete AgentSample.
            sample = generate(state["task"], seed)  # type: ignore[misc]
            return {"samples": [sample]}

        return live_node

    store = spec.store
    gen_id = spec.gen_id
    agent_id = spec.agent_id
    model_key = spec.model_key
    model_id = spec.model_id

    def replay_node(state: VotingState) -> dict:
        record = store[(state["problem_id"], gen_id)]  # type: ignore[index]
        sample: AgentSample = {
            "agent_id": agent_id,
            "model_key": model_key,
            "model_id": model_id,
            "source": "replay",
            "gen_id": gen_id,
            "seed": record.get("seed"),
            "answer": record["answer"],
            "parsed_output": record.get("parsed_output"),
            "latency_s": record["latency_s"],
            "input_tokens": record["input_tokens"],
            "output_tokens": record["output_tokens"],
            "total_tokens": record["total_tokens"],
            "gpu_peak_mb": record["gpu_peak_mb"],
        }
        return {"samples": [sample]}

    return replay_node


def build_majority_voting_graph(
    agent_specs: Sequence[AgentSpec],
    benchmark_spec: BenchmarkSpec,
    *,
    rng,
    vram_rule: Literal["sum_per_model", "coresident_peak"] = "sum_per_model",
) -> CompiledStateGraph:
    """Build the Majority Voting graph: START -> gen_* -> aggregate -> END.

    One <agent_id> node per spec (live or replay), called sequentially
    in-process via synchronous invoke() (a caller may still pre-load every
    agent's model together first for a live, co-resident ensemble).
    aggregate fans in samples, votes, and composes metrics.

    Parameters
    ----------
    agent_specs : sequence of AgentSpec
        One per graph node. Order determines node names and, since sample
        fan-in follows node-addition order under a synchronous invoke(),
        the index a vote's winner_idx refers to.
    benchmark_spec : BenchmarkSpec
        The benchmark's extraction/equivalence rule (e.g.
        utils.voting.MATH500_SPEC), used to group this graph's own
        fanned-in samples.
    rng : random.Random
        Shared across every vote in the run; see utils.voting.majority_vote
        and utils.voting.run_self_consistency, which construct one instance
        and thread it through every graph they compile.
    vram_rule : {"sum_per_model", "coresident_peak"}
        Forwarded to utils.voting.compose_generation_metrics. Keep the
        default for replayed/sequential runs (one model resident per
        call). Pass "coresident_peak" when every agent_specs model was
        pre-loaded together live, or VRAM over-counts.

    Returns
    -------
    CompiledStateGraph
        Invoke once per problem with {"task": ..., "problem_id": ...,
        "dataset": ..., "samples": [], "vote": None, "metrics": None}.
    """
    if not agent_specs:
        raise ValueError("build_majority_voting_graph requires at least one AgentSpec")

    graph = StateGraph(VotingState)

    node_names = []
    seen_agent_ids: set[str] = set()
    for agent_spec in agent_specs:
        if agent_spec.agent_id == "aggregate":
            raise ValueError("agent_id 'aggregate' is reserved for the vote-aggregation node")
        if agent_spec.agent_id in seen_agent_ids:
            raise ValueError(
                f"Duplicate agent_id {agent_spec.agent_id!r} — LangGraph node names "
                "must be unique; give each AgentSpec a distinct agent_id"
            )
        seen_agent_ids.add(agent_spec.agent_id)
        node_name = agent_spec.agent_id
        node_names.append(node_name)
        graph.add_node(node_name, _make_gen_node(agent_spec))
        graph.add_edge(START, node_name)

    def aggregate(state: VotingState) -> dict:
        samples = state["samples"]
        answers = [sample["answer"] for sample in samples]
        labels, predictions = group_generations(answers, benchmark_spec)
        vote = majority_vote(labels, rng=rng)
        if vote["winner_idx"] is not None:
            vote["winner_pred"] = predictions[vote["winner_idx"]]
        metrics = compose_generation_metrics(samples, vram_rule=vram_rule)
        return {"vote": vote, "metrics": metrics}

    graph.add_node("aggregate", aggregate)
    for node_name in node_names:
        graph.add_edge(node_name, "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# BoN-MAV: a separate template (candidates + verifiers + selection, not a
# plain vote).
# ---------------------------------------------------------------------------

class Approval(TypedDict):
    candidate_idx: int
    aspect_id: str
    verdict: Optional[bool]  # None = unparseable; scored as an approval, not a rejection
    raw_output: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    gpu_peak_mb: float
    model_key: str  # shape expected by utils.voting.compose_generation_metrics


class BonMavState(TypedDict):
    task: str
    problem_id: int
    dataset: str
    samples: Annotated[list[AgentSample], operator.add]  # the N replayed candidates
    approvals: Annotated[list[Approval], operator.add]
    selection: Optional[dict]  # {selected_idx, scores, tie, tie_break_used, abstentions, tie_break_stage}


def build_bon_mav_graph(
    candidate_specs: Sequence[AgentSpec],
    aspect_ids: Sequence[str],
    verify: Callable[[str, str, str], Approval],
    select_fn: Callable[..., dict],
    *,
    rng,
) -> CompiledStateGraph:
    """Build the BoN-MAV graph: START -> cand_* -> collect -> verify_* -> select -> END.

    Reuses _make_gen_node/AgentSpec unmodified for the candidate side, the
    same replay path build_majority_voting_graph uses, so a candidate
    AgentSample is byte-identical whether it came from a MAJ run or a
    BoN-MAV run replaying the same store. collect is a barrier node with no
    state update, so every verify_<aspect_id> node fans out only once all
    candidates have landed.

    Parameters
    ----------
    candidate_specs : sequence of AgentSpec
        One per candidate (typically mode="replay", sourced from
        utils.ensembles.sample_slots's rep=0 draw).
    aspect_ids : sequence of str
        One verify_<aspect_id> node per id (e.g.
        utils.mav.MAV_ASPECT_SUBSETS[dataset][subset_id]); each node calls
        verify() once per fanned-in candidate sample.
    verify : Callable[[task, candidate_answer, aspect_id], Approval]
        A live closure over the single, already-loaded verifier model (the
        only model resident in VRAM). The returned Approval need not set
        candidate_idx/aspect_id; the node fills both in regardless, keyed
        on the fanned-in candidate's position and this node's own aspect.
    select_fn : Callable[[n_candidates, approvals, candidate_answers], dict]
        Called as select_fn(n_candidates, approvals, candidate_answers,
        rng=rng), e.g. utils.mav.select_bon_candidate with its own
        spec/rng keywords bound via a notebook-level partial/closure (spec
        is benchmark-specific, not part of this template's signature).
    rng : random.Random
        Threaded into select_fn the same way build_majority_voting_graph
        threads its shared rng into every vote in a run.

    Returns
    -------
    CompiledStateGraph
        Invoke once per problem with {"task": ..., "problem_id": ...,
        "dataset": ..., "samples": [], "approvals": [], "selection": None}.
    """
    if not candidate_specs:
        raise ValueError("build_bon_mav_graph requires at least one candidate AgentSpec")
    if not aspect_ids:
        raise ValueError("build_bon_mav_graph requires at least one aspect_id")

    graph = StateGraph(BonMavState)

    cand_node_names: list[str] = []
    seen_agent_ids: set[str] = set()
    for spec in candidate_specs:
        if spec.agent_id in ("collect", "select"):
            raise ValueError(f"agent_id {spec.agent_id!r} is reserved")
        if spec.agent_id in seen_agent_ids:
            raise ValueError(
                f"Duplicate agent_id {spec.agent_id!r} — LangGraph node names "
                "must be unique; give each AgentSpec a distinct agent_id"
            )
        seen_agent_ids.add(spec.agent_id)
        graph.add_node(spec.agent_id, _make_gen_node(spec))
        graph.add_edge(START, spec.agent_id)
        cand_node_names.append(spec.agent_id)

    def collect(state: BonMavState) -> dict:
        return {}

    graph.add_node("collect", collect)
    for node_name in cand_node_names:
        graph.add_edge(node_name, "collect")

    def _make_verify_node(aspect_id: str) -> Callable[[BonMavState], dict]:
        def verify_node(state: BonMavState) -> dict:
            approvals: list[Approval] = []
            for candidate_idx, sample in enumerate(state["samples"]):
                approval = dict(verify(state["task"], sample["answer"], aspect_id))
                approval["candidate_idx"] = candidate_idx
                approval["aspect_id"] = aspect_id
                approvals.append(approval)  # type: ignore[arg-type]
            return {"approvals": approvals}

        return verify_node

    verify_node_names: list[str] = []
    for aspect_id in aspect_ids:
        node_name = f"verify_{aspect_id}"
        graph.add_node(node_name, _make_verify_node(aspect_id))
        graph.add_edge("collect", node_name)
        verify_node_names.append(node_name)

    def select(state: BonMavState) -> dict:
        n_candidates = len(state["samples"])
        candidate_answers = [sample["answer"] for sample in state["samples"]]
        selection = select_fn(n_candidates, state["approvals"], candidate_answers, rng=rng)
        return {"selection": selection}

    graph.add_node("select", select)
    for node_name in verify_node_names:
        graph.add_edge(node_name, "select")
    graph.add_edge("select", END)

    return graph.compile()
