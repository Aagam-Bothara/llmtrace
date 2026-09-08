"""Bounded configuration experiments derived from findings. Plans only; nothing is executed here.

Given the findings of a recorded run and that run's effective engine
configuration (from its manifest), ``plan_experiments`` proposes a small set
of scheduler/batching changes worth replaying with the same workload, each
with the finding it follows from, the effect it should have if the finding is
the cause, and the request class expected to pay for it. Only knobs of vLLM
0.11.0's ``SchedulerConfig``/``CacheConfig`` that the experiment driver can
pass to ``LLM(...)`` are proposed: ``long_prefill_token_threshold``,
``max_num_batched_tokens``, ``max_num_seqs``, ``gpu_memory_utilization``,
``enable_prefix_caching``. Values are bounded by the current configuration
(halving/doubling, chunk sizes below what was observed, memory utilization at
most 0.95) and the list is capped.

The plan is a JSON file that ``llmtrace run --plan`` executes explicitly, run
by run, under the same workload and repeat count, writing
``<out>/<config>/r<i>``; ``llmtrace decide`` then compares them. No running
server is touched: every candidate is a fresh engine started by the runner.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from llmtrace.control_plane.findings import SUPPORTED, Finding
from llmtrace.manifest import RunManifest
from llmtrace.models.trace import BatchMetadata

KNOWN_KNOBS = ("long_prefill_token_threshold", "max_num_batched_tokens", "max_num_seqs", "gpu_memory_utilization",
               "enable_prefix_caching", "max_model_len")


class Candidate(BaseModel):
    name: str
    scheduling_change: Dict[str, Any]
    source_finding: str
    rationale: str
    expected_effect: str  # what improves if the finding is the cause
    expected_cost: str  # which class or metric is expected to regress
    repeats: int = 3


class ExperimentPlan(BaseModel):
    plan_version: int = 2
    source_run: Optional[str] = None
    workload_hash: Optional[str] = None
    source_engine: Optional[str] = None  # fake | vllm
    source_model: Optional[str] = None
    source_model_revision: Optional[str] = None
    # Engine kwargs that reproduce the source run: its explicit kwargs and scheduling change, plus model revision and
    # parallelism read from the effective config. The baseline runs with exactly these; each candidate applies its
    # scheduling_change on top.
    source_engine_kwargs: Dict[str, Any] = Field(default_factory=dict)
    baseline_config: Dict[str, Any] = Field(default_factory=dict)  # effective knobs the candidates are relative to (display)
    baseline_name: str = "baseline"
    repeats: int = 3
    candidates: List[Candidate] = Field(default_factory=list)
    skipped: List[str] = Field(default_factory=list)  # findings or knobs that produced no candidate, with the reason
    notes: List[str] = Field(default_factory=list)

    def configs(self) -> List[Dict[str, Any]]:
        """Baseline plus candidates as (name, engine_kwargs, scheduling_change) for the runner: every entry carries the
        source run's engine kwargs; a candidate's change is applied on top of them."""
        base = dict(self.source_engine_kwargs)
        return [{"name": self.baseline_name, "engine_kwargs": dict(base), "scheduling_change": {}}] + [
            {"name": c.name, "engine_kwargs": dict(base), "scheduling_change": dict(c.scheduling_change)} for c in self.candidates]

    def format(self) -> str:
        lines = [f"experiment plan from {self.source_run or '?'} (workload {self.workload_hash or '?'}), {self.repeats} repeats each"]
        src = ", ".join(x for x in (self.source_engine, self.source_model, f"revision {self.source_model_revision}" if self.source_model_revision else None) if x)
        if src:
            lines.append(f"source: {src}")
        if self.source_engine_kwargs:
            lines.append("baseline engine kwargs (reproduced for every run): " + ", ".join(f"{k}={v}" for k, v in self.source_engine_kwargs.items()))
        if self.baseline_config:
            lines.append("baseline effective: " + ", ".join(f"{k}={v}" for k, v in self.baseline_config.items() if v is not None))
        lines.append("")
        if not self.candidates:
            lines.append("no candidates: no supported finding maps to a bounded configuration change")
        for c in self.candidates:
            change = " ".join(f"--set {k}={v!s}" for k, v in c.scheduling_change.items()).replace("True", "true").replace("False", "false")
            lines.append(f"[{c.name}] {change}")
            lines.append(f"    from: {c.source_finding}. {c.rationale}")
            lines.append(f"    expect: {c.expected_effect}")
            lines.append(f"    cost: {c.expected_cost}")
        for s in self.skipped:
            lines.append(f"skipped: {s}")
        for n in self.notes:
            lines.append(f"note: {n}")
        return "\n".join(lines)


def effective_knobs(manifest: Optional[RunManifest]) -> Dict[str, Any]:
    """Flatten the knobs this planner reasons about from a manifest (real vLLM sections or the fake engine)."""
    out: Dict[str, Any] = {k: None for k in KNOWN_KNOBS}
    if manifest is None:
        return out
    cfg = manifest.effective_engine_config or {}
    for section in ("scheduler_config", "cache_config", "model_config", "fake_engine"):
        for k, v in (cfg.get(section) or {}).items():
            if k in out and v is not None:
                out[k] = v
    for k, v in {**manifest.engine_kwargs, **manifest.scheduling_change}.items():
        if k in out:
            out[k] = v
    return out


def source_engine_kwargs(manifest: Optional[RunManifest]) -> Dict[str, Any]:
    """Engine kwargs that reproduce the source run: explicit kwargs + its scheduling change, plus revision and
    parallelism from the effective config when they were not explicit."""
    if manifest is None:
        return {}
    out: Dict[str, Any] = {**manifest.engine_kwargs, **manifest.scheduling_change}
    cfg = manifest.effective_engine_config or {}
    par = cfg.get("parallel_config") or {}
    for k in ("tensor_parallel_size", "pipeline_parallel_size"):
        v = par.get(k)
        if v is not None and k not in out and _int(v) not in (None, 1):
            out[k] = _int(v)
    rev = (cfg.get("model_config") or {}).get("revision") or manifest.model_revision
    if rev and "revision" not in out and manifest.engine == "vllm":
        out["revision"] = rev
    return out


def _int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def plan_experiments(findings: List[Finding], manifest: Optional[RunManifest] = None,
                     batches: Optional[List[BatchMetadata]] = None, max_candidates: int = 4, repeats: int = 3,
                     source_run: Optional[str] = None) -> ExperimentPlan:
    knobs = effective_knobs(manifest)
    plan = ExperimentPlan(source_run=source_run, workload_hash=manifest.workload_hash if manifest else None,
                          source_engine=manifest.engine if manifest else None,
                          source_model=(manifest.model if manifest and manifest.engine == "vllm" else None),
                          source_model_revision=manifest.model_revision if manifest else None,
                          source_engine_kwargs=source_engine_kwargs(manifest),
                          baseline_config={k: v for k, v in knobs.items() if v is not None}, repeats=repeats)
    if manifest is None:
        plan.notes.append("no manifest: baseline knobs unknown, candidates use vLLM 0.11.0 defaults as the reference")
    by_h = {f.hypothesis: f for f in findings}
    cands: List[Candidate] = []

    # --- long prompt interference: cap the per-step prefill chunk of long prompts
    f = by_h.get("long_prompt_interference")
    if f is not None and f.status == SUPPORTED:
        biggest = max((max(b.scheduled_tokens.values()) for b in (batches or []) if b.scheduled_tokens), default=None)
        chunk_threshold = _int(f.parameters.get("chunk_threshold")) or 128
        current = _int(knobs.get("long_prefill_token_threshold")) or 0
        ceiling = biggest if biggest else (current if current else 8192)
        values = [v for v in (1024, 512, 256, 128) if v < ceiling and v >= chunk_threshold and (current == 0 or v < current)]
        for v in values[:2]:
            cands.append(Candidate(
                name=f"cap{v}", scheduling_change={"long_prefill_token_threshold": v}, source_finding=f.hypothesis, repeats=repeats,
                rationale=f"steps carrying prefill chunks above {chunk_threshold} tokens were markedly longer; capping long prompts at "
                          f"{v} tokens per step bounds step time" + (f" (largest observed chunk {biggest})" if biggest else ""),
                expected_effect="short-request TTFT p95 and worst inter-token stall fall",
                expected_cost="long-request TTFT rises (prefill spread over more steps); prefill throughput may fall"))
        mnbt = _int(knobs.get("max_num_batched_tokens"))
        if mnbt and mnbt >= 1024 and (not biggest or mnbt // 2 < biggest):
            cands.append(Candidate(
                name=f"budget{mnbt // 2}", scheduling_change={"max_num_batched_tokens": mnbt // 2}, source_finding=f.hypothesis, repeats=repeats,
                rationale=f"halving the per-step token budget from {mnbt} caps every chunk, not only long prompts",
                expected_effect="step time upper bound falls for all requests",
                expected_cost="prefill throughput falls for every class; long-request TTFT rises"))
        if not values and current:
            plan.skipped.append(f"long_prompt_interference: long_prefill_token_threshold is already {current}; no smaller bounded value proposed")

    # --- queue overload: more concurrency if the running count hit max_num_seqs, else a larger token budget
    f = by_h.get("queue_overload")
    if f is not None and f.status == SUPPORTED:
        mns = _int(knobs.get("max_num_seqs"))
        running_max = max((b.num_running for b in (batches or []) if b.num_running is not None), default=None)
        if mns and (running_max is None or running_max >= mns):
            cands.append(Candidate(
                name=f"seqs{mns * 2}", scheduling_change={"max_num_seqs": mns * 2}, source_finding=f.hypothesis, repeats=repeats,
                rationale=f"requests waited in the queue while the running count reached max_num_seqs={mns}"
                          if running_max is not None else f"requests waited in the queue; max_num_seqs={mns} may be the limit",
                expected_effect="queue wait p95 falls; throughput rises if the GPU has headroom",
                expected_cost="per-token latency (TPOT) rises for every class as steps carry more sequences; KV-cache pressure rises"))
        mnbt = _int(knobs.get("max_num_batched_tokens"))
        if mnbt:
            cands.append(Candidate(
                name=f"budget{mnbt * 2}", scheduling_change={"max_num_batched_tokens": mnbt * 2}, source_finding=f.hypothesis, repeats=repeats,
                rationale=f"a larger per-step token budget ({mnbt} -> {mnbt * 2}) admits more prefill per step",
                expected_effect="queue wait and TTFT fall when prefill is the bottleneck",
                expected_cost="steps get longer: TPOT rises for decoding requests (the long_prompt_interference trade-off)"))
        if not mns and not mnbt:
            plan.skipped.append("queue_overload: neither max_num_seqs nor max_num_batched_tokens known from the manifest")
        plan.notes.append("queue_overload can also be a capacity problem: replaying at a lower arrival rate is a workload change, not a configuration change")

    # --- KV-cache pressure: more cache, fewer sequences, or prefix caching
    f = by_h.get("kv_cache_pressure")
    if f is not None and f.status == SUPPORTED:
        gmu = knobs.get("gpu_memory_utilization")
        try:
            gmu_f = float(gmu) if gmu is not None else None
        except (TypeError, ValueError):
            gmu_f = None
        if gmu_f is not None and gmu_f < 0.95:
            new = round(min(0.95, gmu_f + 0.1), 2)
            cands.append(Candidate(
                name=f"mem{int(new * 100)}", scheduling_change={"gpu_memory_utilization": new}, source_finding=f.hypothesis, repeats=repeats,
                rationale=f"KV-cache usage reached capacity with preemptions; gpu_memory_utilization {gmu_f} -> {new} adds cache blocks",
                expected_effect="preemptions fall; TPOT of long-running requests stabilizes",
                expected_cost="less free memory for activations/graphs; may fail to start (recorded as a failed run)"))
        mns = _int(knobs.get("max_num_seqs"))
        if mns and mns >= 2:
            cands.append(Candidate(
                name=f"seqs{mns // 2}", scheduling_change={"max_num_seqs": mns // 2}, source_finding=f.hypothesis, repeats=repeats,
                rationale=f"fewer concurrent sequences ({mns} -> {mns // 2}) keep the cache below capacity",
                expected_effect="no preemptions; per-request TPOT falls",
                expected_cost="queue wait and TTFT rise; throughput falls"))
        if knobs.get("enable_prefix_caching") is False:
            cands.append(Candidate(
                name="prefix_cache", scheduling_change={"enable_prefix_caching": True}, source_finding=f.hypothesis, repeats=repeats,
                rationale="prefix caching was off; shared prefixes would reuse blocks",
                expected_effect="lower KV usage and TTFT only if prompts share prefixes (this workload's synthetic prompts do not)",
                expected_cost="none expected; no effect without shared prefixes"))

    # --- host overhead: amortize per-step host work over more sequences
    f = by_h.get("host_overhead")
    if f is not None and f.status == SUPPORTED:
        mns = _int(knobs.get("max_num_seqs"))
        if mns:
            cands.append(Candidate(
                name=f"seqs{mns * 2}", scheduling_change={"max_num_seqs": mns * 2}, source_finding=f.hypothesis, repeats=repeats,
                rationale="a large host share per step is amortized by carrying more sequences per step",
                expected_effect="throughput per step rises; host share per token falls",
                expected_cost="TPOT rises; only helps if arrivals are high enough to fill the larger batch"))
        plan.notes.append("host_overhead: also compare a run with gpu_step_timing/tracing disabled (a runner option, not an engine knob) "
                          "to separate the tracer's own cost")

    for f in findings:
        if f.status == SUPPORTED and f.hypothesis == "tracer_observer_effect":
            plan.notes.append("tracer_observer_effect is supported: lower collection_interval_s (a tracer option) before trusting any comparison")
        elif f.status != SUPPORTED and f.hypothesis in ("long_prompt_interference", "queue_overload", "kv_cache_pressure", "host_overhead"):
            plan.skipped.append(f"{f.hypothesis}: {f.status}")

    # dedupe by change, keep order, cap
    seen = set()
    for c in cands:
        key = tuple(sorted(c.scheduling_change.items()))
        if key in seen:
            continue
        seen.add(key)
        if len(plan.candidates) < max_candidates:
            plan.candidates.append(c)
        else:
            plan.skipped.append(f"{c.name}: beyond --max-candidates {max_candidates}")
    if plan.candidates:
        plan.notes.append("every candidate is a fresh engine started by llmtrace run; no running server is modified")
    return plan


__all__ = ["Candidate", "ExperimentPlan", "effective_knobs", "source_engine_kwargs", "plan_experiments", "KNOWN_KNOBS"]
