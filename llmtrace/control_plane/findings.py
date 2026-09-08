"""Inspectable findings: hypotheses supported by recorded events, never confident root causes.

Each ``Finding`` names the hypothesis, the affected requests, the supporting
events (with the file/field they came from), what evidence is *missing*, and a
suggested experiment. Five hypotheses are implemented:

* ``queue_overload``: requests waited in the scheduler queue while the engine
  was busy (queue spans from in-process scheduling, or vLLM's own queued-time
  and waiting-count stats).
* ``long_prompt_interference``: short requests shared engine steps with large
  prefill chunks and those steps were markedly longer.
* ``kv_cache_pressure``: KV-cache usage was near capacity while vLLM reported
  preemptions (vLLM stats; request ids are not exposed there).

A finding is only produced when its supporting events exist; otherwise the
hypothesis is reported as ``not_evaluable`` with the missing evidence listed,
so "no finding" is never confused with "nothing happened".
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from llmtrace.control_plane.reporter import percentile
from llmtrace.data_plane.vllm_stats import VLLMIterationRecord
from llmtrace.models.trace import BatchMetadata, RequestTrace

DEFAULT_CHUNK_THRESHOLD = 128


class Evidence(BaseModel):
    source: str  # e.g. "batches_*.jsonl:step_end_monotonic"
    statement: str
    value: Optional[float] = None
    unit: Optional[str] = None


class Finding(BaseModel):
    hypothesis: str
    status: str  # supported | not_supported | not_evaluable
    summary: str
    affected_requests: List[str] = Field(default_factory=list)
    affected_count: int = 0
    supporting_events: List[Evidence] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    suggested_experiment: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


def _kind(request_id: str) -> str:
    return request_id.split("-")[0] if "-" in request_id else "request"


def _step_durations(batches: List[BatchMetadata]) -> Dict[str, float]:
    return {b.batch_id: (b.step_end_monotonic - b.monotonic) * 1000.0
            for b in batches if b.monotonic is not None and b.step_end_monotonic is not None}


# ------------------------------------------------------------------ hypotheses

def check_queue_overload(traces: List[RequestTrace], batches: List[BatchMetadata], stats: List[VLLMIterationRecord],
                         threshold_ms: float = 100.0) -> Finding:
    queued = [(t.request_id, t.queue_duration_ms) for t in traces if t.queue_duration_ms > 0]
    events: List[Evidence] = []
    missing: List[str] = []
    affected = sorted((rid for rid, q in queued if q >= threshold_ms), key=lambda r: r)
    if queued:
        qs = [q for _, q in queued]
        events.append(Evidence(source="traces_*.jsonl:spans[phase=queue]", statement="queue wait p95 across requests with a queue span",
                               value=percentile(qs, 95), unit="ms"))
    else:
        missing.append("queue spans (need the in-process scheduler, VLLM_ENABLE_V1_MULTIPROCESSING=0)")
    waiting = [r.num_waiting_reqs for r in stats if r.num_waiting_reqs is not None]
    if waiting:
        events.append(Evidence(source="vllm_stats_*.jsonl:num_waiting_reqs", statement="max requests waiting in vLLM's queue per step",
                               value=float(max(waiting)), unit="requests"))
        vq = [f.queued_time_s * 1000.0 for r in stats for f in r.finished_requests if f.queued_time_s is not None]
        if vq:
            events.append(Evidence(source="vllm_stats_*.jsonl:finished_requests.queued_time", statement="vLLM's own queued time p95",
                                   value=percentile(vq, 95), unit="ms"))
    else:
        missing.append("vLLM per-step waiting counts (stat_loggers hook)")
    if not queued and not waiting:
        return Finding(hypothesis="queue_overload", status="not_evaluable", summary="No queue evidence recorded.",
                       missing_evidence=missing, parameters={"threshold_ms": threshold_ms})
    supported = bool(affected) or (waiting and max(waiting) > 0 and any(
        e.source.endswith("queued_time") and (e.value or 0) >= threshold_ms for e in events))
    summary = (f"{len(affected)} request(s) waited >= {threshold_ms:.0f} ms before first being scheduled."
               if supported else f"No request waited >= {threshold_ms:.0f} ms in the queue.")
    return Finding(
        hypothesis="queue_overload", status="supported" if supported else "not_supported", summary=summary,
        affected_requests=affected[:50], affected_count=len(affected), supporting_events=events, missing_evidence=missing,
        suggested_experiment=("Replay the same workload at a lower arrival rate, or with a higher max_num_seqs / "
                              "max_num_batched_tokens, and compare queue wait p95 and throughput.") if supported else None,
        parameters={"threshold_ms": threshold_ms},
    )


def check_long_prompt_interference(traces: List[RequestTrace], batches: List[BatchMetadata],
                                   chunk_threshold: int = DEFAULT_CHUNK_THRESHOLD, slowdown_factor: float = 2.0) -> Finding:
    dur = _step_durations(batches)
    if not dur:
        return Finding(hypothesis="long_prompt_interference", status="not_evaluable",
                       summary="No batch metadata: cannot see which requests shared which engine step.",
                       missing_evidence=["batch metadata with step end times (in-process scheduler, VLLM_ENABLE_V1_MULTIPROCESSING=0)"],
                       parameters={"chunk_threshold": chunk_threshold})
    by_id = {b.batch_id: b for b in batches}
    long_steps = {bid for bid, b in by_id.items() if b.scheduled_tokens and max(b.scheduled_tokens.values()) > chunk_threshold and bid in dur}
    if not long_steps:
        return Finding(hypothesis="long_prompt_interference", status="not_supported",
                       summary=f"No engine step carried a prefill chunk larger than {chunk_threshold} tokens.",
                       parameters={"chunk_threshold": chunk_threshold})
    d_long = [dur[b] for b in long_steps]
    d_other = [d for b, d in dur.items() if b not in long_steps]
    med_other = statistics.median(d_other) if d_other else None
    med_long = statistics.median(d_long)
    affected, victims = [], []
    for t in traces:
        shared = [b for b in t.batch_ids if b in long_steps]
        if not shared:
            continue
        # a request is a victim if it was not itself the owner of the large chunk in those steps
        owner = any(max(by_id[b].scheduled_tokens, key=by_id[b].scheduled_tokens.get) == t.request_id for b in shared)
        (affected if owner else victims).append(t.request_id)
    events = [
        Evidence(source="batches_*.jsonl:scheduled_tokens", statement="engine steps carrying a prefill chunk above the threshold",
                 value=float(len(long_steps)), unit="steps"),
        Evidence(source="batches_*.jsonl:step_end_monotonic-monotonic", statement="median duration of those steps", value=med_long, unit="ms"),
        Evidence(source="batches_*.jsonl:step_end_monotonic-monotonic", statement="median duration of all other steps", value=med_other, unit="ms"),
        Evidence(source="traces_*.jsonl:batch_ids", statement="requests that shared such a step without owning the chunk",
                 value=float(len(victims)), unit="requests"),
    ]
    v_ttft = [t.ttft_ms for t in traces if t.request_id in set(victims) and t.ttft_ms is not None]
    o_ttft = [t.ttft_ms for t in traces if t.request_id not in set(victims) and t.request_id not in set(affected) and t.ttft_ms is not None]
    if v_ttft and o_ttft:
        events.append(Evidence(source="traces_*.jsonl:ttft_ms", statement="TTFT p95 of sharing requests", value=percentile(v_ttft, 95), unit="ms"))
        events.append(Evidence(source="traces_*.jsonl:ttft_ms", statement="TTFT p95 of non-sharing requests", value=percentile(o_ttft, 95), unit="ms"))
    supported = bool(victims) and med_other is not None and med_long >= slowdown_factor * med_other
    summary = (f"{len(victims)} request(s) decoded or waited in {len(long_steps)} step(s) that also prefilled a chunk > "
               f"{chunk_threshold} tokens; those steps took {med_long:.1f} ms vs {med_other:.1f} ms for the rest. "
               "They waited while the scheduler spent its token budget on other work."
               if supported else "Large prefill chunks were present but did not make steps markedly longer or affected no other request.")
    return Finding(
        hypothesis="long_prompt_interference", status="supported" if supported else "not_supported", summary=summary,
        affected_requests=victims[:50], affected_count=len(victims), supporting_events=events,
        missing_evidence=[] if v_ttft else ["TTFT of affected requests (needs cumulative/delta outputs; LLM.generate() uses FINAL_ONLY)"],
        suggested_experiment=("Replay with long_prefill_token_threshold set to a smaller per-step chunk (e.g. 256) and compare "
                              "the sharing requests' TTFT p95 and worst inter-token stall against the long requests' TTFT cost.")
        if supported else None,
        parameters={"chunk_threshold": chunk_threshold, "slowdown_factor": slowdown_factor, "chunk_owners": affected[:50]},
    )


def check_kv_cache_pressure(traces: List[RequestTrace], batches: List[BatchMetadata], stats: List[VLLMIterationRecord],
                            usage_threshold: float = 0.9) -> Finding:
    kv_stats = [r for r in stats if r.kv_cache_usage is not None]
    kv_batches = [b for b in batches if b.kv_cache_usage_fraction is not None]
    if not kv_stats and not kv_batches:
        return Finding(hypothesis="kv_cache_pressure", status="not_evaluable", summary="No KV-cache usage recorded.",
                       missing_evidence=["KV-cache usage per step (vLLM stat_loggers hook or in-process scheduler)",
                                         "preemption counts (vLLM stat_loggers hook)"],
                       parameters={"usage_threshold": usage_threshold})
    usage = [r.kv_cache_usage for r in kv_stats] or [b.kv_cache_usage_fraction for b in kv_batches]
    preempt = sum(r.num_preempted_reqs or 0 for r in stats)
    high_steps = sum(1 for u in usage if u >= usage_threshold)
    events_src = "vllm_stats_*.jsonl:kv_cache_usage" if kv_stats else "batches_*.jsonl:kv_cache_usage_fraction"
    events = [Evidence(source=events_src, statement="max KV-cache usage", value=max(usage), unit="fraction"),
              Evidence(source=events_src, statement=f"steps with usage >= {usage_threshold:.0%}", value=float(high_steps), unit="steps")]
    missing = []
    if stats:
        events.append(Evidence(source="vllm_stats_*.jsonl:num_preempted_reqs", statement="preemptions reported by vLLM", value=float(preempt), unit="requests"))
        missing.append("which requests were preempted (vLLM stats carry no request ids)")
    else:
        missing.append("preemption counts (vLLM stat_loggers hook)")
    # Requests alive during high-usage steps (only known with batch membership)
    affected: List[str] = []
    if kv_batches:
        hot = {b.batch_id for b in kv_batches if b.kv_cache_usage_fraction >= usage_threshold}
        affected = sorted({t.request_id for t in traces if any(b in hot for b in t.batch_ids)})
    supported = high_steps > 0 and (preempt > 0 or not stats)
    status = "supported" if supported else "not_supported"
    if high_steps > 0 and stats and preempt == 0:
        summary = f"KV-cache usage reached {max(usage):.0%} in {high_steps} step(s) but vLLM reported no preemptions."
    elif supported:
        summary = (f"KV-cache usage reached {max(usage):.0%} in {high_steps} step(s)" + (f" with {preempt} preemption(s) reported by vLLM." if stats else "."))
    else:
        summary = f"KV-cache usage stayed below {usage_threshold:.0%} (max {max(usage):.0%})."
    return Finding(hypothesis="kv_cache_pressure", status=status, summary=summary, affected_requests=affected[:50],
                   affected_count=len(affected), supporting_events=events, missing_evidence=missing,
                   suggested_experiment=("Replay with a lower max_num_seqs or a higher gpu_memory_utilization and compare "
                                         "preemptions, KV usage max and the affected requests' TPOT.") if supported else None,
                   parameters={"usage_threshold": usage_threshold, "source": events_src})


def check_host_overhead(batches: List[BatchMetadata], gpu_steps: List[Any], share_threshold: float = 0.5) -> Finding:
    """Are engine steps dominated by time the GPU is not spanned (scheduling, input prep, output processing, tracer)?"""
    resolved = [g for g in gpu_steps if g.gpu_span_ms is not None and g.host_step_ms > 0]
    if not resolved:
        return Finding(hypothesis="host_overhead", status="not_evaluable",
                       summary="No GPU step spans recorded: cannot separate GPU time from host time.",
                       missing_evidence=["gpu_steps_*.jsonl (CUDA events around execute_model; needs in-process engine core and torch.cuda)"])
    shares = [max(0.0, g.host_step_ms - g.gpu_span_ms) / g.host_step_ms for g in resolved]
    med_share = statistics.median(shares)
    spans = [g.gpu_span_ms for g in resolved]
    hosts = [g.host_step_ms for g in resolved]
    events = [Evidence(source="gpu_steps_*.jsonl:gpu_span_ms", statement="median GPU span per step", value=statistics.median(spans), unit="ms"),
              Evidence(source="gpu_steps_*.jsonl:host_step_ms", statement="median host step time", value=statistics.median(hosts), unit="ms"),
              Evidence(source="gpu_steps_*.jsonl:host_overhead_ms/host_step_ms", statement="median share of step time not spanned by the GPU",
                       value=med_share, unit="fraction"),
              Evidence(source="gpu_steps_*.jsonl", statement="steps with a resolved GPU span", value=float(len(resolved)), unit="steps")]
    supported = med_share >= share_threshold
    return Finding(hypothesis="host_overhead", status="supported" if supported else "not_supported",
                   summary=(f"The GPU was spanned for only {1 - med_share:.0%} of a typical step; {med_share:.0%} is host-side work "
                            "(scheduling, input preparation, output processing, tracing). Note: the span is an upper bound on GPU busy time."
                            if supported else f"Host-side time is {med_share:.0%} of a typical step; steps are GPU-bound."),
                   supporting_events=events, suggested_experiment=(
                       "Replay with fewer, larger batches (raise max_num_batched_tokens / lower arrival rate) or with tracing "
                       "disabled, and compare host share per step and throughput." if supported else None),
                   missing_evidence=["GPU busy time (CUPTI); the CUDA-event span includes launch gaps"],
                   parameters={"share_threshold": share_threshold})


def check_tracer_self_effect(batches: List[BatchMetadata], collector_events: List[Any], factor: float = 2.0) -> Finding:
    """Flag engine steps overlapping a tracer collector drain that are much longer than the median."""
    dur = _step_durations(batches)
    if not dur or not collector_events:
        return Finding(hypothesis="tracer_observer_effect", status="not_evaluable", summary="No collector events or step timings.",
                       missing_evidence=["collector_*.jsonl (written by LLMTracer) and batch metadata"])
    med = statistics.median(dur.values())
    by_id = {b.batch_id: b for b in batches}
    hits = []
    for ev in collector_events:
        a, b_ = ev.monotonic, ev.monotonic + ev.duration_ms / 1000.0
        for bid, d in dur.items():
            b = by_id[bid]
            if not (b.monotonic <= b_ and b.step_end_monotonic >= a) or d < factor * med:
                continue
            # A long step that merely overlaps a short drain is not the tracer's doing: the drain must be a
            # material part of the step's excess over the median (a 100 ms prefill step overlapping a 0.2 ms
            # drain is explained by its tokens, not by llmtrace).
            excess = d - med
            if ev.duration_ms >= max(1.0, 0.25 * excess):
                hits.append((bid, d, ev.duration_ms))
    events = [Evidence(source="collector_*.jsonl", statement="collector drains recorded", value=float(len(collector_events)), unit="drains"),
              Evidence(source="collector_*.jsonl:duration_ms", statement="longest drain",
                       value=max(e.duration_ms for e in collector_events), unit="ms")]
    supported = bool(hits)
    return Finding(hypothesis="tracer_observer_effect", status="supported" if supported else "not_supported",
                   summary=(f"{len(hits)} engine step(s) overlapping a tracer drain took >= {factor:.0f}x the median step time; "
                            "llmtrace itself may have stalled the engine." if supported
                            else "No engine step overlapping a tracer drain was unusually long."),
                   supporting_events=events, parameters={"factor": factor, "steps": [h[0] for h in hits][:20]},
                   suggested_experiment="Lower collection_interval_s (smaller drains) or raise it past the run, and re-check." if supported else None)


def evaluate_all(traces: List[RequestTrace], batches: List[BatchMetadata], stats: List[VLLMIterationRecord],
                 collector_events: Optional[List[Any]] = None, chunk_threshold: int = DEFAULT_CHUNK_THRESHOLD,
                 queue_threshold_ms: float = 100.0, kv_threshold: float = 0.9, gpu_steps: Optional[List[Any]] = None) -> List[Finding]:
    out = [check_queue_overload(traces, batches, stats, queue_threshold_ms),
           check_long_prompt_interference(traces, batches, chunk_threshold),
           check_kv_cache_pressure(traces, batches, stats, kv_threshold),
           check_host_overhead(batches, gpu_steps or [])]
    if collector_events is not None:
        out.append(check_tracer_self_effect(batches, collector_events))
    return out


def format_findings(findings: List[Finding]) -> str:
    lines = []
    for f in findings:
        lines.append(f"[{f.status}] {f.hypothesis}: {f.summary}")
        if f.affected_count:
            shown = ", ".join(f.affected_requests[:8]) + (" ..." if f.affected_count > 8 else "")
            lines.append(f"    affected ({f.affected_count}): {shown}")
        for e in f.supporting_events:
            v = "" if e.value is None else f" = {e.value:.3g}{(' ' + e.unit) if e.unit else ''}"
            lines.append(f"    evidence: {e.statement}{v}  [{e.source}]")
        for m in f.missing_evidence:
            lines.append(f"    missing: {m}")
        if f.suggested_experiment:
            lines.append(f"    experiment: {f.suggested_experiment}")
    return "\n".join(lines)
