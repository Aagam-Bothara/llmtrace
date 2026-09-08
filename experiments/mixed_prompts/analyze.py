"""Diagnosis for the mixed-prompt experiment, computed only from llmtrace's recorded files.

Given a run directory (traces_*, batches_*, gpu_*):

1. Per request class (short/long): TTFT and TPOT percentiles; ``ttft_sched``
   (TTFT from the *intended* arrival, i.e. engine TTFT plus load-generator
   delay from the manifest); ``step_ms`` (durations of the engine steps the
   request was scheduled in, a per-token compute proxy); and ``itl_ms``, the
   real inter-token latency: intervals between the request's successive step
   ends from its first-token step onward, including steps it was not scheduled
   in. A request's average TPOT hides a single slow interval; ITL max does not.
2. Per engine step (from batch metadata): duration, scheduled tokens, and whether
   the step carried a "long prefill chunk" (a single request scheduled with more
   than ``chunk_threshold`` tokens).
3. Interference attribution for short requests: how much of each short request's
   decode time was spent in steps that also carried a long prefill chunk, and how
   much slower those steps were. This is co-occurrence evidence from the traces;
   the controlled comparison (``compare``) is the causal test.
4. Step-time model: least-squares slope of step duration vs scheduled tokens.

``compare(baseline, candidate)`` reports the change in the short-request tail
and the long-request cost side by side and states whether the candidate improved
the short tail beyond ``min_improvement_pct``.

Verdict metrics. The mechanism produces *stalls*: a short request arriving
during a big prefill step waits for the whole step (TTFT), and short requests
already decoding lose one token interval to it (ITL max). Capping long prefill
bounds the stall but spreads it over more steps, so it raises the *number* of
affected steps while lowering their severity; an all-token ITL p99 therefore
moves against the cap by construction when affected steps are rare (<1%). The
verdict uses short-request TTFT p95 and ITL max (both must improve by the
threshold; neither may regress) and reports ITL p99, TPOT and the long-request
TTFT cost alongside so the trade-off is visible. Without batch metadata the
ITL part falls back to TPOT p95.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmtrace import io
from llmtrace.control_plane.reporter import percentile
from llmtrace.control_plane.steps import per_request_intervals, ttft_from_scheduled_ms
from llmtrace.manifest import RunManifest
from llmtrace.models.trace import BatchMetadata, RequestTrace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workload import kind_of  # noqa: E402


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    return {"n": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95),
            "p99": percentile(values, 99), "max": max(values) if values else None}


def analyze_run(traces: List[RequestTrace], batches: List[BatchMetadata], chunk_threshold: int = 128,
                arrival_delays_ms: Optional[Dict[str, float]] = None, gpu_steps: Optional[List[Any]] = None) -> Dict[str, Any]:
    by_kind: Dict[str, List[RequestTrace]] = {}
    for t in traces:
        by_kind.setdefault(kind_of(t.request_id), []).append(t)
    step_ms, itl_ms = per_request_intervals(traces, batches)
    delays = arrival_delays_ms or {}
    latency = {
        kind: {
            "requests": len(ts),
            "completed": sum(1 for t in ts if t.status.value == "completed"),
            "ttft_ms": _stats([t.ttft_ms for t in ts if t.ttft_ms is not None]),
            "ttft_mean_ms": statistics.mean([t.ttft_ms for t in ts if t.ttft_ms is not None]) if any(t.ttft_ms is not None for t in ts) else None,
            "queue_mean_ms": statistics.mean([t.queue_duration_ms for t in ts]) if ts else None,
            "prefill_mean_ms": statistics.mean([t.prefill_duration_ms for t in ts]) if ts else None,
            "ttft_span_resolved": all(t.scheduler_visible for t in ts) if ts else False,
            "ttft_sched_ms": _stats([v for v in (ttft_from_scheduled_ms(t, delays.get(t.request_id)) for t in ts) if v is not None]),
            "arrival_delay_ms": _stats([delays[t.request_id] for t in ts if t.request_id in delays]),
            "tpot_ms": _stats([t.tpot_ms for t in ts if t.tpot_ms is not None]),
            "step_ms": _stats([v for t in ts for v in step_ms.get(t.request_id, [])]),
            "itl_ms": _stats([v for t in ts for v in itl_ms.get(t.request_id, [])]),
            "queue_ms": _stats([t.queue_duration_ms for t in ts if t.queue_duration_ms > 0]),
        }
        for kind, ts in sorted(by_kind.items())
    }

    steps = []
    for b in batches:
        if b.monotonic is None or b.step_end_monotonic is None:
            continue
        biggest = max(b.scheduled_tokens.values()) if b.scheduled_tokens else 0
        steps.append({
            "batch_id": b.batch_id, "step_index": b.step_index,
            "duration_ms": (b.step_end_monotonic - b.monotonic) * 1000.0,
            "scheduled_tokens": b.total_scheduled_tokens, "num_requests": b.num_requests,
            "num_prefill": b.num_prefill, "biggest_chunk": biggest,
            "long_chunk": biggest > chunk_threshold, "request_ids": set(b.request_ids),
        })
    span_by = {g.step_index: g for g in (gpu_steps or []) if g.gpu_span_ms is not None}
    for s in steps:
        g = span_by.get(s["step_index"])
        s["gpu_span_ms"] = g.gpu_span_ms if g else None
        s["host_overhead_ms"] = (s["duration_ms"] - g.gpu_span_ms) if g else None
    with_chunk = [s for s in steps if s["long_chunk"]]
    without = [s for s in steps if not s["long_chunk"]]
    gpu_split = None
    if span_by:
        resolved = [s for s in steps if s["gpu_span_ms"] is not None]
        gpu_split = {
            "steps_with_gpu_span": len(resolved),
            "gpu_span_ms": _stats([s["gpu_span_ms"] for s in resolved]),
            "host_overhead_ms": _stats([s["host_overhead_ms"] for s in resolved]),
            "gpu_span_ms_with_long_chunk": _stats([s["gpu_span_ms"] for s in resolved if s["long_chunk"]]),
            "gpu_span_ms_without": _stats([s["gpu_span_ms"] for s in resolved if not s["long_chunk"]]),
            "host_share_median": statistics.median([max(0.0, s["host_overhead_ms"]) / s["duration_ms"] for s in resolved if s["duration_ms"] > 0]),
        }
    step_summary = {
        "steps": len(steps),
        "steps_with_long_chunk": len(with_chunk),
        "chunk_threshold_tokens": chunk_threshold,
        "duration_ms_with_long_chunk": _stats([s["duration_ms"] for s in with_chunk]),
        "duration_ms_without": _stats([s["duration_ms"] for s in without]),
        "biggest_chunk_tokens_max": max((s["biggest_chunk"] for s in steps), default=0),
    }

    # Step-time model: duration = a + b * scheduled_tokens (least squares).
    model = None
    if len(steps) >= 3:
        xs = [s["scheduled_tokens"] for s in steps]
        ys = [s["duration_ms"] for s in steps]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx > 0:
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
            intercept = my - slope * mx
            ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
            ss_tot = sum((y - my) ** 2 for y in ys)
            model = {"intercept_ms": intercept, "us_per_token": slope * 1000.0,
                     "r2": 1 - ss_res / ss_tot if ss_tot > 0 else None}

    # Stalls the token model cannot explain: residual = duration - (a + b * tokens). These are
    # candidates for effects outside the scheduler (warm-up, GC, the tracer's own collector, ...).
    unexplained = None
    if model is not None and steps:
        t_first = min(s["step_index"] for s in steps)
        first_mono = min(b.monotonic for b in batches if b.monotonic is not None)
        resid = []
        for s, b in zip(steps, [b for b in batches if b.monotonic is not None and b.step_end_monotonic is not None]):
            pred = model["intercept_ms"] + model["us_per_token"] / 1000.0 * s["scheduled_tokens"]
            resid.append({"step_index": s["step_index"], "offset_s": b.monotonic - first_mono, "duration_ms": s["duration_ms"],
                          "predicted_ms": pred, "residual_ms": s["duration_ms"] - pred, "scheduled_tokens": s["scheduled_tokens"],
                          "long_chunk": s["long_chunk"]})
        base = statistics.median(s["duration_ms"] for s in steps)
        big = sorted((r for r in resid if r["residual_ms"] > 2.0 * base), key=lambda r: -r["residual_ms"])
        unexplained = {"threshold_ms": 2.0 * base, "count": len(big), "top": big[:8], "first_step_index": t_first}

    # Interference attribution for short requests, over the steps each request participated in.
    step_by_id = {s["batch_id"]: s for s in steps}
    per_short = []
    for t in by_kind.get("short", []):
        mine = [step_by_id[b] for b in t.batch_ids if b in step_by_id]
        if not mine:
            continue
        shared = [s for s in mine if s["long_chunk"]]
        total = sum(s["duration_ms"] for s in mine)
        in_shared = sum(s["duration_ms"] for s in shared)
        per_short.append({"request_id": t.request_id, "steps": len(mine), "steps_with_long_chunk": len(shared),
                          "step_time_ms": total, "step_time_in_long_chunk_steps_ms": in_shared,
                          "share_in_long_chunk_steps": (in_shared / total) if total > 0 else 0.0,
                          "tpot_ms": t.tpot_ms, "ttft_ms": t.ttft_ms})
    affected = [p for p in per_short if p["steps_with_long_chunk"] > 0]
    unaffected = [p for p in per_short if p["steps_with_long_chunk"] == 0]
    interference = {
        "short_requests_with_step_data": len(per_short),
        "short_requests_sharing_a_long_chunk_step": len(affected),
        "share_of_short_step_time_in_long_chunk_steps": (
            sum(p["step_time_in_long_chunk_steps_ms"] for p in per_short) / sum(p["step_time_ms"] for p in per_short)
            if per_short and sum(p["step_time_ms"] for p in per_short) > 0 else None),
        "tpot_ms_affected": _stats([p["tpot_ms"] for p in affected if p["tpot_ms"] is not None]),
        "tpot_ms_unaffected": _stats([p["tpot_ms"] for p in unaffected if p["tpot_ms"] is not None]),
        "ttft_ms_affected": _stats([p["ttft_ms"] for p in affected if p["ttft_ms"] is not None]),
        "ttft_ms_unaffected": _stats([p["ttft_ms"] for p in unaffected if p["ttft_ms"] is not None]),
    }
    return {"latency": latency, "steps": step_summary, "step_time_model": model, "interference": interference,
            "unexplained_stalls": unexplained, "gpu_split": gpu_split, "batch_metadata_available": bool(steps)}


def explain(a: Dict[str, Any]) -> str:
    lines = []
    for kind, m in a["latency"].items():
        lines.append(f"{kind:5} n={m['requests']} completed={m['completed']}  TTFT p50/p95 = "
                     f"{_f(m['ttft_ms']['p50'])}/{_f(m['ttft_ms']['p95'])} ms"
                     + (f" (from intended arrival p95 {_f(m['ttft_sched_ms']['p95'])} ms, arrival delay max "
                        f"{_f(m['arrival_delay_ms']['max'])} ms)" if m["ttft_sched_ms"]["n"] else "")
                     + f"   TPOT p50/p95 = {_f(m['tpot_ms']['p50'])}/{_f(m['tpot_ms']['p95'])} ms   "
                     f"ITL p50/p99/max = {_f(m['itl_ms']['p50'])}/{_f(m['itl_ms']['p99'])}/{_f(m['itl_ms']['max'])} ms   "
                     f"step p50/max = {_f(m['step_ms']['p50'])}/{_f(m['step_ms']['max'])} ms")
    if not a["batch_metadata_available"]:
        lines.append("No batch metadata: step-level diagnosis unavailable (run with VLLM_ENABLE_V1_MULTIPROCESSING=0).")
        return "\n".join(lines)
    s = a["steps"]
    lines.append(f"steps: {s['steps']}, of which {s['steps_with_long_chunk']} carried a prefill chunk > "
                 f"{s['chunk_threshold_tokens']} tokens (largest {s['biggest_chunk_tokens_max']}).")
    lines.append(f"  step duration p50: with long chunk {_f(s['duration_ms_with_long_chunk']['p50'])} ms vs "
                 f"without {_f(s['duration_ms_without']['p50'])} ms")
    if a["step_time_model"]:
        m = a["step_time_model"]
        lines.append(f"  step time ~ {_f(m['intercept_ms'])} ms + {_f(m['us_per_token'], 3)} us/token (r2={_f(m['r2'], 3)})")
    g = a.get("gpu_split")
    if g:
        lines.append(f"  GPU span (CUDA events) on {g['steps_with_gpu_span']} steps: p50 {_f(g['gpu_span_ms']['p50'])} ms "
                     f"(long-chunk steps {_f(g['gpu_span_ms_with_long_chunk']['p50'])} ms, others {_f(g['gpu_span_ms_without']['p50'])} ms); "
                     f"host overhead p50 {_f(g['host_overhead_ms']['p50'])} ms, median host share {g['host_share_median']:.0%}")
    else:
        lines.append("  GPU span per step: unavailable (needs in-process engine core and torch.cuda; see gpu_steps_*.jsonl)")
    u = a.get("unexplained_stalls")
    if u:
        lines.append(f"  {u['count']} step(s) exceed the token model by > {_f(u['threshold_ms'])} ms (unexplained stalls):")
        for r in u["top"][:5]:
            lines.append(f"    step {r['step_index']} at {r['offset_s']:.3f}s: {_f(r['duration_ms'])} ms measured vs "
                         f"{_f(r['predicted_ms'])} ms predicted for {r['scheduled_tokens']} tokens"
                         f"{' (long chunk)' if r['long_chunk'] else ''}")
    i = a["interference"]
    lines.append(f"short requests: {i['short_requests_sharing_a_long_chunk_step']}/{i['short_requests_with_step_data']} "
                 f"shared at least one step with a long prefill chunk; "
                 f"{_pct(i['share_of_short_step_time_in_long_chunk_steps'])} of all short-request step time was in such steps.")
    lines.append(f"  TPOT p95: affected {_f(i['tpot_ms_affected']['p95'])} ms vs unaffected {_f(i['tpot_ms_unaffected']['p95'])} ms; "
                 f"TTFT p95: affected {_f(i['ttft_ms_affected']['p95'])} ms vs unaffected {_f(i['ttft_ms_unaffected']['p95'])} ms")
    return "\n".join(lines)


def compare(base: Dict[str, Any], cand: Dict[str, Any], min_improvement_pct: float = 20.0) -> Dict[str, Any]:
    def pick(a: Dict[str, Any], kind: str, metric: str, stat: str) -> Optional[float]:
        return a["latency"].get(kind, {}).get(metric, {}).get(stat)

    rows = {}
    for kind, metric, stat in (("short", "itl_ms", "p99"), ("short", "itl_ms", "max"), ("short", "step_ms", "max"),
                               ("short", "tpot_ms", "p95"), ("short", "tpot_ms", "p50"), ("short", "ttft_ms", "p95"),
                               ("short", "ttft_sched_ms", "p95"), ("long", "ttft_ms", "p50"), ("long", "tpot_ms", "p50")):
        b, c = pick(base, kind, metric, stat), pick(cand, kind, metric, stat)
        rows[f"{kind}_{metric}_{stat}"] = {"baseline": b, "candidate": c,
                                           "change_pct": ((c - b) / b * 100.0) if b and c is not None else None}
    # Verdict metrics: short-request TTFT p95 (arrival stall) and real ITL max (in-flight stall).
    # Without batch metadata in *both* runs the stall metric falls back to TPOT p95. Every verdict
    # metric must be present in both runs; otherwise the verdict is "unavailable" and says what is missing.
    has_itl = all(a["latency"].get("short", {}).get("itl_ms", {}).get("n") for a in (base, cand))
    stall_metric = "short_itl_ms_max" if has_itl else "short_tpot_ms_p95"
    tail_metrics = {"short_ttft_ms_p95": rows["short_ttft_ms_p95"]["change_pct"], stall_metric: rows[stall_metric]["change_pct"]}
    long_cost = rows["long_ttft_ms_p50"]["change_pct"]
    missing = [k for k, v in tail_metrics.items() if v is None]
    changes = list(tail_metrics.values())
    if missing:
        verdict = "unavailable"
    elif any(c >= min_improvement_pct for c in changes):
        verdict = "worse"
    elif all(c <= -min_improvement_pct for c in changes):
        verdict = "improved"
    else:
        verdict = "no_meaningful_change"
    return {"rows": rows, "verdict_metrics": tail_metrics, "missing_metrics": missing, "long_ttft_change_pct": long_cost,
            "verdict": verdict, "min_improvement_pct": min_improvement_pct,
            "ttft_decomposition": decompose_ttft_change(base, cand)}


def decompose_ttft_change(base: Dict[str, Any], cand: Dict[str, Any], kind: str = "short") -> Dict[str, Any]:
    """Attribute the change in mean TTFT between two runs to queue wait vs prefill.

    With the in-process scheduler every request's TTFT is exactly queue span + prefill
    span (both step-granular), so mean TTFT decomposes exactly. Without it the boundary is
    not exposed and the decomposition is reported as unavailable, not estimated.
    """
    a, b = base["latency"].get(kind, {}), cand["latency"].get(kind, {})
    if not a or not b or a.get("ttft_mean_ms") is None or b.get("ttft_mean_ms") is None:
        return {"available": False, "reason": f"no TTFT for class '{kind}' in one of the runs"}
    if not (a.get("ttft_span_resolved") and b.get("ttft_span_resolved")):
        return {"available": False, "reason": "queue/prefill boundary not exposed (in-process scheduler required in both runs)"}
    d_ttft = b["ttft_mean_ms"] - a["ttft_mean_ms"]
    d_queue = b["queue_mean_ms"] - a["queue_mean_ms"]
    d_prefill = b["prefill_mean_ms"] - a["prefill_mean_ms"]
    residual = d_ttft - d_queue - d_prefill
    parts = {"queue": d_queue, "prefill": d_prefill}
    dominant = max(parts, key=lambda k: abs(parts[k])) if any(parts.values()) else None
    return {"available": True, "class": kind, "ttft_mean_ms": {"baseline": a["ttft_mean_ms"], "candidate": b["ttft_mean_ms"]},
            "delta_ttft_ms": d_ttft, "delta_queue_ms": d_queue, "delta_prefill_ms": d_prefill, "residual_ms": residual,
            "dominant_component": dominant,
            "share_of_change": {k: (v / d_ttft if abs(d_ttft) > 1e-9 else None) for k, v in parts.items()}}


def _f(v: Optional[float], d: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 100:.0f}%"


def load_and_analyze(run_dir: str, chunk_threshold: int) -> Dict[str, Any]:
    manifest = RunManifest.read(run_dir)
    delays = {a.request_id: a.delay_ms for a in manifest.arrivals if a.delay_ms is not None} if manifest else None
    return analyze_run(io.load_traces([run_dir]), io.load_batches([run_dir]), chunk_threshold, delays, io.load_gpu_steps([run_dir]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--compare", help="candidate run directory (run_dir is then the baseline)")
    parser.add_argument("--chunk-threshold", type=int, default=128)
    parser.add_argument("--json", help="write full analysis JSON here")
    args = parser.parse_args()

    base = load_and_analyze(args.run_dir, args.chunk_threshold)
    info = Path(args.run_dir) / "run_info.json"
    if info.exists() and json.loads(info.read_text()).get("synthetic"):
        print("SYNTHETIC RUN (fake engine): numbers demonstrate the pipeline only.")
    print(f"== {args.run_dir}")
    print(explain(base))
    result: Dict[str, Any] = {"baseline": base}
    if args.compare:
        cand = load_and_analyze(args.compare, args.chunk_threshold)
        print(f"\n== {args.compare}")
        print(explain(cand))
        cmp = compare(base, cand)
        result["candidate"] = cand
        result["comparison"] = cmp
        print("\n== comparison (candidate vs baseline; negative = faster)")
        for name, r in cmp["rows"].items():
            print(f"  {name:22} {_f(r['baseline'])} -> {_f(r['candidate'])} ms  ({_f(r['change_pct'], 1)}%)")
        dec = cmp["ttft_decomposition"]
        if dec.get("available"):
            print(f"short TTFT (mean) {_f(dec['ttft_mean_ms']['baseline'])} -> {_f(dec['ttft_mean_ms']['candidate'])} ms: "
                  f"queue {dec['delta_queue_ms']:+.2f} ms, prefill {dec['delta_prefill_ms']:+.2f} ms, "
                  f"residual {_f(dec['residual_ms'], 3)} ms; dominant component: {dec['dominant_component']}")
        else:
            print(f"TTFT decomposition unavailable: {dec.get('reason')}")
        metrics = ", ".join(f"{k} {_f(v, 1)}%" for k, v in cmp["verdict_metrics"].items())
        print(f"verdict: {cmp['verdict']} on [{metrics}] (threshold {cmp['min_improvement_pct']}%); "
              f"long-request TTFT p50 change {_f(cmp['long_ttft_change_pct'], 1)}% (cost)"
              + (f"; missing: {', '.join(cmp['missing_metrics'])}" if cmp["missing_metrics"] else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=lambda o: sorted(o) if isinstance(o, set) else str(o)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
