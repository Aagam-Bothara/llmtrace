"""Step-level helpers shared by the experiment analysis, findings and visualization.

Two distinct per-request quantities are derived from batch metadata:

* ``step_ms``: durations of the engine steps a request was scheduled in. A
  proxy for how long each of its tokens took to compute, not an inter-token
  latency.
* ``itl_ms`` (inter-token latency): intervals between the *end times* of the
  request's successive steps from its first-token step onward. Each decode
  step yields one token for the request, so these are the intervals at which
  its tokens became visible, including any steps it was not scheduled in
  (queueing, preemption). Requires cumulative or delta outputs and batch
  metadata; empty otherwise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from llmtrace.models.trace import BatchMetadata, RequestTrace


def step_durations(batches: List[BatchMetadata]) -> Dict[str, float]:
    """batch_id -> step duration in ms (only batches with both timestamps)."""
    return {b.batch_id: (b.step_end_monotonic - b.monotonic) * 1000.0
            for b in batches if b.monotonic is not None and b.step_end_monotonic is not None}


def step_ends(batches: List[BatchMetadata]) -> Dict[str, float]:
    return {b.batch_id: b.step_end_monotonic for b in batches if b.step_end_monotonic is not None}


def request_step_ms(trace: RequestTrace, durations: Dict[str, float]) -> List[float]:
    return [durations[b] for b in trace.batch_ids if b in durations]


def request_itl_ms(trace: RequestTrace, ends: Dict[str, float]) -> List[float]:
    """Intervals between successive step ends from the first-token step onward (ms)."""
    if trace.first_token_monotonic is None:
        return []
    ts = sorted(ends[b] for b in trace.batch_ids if b in ends and ends[b] >= trace.first_token_monotonic - 1e-9)
    return [(b - a) * 1000.0 for a, b in zip(ts, ts[1:])]


def per_request_intervals(traces: List[RequestTrace], batches: List[BatchMetadata]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """(step_ms per request, itl_ms per request)."""
    d, e = step_durations(batches), step_ends(batches)
    return ({t.request_id: request_step_ms(t, d) for t in traces}, {t.request_id: request_itl_ms(t, e) for t in traces})


def gpu_span_by_step(gpu_steps: List[Any]) -> Dict[int, Any]:
    """step_index -> StepGpuTiming with a resolved gpu span."""
    return {g.step_index: g for g in gpu_steps if g.gpu_span_ms is not None}


def ttft_from_scheduled_ms(trace: RequestTrace, arrival_delay_ms: Optional[float]) -> Optional[float]:
    """TTFT measured from the *intended* arrival: engine TTFT plus load-generator delay."""
    if trace.ttft_ms is None or arrival_delay_ms is None:
        return None
    return trace.ttft_ms + max(arrival_delay_ms, 0.0)
