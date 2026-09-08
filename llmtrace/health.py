"""One reading of a tracer health record, shared by the runner, ``doctor`` and ``decide``.

``LLMTracer.health()`` is a nested dict (instrumentation, cuda_timing,
gpu_sampler, vllm_stats, writer, collection errors). ``assess_health`` turns
it into two lists:

* ``problems``: the run's request record is not trustworthy for any
  comparison (instrumentation errors, requests leaked at stop, dropped or
  unwritten traces, writer errors, collector failures);
* ``telemetry_problems``: a secondary signal is missing or lossy (GPU
  telemetry, GPU step spans, vLLM stats, batch metadata, collector events).
  Latency targets are unaffected; the metrics that depend on that signal are
  reported as unavailable instead of being compared.

Both the full health dict and the nested ``instrumentation`` dict alone are
accepted (older manifests stored only the latter).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthAssessment(BaseModel):
    ok: bool  # no problems (telemetry problems do not count)
    problems: List[str] = Field(default_factory=list)
    telemetry_problems: List[str] = Field(default_factory=list)
    gpu_telemetry_ok: Optional[bool] = None  # None when the record says nothing about the sampler
    gpu_steps_ok: Optional[bool] = None
    vllm_stats_ok: Optional[bool] = None


def _n(d: Any, key: str) -> int:
    try:
        return int((d or {}).get(key) or 0)
    except (TypeError, ValueError, AttributeError):
        return 0


def assess_health(health: Optional[Dict[str, Any]]) -> HealthAssessment:
    if not isinstance(health, dict) or not health:
        return HealthAssessment(ok=False, problems=["no tracer health record"])
    full = "instrumentation" in health and isinstance(health.get("instrumentation"), dict)
    inst = health["instrumentation"] if full else health
    problems: List[str] = []
    tele: List[str] = []

    if _n(inst, "instrumentation_errors"):
        problems.append(f"{_n(inst, 'instrumentation_errors')} instrumentation error(s): {inst.get('last_instrumentation_error')}")
    if _n(inst, "active_requests"):
        problems.append(f"{_n(inst, 'active_requests')} request(s) still active at stop")
    if _n(inst, "dropped_traces"):
        problems.append(f"{_n(inst, 'dropped_traces')} trace(s) dropped by the instrumentation buffer")
    if _n(inst, "dropped_batches"):
        tele.append(f"{_n(inst, 'dropped_batches')} batch record(s) dropped by the instrumentation buffer")

    if full:
        writer = health.get("writer") or {}
        if _n(writer, "write_errors"):
            problems.append(f"{_n(writer, 'write_errors')} writer error(s): {writer.get('last_error')}")
        dropped = writer.get("dropped") or {}
        lossy = set()  # data types whose records were dropped before reaching disk
        for kind, count in dropped.items():
            if not count:
                continue
            if kind == "traces":
                problems.append(f"{count} trace(s) dropped by the writer queue")
            else:
                tele.append(f"{count} {kind} record(s) dropped by the writer queue")
                lossy.add(kind)
        if _n(health, "collection_errors"):
            problems.append(f"{_n(health, 'collection_errors')} collector error(s): {health.get('last_collection_error')}")

        gpu = health.get("gpu_sampler")
        gpu_ok: Optional[bool] = None
        if isinstance(gpu, dict):
            gpu_ok = True
            if not gpu.get("available", True):
                tele.append(f"GPU telemetry unavailable: {gpu.get('unavailable_reason')}")
                gpu_ok = False
            if _n(gpu, "read_errors"):
                tele.append(f"{_n(gpu, 'read_errors')} GPU sampler read error(s): {gpu.get('last_error')}")
                gpu_ok = False
            if _n(gpu, "dropped"):
                tele.append(f"{_n(gpu, 'dropped')} GPU sample(s) dropped")
                gpu_ok = False
            if "gpu" in lossy:
                gpu_ok = False  # samples lost between the sampler and the file: the energy integral has holes
        cuda = health.get("cuda_timing")
        steps_ok: Optional[bool] = None
        if isinstance(cuda, dict):
            steps_ok = bool(cuda.get("available", False))
            if not steps_ok:
                tele.append(f"GPU step spans unavailable: {cuda.get('unavailable_reason')}")
            if _n(cuda, "errors") or _n(cuda, "dropped"):
                tele.append(f"GPU step timing: {_n(cuda, 'errors')} error(s), {_n(cuda, 'dropped')} dropped")
                steps_ok = False
            if "gpu_steps" in lossy:
                steps_ok = False
        stats = health.get("vllm_stats")
        stats_ok: Optional[bool] = None
        if isinstance(stats, dict):
            stats_ok = stats.get("unavailable_reason") in (None, "")
            if not stats_ok:
                tele.append(f"vLLM stats unavailable: {stats.get('unavailable_reason')}")
            if _n(stats, "errors") or _n(stats, "dropped"):
                tele.append(f"vLLM stats: {_n(stats, 'errors')} error(s), {_n(stats, 'dropped')} dropped")
                stats_ok = False
            if "vllm_stats" in lossy:
                stats_ok = False
        return HealthAssessment(ok=not problems, problems=problems, telemetry_problems=tele,
                                gpu_telemetry_ok=gpu_ok, gpu_steps_ok=steps_ok, vllm_stats_ok=stats_ok)
    return HealthAssessment(ok=not problems, problems=problems, telemetry_problems=tele)


__all__ = ["HealthAssessment", "assess_health"]
