"""Configuration comparison against a stated latency target. Advisory only.

Input: named configurations, each a list of run directories (repeats), and a
target such as ``short ttft_p95 <= 300`` (ms). Output: per configuration, the
target metric per repeat, whether every repeat met it, throughput, energy per
output token with telemetry coverage, run-to-run variability, and whether any
repeat failed (a ``run_info.json`` / ``manifest.json`` with ``status: failed``,
or no traces). No setting is changed anywhere; the table is the deliverable.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from llmtrace import io
from llmtrace.control_plane.correlator import Correlator
from llmtrace.control_plane.reporter import percentile
from llmtrace.models.config import EnergyConfig
from llmtrace.models.trace import RequestTrace

_TARGET_RE = re.compile(r"^\s*(?P<cls>[\w*]+)\s+(?P<metric>ttft|tpot|e2e)_(?P<stat>p50|p90|p95|p99|max)\s*(?P<op><=|<)\s*(?P<value>[\d.]+)\s*(ms)?\s*$")


class Target(BaseModel):
    request_class: str  # "*" for all
    metric: str  # ttft | tpot | e2e
    stat: str  # p50 | p90 | p95 | p99 | max
    value_ms: float

    @classmethod
    def parse(cls, text: str) -> "Target":
        m = _TARGET_RE.match(text)
        if not m:
            raise ValueError(f"Cannot parse target {text!r}; expected e.g. 'short ttft_p95 <= 300ms'")
        return cls(request_class=m["cls"], metric=m["metric"], stat=m["stat"], value_ms=float(m["value"]))

    def describe(self) -> str:
        return f"{self.request_class} {self.metric}_{self.stat} <= {self.value_ms:g} ms"


class RepeatResult(BaseModel):
    run_dir: str
    status: str  # ok | failed | empty
    error: Optional[str] = None
    target_value_ms: Optional[float] = None
    meets_target: Optional[bool] = None
    attainment_fraction: Optional[float] = None  # share of requests individually under the target
    requests: int = 0
    completed: int = 0
    duration_s: Optional[float] = None
    output_tokens: int = 0
    tokens_per_s: Optional[float] = None
    requests_per_s: Optional[float] = None
    device_joules: Optional[float] = None
    joules_per_output_token: Optional[float] = None
    telemetry_coverage: Optional[float] = None
    work_signature: Optional[str] = None  # sorted output token counts, to check work-identical replay


class ConfigResult(BaseModel):
    name: str
    repeats: List[RepeatResult]
    all_ok: bool
    meets_target_all_repeats: Optional[bool] = None
    target_median_ms: Optional[float] = None
    target_min_ms: Optional[float] = None
    target_max_ms: Optional[float] = None
    tokens_per_s_median: Optional[float] = None
    joules_per_output_token_median: Optional[float] = None
    coverage_min: Optional[float] = None
    work_identical_across_repeats: Optional[bool] = None


class Decision(BaseModel):
    target: str
    configs: List[ConfigResult]
    candidates: List[str] = Field(default_factory=list)  # configs meeting the target in every ok repeat
    recommendation: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


def _metric_values(traces: List[RequestTrace], target: Target) -> List[float]:
    sel = [t for t in traces if target.request_class in ("*", t.request_id.split("-")[0])]
    if target.metric == "ttft":
        return [t.ttft_ms for t in sel if t.ttft_ms is not None]
    if target.metric == "tpot":
        return [t.tpot_ms for t in sel if t.tpot_ms is not None]
    return [t.total_duration_ms for t in sel]


def _stat(values: List[float], stat: str) -> Optional[float]:
    if not values:
        return None
    return max(values) if stat == "max" else percentile(values, int(stat[1:]))


def evaluate_repeat(run_dir: str, target: Target, attribution: str = "equal_share") -> RepeatResult:
    d = Path(run_dir)
    info: Dict[str, Any] = {}
    for name in ("manifest.json", "run_info.json"):
        if (d / name).exists():
            try:
                info = json.loads((d / name).read_text(encoding="utf-8"))
                break
            except Exception:
                pass
    if info.get("status") == "failed":
        return RepeatResult(run_dir=str(d), status="failed", error=str(info.get("error"))[:300])
    traces = io.load_traces([d])
    if not traces:
        return RepeatResult(run_dir=str(d), status="empty", error="no traces")
    res = Correlator(EnergyConfig(attribution_method=attribution)).correlate(traces, io.load_gpu_samples([d]), io.load_batches([d]))  # type: ignore[arg-type]
    vals = _metric_values(res.traces, target)
    tv = _stat(vals, target.stat)
    start = min(t.start_monotonic if t.start_monotonic is not None else t.start_time for t in res.traces)
    end = max(t.end_monotonic if t.end_monotonic is not None else t.end_time for t in res.traces)
    dur = max(end - start, 1e-9)
    out_tokens = sum(t.output_length for t in res.traces)
    L = res.ledger
    return RepeatResult(
        run_dir=str(d), status="ok", target_value_ms=tv, meets_target=(tv <= target.value_ms) if tv is not None else None,
        attainment_fraction=(sum(1 for v in vals if v <= target.value_ms) / len(vals)) if vals else None,
        requests=len(res.traces), completed=sum(1 for t in res.traces if t.status.value == "completed"),
        duration_s=dur, output_tokens=out_tokens, tokens_per_s=out_tokens / dur, requests_per_s=len(res.traces) / dur,
        device_joules=L.device_joules,
        joules_per_output_token=(L.attributed_joules / out_tokens) if L.device_joules is not None and out_tokens else None,
        telemetry_coverage=L.coverage.coverage_fraction if L.coverage else None,
        work_signature=",".join(str(x) for x in sorted(t.output_length for t in res.traces)),
    )


def evaluate(configs: Dict[str, List[str]], target: Target, attribution: str = "equal_share") -> Decision:
    results: List[ConfigResult] = []
    for name, dirs in configs.items():
        reps = [evaluate_repeat(dd, target, attribution) for dd in dirs]
        ok = [r for r in reps if r.status == "ok"]
        tvals = [r.target_value_ms for r in ok if r.target_value_ms is not None]
        tps = [r.tokens_per_s for r in ok if r.tokens_per_s is not None]
        jpt = [r.joules_per_output_token for r in ok if r.joules_per_output_token is not None]
        cov = [r.telemetry_coverage for r in ok if r.telemetry_coverage is not None]
        sigs = {r.work_signature for r in ok}
        results.append(ConfigResult(
            name=name, repeats=reps, all_ok=len(ok) == len(reps) and bool(reps),
            meets_target_all_repeats=(all(r.meets_target for r in ok if r.meets_target is not None) if tvals else None),
            target_median_ms=statistics.median(tvals) if tvals else None,
            target_min_ms=min(tvals) if tvals else None, target_max_ms=max(tvals) if tvals else None,
            tokens_per_s_median=statistics.median(tps) if tps else None,
            joules_per_output_token_median=statistics.median(jpt) if jpt else None,
            coverage_min=min(cov) if cov else None,
            work_identical_across_repeats=(len(sigs) == 1) if ok else None,
        ))
    candidates = [c.name for c in results if c.all_ok and c.meets_target_all_repeats]
    notes = []
    for c in results:
        if not c.all_ok:
            failed = [r for r in c.repeats if r.status != "ok"]
            notes.append(f"{c.name}: {len(failed)} repeat(s) failed or empty ({failed[0].error})")
        if c.work_identical_across_repeats is False:
            notes.append(f"{c.name}: output token counts differ across repeats (work not identical; use ignore_eos / fixed max_tokens)")
    all_sigs = {r.work_signature for c in results for r in c.repeats if r.status == "ok"}
    if len(all_sigs) > 1:
        notes.append("output token counts differ across configurations; throughput and energy per token are not like-for-like")
    rec = None
    if candidates:
        best = max((c for c in results if c.name in candidates), key=lambda c: c.tokens_per_s_median or 0.0)
        rec = (f"{best.name} meets '{target.describe()}' in every repeat with the highest median throughput "
               f"({best.tokens_per_s_median:.0f} output tokens/s). Advisory: verify on the production model and workload.")
    elif results:
        rec = "No configuration met the target in every repeat; see per-repeat values and notes."
    return Decision(target=target.describe(), configs=results, candidates=candidates, recommendation=rec, notes=notes)


def _f(v: Optional[float], d: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def format_decision(dec: Decision) -> str:
    lines = [f"target: {dec.target}", "",
             f"{'config':14} {'ok':>3} {'meets':>6} {'target ms (median [min..max])':>32} {'tok/s':>8} {'J/tok':>8} {'cov':>5} {'same work':>9}"]
    for c in dec.configs:
        meets = "n/a" if c.meets_target_all_repeats is None else ("yes" if c.meets_target_all_repeats else "no")
        rng = f"{_f(c.target_median_ms)} [{_f(c.target_min_ms)}..{_f(c.target_max_ms)}]"
        same = "n/a" if c.work_identical_across_repeats is None else ("yes" if c.work_identical_across_repeats else "NO")
        lines.append(f"{c.name:14} {('yes' if c.all_ok else 'NO'):>3} {meets:>6} {rng:>32} {_f(c.tokens_per_s_median, 0):>8} "
                     f"{_f(c.joules_per_output_token_median, 4):>8} {_f(c.coverage_min, 2):>5} {same:>9}")
    lines.append("")
    for n in dec.notes:
        lines.append(f"note: {n}")
    lines.append(f"candidates meeting the target in every repeat: {', '.join(dec.candidates) if dec.candidates else 'none'}")
    if dec.recommendation:
        lines.append(f"recommendation (advisory): {dec.recommendation}")
    return "\n".join(lines)
