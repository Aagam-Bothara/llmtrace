"""Configuration comparison against a stated latency target. Advisory only.

Input: named configurations, each a list of run directories (repeats), and a
target such as ``short ttft_p95 <= 300`` (ms). Output: per configuration, the
target metric per repeat, whether every repeat met it, throughput, energy per
output token with telemetry coverage, run-to-run variability, and whether any
repeat failed (a ``run_info.json`` / ``manifest.json`` with ``status: failed``,
or no traces). No setting is changed anywhere; the table is the deliverable.

Uncertainty: besides the per-repeat values (median, min, max across repeats),
the target statistic gets a 95% bootstrap percentile interval computed over
the per-request values pooled across the eligible repeats (seeded, so the
interval is reproducible). A configuration whose every repeat meets the
target but whose interval's upper bound does not is flagged as marginal.

Goodput: optional SLOs per request class (``--slo "short: ttft <= 50ms, tpot
<= 15ms"``) give the share of selected requests that meet every bound of
their class in one run. A request missing one of the bounded metrics counts
as not meeting it, and the share of requests that carried all the metrics is
reported alongside.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from llmtrace import io
from llmtrace.control_plane.correlator import Correlator
from llmtrace.control_plane.reporter import percentile
from llmtrace.control_plane.steps import ttft_from_scheduled_ms
from llmtrace.manifest import RunManifest
from llmtrace.models.config import EnergyConfig
from llmtrace.models.trace import RequestTrace

_TARGET_RE = re.compile(r"^\s*(?P<cls>[\w*]+)\s+(?P<metric>ttft_sched|ttft|tpot|e2e)_(?P<stat>p50|p90|p95|p99|max)\s*(?P<op><=|<)\s*(?P<value>[\d.]+)\s*(ms)?\s*$")


class Target(BaseModel):
    request_class: str  # "*" for all
    metric: str  # ttft | ttft_sched (from intended arrival, needs manifest delays) | tpot | e2e
    stat: str  # p50 | p90 | p95 | p99 | max
    value_ms: float

    @classmethod
    def parse(cls, text: str) -> "Target":
        m = _TARGET_RE.match(text)
        if not m:
            raise ValueError(f"Cannot parse target {text!r}; expected e.g. 'short ttft_p95 <= 300ms' "
                             "(metrics: ttft, ttft_sched, tpot, e2e; stats: p50, p90, p95, p99, max)")
        return cls(request_class=m["cls"], metric=m["metric"], stat=m["stat"], value_ms=float(m["value"]))

    def describe(self) -> str:
        return f"{self.request_class} {self.metric}_{self.stat} <= {self.value_ms:g} ms"


_SLO_RE = re.compile(r"^\s*(?P<cls>[\w*]+)\s*:\s*(?P<bounds>.+?)\s*$")
_BOUND_RE = re.compile(r"^\s*(?P<metric>ttft_sched|ttft|tpot|e2e)\s*(<=|<)\s*(?P<value>[\d.]+)\s*(ms)?\s*$")


class Slo(BaseModel):
    """Per-request bounds for one request class (``*`` for all): a request is 'good' when it meets every bound."""

    request_class: str
    bounds: Dict[str, float]  # metric -> max ms

    @classmethod
    def parse(cls, text: str) -> "Slo":
        m = _SLO_RE.match(text)
        if not m:
            raise ValueError(f"Cannot parse SLO {text!r}; expected e.g. 'short: ttft <= 50ms, tpot <= 15ms'")
        bounds: Dict[str, float] = {}
        for part in m["bounds"].split(","):
            b = _BOUND_RE.match(part)
            if not b:
                raise ValueError(f"Cannot parse SLO bound {part.strip()!r} in {text!r} (metrics: ttft, ttft_sched, tpot, e2e)")
            bounds[b["metric"]] = float(b["value"])
        if not bounds:
            raise ValueError(f"SLO {text!r} has no bounds")
        return cls(request_class=m["cls"], bounds=bounds)

    def describe(self) -> str:
        return f"{self.request_class}: " + ", ".join(f"{k} <= {v:g} ms" for k, v in self.bounds.items())


def _request_metric(t: RequestTrace, metric: str, delay_ms: Optional[float]) -> Optional[float]:
    if metric == "ttft":
        return t.ttft_ms
    if metric == "ttft_sched":
        return ttft_from_scheduled_ms(t, delay_ms)
    if metric == "tpot":
        return t.tpot_ms
    return t.total_duration_ms


def goodput(traces: List[RequestTrace], slos: List[Slo], delays: Optional[Dict[str, float]] = None) -> Tuple[Optional[float], int, Optional[float]]:
    """(share of selected requests meeting every bound of their class, selected count, share carrying all bounded metrics).

    A request is selected when at least one SLO applies to its class (a ``*`` SLO applies to all). A request that
    lacks a bounded metric counts as not good."""
    if not slos:
        return None, 0, None
    good = n = complete = 0
    for t in traces:
        cls = t.request_id.split("-")[0]
        applicable = [s for s in slos if s.request_class in ("*", cls)]
        if not applicable:
            continue
        n += 1
        vals = {m: _request_metric(t, m, (delays or {}).get(t.request_id)) for s in applicable for m in s.bounds}
        has_all = all(v is not None for v in vals.values())
        complete += has_all
        if has_all and all(vals[m] <= s.bounds[m] for s in applicable for m in s.bounds):
            good += 1
    return (good / n if n else None), n, (complete / n if n else None)


def bootstrap_interval(values: List[float], stat: str, resamples: int = 1000, seed: int = 0,
                       level: float = 0.95) -> Optional[Tuple[float, float]]:
    """Seeded percentile-bootstrap interval for ``stat`` (p50/p90/p95/p99/max) of ``values``. None below 2 values."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(_stat([values[rng.randrange(n)] for _ in range(n)], stat) or 0.0 for _ in range(resamples))
    lo = stats[int((1 - level) / 2 * (resamples - 1))]
    hi = stats[int((1 - (1 - level) / 2) * (resamples - 1))]
    return lo, hi


class RepeatResult(BaseModel):
    run_dir: str
    status: str  # ok | ineligible | failed | empty
    error: Optional[str] = None
    problems: List[str] = Field(default_factory=list)  # why a repeat is ineligible
    eligible: bool = False
    target_value_ms: Optional[float] = None
    meets_target: Optional[bool] = None
    metric_coverage: Optional[float] = None  # share of selected requests that have the target metric
    attainment_fraction: Optional[float] = None  # share of requests individually under the target
    target_values_ms: List[float] = Field(default_factory=list)  # per-request values behind target_value_ms (for pooling)
    goodput: Optional[float] = None  # share of SLO-selected requests meeting every bound (None without SLOs)
    goodput_requests: int = 0
    slo_metric_coverage: Optional[float] = None  # share of SLO-selected requests that carried every bounded metric
    requests: int = 0
    expected_requests: Optional[int] = None
    completed: int = 0
    aborted: int = 0
    incomplete: int = 0
    health_ok: Optional[bool] = None
    arrival_delay_ms_max: Optional[float] = None
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
    all_ok: bool  # every repeat ran (status ok or ineligible)
    all_eligible: bool = False  # every repeat is eligible for the target decision
    meets_target_all_repeats: Optional[bool] = None  # True only if all repeats are eligible and meet the target
    target_median_ms: Optional[float] = None
    target_min_ms: Optional[float] = None
    target_max_ms: Optional[float] = None
    target_ci95_ms: Optional[Tuple[float, float]] = None  # bootstrap over per-request values pooled across eligible repeats
    pooled_requests: int = 0
    meets_target_ci_upper: Optional[bool] = None  # the interval's upper bound also meets the target
    goodput_median: Optional[float] = None
    goodput_min: Optional[float] = None
    goodput_max: Optional[float] = None
    tokens_per_s_median: Optional[float] = None
    joules_per_output_token_median: Optional[float] = None
    coverage_min: Optional[float] = None
    work_identical_across_repeats: Optional[bool] = None


class Decision(BaseModel):
    target: str
    slos: List[str] = Field(default_factory=list)
    configs: List[ConfigResult]
    marginal: List[str] = Field(default_factory=list)  # candidates whose bootstrap upper bound misses the target
    candidates: List[str] = Field(default_factory=list)  # configs meeting the target in every ok repeat
    recommendation: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


def _metric_values(traces: List[RequestTrace], target: Target,
                   delays: Optional[Dict[str, float]] = None) -> Tuple[List[float], int]:
    """(values, number of selected requests). Requests missing the metric are excluded from values."""
    sel = [t for t in traces if target.request_class in ("*", t.request_id.split("-")[0])]
    if target.metric == "ttft":
        vals = [t.ttft_ms for t in sel if t.ttft_ms is not None]
    elif target.metric == "ttft_sched":
        vals = [v for v in (ttft_from_scheduled_ms(t, (delays or {}).get(t.request_id)) for t in sel) if v is not None]
    elif target.metric == "tpot":
        vals = [t.tpot_ms for t in sel if t.tpot_ms is not None]
    else:
        vals = [t.total_duration_ms for t in sel]
    return vals, len(sel)


def _stat(values: List[float], stat: str) -> Optional[float]:
    if not values:
        return None
    return max(values) if stat == "max" else percentile(values, int(stat[1:]))


def evaluate_repeat(run_dir: str, target: Target, attribution: str = "equal_share",
                    min_metric_coverage: float = 1.0, exclude_classes: Optional[List[str]] = None,
                    slos: Optional[List[Slo]] = None) -> RepeatResult:
    d = Path(run_dir)
    manifest = None
    try:
        manifest = RunManifest.read(str(d))
    except Exception:
        manifest = None
    info: Dict[str, Any] = {}
    if manifest is None and (d / "run_info.json").exists():
        try:
            info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
        except Exception:
            pass
    if (manifest is not None and manifest.status == "failed") or info.get("status") == "failed":
        err = manifest.error if manifest is not None else info.get("error")
        return RepeatResult(run_dir=str(d), status="failed", error=str(err)[:300])
    traces = io.load_traces([d])
    excluded = 0
    if exclude_classes:
        keep = [t for t in traces if t.request_id.split("-")[0] not in exclude_classes]
        excluded = len(traces) - len(keep)
        traces = keep
    if not traces:
        return RepeatResult(run_dir=str(d), status="empty", error="no traces")
    res = Correlator(EnergyConfig(attribution_method=attribution)).correlate(traces, io.load_gpu_samples([d]), io.load_batches([d]))  # type: ignore[arg-type]
    delays = {a.request_id: a.delay_ms for a in manifest.arrivals if a.delay_ms is not None} if manifest else {}
    vals, n_sel = _metric_values(res.traces, target, delays)
    tv = _stat(vals, target.stat)
    coverage = (len(vals) / n_sel) if n_sel else None
    start = min(t.start_monotonic if t.start_monotonic is not None else t.start_time for t in res.traces)
    end = max(t.end_monotonic if t.end_monotonic is not None else t.end_time for t in res.traces)
    dur = max(end - start, 1e-9)
    out_tokens = sum(t.output_length for t in res.traces)
    L = res.ledger
    completed = sum(1 for t in res.traces if t.status.value == "completed")
    aborted = sum(1 for t in res.traces if t.status.value == "aborted")
    incomplete = len(res.traces) - completed - aborted
    expected = (manifest.expected_requests - excluded) if manifest and manifest.expected_requests else None
    if expected is not None and excluded and manifest and manifest.expected_requests == len(res.traces):
        expected = len(res.traces)  # manifest counted only the kept classes
    health = manifest.health if manifest else info.get("health") or {}
    inst = health.get("instrumentation", health) if isinstance(health, dict) else {}  # full tracer health or the nested dict
    health_ok = None
    if inst:
        health_ok = not inst.get("instrumentation_errors") and not inst.get("active_requests") and not inst.get("dropped_traces")

    problems: List[str] = []
    if n_sel == 0:
        problems.append(f"no requests of class '{target.request_class}'")
    elif tv is None:
        problems.append(f"target metric {target.metric} missing for every selected request")
    elif coverage is not None and coverage < min_metric_coverage:
        problems.append(f"target metric present for only {coverage:.0%} of selected requests (need {min_metric_coverage:.0%})")
    if aborted or incomplete:
        problems.append(f"{aborted} aborted and {incomplete} incomplete request(s)")
    if expected is not None and len(res.traces) != expected:
        problems.append(f"{len(res.traces)} traced requests but {expected} expected")
    if health_ok is False:
        problems.append("tracer health not clean (instrumentation errors, leaked requests or dropped traces)")
    eligible = not problems
    gp, gp_n, gp_cov = goodput(res.traces, slos or [], delays)
    return RepeatResult(
        run_dir=str(d), status="ok" if eligible else "ineligible", problems=problems, eligible=eligible,
        target_value_ms=tv, meets_target=(tv <= target.value_ms) if (tv is not None and eligible) else None,
        metric_coverage=coverage,
        attainment_fraction=(sum(1 for v in vals if v <= target.value_ms) / len(vals)) if vals else None,
        target_values_ms=list(vals), goodput=gp, goodput_requests=gp_n, slo_metric_coverage=gp_cov,
        requests=len(res.traces), expected_requests=expected, completed=completed, aborted=aborted, incomplete=incomplete,
        health_ok=health_ok, arrival_delay_ms_max=manifest.arrival_delay_ms_max if manifest else None,
        duration_s=dur, output_tokens=out_tokens, tokens_per_s=out_tokens / dur, requests_per_s=len(res.traces) / dur,
        device_joules=L.device_joules,
        joules_per_output_token=(L.attributed_joules / out_tokens) if L.device_joules is not None and out_tokens else None,
        telemetry_coverage=L.coverage.coverage_fraction if L.coverage else None,
        work_signature=",".join(str(x) for x in sorted(t.output_length for t in res.traces)),
    )


def evaluate(configs: Dict[str, List[str]], target: Target, attribution: str = "equal_share",
             min_metric_coverage: float = 1.0, exclude_classes: Optional[List[str]] = None,
             slos: Optional[List[Slo]] = None, bootstrap_resamples: int = 1000, seed: int = 0) -> Decision:
    results: List[ConfigResult] = []
    for name, dirs in configs.items():
        reps = [evaluate_repeat(dd, target, attribution, min_metric_coverage, exclude_classes, slos) for dd in dirs]
        ran = [r for r in reps if r.status in ("ok", "ineligible")]
        ok = [r for r in reps if r.eligible]
        tvals = [r.target_value_ms for r in ran if r.target_value_ms is not None]
        pooled = [v for r in ok for v in r.target_values_ms]
        ci = bootstrap_interval(pooled, target.stat, bootstrap_resamples, seed) if pooled else None
        gps = [r.goodput for r in ran if r.goodput is not None]
        tps = [r.tokens_per_s for r in ran if r.tokens_per_s is not None]
        jpt = [r.joules_per_output_token for r in ran if r.joules_per_output_token is not None]
        cov = [r.telemetry_coverage for r in ran if r.telemetry_coverage is not None]
        sigs = {r.work_signature for r in ran}
        all_eligible = bool(reps) and len(ok) == len(reps)
        results.append(ConfigResult(
            name=name, repeats=reps, all_ok=len(ran) == len(reps) and bool(reps), all_eligible=all_eligible,
            # Every repeat must be eligible (valid measurement, complete, healthy) and meet the target.
            meets_target_all_repeats=(all(r.meets_target is True for r in ok) if all_eligible else (False if reps else None)),
            target_median_ms=statistics.median(tvals) if tvals else None,
            target_min_ms=min(tvals) if tvals else None, target_max_ms=max(tvals) if tvals else None,
            target_ci95_ms=ci, pooled_requests=len(pooled),
            meets_target_ci_upper=(ci[1] <= target.value_ms) if ci else None,
            goodput_median=statistics.median(gps) if gps else None,
            goodput_min=min(gps) if gps else None, goodput_max=max(gps) if gps else None,
            tokens_per_s_median=statistics.median(tps) if tps else None,
            joules_per_output_token_median=statistics.median(jpt) if jpt else None,
            coverage_min=min(cov) if cov else None,
            work_identical_across_repeats=(len(sigs) == 1) if ran else None,
        ))
    candidates = [c.name for c in results if c.all_eligible and c.meets_target_all_repeats]
    marginal = [c.name for c in results if c.name in candidates and c.meets_target_ci_upper is False]
    notes = []
    for c in results:
        if c.name in marginal and c.target_ci95_ms:
            notes.append(f"{c.name}: meets the target in every repeat but the 95% bootstrap interval of {target.metric}_{target.stat} "
                         f"over {c.pooled_requests} pooled requests is [{c.target_ci95_ms[0]:.1f}, {c.target_ci95_ms[1]:.1f}] ms, "
                         f"above {target.value_ms:g} ms at the upper end: marginal, add repeats or requests")
        if c.repeats and len([r for r in c.repeats if r.eligible]) < 2:
            notes.append(f"{c.name}: only {len([r for r in c.repeats if r.eligible])} eligible repeat; run-to-run variation is unknown")
    if slos:
        for c in results:
            low_cov = [r for r in c.repeats if r.slo_metric_coverage is not None and r.slo_metric_coverage < 1.0]
            if low_cov:
                notes.append(f"{c.name}: {len(low_cov)} repeat(s) where some SLO-selected requests lack a bounded metric "
                             f"(counted as not meeting the SLO; coverage min {min(r.slo_metric_coverage for r in low_cov):.0%})")
    for c in results:
        failed = [r for r in c.repeats if r.status in ("failed", "empty")]
        if failed:
            notes.append(f"{c.name}: {len(failed)} repeat(s) failed or empty ({failed[0].error})")
        for r in c.repeats:
            if r.status == "ineligible":
                notes.append(f"{c.name} ({Path(r.run_dir).name}): ineligible: " + "; ".join(r.problems))
        if c.work_identical_across_repeats is False:
            notes.append(f"{c.name}: output token counts differ across repeats (work not identical; use ignore_eos / fixed max_tokens)")
    notes.append("throughput (tok/s) is measured over each run's window; with an open-loop (arrival-paced) workload it "
                 "reflects the arrival schedule unless the system saturates, so compare latency and energy per token, and "
                 "use a saturating workload to compare capacity")
    all_sigs = {r.work_signature for c in results for r in c.repeats if r.status in ("ok", "ineligible")}
    if len(all_sigs) > 1:
        notes.append("output token counts differ across configurations; throughput and energy per token are not like-for-like")
    rec = None
    if candidates:
        pool = [c for c in results if c.name in candidates]
        if slos and any(c.goodput_median is not None for c in pool):
            best = max(pool, key=lambda c: (c.goodput_median or 0.0, c.tokens_per_s_median or 0.0))
            rec = (f"{best.name} meets '{target.describe()}' in every repeat with the highest median goodput "
                   f"({best.goodput_median:.0%} of SLO-selected requests). Advisory: verify on the production model and workload.")
        else:
            best = max(pool, key=lambda c: c.tokens_per_s_median or 0.0)
            rec = (f"{best.name} meets '{target.describe()}' in every repeat with the highest median throughput "
                   f"({best.tokens_per_s_median:.0f} output tokens/s). Advisory: verify on the production model and workload.")
        if best.name in marginal:
            rec += " The bootstrap interval makes this marginal (see notes)."
    elif results:
        rec = "No configuration met the target in every repeat; see per-repeat values and notes."
    return Decision(target=target.describe(), slos=[s.describe() for s in (slos or [])], configs=results, candidates=candidates,
                    marginal=marginal, recommendation=rec, notes=notes)


def _f(v: Optional[float], d: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def format_decision(dec: Decision) -> str:
    lines = [f"target: {dec.target}"]
    for s in dec.slos:
        lines.append(f"slo: {s}")
    lines += ["", f"{'config':14} {'ran':>3} {'elig':>4} {'meets':>6} {'target ms (median [min..max])':>32} {'95% CI':>16} "
              f"{'goodput':>8} {'tok/s':>8} {'J/tok':>8} {'cov':>5} {'same work':>9}"]
    for c in dec.configs:
        meets = "n/a" if c.meets_target_all_repeats is None else ("yes" if c.meets_target_all_repeats else "no")
        rng = f"{_f(c.target_median_ms)} [{_f(c.target_min_ms)}..{_f(c.target_max_ms)}]"
        ci = "n/a" if not c.target_ci95_ms else f"[{_f(c.target_ci95_ms[0])}..{_f(c.target_ci95_ms[1])}]"
        gp = "n/a" if c.goodput_median is None else f"{c.goodput_median:.0%}"
        same = "n/a" if c.work_identical_across_repeats is None else ("yes" if c.work_identical_across_repeats else "NO")
        elig = f"{sum(1 for r in c.repeats if r.eligible)}/{len(c.repeats)}"
        lines.append(f"{c.name:14} {('yes' if c.all_ok else 'NO'):>3} {elig:>4} {meets:>6} {rng:>32} {ci:>16} {gp:>8} "
                     f"{_f(c.tokens_per_s_median, 0):>8} {_f(c.joules_per_output_token_median, 4):>8} {_f(c.coverage_min, 2):>5} {same:>9}")
    lines.append("")
    for n in dec.notes:
        lines.append(f"note: {n}")
    lines.append(f"candidates meeting the target in every repeat: {', '.join(dec.candidates) if dec.candidates else 'none'}")
    if dec.recommendation:
        lines.append(f"recommendation (advisory): {dec.recommendation}")
    return "\n".join(lines)
