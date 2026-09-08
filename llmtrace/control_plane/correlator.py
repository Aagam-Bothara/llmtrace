"""Correlator: aligns request windows with GPU telemetry and keeps an energy ledger.

Quantities and their status
---------------------------
* **Telemetry** (measured): ``GPUSample.power_draw_watts`` readings from NVML.
* **Integrated device energy** (estimate from sampling): trapezoidal
  integration of each GPU's power over time, done *per GPU* on that GPU's own
  timestamps, then summed across GPUs. No energy is integrated across gaps
  longer than ``EnergyConfig.max_sample_gap_s``; such time is "uncovered".
* **Attributed energy** (allocation estimate): device energy in each elementary
  time interval is split among the requests active in that interval according
  to ``EnergyConfig.attribution_method``. Membership comes from scheduler batch
  metadata when available, otherwise from the request window (arrival to
  completion, which includes queue wait). ``window_only`` performs no
  allocation.

Conservation: for the run window ``[first arrival, last completion]``,
``device_joules == attributed + idle + unattributable`` up to floating point
rounding, where *idle* is energy in intervals with no active request and
*unattributable* is energy in intervals whose requests had insufficient
telemetry coverage (or, under ``window_only``, all active-interval energy).
Energy outside the run window is not accounted at all.

Insufficient telemetry never yields a zero figure; it yields ``None`` with a
reason.
"""

from __future__ import annotations

import logging
import math
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from llmtrace.models.config import EnergyConfig
from llmtrace.models.trace import (
    BatchMetadata,
    EnergyAttribution,
    EnergyCoverage,
    GPUSample,
    RequestSpan,
    RequestTrace,
    RunEnergyLedger,
    SpanPhase,
)

logger = logging.getLogger(__name__)

CONSERVATION_REL_TOL = 1e-9
CONSERVATION_ABS_TOL = 1e-6


class CumulativePower:
    """Piecewise-linear power curve for one GPU with O(log n) energy queries."""

    def __init__(self, points: Sequence[Tuple[float, float]], max_gap_s: float):
        dedup: Dict[float, float] = {}
        for t, p in points:
            dedup[t] = p  # identical timestamps: keep the last reading
        items = sorted(dedup.items())
        self.t = [t for t, _ in items]
        self.p = [p for _, p in items]
        n = len(self.t)
        self.cum = [0.0] * n
        self.covered = [False] * max(n - 1, 0)
        for i in range(n - 1):
            dt = self.t[i + 1] - self.t[i]
            e = 0.0
            if 0 < dt <= max_gap_s:
                e = 0.5 * (self.p[i] + self.p[i + 1]) * dt
                self.covered[i] = True
            self.cum[i + 1] = self.cum[i] + e

    @property
    def num_points(self) -> int:
        return len(self.t)

    def _energy_to(self, x: float) -> float:
        n = len(self.t)
        if n == 0 or x <= self.t[0]:
            return 0.0
        if x >= self.t[-1]:
            return self.cum[-1]
        i = bisect_right(self.t, x) - 1  # t[i] <= x < t[i+1]
        e = self.cum[i]
        if self.covered[i]:
            frac = (x - self.t[i]) / (self.t[i + 1] - self.t[i])
            px = self.p[i] + (self.p[i + 1] - self.p[i]) * frac
            e += 0.5 * (self.p[i] + px) * (x - self.t[i])
        return e

    def energy(self, a: float, b: float) -> float:
        if b <= a:
            return 0.0
        return self._energy_to(b) - self._energy_to(a)

    def covered_seconds(self, a: float, b: float) -> float:
        total = 0.0
        for i in range(len(self.t) - 1):
            if not self.covered[i]:
                continue
            lo, hi = max(a, self.t[i]), min(b, self.t[i + 1])
            if hi > lo:
                total += hi - lo
        return total

    def samples_in(self, a: float, b: float) -> int:
        return sum(1 for t in self.t if a <= t <= b)

    def max_gap_in(self, a: float, b: float) -> Optional[float]:
        inside = [t for t in self.t if a <= t <= b]
        if len(inside) < 2:
            return None
        return max(inside[i + 1] - inside[i] for i in range(len(inside) - 1))


@dataclass
class _Interval:
    start: float
    end: float


@dataclass
class _Alloc:
    joules: float = 0.0
    phase: Dict[SpanPhase, float] = field(default_factory=dict)


@dataclass
class CorrelationResult:
    traces: List[RequestTrace]
    ledger: RunEnergyLedger


class Correlator:
    """Aligns request traces with GPU samples and computes the energy ledger."""

    def __init__(self, energy_config: EnergyConfig):
        self.config = energy_config
        self._unusable_batches = 0

    # ------------------------------------------------------------------ public

    def correlate_traces(
        self,
        traces: List[RequestTrace],
        gpu_samples: List[GPUSample],
        batches: Optional[List[BatchMetadata]] = None,
    ) -> List[RequestTrace]:
        return self.correlate(traces, gpu_samples, batches).traces

    def correlate(
        self,
        traces: List[RequestTrace],
        gpu_samples: List[GPUSample],
        batches: Optional[List[BatchMetadata]] = None,
    ) -> CorrelationResult:
        clock = self._choose_clock(traces, gpu_samples)
        if not traces:
            return CorrelationResult([], RunEnergyLedger(window_start=0.0, window_end=0.0, clock=clock))

        starts = [self._trace_bounds(t, clock)[0] for t in traces]
        ends = [self._trace_bounds(t, clock)[1] for t in traces]
        run_start, run_end = min(starts), max(ends)

        # Attach raw samples in each request window (for diagnosis rules and inspection).
        by_ts = sorted(gpu_samples, key=lambda s: self._sample_time(s, clock))
        for t in traces:
            a, b = self._trace_bounds(t, clock)
            t.gpu_samples = [s for s in by_ts if a <= self._sample_time(s, clock) <= b]

        ledger = RunEnergyLedger(
            window_start=run_start,
            window_end=run_end,
            clock=clock,
            allocation_policy=self.config.attribution_method,
            num_requests=len(traces),
        )
        if not self.config.enabled:
            ledger.notes.append("energy accounting disabled by config")
            return CorrelationResult(traces, ledger)

        curves, no_power = self._build_curves(gpu_samples, clock)
        ledger.samples_without_power = no_power
        if not curves:
            ledger.notes.append("no GPU power samples; energy unavailable")
            for t in traces:
                t.energy = self._unavailable(t, "no GPU power telemetry", ledger.membership_source)
            ledger.num_requests_without_telemetry = len(traces)
            return CorrelationResult(traces, ledger)

        ledger.per_gpu_joules = {g: c.energy(run_start, run_end) for g, c in curves.items()}
        ledger.device_joules = sum(ledger.per_gpu_joules.values())
        ledger.coverage = self._coverage(curves, run_start, run_end, clock)

        self._unusable_batches = 0
        intervals, membership = self._membership(traces, batches, clock)
        ledger.membership_source = membership
        if self._unusable_batches:
            ledger.notes.append(
                f"{self._unusable_batches} batch record(s) without a step end time were ignored for membership"
            )

        allocs, idle = self._sweep(traces, intervals, curves, run_start, run_end, clock)
        ledger.idle_joules = idle

        attributed_total = 0.0
        unattributable = 0.0
        for t in traces:
            a, b = self._trace_bounds(t, clock)
            cov = self._coverage(curves, a, b, clock)
            window_j = sum(c.energy(a, b) for c in curves.values())
            alloc = allocs.get(t.request_id, _Alloc())
            enough = cov.num_samples >= 2 and cov.coverage_fraction >= self.config.min_coverage_fraction
            if not enough:
                t.energy = self._unavailable(
                    t,
                    f"insufficient telemetry: {cov.num_samples} power samples, "
                    f"{cov.coverage_fraction:.0%} of window covered",
                    membership,
                    coverage=cov,
                )
                unattributable += alloc.joules
                ledger.num_requests_without_telemetry += 1
                continue
            if self.config.attribution_method == "window_only":
                t.energy = EnergyAttribution(
                    request_id=t.request_id,
                    window_device_joules=window_j,
                    allocation_policy="window_only",
                    membership_source=membership,
                    is_allocated=False,
                    unavailable_reason="window_only policy: device energy over the request window "
                    "is reported; it is shared with concurrent requests and not per-request consumption",
                    coverage=cov,
                )
                unattributable += alloc.joules
                continue
            phases = {p: alloc.phase.get(p) for p in SpanPhase}
            t.energy = EnergyAttribution(
                request_id=t.request_id,
                window_device_joules=window_j,
                attributed_joules=alloc.joules,
                joules_per_output_token=(alloc.joules / t.output_length) if t.output_length > 0 else None,
                queue_joules=phases[SpanPhase.QUEUE],
                prefill_joules=phases[SpanPhase.PREFILL],
                time_to_first_token_joules=phases[SpanPhase.TIME_TO_FIRST_TOKEN],
                decode_joules=phases[SpanPhase.DECODE],
                cost_usd=self._cost(alloc.joules),
                energy_price_usd_per_kwh=self.config.energy_price_usd_per_kwh,
                allocation_policy=self.config.attribution_method,
                membership_source=membership,
                is_allocated=True,
                coverage=cov,
            )
            attributed_total += alloc.joules
            ledger.num_requests_allocated += 1

        ledger.attributed_joules = attributed_total
        ledger.unattributable_joules = unattributable
        total = ledger.device_joules
        err = abs(attributed_total + idle + unattributable - total)
        ledger.conservation_error_joules = err
        if err > max(CONSERVATION_ABS_TOL, CONSERVATION_REL_TOL * abs(total)):
            msg = f"energy conservation violated: error {err:.6g} J of {total:.6g} J"
            logger.error(msg)
            ledger.notes.append(msg)
        if membership == "request_window":
            ledger.notes.append(
                "membership from request windows (arrival to completion, includes queue wait); "
                "scheduler batch metadata was not available"
            )
        return CorrelationResult(traces, ledger)

    # ------------------------------------------------------------ clock utils

    @staticmethod
    def _choose_clock(traces: List[RequestTrace], samples: List[GPUSample]) -> str:
        domains = {t.clock_domain for t in traces} | {s.clock_domain for s in samples}
        have_mono = all(t.start_monotonic is not None and t.end_monotonic is not None for t in traces) and all(
            s.monotonic is not None for s in samples
        )
        if traces and samples and have_mono and len(domains) == 1 and None not in domains:
            return "monotonic"
        return "wall"

    @staticmethod
    def _trace_bounds(t: RequestTrace, clock: str) -> Tuple[float, float]:
        if clock == "monotonic" and t.start_monotonic is not None and t.end_monotonic is not None:
            return t.start_monotonic, t.end_monotonic
        return t.start_time, t.end_time

    @staticmethod
    def _span_bounds(s: RequestSpan, clock: str) -> Tuple[float, float]:
        if clock == "monotonic" and s.start_monotonic is not None and s.end_monotonic is not None:
            return s.start_monotonic, s.end_monotonic
        return s.start_time, s.end_time

    @staticmethod
    def _sample_time(s: GPUSample, clock: str) -> float:
        if clock == "monotonic" and s.monotonic is not None:
            return s.monotonic
        return s.timestamp

    # ------------------------------------------------------------- building

    def _build_curves(self, samples: List[GPUSample], clock: str) -> Tuple[Dict[int, CumulativePower], int]:
        by_gpu: Dict[int, List[Tuple[float, float]]] = {}
        no_power = 0
        for s in samples:
            if s.power_draw_watts is None or not math.isfinite(s.power_draw_watts):
                no_power += 1
                continue
            by_gpu.setdefault(s.gpu_id, []).append((self._sample_time(s, clock), float(s.power_draw_watts)))
        curves = {g: CumulativePower(pts, self.config.max_sample_gap_s) for g, pts in by_gpu.items()}
        return {g: c for g, c in curves.items() if c.num_points > 0}, no_power

    def _coverage(self, curves: Dict[int, CumulativePower], a: float, b: float, clock: str) -> EnergyCoverage:
        window = max(b - a, 0.0)
        covered = [c.covered_seconds(a, b) for c in curves.values()]
        gaps = [g for g in (c.max_gap_in(a, b) for c in curves.values()) if g is not None]
        mean_covered = sum(covered) / len(covered) if covered else 0.0
        return EnergyCoverage(
            window_s=window,
            covered_s=mean_covered,
            coverage_fraction=(mean_covered / window) if window > 0 else 0.0,
            num_samples=sum(c.samples_in(a, b) for c in curves.values()),
            gpu_ids=sorted(curves.keys()),
            max_gap_s=max(gaps) if gaps else None,
            clock=clock,
        )

    def _membership(
        self, traces: List[RequestTrace], batches: Optional[List[BatchMetadata]], clock: str
    ) -> Tuple[Dict[str, List[_Interval]], str]:
        """Return active intervals per request and the membership source used."""
        if self.config.attribution_method == "window_only":
            return {t.request_id: [_Interval(*self._trace_bounds(t, clock))] for t in traces}, "request_window"

        if batches and clock == "monotonic":
            usable = [b for b in batches if b.monotonic is not None and b.step_end_monotonic is not None and b.request_ids]
            self._unusable_batches = len(batches) - len(usable)
            if usable:
                ids = {t.request_id for t in traces}
                intervals: Dict[str, List[_Interval]] = {}
                for b in usable:
                    for rid in b.request_ids:
                        if rid in ids and b.step_end_monotonic > b.monotonic:
                            intervals.setdefault(rid, []).append(_Interval(b.monotonic, b.step_end_monotonic))
                if intervals:
                    return intervals, "batch_metadata"
        return {t.request_id: [_Interval(*self._trace_bounds(t, clock))] for t in traces}, "request_window"

    def _weight(self, t: RequestTrace) -> float:
        if self.config.attribution_method == "proportional_tokens":
            return float((t.prompt_length or 0) + t.output_length)
        return 1.0

    def _sweep(
        self,
        traces: List[RequestTrace],
        intervals: Dict[str, List[_Interval]],
        curves: Dict[int, CumulativePower],
        run_start: float,
        run_end: float,
        clock: str,
    ) -> Tuple[Dict[str, _Alloc], float]:
        by_id = {t.request_id: t for t in traces}
        events: List[Tuple[float, int, str]] = []  # (time, +1 start / -1 end, request_id)
        for rid, ivs in intervals.items():
            for iv in ivs:
                a, b = max(iv.start, run_start), min(iv.end, run_end)
                if b > a:
                    events.append((a, 1, rid))
                    events.append((b, -1, rid))
        # Span edges are boundaries too, so every elementary interval lies wholly inside or
        # outside each phase span and phase energy is the integrated curve, not a time fraction.
        span_edges = set()
        for t in traces:
            for span in t.spans:
                for x in self._span_bounds(span, clock):
                    if run_start < x < run_end:
                        span_edges.add(x)
        boundaries = sorted({run_start, run_end} | {e[0] for e in events} | span_edges)
        events.sort(key=lambda e: (e[0], e[1]))  # ends before starts at equal times

        allocs: Dict[str, _Alloc] = {}
        active: Dict[str, int] = {}
        idle = 0.0
        ei = 0
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i + 1]
            while ei < len(events) and events[ei][0] <= a:
                _, kind, rid = events[ei]
                active[rid] = active.get(rid, 0) + kind
                if active[rid] <= 0:
                    del active[rid]
                ei += 1
            e = sum(c.energy(a, b) for c in curves.values())
            if e == 0.0:
                continue
            if not active:
                idle += e
                continue
            weights = {rid: self._weight(by_id[rid]) for rid in active}
            wsum = sum(weights.values())
            if wsum <= 0:
                weights = {rid: 1.0 for rid in active}
                wsum = float(len(active))
            for rid, w in weights.items():
                share = e * w / wsum
                alloc = allocs.setdefault(rid, _Alloc())
                alloc.joules += share
                for span in by_id[rid].spans:
                    sa, sb = self._span_bounds(span, clock)
                    if sa <= a and sb >= b:  # interval lies inside the span (never partial, see above)
                        alloc.phase[span.phase] = alloc.phase.get(span.phase, 0.0) + share
        return allocs, idle

    # ---------------------------------------------------------------- helpers

    def _cost(self, joules: float) -> Optional[float]:
        if self.config.energy_price_usd_per_kwh is None:
            return None
        return joules / 3.6e6 * self.config.energy_price_usd_per_kwh

    def _unavailable(
        self, t: RequestTrace, reason: str, membership: str, coverage: Optional[EnergyCoverage] = None
    ) -> EnergyAttribution:
        return EnergyAttribution(
            request_id=t.request_id,
            allocation_policy=self.config.attribution_method,
            membership_source=membership,
            is_allocated=False,
            unavailable_reason=reason,
            coverage=coverage,
        )

    # Kept for tests and simple callers: integrate one GPU's samples on its own timestamps.
    def integrate_power(self, samples: List[GPUSample], clock: str = "wall") -> Optional[float]:
        pts = [
            (self._sample_time(s, clock), float(s.power_draw_watts))
            for s in samples
            if s.power_draw_watts is not None
        ]
        if len(pts) < 2:
            return None
        curve = CumulativePower(pts, self.config.max_sample_gap_s)
        return curve.energy(curve.t[0], curve.t[-1])
