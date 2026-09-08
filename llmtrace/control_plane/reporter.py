"""Reporter for generating analysis reports and baseline comparisons."""

from __future__ import annotations

import logging
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    RICH_AVAILABLE = False

from llmtrace.models.config import ReporterConfig
from llmtrace.models.trace import (
    DiagnosisCategory,
    MetricComparison,
    RequestTrace,
    RunEnergyLedger,
    TraceAnalysis,
)

logger = logging.getLogger(__name__)


def percentile(values: List[float], p: float) -> Optional[float]:
    """Nearest-rank percentile (p in [0, 100]); None for empty input."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def compare_metric(
    name: str, baseline: Optional[float], current: Optional[float], higher_is_worse: bool = True
) -> MetricComparison:
    """Percent change with explicit handling of missing values and zero baselines."""
    cmp = MetricComparison(metric=name, baseline=baseline, current=current, higher_is_worse=higher_is_worse)
    if baseline is None:
        cmp.status = "missing_baseline"
    elif current is None:
        cmp.status = "missing_current"
    elif baseline == 0 and current == 0:
        cmp.pct_change = 0.0
        cmp.status = "ok"
    elif baseline == 0:
        cmp.status = "zero_baseline"
    else:
        cmp.pct_change = (current - baseline) / baseline * 100.0
        cmp.status = "ok"
    return cmp


def _fmt(v: Optional[float], unit: str = "", digits: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}{unit}"


class Reporter:
    def __init__(self, config: Optional[ReporterConfig] = None):
        self.config = config or ReporterConfig()
        self.console = Console() if RICH_AVAILABLE and self.config.cli_rich_output else None

    # --------------------------------------------------------------- analysis

    def generate_analysis(
        self,
        traces: List[RequestTrace],
        baseline_traces: Optional[List[RequestTrace]] = None,
        ledger: Optional[RunEnergyLedger] = None,
        baseline_ledger: Optional[RunEnergyLedger] = None,
    ) -> TraceAnalysis:
        if not traces:
            return self._empty_analysis()

        start = min(t.start_time for t in traces)
        end = max(t.end_time for t in traces)

        ttfts = [t.ttft_ms for t in traces if t.ttft_ms is not None]
        tpots = [t.tpot_ms for t in traces if t.tpot_ms is not None]
        allocated = [t.energy.attributed_joules for t in traces if t.energy and t.energy.attributed_joules is not None]
        jpt = [
            t.energy.joules_per_output_token
            for t in traces
            if t.energy and t.energy.joules_per_output_token is not None
        ]

        all_samples = [s for t in traces for s in t.gpu_samples]
        utils = [s.gpu_utilization_pct for s in all_samples if s.gpu_utilization_pct is not None]
        powers = [s.power_draw_watts for s in all_samples if s.power_draw_watts is not None]

        diagnoses = [t.diagnosis for t in traces if t.diagnosis]
        counts: Dict[DiagnosisCategory, int] = {}
        for d in diagnoses:
            cat = DiagnosisCategory(d.category)
            counts[cat] = counts.get(cat, 0) + 1
        top = sorted(counts, key=lambda c: counts[c], reverse=True)

        status_counts: Dict[str, int] = {}
        for t in traces:
            key = t.status.value if hasattr(t.status, "value") else str(t.status)
            status_counts[key] = status_counts.get(key, 0) + 1

        analysis = TraceAnalysis(
            num_requests=len(traces),
            start_time=start,
            end_time=end,
            duration_s=end - start,
            num_with_ttft=len(ttfts),
            avg_ttft_ms=statistics.mean(ttfts) if ttfts else None,
            p50_ttft_ms=percentile(ttfts, 50),
            p95_ttft_ms=percentile(ttfts, 95),
            p99_ttft_ms=percentile(ttfts, 99),
            num_with_tpot=len(tpots),
            avg_tpot_ms=statistics.mean(tpots) if tpots else None,
            p50_tpot_ms=percentile(tpots, 50),
            p95_tpot_ms=percentile(tpots, 95),
            p99_tpot_ms=percentile(tpots, 99),
            energy_ledger=ledger,
            num_with_energy=len(allocated),
            total_device_joules=ledger.device_joules if ledger else None,
            attributed_joules=ledger.attributed_joules if ledger and ledger.device_joules is not None else None,
            unallocated_joules=(ledger.idle_joules + ledger.unattributable_joules)
            if ledger and ledger.device_joules is not None
            else None,
            avg_attributed_joules_per_request=statistics.mean(allocated) if allocated else None,
            avg_joules_per_output_token=statistics.mean(jpt) if jpt else None,
            diagnoses=diagnoses,
            top_issues=top,
            num_gpu_samples=len(all_samples),
            avg_gpu_utilization_pct=statistics.mean(utils) if utils else None,
            avg_power_draw_watts=statistics.mean(powers) if powers else None,
            throttle_incidents=sum(1 for s in all_samples if s.is_throttled),
            status_counts=status_counts,
        )

        if baseline_traces is not None:
            base = self.generate_analysis(baseline_traces, None, baseline_ledger) if baseline_traces else None
            analysis.regressions = self.compare(analysis, base)
        return analysis

    def compare(self, current: TraceAnalysis, baseline: Optional[TraceAnalysis]) -> Dict[str, MetricComparison]:
        def b(attr: str) -> Optional[float]:
            return getattr(baseline, attr) if baseline is not None else None

        cur_thr = current.throttle_incidents / current.num_requests if current.num_requests else None
        base_thr = (
            baseline.throttle_incidents / baseline.num_requests if baseline and baseline.num_requests else None
        )
        return {
            "p95_ttft_ms": compare_metric("p95_ttft_ms", b("p95_ttft_ms"), current.p95_ttft_ms),
            "p95_tpot_ms": compare_metric("p95_tpot_ms", b("p95_tpot_ms"), current.p95_tpot_ms),
            "joules_per_output_token": compare_metric(
                "joules_per_output_token", b("avg_joules_per_output_token"), current.avg_joules_per_output_token
            ),
            "throttle_incidents_per_request": compare_metric(
                "throttle_incidents_per_request", base_thr, cur_thr
            ),
        }

    # --------------------------------------------------------------- printing

    def print_analysis(self, analysis: TraceAnalysis) -> None:
        if self.console is not None:
            self._print_rich(analysis)
        else:
            print(analysis.summary())

    def _print_rich(self, a: TraceAnalysis) -> None:
        c = self.console
        assert c is not None
        c.print(Panel.fit("[bold cyan]llmtrace Analysis Report[/bold cyan]", border_style="cyan"))

        overview = Table(title="Overview", show_header=False)
        overview.add_column("Metric", style="cyan")
        overview.add_column("Value", style="green")
        overview.add_row("Requests", f"{a.num_requests} {a.status_counts}")
        overview.add_row("Duration", f"{a.duration_s:.2f}s")
        overview.add_row("Throughput", f"{a.num_requests / max(a.duration_s, 1e-3):.2f} req/s")
        c.print(overview)

        latency = Table(title="Latency (step-granular, measured at engine step boundaries)")
        for col in ("Metric", "N", "Avg", "P50", "P95", "P99"):
            latency.add_column(col)
        latency.add_row(
            "TTFT (ms)", str(a.num_with_ttft), _fmt(a.avg_ttft_ms), _fmt(a.p50_ttft_ms), _fmt(a.p95_ttft_ms), _fmt(a.p99_ttft_ms)
        )
        latency.add_row(
            "TPOT (ms)", str(a.num_with_tpot), _fmt(a.avg_tpot_ms), _fmt(a.p50_tpot_ms), _fmt(a.p95_tpot_ms), _fmt(a.p99_tpot_ms)
        )
        c.print(latency)

        energy = Table(title="Energy (integrated from sampled power; allocations are estimates)", show_header=False)
        energy.add_column("Metric", style="cyan")
        energy.add_column("Value", style="green")
        energy.add_row("Device energy in run window", _fmt(a.total_device_joules, " J"))
        energy.add_row("Attributed to requests", _fmt(a.attributed_joules, " J"))
        energy.add_row("Unallocated (idle/unattributable)", _fmt(a.unallocated_joules, " J"))
        energy.add_row("Requests with allocation", f"{a.num_with_energy}/{a.num_requests}")
        energy.add_row("Avg attributed per request", _fmt(a.avg_attributed_joules_per_request, " J"))
        energy.add_row("Avg per output token", _fmt(a.avg_joules_per_output_token, " J", 4))
        if a.energy_ledger:
            energy.add_row("Allocation policy", a.energy_ledger.allocation_policy)
            energy.add_row("Membership source", a.energy_ledger.membership_source)
            if a.energy_ledger.coverage:
                energy.add_row("Telemetry coverage", f"{a.energy_ledger.coverage.coverage_fraction:.0%}")
        energy.add_row("GPU samples", str(a.num_gpu_samples))
        energy.add_row("Avg GPU utilization", _fmt(a.avg_gpu_utilization_pct, "%", 1))
        energy.add_row("Avg power draw", _fmt(a.avg_power_draw_watts, " W", 1))
        energy.add_row("Throttle incidents", str(a.throttle_incidents))
        c.print(energy)

        if a.energy_ledger and a.energy_ledger.notes:
            for note in a.energy_ledger.notes:
                c.print(f"[yellow]note:[/yellow] {note}")

        if a.top_issues:
            issues = Table(title="Top Diagnosed Issues")
            issues.add_column("Issue", style="red")
            issues.add_column("Occurrences", style="yellow")
            for issue in a.top_issues[:5]:
                n = sum(1 for d in a.diagnoses if DiagnosisCategory(d.category) == issue)
                issues.add_row(issue.value, str(n))
            c.print(issues)

        if a.regressions:
            reg = Table(title="Comparison vs Baseline (positive = worse)")
            for col in ("Metric", "Baseline", "Current", "Change"):
                reg.add_column(col)
            for name, cmp in a.regressions.items():
                if cmp.pct_change is None:
                    change = f"[yellow]{cmp.status}[/yellow]"
                else:
                    color = "red" if cmp.pct_change > 0 else "green"
                    change = f"[{color}]{cmp.pct_change:+.2f}%[/{color}]"
                reg.add_row(name, _fmt(cmp.baseline, "", 4), _fmt(cmp.current, "", 4), change)
            c.print(reg)

    # ---------------------------------------------------------------- helpers

    def _empty_analysis(self) -> TraceAnalysis:
        return TraceAnalysis(num_requests=0, start_time=0.0, end_time=0.0, duration_s=0.0)

    def export_to_file(self, analysis: TraceAnalysis, filepath: Path) -> None:
        filepath = Path(filepath)
        if filepath.suffix == ".json":
            filepath.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
        else:
            filepath.write_text(analysis.summary(), encoding="utf-8")
        logger.info("Analysis exported to %s", filepath)
