"""Reporter for generating analysis reports and exports."""

import logging
import statistics
from typing import List, Dict, Optional
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from llmtrace.models.config import ReporterConfig
from llmtrace.models.trace import RequestTrace, TraceAnalysis, DiagnosisCategory

logger = logging.getLogger(__name__)


class Reporter:
    """
    Generates analysis reports and exports.

    Supports:
    - Rich CLI output
    - Plain text summaries
    - Export to various formats
    """

    def __init__(self, config: ReporterConfig):
        self.config = config
        self.console = Console() if RICH_AVAILABLE and config.cli_rich_output else None

    def generate_analysis(
        self,
        traces: List[RequestTrace],
        baseline_traces: Optional[List[RequestTrace]] = None,
    ) -> TraceAnalysis:
        """
        Generate comprehensive analysis from traces.

        Args:
            traces: Current traces to analyze
            baseline_traces: Optional baseline traces for regression detection

        Returns:
            TraceAnalysis object
        """
        if not traces:
            logger.warning("No traces to analyze")
            return self._empty_analysis()

        # Compute aggregate metrics
        num_requests = len(traces)
        start_time = min(t.start_time for t in traces)
        end_time = max(t.end_time for t in traces)
        duration_s = end_time - start_time

        # TTFT metrics
        ttfts = [t.ttft_ms for t in traces if t.ttft_ms is not None]
        avg_ttft = statistics.mean(ttfts) if ttfts else 0.0
        p50_ttft = statistics.median(ttfts) if ttfts else 0.0
        p95_ttft = self._percentile(ttfts, 95) if ttfts else 0.0
        p99_ttft = self._percentile(ttfts, 99) if ttfts else 0.0

        # TPOT metrics
        tpots = [t.tpot_ms for t in traces if t.tpot_ms is not None]
        avg_tpot = statistics.mean(tpots) if tpots else 0.0
        p50_tpot = statistics.median(tpots) if tpots else 0.0
        p95_tpot = self._percentile(tpots, 95) if tpots else 0.0
        p99_tpot = self._percentile(tpots, 99) if tpots else 0.0

        # Energy metrics
        energies = [t.energy.total_joules for t in traces if t.energy]
        total_joules = sum(energies)
        avg_joules_per_request = statistics.mean(energies) if energies else 0.0

        joules_per_token_list = [t.energy.joules_per_token for t in traces if t.energy]
        avg_joules_per_token = statistics.mean(joules_per_token_list) if joules_per_token_list else 0.0

        # GPU stats
        all_samples = [s for t in traces for s in t.gpu_samples]
        avg_gpu_util = statistics.mean(s.gpu_utilization_pct for s in all_samples) if all_samples else 0.0
        avg_power = statistics.mean(s.power_draw_watts for s in all_samples) if all_samples else 0.0
        throttle_incidents = sum(1 for s in all_samples if s.is_throttled)

        # Collect diagnoses
        diagnoses = [t.diagnosis for t in traces if t.diagnosis]

        # Count top issues
        issue_counts: Dict[DiagnosisCategory, int] = {}
        for diag in diagnoses:
            issue_counts[diag.category] = issue_counts.get(diag.category, 0) + 1

        top_issues = sorted(issue_counts.keys(), key=lambda c: issue_counts[c], reverse=True)

        # Regression detection
        regressions = {}
        if baseline_traces:
            regressions = self._detect_regressions(traces, baseline_traces)

        analysis = TraceAnalysis(
            num_requests=num_requests,
            start_time=start_time,
            end_time=end_time,
            duration_s=duration_s,
            avg_ttft_ms=avg_ttft,
            p50_ttft_ms=p50_ttft,
            p95_ttft_ms=p95_ttft,
            p99_ttft_ms=p99_ttft,
            avg_tpot_ms=avg_tpot,
            p50_tpot_ms=p50_tpot,
            p95_tpot_ms=p95_tpot,
            p99_tpot_ms=p99_tpot,
            total_joules=total_joules,
            avg_joules_per_request=avg_joules_per_request,
            avg_joules_per_token=avg_joules_per_token,
            diagnoses=diagnoses,
            top_issues=top_issues,
            regressions=regressions,
            avg_gpu_utilization_pct=avg_gpu_util,
            avg_power_draw_watts=avg_power,
            throttle_incidents=throttle_incidents,
        )

        return analysis

    def print_analysis(self, analysis: TraceAnalysis) -> None:
        """Print analysis to console."""
        if self.console and RICH_AVAILABLE:
            self._print_rich_analysis(analysis)
        else:
            # Fallback to plain text
            print(analysis.summary())

    def _print_rich_analysis(self, analysis: TraceAnalysis) -> None:
        """Print analysis with rich formatting."""
        # Title
        self.console.print(
            Panel.fit(
                "[bold cyan]llmtrace Analysis Report[/bold cyan]",
                border_style="cyan",
            )
        )

        # Overview
        overview = Table(title="Overview", show_header=False)
        overview.add_column("Metric", style="cyan")
        overview.add_column("Value", style="green")
        overview.add_row("Requests", str(analysis.num_requests))
        overview.add_row("Duration", f"{analysis.duration_s:.2f}s")
        overview.add_row("Throughput", f"{analysis.num_requests / max(analysis.duration_s, 0.001):.2f} req/s")
        self.console.print(overview)

        # Latency
        latency = Table(title="Latency Metrics", show_header=True)
        latency.add_column("Metric", style="cyan")
        latency.add_column("Avg", style="yellow")
        latency.add_column("P50", style="yellow")
        latency.add_column("P95", style="red")
        latency.add_column("P99", style="bold red")

        latency.add_row(
            "TTFT (ms)",
            f"{analysis.avg_ttft_ms:.2f}",
            f"{analysis.p50_ttft_ms:.2f}",
            f"{analysis.p95_ttft_ms:.2f}",
            f"{analysis.p99_ttft_ms:.2f}",
        )
        latency.add_row(
            "TPOT (ms)",
            f"{analysis.avg_tpot_ms:.2f}",
            f"{analysis.p50_tpot_ms:.2f}",
            f"{analysis.p95_tpot_ms:.2f}",
            f"{analysis.p99_tpot_ms:.2f}",
        )
        self.console.print(latency)

        # Energy
        energy = Table(title="Energy & Efficiency", show_header=False)
        energy.add_column("Metric", style="cyan")
        energy.add_column("Value", style="green")
        energy.add_row("Total Energy", f"{analysis.total_joules:.2f} J")
        energy.add_row("Avg per Request", f"{analysis.avg_joules_per_request:.2f} J")
        energy.add_row("Avg per Token", f"{analysis.avg_joules_per_token:.4f} J")
        energy.add_row("Avg GPU Utilization", f"{analysis.avg_gpu_utilization_pct:.1f}%")
        energy.add_row("Avg Power Draw", f"{analysis.avg_power_draw_watts:.1f} W")
        energy.add_row("Throttle Incidents", str(analysis.throttle_incidents))
        self.console.print(energy)

        # Top Issues
        if analysis.top_issues:
            issues = Table(title="Top Diagnosed Issues", show_header=True)
            issues.add_column("Issue", style="red")
            issues.add_column("Occurrences", style="yellow")

            for issue in analysis.top_issues[:5]:
                count = sum(1 for d in analysis.diagnoses if d.category == issue)
                issues.add_row(issue.value, str(count))

            self.console.print(issues)

        # Regressions
        if analysis.regressions:
            regressions = Table(title="Regressions vs Baseline", show_header=True)
            regressions.add_column("Metric", style="cyan")
            regressions.add_column("Change", style="red")

            for metric, pct_change in analysis.regressions.items():
                sign = "+" if pct_change > 0 else ""
                color = "red" if pct_change > 0 else "green"
                regressions.add_row(metric, f"[{color}]{sign}{pct_change:.2f}%[/{color}]")

            self.console.print(regressions)

    def _detect_regressions(
        self, current: List[RequestTrace], baseline: List[RequestTrace]
    ) -> Dict[str, float]:
        """Detect regressions compared to baseline."""
        regressions = {}

        # TTFT regression
        current_ttfts = [t.ttft_ms for t in current if t.ttft_ms]
        baseline_ttfts = [t.ttft_ms for t in baseline if t.ttft_ms]

        if current_ttfts and baseline_ttfts:
            current_p95_ttft = self._percentile(current_ttfts, 95)
            baseline_p95_ttft = self._percentile(baseline_ttfts, 95)
            pct_change = ((current_p95_ttft - baseline_p95_ttft) / baseline_p95_ttft) * 100
            regressions["p95_ttft"] = pct_change

        # Energy regression
        current_energies = [t.energy.joules_per_token for t in current if t.energy]
        baseline_energies = [t.energy.joules_per_token for t in baseline if t.energy]

        if current_energies and baseline_energies:
            current_avg_energy = statistics.mean(current_energies)
            baseline_avg_energy = statistics.mean(baseline_energies)
            pct_change = ((current_avg_energy - baseline_avg_energy) / baseline_avg_energy) * 100
            regressions["joules_per_token"] = pct_change

        # Throttling regression
        current_throttles = sum(
            1 for t in current for s in t.gpu_samples if s.is_throttled
        )
        baseline_throttles = sum(
            1 for t in baseline for s in t.gpu_samples if s.is_throttled
        )

        if baseline_throttles > 0:
            pct_change = ((current_throttles - baseline_throttles) / baseline_throttles) * 100
            regressions["throttle_incidents"] = pct_change

        return regressions

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def _empty_analysis(self) -> TraceAnalysis:
        """Return empty analysis."""
        return TraceAnalysis(
            num_requests=0,
            start_time=0.0,
            end_time=0.0,
            duration_s=0.0,
            avg_ttft_ms=0.0,
            p50_ttft_ms=0.0,
            p95_ttft_ms=0.0,
            p99_ttft_ms=0.0,
            avg_tpot_ms=0.0,
            p50_tpot_ms=0.0,
            p95_tpot_ms=0.0,
            p99_tpot_ms=0.0,
            total_joules=0.0,
            avg_joules_per_request=0.0,
            avg_joules_per_token=0.0,
        )

    def export_to_file(self, analysis: TraceAnalysis, filepath: Path) -> None:
        """Export analysis to file."""
        with open(filepath, "w") as f:
            f.write(analysis.summary())

        logger.info(f"Analysis exported to {filepath}")
