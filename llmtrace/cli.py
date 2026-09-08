"""Command-line interface for llmtrace (offline analysis; no GPU or vLLM needed)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import click

from llmtrace import __version__, io
from llmtrace.control_plane.correlator import Correlator, CorrelationResult
from llmtrace.control_plane.reporter import Reporter
from llmtrace.control_plane.rules_engine import RulesEngine
from llmtrace.models.config import AutopsyConfig, EnergyConfig, ReporterConfig, TracerConfig
from llmtrace.models.trace import MetricComparison

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_USAGE = 2
EXIT_NOT_IMPLEMENTED = 3


@click.group()
@click.version_option(version=__version__)
@click.option("-v", "--verbose", is_flag=True, help="Enable INFO logging")
def main(verbose: bool) -> None:
    """llmtrace - flight recorder, attribution and autopsy for vLLM inference."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _correlate_dir(
    trace_paths: List[str], gpu_paths: Optional[List[str]], attribution: str
) -> CorrelationResult:
    traces = io.load_traces(trace_paths)
    dirs = io.run_directories_for(trace_paths)
    samples = io.load_gpu_samples(gpu_paths if gpu_paths else dirs)
    batches = io.load_batches(dirs)
    correlator = Correlator(EnergyConfig(attribution_method=attribution))  # type: ignore[arg-type]
    return correlator.correlate(traces, samples, batches)


@main.command()
@click.argument("trace_paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--gpu-samples", "gpu_paths", multiple=True, type=click.Path(exists=True),
              help="GPU sample file(s)/dir(s); default: gpu_*.jsonl next to the trace files")
@click.option("--baseline", type=click.Path(exists=True), help="Baseline run directory for comparison")
@click.option("--attribution", default="equal_share",
              type=click.Choice(["equal_share", "proportional_tokens", "window_only"]))
@click.option("--output", type=click.Path(), help="Write report (.json for machine-readable, else text)")
@click.option("--no-rich", is_flag=True, help="Plain text output")
def analyze(trace_paths: Tuple[str, ...], gpu_paths: Tuple[str, ...], baseline: Optional[str],
            attribution: str, output: Optional[str], no_rich: bool) -> None:
    """Analyze trace files or run directories."""
    result = _correlate_dir(list(trace_paths), list(gpu_paths) or None, attribution)
    if not result.traces:
        click.echo("No traces found", err=True)
        sys.exit(EXIT_USAGE)
    rules = RulesEngine(AutopsyConfig())
    for t in result.traces:
        t.diagnosis = rules.diagnose_request(t)

    baseline_traces = baseline_ledger = None
    if baseline:
        b = _correlate_dir([baseline], None, attribution)
        baseline_traces, baseline_ledger = b.traces, b.ledger
        if not baseline_traces:
            click.echo(f"Warning: no baseline traces in {baseline}", err=True)

    vllm_stats = io.load_vllm_stats(io.run_directories_for(list(trace_paths)))
    if vllm_stats:
        from llmtrace.control_plane.reporter import summarize_vllm_stats
        result.ledger.notes.append("vLLM engine stats (stat_loggers hook): " + summarize_vllm_stats(vllm_stats)["text"])
    reporter = Reporter(ReporterConfig(cli_rich_output=not no_rich))
    analysis = reporter.generate_analysis(result.traces, baseline_traces, result.ledger, baseline_ledger)
    reporter.print_analysis(analysis)
    if output:
        reporter.export_to_file(analysis, Path(output))
        click.echo(f"Report written to {output}")


def evaluate_regressions(
    comparisons: dict, thresholds: dict
) -> Tuple[List[str], List[str], List[str]]:
    """Classify comparisons. Returns (regressions, improvements_or_ok, unavailable).

    A metric regresses only when its percent change is *positive* and exceeds
    the threshold (all compared metrics are higher-is-worse). Negative changes
    are improvements and never fail the check.
    """
    regressions, ok, unavailable = [], [], []
    for name, threshold in thresholds.items():
        cmp: Optional[MetricComparison] = comparisons.get(name)
        if cmp is None or cmp.pct_change is None:
            status = cmp.status if cmp is not None else "missing"
            unavailable.append(f"{name}: {status} (baseline={cmp.baseline if cmp else None}, "
                               f"current={cmp.current if cmp else None})")
            continue
        line = f"{name}: {cmp.pct_change:+.2f}% (threshold +{threshold}%)"
        if cmp.pct_change > threshold:
            regressions.append(line)
        else:
            ok.append(line)
    return regressions, ok, unavailable


@main.command()
@click.option("--baseline", required=True, type=click.Path(exists=True), help="Baseline run directory")
@click.option("--current", required=True, type=click.Path(exists=True), help="Current run directory")
@click.option("--ttft-threshold", default=5.0, type=float, help="Max allowed p95 TTFT increase (%)")
@click.option("--tpot-threshold", default=None, type=float, help="Max allowed p95 TPOT increase (%)")
@click.option("--energy-threshold", default=10.0, type=float, help="Max allowed J/output-token increase (%)")
@click.option("--attribution", default="equal_share",
              type=click.Choice(["equal_share", "proportional_tokens", "window_only"]))
@click.option("--fail-on-regression", is_flag=True, help="Exit 1 if any threshold is exceeded")
@click.option("--fail-on-missing", is_flag=True, help="Exit 1 if a thresholded metric is unavailable")
@click.option("--no-rich", is_flag=True, help="Plain text output")
def compare(baseline: str, current: str, ttft_threshold: float, tpot_threshold: Optional[float],
            energy_threshold: float, attribution: str, fail_on_regression: bool, fail_on_missing: bool,
            no_rich: bool) -> None:
    """Compare a current run against a baseline run (positive change = worse)."""
    base = _correlate_dir([baseline], None, attribution)
    cur = _correlate_dir([current], None, attribution)
    if not base.traces or not cur.traces:
        click.echo(f"Error: baseline has {len(base.traces)} traces, current has {len(cur.traces)}", err=True)
        sys.exit(EXIT_USAGE)

    reporter = Reporter(ReporterConfig(cli_rich_output=not no_rich))
    analysis = reporter.generate_analysis(cur.traces, base.traces, cur.ledger, base.ledger)
    reporter.print_analysis(analysis)

    thresholds = {"p95_ttft_ms": ttft_threshold, "joules_per_output_token": energy_threshold}
    if tpot_threshold is not None:
        thresholds["p95_tpot_ms"] = tpot_threshold
    regressions, ok, unavailable = evaluate_regressions(analysis.regressions, thresholds)

    click.echo("\nRegression check (positive change = worse):")
    for line in ok:
        click.echo(f"  ok         {line}")
    for line in unavailable:
        click.echo(f"  unavailable {line}")
    for line in regressions:
        click.echo(f"  REGRESSION {line}")

    code = EXIT_OK
    if regressions and fail_on_regression:
        code = EXIT_REGRESSION
    if unavailable and fail_on_missing:
        code = EXIT_REGRESSION
    if code != EXIT_OK:
        click.echo("FAILED")
    elif regressions:
        click.echo("Regressions found (not failing: --fail-on-regression not set)")
    else:
        click.echo("PASSED")
    sys.exit(code)


@main.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--compare", "compare_dir", type=click.Path(exists=True), help="Second run directory for side-by-side report")
@click.option("--trace-out", type=click.Path(), help="Write a Chrome/Perfetto trace JSON here (open at ui.perfetto.dev)")
@click.option("--html-out", type=click.Path(), help="Write a self-contained HTML report here")
@click.option("--chunk-threshold", type=int, default=128, help="Highlight steps whose largest prefill chunk exceeds this")
@click.option("--title", default="llmtrace run report")
def visualize(run_dir: str, compare_dir: Optional[str], trace_out: Optional[str], html_out: Optional[str],
              chunk_threshold: int, title: str) -> None:
    """Export a Perfetto trace and/or an HTML report from a recorded run directory."""
    from llmtrace.visualize import RunData, export_chrome_trace, render_html_report

    if not trace_out and not html_out:
        click.echo("Nothing to do: pass --trace-out and/or --html-out", err=True)
        sys.exit(EXIT_USAGE)
    run = RunData.load(run_dir)
    if not run.traces:
        click.echo(f"No traces in {run_dir}", err=True)
        sys.exit(EXIT_USAGE)
    if trace_out:
        counts = export_chrome_trace(run, trace_out)
        click.echo(f"Perfetto trace written to {trace_out} ({counts['events']} events; open at https://ui.perfetto.dev)")
    if html_out:
        cmp = RunData.load(compare_dir) if compare_dir else None
        render_html_report(run, html_out, cmp, chunk_threshold, title)
        click.echo(f"HTML report written to {html_out}")


@main.command()
@click.option("--pid", type=int, help="(not implemented)")
def monitor(pid: Optional[int]) -> None:
    """Attach to a running vLLM process. NOT IMPLEMENTED."""
    click.echo(
        "monitor is not implemented: llmtrace cannot attach to an external process. "
        "Use LLMTracer.instrument_engine() inside the vLLM process.",
        err=True,
    )
    sys.exit(EXIT_NOT_IMPLEMENTED)


@main.command("init-config")
@click.option("--output", default="llmtrace_config.json", help="Output config file path")
def init_config(output: str) -> None:
    """Write a default configuration file (load with LLMTracer.from_config_file)."""
    with open(output, "w", encoding="utf-8") as f:
        json.dump(TracerConfig().model_dump(), f, indent=2)
    click.echo(f"Default configuration written to {output}")


if __name__ == "__main__":
    main()
