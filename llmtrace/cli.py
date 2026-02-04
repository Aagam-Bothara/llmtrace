"""Command-line interface for llmtrace."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from llmtrace import LLMTracer
from llmtrace.models.config import TracerConfig
from llmtrace.control_plane.reporter import Reporter, ReporterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def main():
    """llmtrace - Flight recorder, attribution, and autopsy for vLLM inference."""
    pass


@main.command()
@click.option("--pid", type=int, help="vLLM process PID to monitor")
@click.option("--output-dir", default="./traces", help="Output directory for traces")
@click.option(
    "--sample-interval",
    default=100,
    type=int,
    help="GPU sampling interval in milliseconds",
)
def monitor(pid: Optional[int], output_dir: str, sample_interval: int):
    """Monitor a running vLLM process and collect traces."""
    click.echo(f"Monitoring vLLM process (PID: {pid})")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"GPU sample interval: {sample_interval}ms")

    # For now, this is a placeholder
    # Full implementation would attach to process, inject instrumentation, etc.
    # This is complex and would require process introspection
    click.echo("\nNote: Live monitoring requires vLLM integration.")
    click.echo("Use LLMTracer programmatically in your vLLM server code for now.")


@main.command()
@click.argument("trace_files", nargs=-1, type=click.Path(exists=True))
@click.option("--baseline", type=click.Path(exists=True), help="Baseline trace directory")
@click.option("--output", type=click.Path(), help="Output file for report")
def analyze(trace_files: tuple, baseline: Optional[str], output: Optional[str]):
    """Analyze trace files and generate report."""
    if not trace_files:
        click.echo("Error: No trace files specified", err=True)
        sys.exit(1)

    click.echo(f"Analyzing {len(trace_files)} trace file(s)")

    # Run async analysis
    asyncio.run(_analyze_async(trace_files, baseline, output))


async def _analyze_async(trace_files: tuple, baseline: Optional[str], output: Optional[str]):
    """Async analysis implementation."""
    from llmtrace.models.trace import RequestTrace, GPUSample
    import json

    # Load traces
    traces = []
    for trace_file in trace_files:
        path = Path(trace_file)

        if path.is_dir():
            # Load all traces from directory
            for jsonl_file in path.glob("traces_*.jsonl"):
                with open(jsonl_file) as f:
                    for line in f:
                        if line.strip():
                            trace = RequestTrace.model_validate(json.loads(line))
                            traces.append(trace)
        else:
            # Single file
            with open(path) as f:
                for line in f:
                    if line.strip():
                        trace = RequestTrace.model_validate(json.loads(line))
                        traces.append(trace)

    if not traces:
        click.echo("No traces found to analyze", err=True)
        return

    # Load GPU samples (look for gpu_*.jsonl in same directory as first trace file)
    gpu_samples = []
    first_trace_dir = Path(trace_files[0]) if Path(trace_files[0]).is_dir() else Path(trace_files[0]).parent

    for gpu_file in first_trace_dir.glob("gpu_*.jsonl"):
        with open(gpu_file) as f:
            for line in f:
                if line.strip():
                    sample = GPUSample.model_validate(json.loads(line))
                    gpu_samples.append(sample)

    # Create tracer components for analysis
    from llmtrace.control_plane.correlator import Correlator
    from llmtrace.control_plane.rules_engine import RulesEngine
    from llmtrace.models.config import EnergyConfig, AutopsyConfig

    correlator = Correlator(EnergyConfig())
    rules_engine = RulesEngine(AutopsyConfig())
    reporter = Reporter(ReporterConfig(cli_rich_output=True, export_formats=["jsonl"]))

    # Correlate
    correlated = await correlator.correlate_traces(traces, gpu_samples)

    # Diagnose
    for trace in correlated:
        trace.diagnosis = await rules_engine.diagnose_request(trace)

    # Load baseline if provided
    baseline_traces = None
    if baseline:
        baseline_traces = []
        baseline_dir = Path(baseline)
        for jsonl_file in baseline_dir.glob("traces_*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        trace = RequestTrace.model_validate(json.loads(line))
                        baseline_traces.append(trace)

    # Generate analysis
    analysis = reporter.generate_analysis(correlated, baseline_traces)

    # Print to console
    reporter.print_analysis(analysis)

    # Export if requested
    if output:
        reporter.export_to_file(analysis, Path(output))
        click.echo(f"\nReport exported to {output}")


@main.command()
@click.option("--baseline", required=True, type=click.Path(exists=True), help="Baseline trace directory")
@click.option("--current", required=True, type=click.Path(exists=True), help="Current trace directory")
@click.option("--ttft-threshold", default=5.0, type=float, help="TTFT regression threshold (%)")
@click.option("--energy-threshold", default=10.0, type=float, help="Energy regression threshold (%)")
@click.option("--fail-on-regression", is_flag=True, help="Exit with error code if regression detected")
def compare(
    baseline: str,
    current: str,
    ttft_threshold: float,
    energy_threshold: float,
    fail_on_regression: bool,
):
    """
    Compare current traces against baseline for CI regression detection.

    This is Feature 3: Energy Regression Guardrail.
    """
    click.echo("Comparing traces for regression detection")
    click.echo(f"Baseline: {baseline}")
    click.echo(f"Current: {current}")
    click.echo(f"Thresholds: TTFT={ttft_threshold}%, Energy={energy_threshold}%")

    # Run async comparison
    exit_code = asyncio.run(
        _compare_async(
            baseline, current, ttft_threshold, energy_threshold, fail_on_regression
        )
    )

    if exit_code != 0:
        sys.exit(exit_code)


async def _compare_async(
    baseline_dir: str,
    current_dir: str,
    ttft_threshold: float,
    energy_threshold: float,
    fail_on_regression: bool,
) -> int:
    """Async comparison implementation."""
    from llmtrace.models.trace import RequestTrace, GPUSample
    from llmtrace.control_plane.correlator import Correlator
    from llmtrace.control_plane.reporter import Reporter
    from llmtrace.models.config import EnergyConfig, ReporterConfig
    import json

    # Load baseline traces
    baseline_traces = []
    for trace_file in Path(baseline_dir).glob("traces_*.jsonl"):
        with open(trace_file) as f:
            for line in f:
                if line.strip():
                    trace = RequestTrace.model_validate(json.loads(line))
                    baseline_traces.append(trace)

    # Load baseline GPU samples
    baseline_gpu_samples = []
    for gpu_file in Path(baseline_dir).glob("gpu_*.jsonl"):
        with open(gpu_file) as f:
            for line in f:
                if line.strip():
                    sample = GPUSample.model_validate(json.loads(line))
                    baseline_gpu_samples.append(sample)

    # Load current traces
    current_traces = []
    for trace_file in Path(current_dir).glob("traces_*.jsonl"):
        with open(trace_file) as f:
            for line in f:
                if line.strip():
                    trace = RequestTrace.model_validate(json.loads(line))
                    current_traces.append(trace)

    # Load current GPU samples
    current_gpu_samples = []
    for gpu_file in Path(current_dir).glob("gpu_*.jsonl"):
        with open(gpu_file) as f:
            for line in f:
                if line.strip():
                    sample = GPUSample.model_validate(json.loads(line))
                    current_gpu_samples.append(sample)

    if not baseline_traces or not current_traces:
        click.echo("Error: Missing baseline or current traces", err=True)
        return 1

    # Correlate both
    correlator = Correlator(EnergyConfig())
    baseline_correlated = await correlator.correlate_traces(baseline_traces, baseline_gpu_samples)
    current_correlated = await correlator.correlate_traces(current_traces, current_gpu_samples)

    # Generate analysis with comparison
    reporter = Reporter(ReporterConfig(cli_rich_output=True, export_formats=["jsonl"]))
    analysis = reporter.generate_analysis(current_correlated, baseline_correlated)

    # Print analysis
    reporter.print_analysis(analysis)

    # Check regressions
    regressions_detected = []

    if "p95_ttft" in analysis.regressions:
        if abs(analysis.regressions["p95_ttft"]) > ttft_threshold:
            regressions_detected.append(
                f"TTFT regression: {analysis.regressions['p95_ttft']:.2f}% "
                f"(threshold: {ttft_threshold}%)"
            )

    if "joules_per_token" in analysis.regressions:
        if analysis.regressions["joules_per_token"] > energy_threshold:
            regressions_detected.append(
                f"Energy regression: {analysis.regressions['joules_per_token']:.2f}% "
                f"(threshold: {energy_threshold}%)"
            )

    if regressions_detected:
        click.echo("\n" + "=" * 50)
        click.echo("REGRESSIONS DETECTED:")
        for reg in regressions_detected:
            click.echo(f"  - {reg}")
        click.echo("=" * 50)

        if fail_on_regression:
            click.echo("\nFailing due to regressions (--fail-on-regression enabled)")
            return 1

    else:
        click.echo("\nNo significant regressions detected. ✓")

    return 0


@main.command()
@click.option("--output", default="llmtrace_config.json", help="Output config file path")
def init_config(output: str):
    """Generate a default configuration file."""
    config = TracerConfig()

    import json

    with open(output, "w") as f:
        json.dump(config.model_dump(), f, indent=2)

    click.echo(f"Default configuration written to {output}")
    click.echo("Edit this file and use with: llmtrace --config <file>")


if __name__ == "__main__":
    main()
