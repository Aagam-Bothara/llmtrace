"""Example: CI regression detection (Feature 3)."""

import asyncio
import sys
from pathlib import Path
from llmtrace.models.trace import RequestTrace, GPUSample
from llmtrace.control_plane.correlator import Correlator
from llmtrace.control_plane.reporter import Reporter
from llmtrace.models.config import EnergyConfig, ReporterConfig


async def main():
    """
    Example CI script for regression detection.

    This would typically be run in CI after performance benchmarks.
    """
    # Paths to baseline and current traces
    baseline_dir = Path("./traces/baseline")
    current_dir = Path("./traces/current")

    # Thresholds
    ttft_threshold = 5.0  # 5% regression in TTFT
    energy_threshold = 10.0  # 10% regression in energy per token

    print("=" * 60)
    print("llmtrace CI Regression Detection")
    print("=" * 60)
    print(f"Baseline: {baseline_dir}")
    print(f"Current:  {current_dir}")
    print(f"TTFT Threshold: {ttft_threshold}%")
    print(f"Energy Threshold: {energy_threshold}%")
    print("=" * 60)

    # Helper to load traces and samples
    async def load_data(directory):
        import json

        traces = []
        for trace_file in directory.glob("traces_*.jsonl"):
            with open(trace_file) as f:
                for line in f:
                    if line.strip():
                        trace = RequestTrace.model_validate(json.loads(line))
                        traces.append(trace)

        samples = []
        for gpu_file in directory.glob("gpu_*.jsonl"):
            with open(gpu_file) as f:
                for line in f:
                    if line.strip():
                        sample = GPUSample.model_validate(json.loads(line))
                        samples.append(sample)

        return traces, samples

    # Load baseline
    print("\nLoading baseline traces...")
    baseline_traces, baseline_samples = await load_data(baseline_dir)
    print(f"  Loaded {len(baseline_traces)} traces")

    # Load current
    print("Loading current traces...")
    current_traces, current_samples = await load_data(current_dir)
    print(f"  Loaded {len(current_traces)} traces")

    if not baseline_traces or not current_traces:
        print("\nError: Missing baseline or current traces")
        sys.exit(1)

    # Correlate
    correlator = Correlator(EnergyConfig())
    print("\nCorrelating traces with GPU telemetry...")
    baseline_correlated = await correlator.correlate_traces(
        baseline_traces, baseline_samples
    )
    current_correlated = await correlator.correlate_traces(
        current_traces, current_samples
    )

    # Generate analysis
    print("Generating comparison analysis...")
    reporter = Reporter(
        ReporterConfig(cli_rich_output=True, export_formats=["jsonl"])
    )
    analysis = reporter.generate_analysis(current_correlated, baseline_correlated)

    # Print report
    reporter.print_analysis(analysis)

    # Check regressions
    print("\n" + "=" * 60)
    print("Regression Check")
    print("=" * 60)

    regressions_found = []

    if "p95_ttft" in analysis.regressions:
        regression_pct = analysis.regressions["p95_ttft"]
        if abs(regression_pct) > ttft_threshold:
            regressions_found.append(
                f"TTFT: {regression_pct:+.2f}% (threshold: ±{ttft_threshold}%)"
            )
            print(f"❌ TTFT regression detected: {regression_pct:+.2f}%")
        else:
            print(f"✓ TTFT within threshold: {regression_pct:+.2f}%")

    if "joules_per_token" in analysis.regressions:
        regression_pct = analysis.regressions["joules_per_token"]
        if regression_pct > energy_threshold:
            regressions_found.append(
                f"Energy: {regression_pct:+.2f}% (threshold: {energy_threshold}%)"
            )
            print(f"❌ Energy regression detected: {regression_pct:+.2f}%")
        else:
            print(f"✓ Energy within threshold: {regression_pct:+.2f}%")

    # Final result
    print("\n" + "=" * 60)
    if regressions_found:
        print("REGRESSION TEST FAILED")
        print("\nRegressions found:")
        for reg in regressions_found:
            print(f"  - {reg}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("REGRESSION TEST PASSED")
        print("No significant regressions detected.")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
