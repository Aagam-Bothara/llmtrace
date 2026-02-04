"""Example: Tail latency explainer (Feature 2)."""

import asyncio
import json
from pathlib import Path
from llmtrace.models.trace import RequestTrace, GPUSample
from llmtrace.control_plane.correlator import Correlator
from llmtrace.control_plane.rules_engine import RulesEngine
from llmtrace.models.config import EnergyConfig, AutopsyConfig
from llmtrace.utils.latency_explainer import LatencyExplainer


async def main():
    # Load traces
    trace_dir = Path("./traces")  # Update to your trace directory

    traces = []
    for trace_file in trace_dir.glob("traces_*.jsonl"):
        with open(trace_file) as f:
            for line in f:
                if line.strip():
                    trace = RequestTrace.model_validate(json.loads(line))
                    traces.append(trace)

    if not traces:
        print("No traces found. Run basic_usage.py first.")
        return

    # Load GPU samples
    gpu_samples = []
    for gpu_file in trace_dir.glob("gpu_*.jsonl"):
        with open(gpu_file) as f:
            for line in f:
                if line.strip():
                    sample = GPUSample.model_validate(json.loads(line))
                    gpu_samples.append(sample)

    print(f"Loaded {len(traces)} traces and {len(gpu_samples)} GPU samples")

    # Correlate and diagnose
    correlator = Correlator(EnergyConfig())
    rules_engine = RulesEngine(AutopsyConfig())

    correlated = await correlator.correlate_traces(traces, gpu_samples)

    for trace in correlated:
        trace.diagnosis = await rules_engine.diagnose_request(trace)

    # Use latency explainer
    explainer = LatencyExplainer()

    # Explain batch tail latency
    batch_summary = explainer.explain_tail_latency_batch(correlated, tail_percentile=95)
    explainer.print_batch_explanation(batch_summary)

    # Explain individual tail requests
    print("\n\nExample Tail Request Explanations:")
    print("=" * 60)

    for trace in correlated:
        if trace.diagnosis:
            explanation = explainer.explain_request(trace)
            explainer.print_explanation(explanation)
            break  # Just show one example

    # Generate diagnosis report
    report = explainer.generate_diagnosis_report(correlated)
    print("\n\nDiagnosis Report:")
    print("=" * 60)
    print(f"Total Requests: {report['total_requests']}")
    print(f"Diagnosed Requests: {report['diagnosed_requests']}")
    print(f"Diagnosis Rate: {report['diagnosis_rate']:.1f}%")

    print("\nCategory Breakdown:")
    for category, stats in report["category_breakdown"].items():
        print(f"\n  {category}:")
        print(f"    Count: {stats['count']}")
        print(f"    Percentage: {stats['percentage']:.1f}%")
        print(f"    Avg Latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"    Avg Confidence: {stats['avg_confidence']*100:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
