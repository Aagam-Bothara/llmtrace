"""Tail latency explanation from recorded traces (offline, CPU-only).

    python examples/latency_diagnosis.py ./traces
"""

import sys

from llmtrace import io
from llmtrace.control_plane.correlator import Correlator
from llmtrace.control_plane.rules_engine import RulesEngine
from llmtrace.models.config import AutopsyConfig, EnergyConfig
from llmtrace.utils.latency_explainer import LatencyExplainer


def main() -> None:
    trace_dir = sys.argv[1] if len(sys.argv) > 1 else "./traces"
    traces = io.load_traces([trace_dir])
    if not traces:
        print(f"No traces in {trace_dir}. Run examples/synthetic_replay.py or basic_usage.py first.")
        return
    samples = io.load_gpu_samples([trace_dir])
    batches = io.load_batches([trace_dir])
    print(f"Loaded {len(traces)} traces, {len(samples)} GPU samples, {len(batches)} batches")

    result = Correlator(EnergyConfig()).correlate(traces, samples, batches)
    rules = RulesEngine(AutopsyConfig())
    for t in result.traces:
        t.diagnosis = rules.diagnose_request(t)

    explainer = LatencyExplainer()
    explainer.print_batch_explanation(explainer.explain_tail_latency_batch(result.traces, tail_percentile=95))
    for t in result.traces:
        if t.diagnosis:
            explainer.print_explanation(explainer.explain_request(t))
            break
    report = explainer.generate_diagnosis_report(result.traces)
    print(f"\nDiagnosed {report['diagnosed_requests']}/{report['total_requests']} requests")
    for category, stats in report["category_breakdown"].items():
        print(f"  {category}: {stats['count']} (avg latency {stats['avg_latency_ms']:.1f} ms, avg rule score {stats['avg_score']:.2f})")


if __name__ == "__main__":
    main()
