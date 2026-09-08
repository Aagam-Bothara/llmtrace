"""Batch/scheduler visibility from recorded batch metadata (offline, CPU-only).

Batch metadata is only recorded when the vLLM scheduler ran in-process
(VLLM_ENABLE_V1_MULTIPROCESSING=0). Otherwise there is nothing to analyse.

    python examples/batch_analysis.py ./traces
"""

import sys

from llmtrace import io
from llmtrace.utils.batch_analyzer import BatchAnalyzer


def main() -> None:
    trace_dir = sys.argv[1] if len(sys.argv) > 1 else "./traces"
    batches = io.load_batches([trace_dir])
    if not batches:
        print(f"No batch metadata in {trace_dir}. It is only recorded with an in-process scheduler "
              "(VLLM_ENABLE_V1_MULTIPROCESSING=0) or from the synthetic example.")
        return
    print(f"Loaded {len(batches)} batches (source: {batches[0].source})")
    analyzer = BatchAnalyzer()
    analyzer.print_batch_summary(batches)
    timeline = analyzer.generate_batch_timeline(batches)
    print(f"\nTimeline entries: {len(timeline)}")
    for ineff in analyzer.detect_batching_inefficiencies(batches):
        print(f"- [{ineff['severity']}] {ineff['type']}: {ineff['description']}")


if __name__ == "__main__":
    main()
