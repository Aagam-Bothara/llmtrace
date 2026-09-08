"""CI regression check as a script (offline, CPU-only). Equivalent to ``llmtrace compare``.

    python examples/ci_regression.py ./traces/baseline ./traces/current

Sign convention: percent change = (current - baseline) / baseline. All compared
metrics are higher-is-worse, so only a positive change above the threshold is
a regression; improvements never fail.
"""

import sys

from llmtrace import io
from llmtrace.cli import evaluate_regressions
from llmtrace.control_plane.correlator import Correlator
from llmtrace.control_plane.reporter import Reporter
from llmtrace.models.config import EnergyConfig, ReporterConfig

TTFT_THRESHOLD_PCT = 5.0
ENERGY_THRESHOLD_PCT = 10.0


def load(directory: str):
    traces = io.load_traces([directory])
    return Correlator(EnergyConfig()).correlate(traces, io.load_gpu_samples([directory]), io.load_batches([directory]))


def main() -> int:
    baseline_dir = sys.argv[1] if len(sys.argv) > 1 else "./traces/baseline"
    current_dir = sys.argv[2] if len(sys.argv) > 2 else "./traces/current"
    base, cur = load(baseline_dir), load(current_dir)
    if not base.traces or not cur.traces:
        print(f"Missing traces: baseline={len(base.traces)} current={len(cur.traces)}")
        return 2

    reporter = Reporter(ReporterConfig(cli_rich_output=False))
    analysis = reporter.generate_analysis(cur.traces, base.traces, cur.ledger, base.ledger)
    reporter.print_analysis(analysis)

    regressions, ok, unavailable = evaluate_regressions(
        analysis.regressions, {"p95_ttft_ms": TTFT_THRESHOLD_PCT, "joules_per_output_token": ENERGY_THRESHOLD_PCT}
    )
    for line in ok:
        print("ok         ", line)
    for line in unavailable:
        print("unavailable", line)
    for line in regressions:
        print("REGRESSION ", line)
    print("FAILED" if regressions else "PASSED")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
