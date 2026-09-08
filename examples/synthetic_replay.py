"""SYNTHETIC end-to-end example. Runs on CPU with no GPU, NVML or vLLM installed.

Everything here is fabricated: a fake engine that mimics the vLLM 0.11.0
``LLMEngine`` call surface, and a fake NVML backend that reports constant
power. The numbers it prints demonstrate the pipeline (instrument -> collect ->
write -> correlate -> diagnose -> report -> compare); they say nothing about
real vLLM behaviour or real hardware.

    python examples/synthetic_replay.py [--out ./traces_synthetic]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from llmtrace.testing.fakes import FakeLLMEngine, FakeNVMLBackend, SamplingParams  # noqa: E402

from llmtrace import LLMTracer, TracerConfig  # noqa: E402


def run_session(out_dir: Path, power_watts: float, step_seconds: float, label: str) -> LLMTracer:
    print(f"\n=== SYNTHETIC session '{label}' -> {out_dir} (fake power {power_watts} W, fake step {step_seconds*1000:.0f} ms)")
    config = TracerConfig(
        output_dir=str(out_dir),
        collection_interval_s=0.05,
        gpu_sampler={"sample_interval_ms": 10},
    )
    tracer = LLMTracer(config, gpu_backend=FakeNVMLBackend({0: power_watts, 1: power_watts / 2}))

    # Fake engine: in-process scheduler visible, chunked prefill of 8 tokens per step, and a real
    # sleep inside step() so batch execution intervals span real time for the energy ledger.
    engine = FakeLLMEngine(in_process_scheduler=True, prefill_chunk=8, step_seconds=0.0, step_sleep_s=step_seconds)
    tracer.instrument_engine(engine)

    prompts = [list(range(5)), list(range(20)), list(range(3)), list(range(40))]
    for i, ids in enumerate(prompts):
        engine.add_request(f"req-{i}", {"prompt_token_ids": ids}, SamplingParams(max_tokens=6 + i))

    # Drive the engine like vllm.LLM._run_engine does; the sampler thread samples meanwhile.
    while engine.has_unfinished_requests():
        engine.step()

    tracer.stop()
    print("health:", tracer.health()["instrumentation"])
    return tracer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./traces_synthetic")
    args = parser.parse_args()
    root = Path(args.out)

    baseline = run_session(root / "baseline", power_watts=200.0, step_seconds=0.02, label="baseline")
    analysis = baseline.analyze()
    baseline.print_analysis(analysis)

    current = run_session(root / "current", power_watts=260.0, step_seconds=0.03, label="current (slower, hotter)")
    current.print_analysis(current.analyze(baseline_dir=str(root / "baseline")))

    print("\n=== CLI comparison (synthetic data; positive change = worse)")
    cmd = [sys.executable, "-m", "llmtrace.cli", "compare", "--baseline", str(root / "baseline"),
           "--current", str(root / "current"), "--no-rich", "--fail-on-regression"]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    print(proc.stdout[-1500:])
    print(f"compare exit code: {proc.returncode} (1 expected: the synthetic 'current' run is slower and hotter)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
