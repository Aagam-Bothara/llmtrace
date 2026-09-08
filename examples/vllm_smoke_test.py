"""Smoke test for llmtrace against real vLLM 0.11.0 on an NVIDIA GPU.

This is the first thing to run on GPU hardware. It has NOT been run by the
authors yet (developed without a GPU); see docs/GPU_VALIDATION.md for the
checklist it feeds.

Requirements (Linux, NVIDIA GPU):
    pip install -e ".[vllm]"            # pins vllm==0.11.0 and nvidia-ml-py
    export VLLM_ENABLE_V1_MULTIPROCESSING=0   # optional: exposes the scheduler for batch metadata

Run:
    python examples/vllm_smoke_test.py --model facebook/opt-125m --out ./traces_smoke
    python examples/vllm_smoke_test.py --model facebook/opt-125m --out ./traces_smoke_untraced --no-trace

Compare traced vs untraced wall time (run each a few times):
    python examples/vllm_smoke_test.py --no-trace --repeat 3
    python examples/vllm_smoke_test.py --repeat 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def check(cond: bool, msg: str, failures: list) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--out", default="./traces_smoke")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no-trace", action="store_true", help="Run generation without llmtrace")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    try:
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM is not installed. pip install -e '.[vllm]' on a Linux machine with an NVIDIA GPU.")
        return 2

    from llmtrace import LLMTracer, TracerConfig, io
    from llmtrace.data_plane.vllm_instrumentation import TARGET_VLLM_VERSION

    failures: list = []
    print(f"vLLM {vllm.__version__} (llmtrace verified against {TARGET_VLLM_VERSION})")
    print(f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '<unset: defaults to 1>')}")
    check(vllm.__version__ == TARGET_VLLM_VERSION, "vLLM version matches the verified target", failures)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization)
    engine = llm.llm_engine
    print(f"engine: {type(engine).__module__}.{type(engine).__name__}; engine_core client: {type(engine.engine_core).__name__}")

    prompts = [f"Prompt number {i}: tell me something about the number {i}." for i in range(args.num_prompts)]
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Warm-up, untraced.
    llm.generate(prompts[:1], sampling)

    for rep in range(args.repeat):
        if args.no_trace:
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling)
            print(f"[untraced] run {rep}: {time.perf_counter() - t0:.3f}s wall, {len(outputs)} outputs")
            continue

        out_dir = Path(args.out) / f"run{rep}"
        tracer = LLMTracer(TracerConfig(output_dir=str(out_dir), gpu_sampler={"sample_interval_ms": 50}))
        tracer.instrument_engine(engine)
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        wall = time.perf_counter() - t0
        tracer.stop()
        print(f"[traced] run {rep}: {wall:.3f}s wall, {len(outputs)} outputs")

        health = tracer.health()
        print(json.dumps(health, indent=2, default=str))
        inst = health["instrumentation"]
        check(inst["instrumentation_errors"] == 0, "no instrumentation errors", failures)
        check(inst["active_requests"] == 0, "no active requests leaked", failures)
        check(not inst["instrumented"], "engine methods restored", failures)
        check("step" not in engine.__dict__, "LLMEngine.step is the original again", failures)
        check(health["gpu_sampler"]["available"], f"GPU telemetry available ({health['gpu_sampler']['unavailable_reason']})", failures)
        check(health["gpu_sampler"]["samples_taken"] > 0, "GPU samples were taken during generate()", failures)
        check(all(v == 0 for v in health["writer"]["dropped"].values()), "no dropped writes", failures)

        files = tracer.get_output_files()
        traces = io.load_traces(files.get("traces", []))
        check(len(traces) == len(prompts), f"{len(traces)} traces for {len(prompts)} prompts", failures)
        completed = [t for t in traces if t.status.value == "completed"]
        check(len(completed) == len(traces), "all traces completed", failures)
        by_id = {t.request_id: t for t in traces}
        for o in outputs:
            t = by_id.get(o.request_id)
            if t is None:
                failures.append(f"missing trace for {o.request_id}")
                continue
            n_engine = sum(len(c.token_ids) for c in o.outputs)
            check(t.output_length == n_engine, f"{o.request_id}: output tokens {t.output_length} == engine {n_engine}", failures)
            check(t.prompt_length == len(o.prompt_token_ids), f"{o.request_id}: prompt tokens match engine", failures)
            check(t.ttft_ms is not None and t.ttft_ms > 0, f"{o.request_id}: TTFT measured ({t.ttft_ms})", failures)
            if t.output_length >= 2:
                check(t.tpot_ms is not None and t.tpot_ms > 0, f"{o.request_id}: TPOT measured ({t.tpot_ms})", failures)
            check(t.total_duration_ms <= wall * 1000 + 1, f"{o.request_id}: duration within generate() wall time", failures)
        batches = io.load_batches(files.get("batches", []))
        print(f"scheduler_visible={inst['scheduler_visible']} reason={inst['scheduler_unavailable_reason']} batches={len(batches)}")
        if inst["scheduler_visible"]:
            check(len(batches) > 0, "batch metadata recorded", failures)
            check(all(t.batch_ids for t in traces), "every trace linked to batches", failures)

        analysis = tracer.analyze()
        tracer.print_analysis(analysis)
        if analysis.energy_ledger:
            L = analysis.energy_ledger
            check(L.conservation_error_joules is None or L.conservation_error_joules < 1e-6, "energy ledger conserved", failures)

    print("\nRESULT:", "ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED")
    for f in failures:
        print("  -", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
