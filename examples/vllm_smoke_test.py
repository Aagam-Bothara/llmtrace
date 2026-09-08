"""Smoke test for llmtrace against real vLLM 0.11.0 on an NVIDIA GPU.

This is the first thing to run on GPU hardware. It has NOT been run by the
authors yet (developed without a GPU); see docs/GPU_VALIDATION.md.

Two traced phases, because ``LLM.generate()`` forces FINAL_ONLY outputs
(``LLM._validate_and_add_requests`` in vLLM 0.11.0) and therefore cannot expose
first-token timing:

  Phase A  ``LLM.generate()``: completion, token counts, restoration, telemetry,
           energy ledger, and *unavailable* TTFT/TPOT with the FINAL_ONLY reason.
  Phase B  raw synchronous engine loop with CUMULATIVE outputs
           (``llmtrace.vllm_helpers.run_engine_with_timing``): TTFT/TPOT checks.

Requirements (Linux, NVIDIA GPU):
    pip install -e ".[vllm]"                  # pins vllm==0.11.0 and nvidia-ml-py
    export VLLM_ENABLE_V1_MULTIPROCESSING=0   # optional: exposes the scheduler for batch metadata

Run:
    python examples/vllm_smoke_test.py --model facebook/opt-125m --out ./traces_smoke
    python examples/vllm_smoke_test.py --no-trace --repeat 3     # untraced timing reference
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


class Checks:
    def __init__(self) -> None:
        self.failures: list = []

    def __call__(self, cond: bool, msg: str) -> None:
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        if not cond:
            self.failures.append(msg)


def common_checks(check: Checks, tracer, engine, files, n_expected: int, wall: float, expect_scheduler: bool) -> list:
    from llmtrace import io

    health = tracer.health()
    print(json.dumps(health, indent=2, default=str))
    inst = health["instrumentation"]
    # VLLMInstrumentation resets scheduler_visible on restore; use the value the tracer captured at start.
    scheduler_visible = bool(health["scheduler_visible_during_run"])
    reason = health["scheduler_unavailable_reason_during_run"]
    check(scheduler_visible == expect_scheduler,
          f"scheduler visible during run == {expect_scheduler} (was {scheduler_visible}; reason: {reason})")
    check(inst["instrumentation_errors"] == 0, "no instrumentation errors")
    check(inst["active_requests"] == 0, "no active requests leaked")
    check(not inst["instrumented"], "engine methods restored")
    check("step" not in engine.__dict__ and "add_request" not in engine.__dict__, "LLMEngine.step/add_request are the originals again")
    check(health["gpu_sampler"]["available"], f"GPU telemetry available ({health['gpu_sampler']['unavailable_reason']})")
    check(health["gpu_sampler"]["samples_taken"] > 0, "GPU samples were taken during inference")
    check(all(v == 0 for v in health["writer"]["dropped"].values()), "no dropped writes")
    traces = io.load_traces(files.get("traces", []))
    check(len(traces) == n_expected, f"{len(traces)} traces for {n_expected} requests")
    check(all(t.status.value == "completed" for t in traces), "all traces completed")
    check(all(t.total_duration_ms <= wall * 1000 + 1 for t in traces), "every duration within inference wall time")
    batches = io.load_batches(files.get("batches", []))
    print(f"scheduler_visible_during_run={scheduler_visible} reason={reason} batches={len(batches)}")
    if scheduler_visible:
        check(len(batches) > 0, "batch metadata recorded")
        check(all(t.batch_ids for t in traces), "every trace linked to batches")
        check(all(t.scheduler_visible for t in traces), "every trace flagged scheduler_visible")
        check(all(b.step_end_monotonic is not None for b in batches), "every batch has a step end time")
        check(batches and batches[0].num_prefill == batches[0].num_requests, "first batch is all prefill")
    else:
        check(len(batches) == 0, "no batch metadata without in-process scheduler")
    analysis = tracer.analyze()
    tracer.print_analysis(analysis)
    if analysis.energy_ledger and analysis.energy_ledger.device_joules is not None:
        L = analysis.energy_ledger
        check(L.conservation_error_joules is not None and L.conservation_error_joules < 1e-6, "energy ledger conserved")
        check(L.membership_source == ("batch_metadata" if scheduler_visible else "request_window"),
              f"membership source {L.membership_source}")
    return traces


def texts_in_input_order(outputs, expected_ids) -> list:
    """Order outputs by request id so comparisons do not depend on completion order."""
    by_id = {str(o.request_id): o for o in outputs}
    return [by_id[rid].outputs[0].text if rid in by_id else None for rid in expected_ids]


def token_checks(check: Checks, traces, outputs) -> None:
    by_id = {t.request_id: t for t in traces}
    for o in outputs:
        t = by_id.get(o.request_id)
        if t is None:
            check(False, f"missing trace for {o.request_id}")
            continue
        n_engine = sum(len(c.token_ids) for c in o.outputs)
        check(t.output_length == n_engine, f"{o.request_id}: output tokens {t.output_length} == engine {n_engine}")
        check(t.prompt_length == len(o.prompt_token_ids) and t.prompt_length_source == "engine_prompt_token_ids",
              f"{o.request_id}: prompt tokens match engine ({t.prompt_length})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--out", default="./traces_smoke")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no-trace", action="store_true", help="Run generation without llmtrace (timing reference)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    try:
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM is not installed. pip install -e '.[vllm]' on a Linux machine with an NVIDIA GPU.")
        return 2

    from llmtrace import LLMTracer, TracerConfig
    from llmtrace.data_plane.vllm_instrumentation import TARGET_VLLM_VERSION
    from llmtrace.vllm_helpers import run_engine_with_timing

    check = Checks()
    print(f"vLLM {vllm.__version__} (llmtrace verified against {TARGET_VLLM_VERSION})")
    print(f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '<unset: defaults to 1>')}")
    check(vllm.__version__ == TARGET_VLLM_VERSION, "vLLM version matches the verified target")

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization)
    engine = llm.llm_engine
    print(f"engine: {type(engine).__module__}.{type(engine).__name__}; engine_core client: {type(engine.engine_core).__name__}")

    prompts = [f"Prompt number {i}: tell me something about the number {i}." for i in range(args.num_prompts)]
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    llm.generate(prompts[:1], sampling)  # warm-up, untraced
    ref_outputs = llm.generate(prompts, sampling)  # LLM.generate() returns outputs in input order
    reference_text = [o.outputs[0].text for o in ref_outputs]
    # The scheduler is only reachable in-process: expect it exactly when multiprocessing is disabled.
    expect_scheduler = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") == "0"

    for rep in range(args.repeat):
        if args.no_trace:
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling)
            print(f"[untraced] run {rep}: {time.perf_counter() - t0:.3f}s wall, {len(outputs)} outputs")
            continue

        # ---------------- Phase A: LLM.generate() (FINAL_ONLY outputs) ----------------
        print(f"\n=== Phase A (run {rep}): LLM.generate(), FINAL_ONLY outputs; timing expected unavailable")
        out_dir = Path(args.out) / f"run{rep}_generate"
        tracer = LLMTracer(TracerConfig(output_dir=str(out_dir), gpu_sampler={"sample_interval_ms": 50}))
        tracer.instrument_engine(engine)
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling)
        wall = time.perf_counter() - t0
        tracer.stop()
        print(f"[traced generate] run {rep}: {wall:.3f}s wall")
        check([o.outputs[0].text for o in outputs] == reference_text, "generated text identical to untraced run")
        traces = common_checks(check, tracer, engine, tracer.get_output_files(), len(prompts), wall, expect_scheduler)
        token_checks(check, traces, outputs)
        for t in traces:
            check(t.output_kind == "final_only", f"{t.request_id}: output_kind recorded as final_only ({t.output_kind})")
            check(t.ttft_ms is None and "FINAL_ONLY" in (t.ttft_unavailable_reason or ""),
                  f"{t.request_id}: TTFT unavailable with FINAL_ONLY reason ({t.ttft_unavailable_reason})")
            check(t.tpot_ms is None, f"{t.request_id}: TPOT unavailable under FINAL_ONLY")

        # ---------------- Phase B: raw engine loop, CUMULATIVE outputs ----------------
        print(f"\n=== Phase B (run {rep}): raw LLMEngine loop with CUMULATIVE outputs; timing expected")
        out_dir = Path(args.out) / f"run{rep}_engine"
        tracer = LLMTracer(TracerConfig(output_dir=str(out_dir), gpu_sampler={"sample_interval_ms": 50}))
        tracer.instrument_engine(engine)
        request_ids = [f"smoke-{rep}-{i}" for i in range(len(prompts))]
        t0 = time.perf_counter()
        outputs = run_engine_with_timing(engine, prompts, sampling, request_ids=request_ids)
        wall = time.perf_counter() - t0
        tracer.stop()
        print(f"[traced engine loop] run {rep}: {wall:.3f}s wall")
        check(len(outputs) == len(prompts), f"{len(outputs)} finished outputs")
        check([str(o.request_id) for o in outputs] == request_ids, "engine-loop outputs returned in input order")
        check(texts_in_input_order(outputs, request_ids) == reference_text,
              "engine-loop text identical to generate() text (matched by request id)")
        traces = common_checks(check, tracer, engine, tracer.get_output_files(), len(prompts), wall, expect_scheduler)
        token_checks(check, traces, outputs)
        for t in traces:
            check(t.output_kind == "cumulative", f"{t.request_id}: output_kind cumulative ({t.output_kind})")
            check(t.ttft_ms is not None and t.ttft_ms > 0, f"{t.request_id}: TTFT measured ({t.ttft_ms})")
            if t.output_length >= 2:
                check(t.tpot_ms is not None and t.tpot_ms > 0, f"{t.request_id}: TPOT measured ({t.tpot_ms})")
            check(t.tokens_at_first_observation == 1, f"{t.request_id}: one token at first observation (no spec decode)")
            if t.scheduler_visible:
                check(abs(t.queue_duration_ms + t.prefill_duration_ms - (t.ttft_ms or 0)) < 1e-6,
                      f"{t.request_id}: queue + prefill == TTFT")

    print("\nRESULT:", "ALL CHECKS PASSED" if not check.failures else f"{len(check.failures)} FAILED")
    for f in check.failures:
        print("  -", f)
    return 0 if not check.failures else 1


if __name__ == "__main__":
    sys.exit(main())
