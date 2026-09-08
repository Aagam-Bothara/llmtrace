"""Smoke test for llmtrace on vLLM 0.11.0 ``AsyncLLM`` (the engine behind the OpenAI server).

Runs N concurrent generate() streams plus one that the client cancels mid-way,
under LLMTracer.instrument_async_engine(). Checks request-level traces, the
cancellation path, vLLM's per-step stats via the stat_loggers hook, and that
scheduler/executor evidence is reported unavailable with the AsyncLLM reason.
Run on hardware 2026-09-08 (RTX 4000 Ada, opt-125m): ALL CHECKS PASSED; see docs/GPU_VALIDATION.md.

    python examples/vllm_async_smoke_test.py --model facebook/opt-125m --out ./traces_async
"""

from __future__ import annotations

import argparse
import asyncio
import json
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


async def run(args) -> int:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    from llmtrace import LLMTracer, TracerConfig, io

    check = Checks()
    tracer = LLMTracer(TracerConfig(output_dir=args.out, gpu_sampler={"sample_interval_ms": 50}))
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(model=args.model, gpu_memory_utilization=0.5, max_model_len=2048),
                                       stat_loggers=[tracer.stat_logger_factory()])
    try:
        tracer.instrument_async_engine(engine)
        prompts = [f"Prompt number {i}: tell me something about the number {i}." for i in range(args.num_prompts)]
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

        async def consume(i: int) -> dict:
            n_tokens = 0
            async for out in engine.generate(prompts[i], sampling, f"async-{i}"):
                n_tokens = sum(len(c.token_ids) for c in out.outputs)
            return {"request_id": f"async-{i}", "tokens": n_tokens}

        async def cancelled_consumer() -> None:
            async for out in engine.generate(prompts[0], sampling, "async-cancelled"):
                if sum(len(c.token_ids) for c in out.outputs) >= 4:
                    break  # client stops reading: closes the generator -> AsyncLLM aborts the request

        t0 = time.perf_counter()
        results = await asyncio.gather(*(consume(i) for i in range(args.num_prompts)), cancelled_consumer())
        wall = time.perf_counter() - t0
        outputs = [r for r in results if isinstance(r, dict)]
    finally:
        tracer.stop()
        engine.shutdown()

    health = tracer.health()
    print(json.dumps(health, indent=2, default=str))
    inst = health["instrumentation"]
    check(inst.get("engine_kind") == "AsyncLLM", "AsyncLLM instrumentation used")
    check(inst["instrumentation_errors"] == 0 and inst["active_requests"] == 0, "no instrumentation errors, no leaked requests")
    check("generate" not in engine.__dict__ and "abort" not in engine.__dict__, "AsyncLLM.generate/abort restored")
    check("AsyncLLM" in (health["scheduler_unavailable_reason_during_run"] or ""), "scheduler reported unavailable with the AsyncLLM reason")
    check("AsyncLLM" in (health["executor_unavailable_reason_during_run"] or ""), "executor reported unavailable with the AsyncLLM reason")
    check(health["gpu_sampler"]["available"] and health["gpu_sampler"]["samples_taken"] > 0, "GPU telemetry sampled")

    files = tracer.get_output_files()
    traces = {t.request_id: t for t in io.load_traces(files.get("traces", []))}
    check(len(traces) == args.num_prompts + 1, f"{len(traces)} traces for {args.num_prompts} + 1 requests")
    for o in outputs:
        t = traces.get(o["request_id"])
        check(t is not None and t.status.value == "completed" and t.output_length == o["tokens"],
              f"{o['request_id']}: completed with {o['tokens']} tokens")
        check(t is not None and t.ttft_ms is not None and t.ttft_ms > 0, f"{o['request_id']}: TTFT measured ({t.ttft_ms if t else None})")
        check(t is not None and t.prompt_length_source == "engine_prompt_token_ids", f"{o['request_id']}: prompt tokens from engine")
        check(t is not None and t.total_duration_ms <= wall * 1000 + 1, f"{o['request_id']}: duration within wall time")
    tc = traces.get("async-cancelled")
    check(tc is not None and tc.status.value == "aborted" and tc.metadata.get("abort_cause") == "client_cancelled"
          and 0 < tc.output_length < args.max_tokens,
          f"cancelled stream recorded as aborted (cause client_cancelled) after {tc.output_length if tc else None} tokens")
    check("batches" not in files, "no batch metadata (out-of-process core)")
    stats = io.load_vllm_stats(files.get("vllm_stats", []))
    check(len(stats) > 0 and any(s.kv_cache_usage is not None for s in stats), f"vLLM per-step stats recorded ({len(stats)} records)")
    check(health["vllm_stats"]["unavailable_reason"] is None, "stat logger attached")

    print("\nRESULT:", "ALL CHECKS PASSED" if not check.failures else f"{len(check.failures)} FAILED")
    for f in check.failures:
        print("  -", f)
    return 0 if not check.failures else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--out", default="./traces_async")
    parser.add_argument("--num-prompts", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    try:
        import vllm  # noqa: F401
    except ImportError:
        print("vLLM is not installed.")
        return 2
    Path(args.out).mkdir(parents=True, exist_ok=True)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
