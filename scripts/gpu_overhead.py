"""Traced vs untraced wall time on real vLLM 0.11.0 (Linux + NVIDIA GPU).

Interleaves four configurations REP times on one engine so they share warm-up
and thermal state:

  untraced_generate    LLM.generate()            (FINAL_ONLY outputs, vLLM default)
  untraced_engine_loop run_engine_with_timing()  (CUMULATIVE outputs, no llmtrace)
  traced_generate      LLM.generate() under LLMTracer
  traced_engine_loop   run_engine_with_timing() under LLMTracer

Compare traced vs untraced *within the same output kind*: cumulative outputs
make vLLM's own output processor do more work per step, independent of tracing.
The per-step figure divides by the engine steps the tracer actually observed
(``health()["instrumentation"]["steps_observed"]``), not by ``max_tokens``.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 python scripts/gpu_overhead.py --model facebook/opt-125m

Run it from a directory that does not contain the repository checkout named
``llmtrace`` (or the checkout directory shadows the installed package).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--sample-interval-ms", type=int, default=50)
    parser.add_argument("--out", default="./overhead_traces")
    parser.add_argument("--json", default="./overhead.json")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    from llmtrace import LLMTracer, TracerConfig
    from llmtrace.vllm_helpers import run_engine_with_timing

    llm = LLM(model=args.model, gpu_memory_utilization=0.5)
    eng = llm.llm_engine
    prompts = [f"Prompt number {i}: tell me something about the number {i}." for i in range(args.num_prompts)]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    llm.generate(prompts[:4], sp)  # warm-up

    res: dict = {k: [] for k in ("untraced_generate", "untraced_engine_loop", "traced_generate", "traced_engine_loop")}

    steps: dict = {"traced_generate": [], "traced_engine_loop": []}
    finish: dict = {}

    def traced(fn, name: str, i: int) -> float:
        tr = LLMTracer(TracerConfig(output_dir=f"{args.out}/{name}_{i}",
                                    gpu_sampler={"sample_interval_ms": args.sample_interval_ms}))
        tr.instrument_engine(eng)
        t0 = time.perf_counter()
        fn()
        wall = time.perf_counter() - t0
        tr.stop()
        h = tr.health()["instrumentation"]
        assert h["instrumentation_errors"] == 0 and h["active_requests"] == 0, h
        key = "traced_generate" if name == "gen" else "traced_engine_loop"
        steps[key].append(h["steps_observed"])  # actual engine steps, not max_tokens
        from llmtrace import io
        from collections import Counter
        finish[f"{key}_{i}"] = dict(Counter(t.finish_reason for t in io.load_traces(tr.get_output_files()["traces"])))
        return wall

    for i in range(args.repeat):
        t0 = time.perf_counter(); llm.generate(prompts, sp); res["untraced_generate"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); run_engine_with_timing(eng, prompts, sp); res["untraced_engine_loop"].append(time.perf_counter() - t0)
        res["traced_generate"].append(traced(lambda: llm.generate(prompts, sp), "gen", i))
        res["traced_engine_loop"].append(traced(lambda: run_engine_with_timing(eng, prompts, sp), "loop", i))

    for k, v in res.items():
        print(f"{k:22} median {statistics.median(v):.4f}s  all {[round(x, 4) for x in v]}")
    print("engine steps observed per traced run:", steps)
    print("finish reasons per traced run:", finish)
    for kind in ("generate", "engine_loop"):
        u, t = statistics.median(res[f"untraced_{kind}"]), statistics.median(res[f"traced_{kind}"])
        n_steps = statistics.median(steps[f"traced_{kind}"])
        # Per-step figure uses the traced runs' recorded step counts; the untraced runs are assumed to
        # take the same number of steps (same prompts, temperature 0), which holds only if outputs match.
        print(f"{kind:12} traced/untraced = {t / u:.3f}  (+{(t - u) * 1000:.1f} ms per run over "
              f"{n_steps:.0f} recorded steps = +{(t - u) * 1e3 / n_steps:.3f} ms per step)")
    with open(args.json, "w") as f:
        json.dump({"wall_s": res, "steps_observed": steps, "finish_reasons": finish,
                   "config": vars(args)}, f, indent=1)
    print("OVERHEAD_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
