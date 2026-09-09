"""Drive an engine through the mixed workload under llmtrace, on either the fake engine
(CPU, synthetic) or real vLLM 0.11.0 (GPU).

The driver mimics a serving loop on the synchronous engine: requests are added
when their arrival time has passed, the engine is stepped while it has work,
and the loop idles until the next arrival otherwise. Both engines are driven by
the same code; only the clock differs (fake clock vs ``time.monotonic``).

Synthetic (CPU):
    python experiments/mixed_prompts/run.py --engine fake --config baseline --out ./exp/fake_baseline
    python experiments/mixed_prompts/run.py --engine fake --config capped   --out ./exp/fake_capped

Real (GPU, in-process engine core required for batch metadata):
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python experiments/mixed_prompts/run.py --engine vllm --config baseline --out ./exp/baseline
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python experiments/mixed_prompts/run.py --engine vllm --config capped --out ./exp/capped
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workload import RequestSpec, WorkloadConfig, build_workload  # noqa: E402

from llmtrace.manifest import RunManifest, engine_effective_config, gpu_info, llmtrace_version, workload_hash  # noqa: E402
from llmtrace.provenance import record_provenance  # noqa: E402
from llmtrace.runner import drive, prepare_run_dir  # noqa: E402  (the serving-loop driver moved into the library unchanged)

# The one scheduling change under test. Everything else stays at vLLM defaults.
CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline": {},
    "capped": {"long_prefill_token_threshold": 256},
}


def run_fake(config: Dict[str, Any], specs: List[RequestSpec], out_dir: str, wl: WorkloadConfig) -> Dict[str, Any]:
    from llmtrace.testing.fakes import FakeClock, FakeLLMEngine, FakeNVMLBackend, RequestOutputKind, SamplingParams

    from llmtrace import LLMTracer, TracerConfig

    clock = FakeClock()
    # Synthetic cost model: 1.5 ms per step + 8 us per scheduled token, so a 1536-token
    # prefill chunk makes a step ~10x slower than a decode-only step. Numbers are invented.
    engine = FakeLLMEngine(clock=clock, in_process_scheduler=True, step_seconds=0.0015,
                           step_seconds_per_token=8e-6, max_num_batched_tokens=8192, **config)
    tracer = LLMTracer(TracerConfig(output_dir=out_dir, collection_interval_s=0.05, gpu_sampler={"sample_interval_ms": 10}),
                       gpu_backend=FakeNVMLBackend({0: 150.0}))
    # The tracer must read the fake clock so traces and batches are consistent with it.
    tracer.vllm_instrumentation._monotonic = clock.monotonic
    tracer.vllm_instrumentation._wall = clock.time
    tracer.instrument_engine(engine)
    res = drive(engine, specs,
                lambda s: SamplingParams(max_tokens=s.max_tokens, output_kind=RequestOutputKind.CUMULATIVE),
                clock.monotonic, lambda t: clock.advance(max(0.0, t - clock.mono)), 50000, wl.seed)
    tracer.stop()
    return {"engine": "fake", "steps": res["steps"], "wall_s": res["wall_s"], "health": tracer.health(),
            "finished": len(res["finished"]), "arrivals": res["arrivals"], "tracer_config": tracer.config.model_dump()}


def run_vllm(config: Dict[str, Any], specs: List[RequestSpec], out_dir: str, wl: WorkloadConfig, model: str,
             engine_kwargs: Dict[str, Any], collection_interval_s: float, ignore_eos: bool = True,
             enable_nvtx: bool = False) -> Dict[str, Any]:
    from vllm import LLM, SamplingParams

    from llmtrace import LLMTracer, TracerConfig
    from llmtrace.vllm_helpers import with_cumulative_outputs

    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0":
        print("WARNING: VLLM_ENABLE_V1_MULTIPROCESSING is not 0; batch metadata (needed by the diagnosis) will be missing")
    kwargs = {"model": model, "max_model_len": 2048, "gpu_memory_utilization": 0.5, "enable_chunked_prefill": True}
    kwargs.update(engine_kwargs)
    kwargs.update(config)
    llm = LLM(**kwargs)
    engine = llm.llm_engine
    vocab = int(engine.model_config.get_vocab_size()) - 1000 if hasattr(engine.model_config, "get_vocab_size") else 50000
    sched_cfg = engine.vllm_config.scheduler_config
    effective = {k: getattr(sched_cfg, k, None) for k in
                 ("max_num_batched_tokens", "max_num_seqs", "enable_chunked_prefill", "long_prefill_token_threshold", "policy")}
    print("effective scheduler config:", effective)
    # Warm-up (untraced): replay the whole workload once with the same arrival schedule, so every
    # batch shape the measured run will hit has been seen. GPU runs 1-3 showed first-time-shape
    # stalls (~23 ms at the first 5-request mixed batch, ~8 ms at the first lone 32-token prefill)
    # that a smaller warm-up did not cover and that dominated the ITL max comparison.
    # ignore_eos=True makes every request generate exactly max_tokens, so the work per request is identical
    # across configurations and repeats (batch composition otherwise shifts where EOS lands).
    def params(s: RequestSpec) -> Any:
        return with_cumulative_outputs(SamplingParams(temperature=0.0, max_tokens=s.max_tokens, ignore_eos=ignore_eos))

    warm = [RequestSpec(f"warm-{s.request_id}", s.kind, s.arrival_s, s.prompt_len, s.max_tokens) for s in specs]
    drive(engine, warm, params, time.monotonic, lambda t: time.sleep(max(0.0, t - time.monotonic())), vocab, wl.seed)

    tracer = LLMTracer(TracerConfig(output_dir=out_dir, gpu_sampler={"sample_interval_ms": 50},
                                    collection_interval_s=collection_interval_s, enable_nvtx=enable_nvtx))
    tracer.instrument_engine(engine)
    # Traced settling phase, excluded from the request classes ("settle-*" -> kind "other"): GPU run 2
    # showed the first traced step of a run taking ~8-9 ms for 32 tokens, which otherwise sets ITL max.
    settle = [RequestSpec(f"settle-{i}", "other", 0.0, wl.short_prompt_len, 4) for i in range(4)]
    drive(engine, settle, params, time.monotonic, lambda t: None, vocab, wl.seed)
    res = drive(engine, specs, params, time.monotonic, lambda t: time.sleep(max(0.0, t - time.monotonic())), vocab, wl.seed)
    tracer.stop()
    return {"engine": "vllm", "model": model, "effective_scheduler_config": effective, "steps": res["steps"],
            "wall_s": res["wall_s"], "health": tracer.health(), "finished": len(res["finished"]),
            "collection_interval_s": collection_interval_s, "warmup_requests": len(warm), "settle_requests": len(settle),
            "arrivals": res["arrivals"], "tracer_config": tracer.config.model_dump(), "ignore_eos": ignore_eos,
            "effective_engine_config": engine_effective_config(engine),
            "engine_version": __import__("vllm").__version__,
            "model_revision": getattr(engine.model_config, "revision", None)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["fake", "vllm"], required=True)
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--engine-kwargs", default="{}", help="JSON of extra LLM(...) kwargs (vllm only)")
    parser.add_argument("--num-short", type=int, default=WorkloadConfig.num_short)
    parser.add_argument("--short-rate", type=float, default=WorkloadConfig.short_rate_per_s)
    parser.add_argument("--num-long", type=int, default=WorkloadConfig.num_long)
    parser.add_argument("--long-prompt-len", type=int, default=WorkloadConfig.long_prompt_len)
    parser.add_argument("--long-every", type=float, default=WorkloadConfig.long_every_s, help="seconds between long prompts")
    parser.add_argument("--short-tokens", type=int, default=WorkloadConfig.short_max_tokens)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ignore-eos", action="store_true", help="Let requests stop at EOS (work then differs across configs)")
    parser.add_argument("--enable-nvtx", action="store_true", help="Push an NVTX range per engine step (for nsys; see scripts/nsys_step_compare.py)")
    parser.add_argument("--overwrite", action="store_true", help="Remove a previous run's files from --out instead of refusing it")
    parser.add_argument("--collection-interval", type=float, default=0.1,
                        help="llmtrace collector drain interval (s). Large drains serialize many batch records under "
                             "the GIL and stall the engine thread; GPU run 1 saw ~10 ms stalls at 1.0 s intervals.")
    args = parser.parse_args()

    wl = WorkloadConfig(num_short=args.num_short, short_rate_per_s=args.short_rate, num_long=args.num_long,
                        long_prompt_len=args.long_prompt_len, long_every_s=args.long_every,
                        short_max_tokens=args.short_tokens, seed=args.seed)
    specs = build_workload(wl)
    try:
        out = prepare_run_dir(args.out, overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"ERROR: {exc}")
        return 2
    config = CONFIGS[args.config]
    engine_kwargs = json.loads(args.engine_kwargs)
    manifest = RunManifest(label=out.name, engine=args.engine, synthetic=args.engine == "fake", model=args.model,
                           llmtrace_version=llmtrace_version(), **record_provenance(str(out), str(Path(__file__).resolve().parents[2])),
                           gpu=gpu_info(), workload=wl.to_dict(), workload_hash=workload_hash(specs), seed=wl.seed,
                           config_name=args.config, scheduling_change=config, engine_kwargs=engine_kwargs)
    try:
        if args.engine == "fake":
            info = run_fake(config, specs, str(out), wl)
        else:
            info = run_vllm(config, specs, str(out), wl, args.model, engine_kwargs, args.collection_interval,
                            ignore_eos=not args.no_ignore_eos, enable_nvtx=args.enable_nvtx)
    except BaseException as exc:  # OOM, engine start failure, ...: record it and keep the run directory
        manifest.status, manifest.error = "failed", f"{type(exc).__name__}: {exc}"[:2000]
        manifest.write(str(out))
        (out / "run_info.json").write_text(json.dumps({"status": "failed", "error": manifest.error, "config_name": args.config}, indent=2))
        print(f"FAILED: {manifest.error}")
        return 1
    info.update({"config_name": args.config, "scheduling_change": config, "workload": wl.to_dict(),
                 "synthetic": args.engine == "fake"})
    manifest.arrivals = info.pop("arrivals", [])
    manifest.tracer_config = info.pop("tracer_config", {})
    manifest.effective_engine_config = info.get("effective_engine_config", {"scheduler_config": info.get("effective_scheduler_config", {})})
    manifest.engine_version = info.get("engine_version")
    manifest.model_revision = info.get("model_revision")
    manifest.steps, manifest.wall_s, manifest.finished, manifest.health = info["steps"], info["wall_s"], info["finished"], info["health"]
    manifest.expected_requests = len(specs) + int(info.get("settle_requests", 0))  # settle-* are traced too
    manifest.extra = {k: v for k, v in info.items() if k not in ("health", "effective_engine_config")}
    manifest.write(str(out))
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps({k: v for k, v in info.items() if k not in ("health", "effective_engine_config")}, indent=2, default=str))
    print(f"arrival delay p50/max: {manifest.arrival_delay_ms_p50} / {manifest.arrival_delay_ms_max} ms")
    h = info["health"]["instrumentation"]
    print(f"scheduler visible: {info['health']['scheduler_visible_during_run']} ({info['health']['scheduler_unavailable_reason_during_run']}); "
          f"executor visible: {info['health']['executor_visible_during_run']} ({info['health']['executor_unavailable_reason_during_run']})")
    if h["instrumentation_errors"] or h["active_requests"] or info["finished"] != len(specs):
        print("PROBLEM: instrumentation errors or unfinished requests", h)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
