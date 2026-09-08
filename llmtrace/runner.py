"""Replay a :class:`~llmtrace.workload.WorkloadSpec` on an engine under llmtrace and write a run directory.

The driver mimics a serving loop on the synchronous engine: requests are added
when their arrival offset has passed, the engine is stepped while it has work,
and the loop idles until the next arrival otherwise. The synthetic engine
(``llmtrace.testing.fakes.FakeLLMEngine``, CPU) and real vLLM 0.11.0 (GPU) are
driven by the same code; only the clock differs.

A run directory holds raw data only (``traces_*``, ``batches_*``, ``gpu_*``,
``gpu_steps_*``, ``vllm_stats_*``, ``collector_*``), the workload that was
replayed (``workload.json``), the manifest (``manifest.json``) and
``run_info.json``. Derived summaries come from ``llmtrace analyze`` /
``findings`` / ``decide`` / ``visualize`` and are written wherever those are
pointed, never into the run directory by this module.

Real-engine protocol (as validated on GPU by ``experiments/mixed_prompts/run.py``,
from which this module was extracted): an untraced warm-up replay of the whole
workload so every batch shape has been seen once; a short traced settling
phase (``settle-*`` requests, class ``settle``); then the measured replay.
``ignore_eos=True`` makes every request generate exactly ``max_tokens`` so the
work is identical across configurations and repeats.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from llmtrace.manifest import ArrivalRecord, RunManifest, engine_effective_config, git_commit, gpu_info, llmtrace_version, workload_hash
from llmtrace.workload import RequestSpec, WorkloadSpec, make_prompt


def drive(engine: Any, specs: List[RequestSpec], make_params: Callable[[RequestSpec], Any],
          now: Callable[[], float], wait_until: Callable[[float], None], vocab_size: int, seed: int,
          min_token_id: int = 100) -> Dict[str, Any]:
    """Serve ``specs`` on a synchronous engine. Returns finished outputs by request id, step count,
    wall time and per-request arrival records (scheduled vs actual, stamped before ``add_request``)."""
    pending = list(specs)
    finished: Dict[str, Any] = {}
    arrivals: List[ArrivalRecord] = []
    t0 = now()
    steps = 0
    eps = 1e-6  # tolerate floating-point rounding in t0 + arrival (a fake clock can otherwise never reach it)
    while pending or engine.has_unfinished_requests():
        while pending and now() - t0 + eps >= pending[0].arrival_s:
            spec = pending.pop(0)
            prompt, params = make_prompt(spec, vocab_size, seed, min_token_id), make_params(spec)
            submit = now()  # submission time: stamped before the call, not after it returns
            engine.add_request(spec.request_id, prompt, params)
            arrivals.append(ArrivalRecord(request_id=spec.request_id, scheduled_s=spec.arrival_s, actual_s=submit - t0,
                                          delay_ms=(submit - t0 - spec.arrival_s) * 1000.0, submit_ms=(now() - submit) * 1000.0))
        if engine.has_unfinished_requests():
            steps += 1
            for out in engine.step():
                if getattr(out, "finished", False):
                    finished[str(out.request_id)] = out
        elif pending:
            before = now()
            wait_until(t0 + pending[0].arrival_s)
            if now() <= before:  # clock did not advance: treat the next request as due rather than spin forever
                t0 = min(t0, now() - pending[0].arrival_s)
    return {"finished": finished, "steps": steps, "wall_s": now() - t0, "arrivals": arrivals}


@dataclass
class RunOptions:
    engine: str = "fake"  # fake | vllm
    out_dir: str = "./run"
    model: str = "facebook/opt-125m"
    label: Optional[str] = None
    config_name: str = "default"
    scheduling_change: Dict[str, Any] = field(default_factory=dict)  # the engine kwargs under test (recorded separately)
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)  # other engine kwargs (vllm: LLM(...); fake: FakeLLMEngine(...))
    collection_interval_s: float = 0.1
    gpu_sample_interval_ms: int = 50
    enable_nvtx: bool = False
    ignore_eos: bool = True
    warmup: bool = True  # vllm only: untraced full-workload replay first
    settle_requests: int = 4  # vllm only: traced settling requests (class "settle") before the measured replay
    fake_power_w: float = 150.0  # fake engine: constant synthetic GPU power
    repo_dir: Optional[str] = None  # for the git commit in the manifest


_FAKE_DEFAULTS = {"step_seconds": 0.0015, "step_seconds_per_token": 8e-6, "max_num_batched_tokens": 8192}


def run_workload(spec: WorkloadSpec, opts: RunOptions) -> RunManifest:
    """Replay ``spec`` under llmtrace and write ``opts.out_dir``. Never raises for engine failures:
    a failed run leaves a manifest with ``status: failed`` and the error."""
    specs = spec.generate()
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec.save(str(out / "workload.json"))
    manifest = RunManifest(label=opts.label or out.name, engine=opts.engine, synthetic=opts.engine == "fake",
                           model=opts.model if opts.engine == "vllm" else "fake",
                           llmtrace_version=llmtrace_version(), llmtrace_git_commit=git_commit(opts.repo_dir),
                           gpu=gpu_info() if opts.engine == "vllm" else None,
                           workload=spec.model_dump(exclude_none=True), workload_hash=workload_hash(specs), seed=spec.seed,
                           config_name=opts.config_name, scheduling_change=dict(opts.scheduling_change),
                           engine_kwargs=dict(opts.engine_kwargs))
    try:
        if opts.engine == "fake":
            info = _run_fake(spec, specs, opts)
        elif opts.engine == "vllm":
            info = _run_vllm(spec, specs, opts)
        else:
            raise ValueError(f"unknown engine {opts.engine!r} (fake | vllm)")
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # OOM, engine start failure, ...: record it and keep the run directory
        manifest.status, manifest.error = "failed", f"{type(exc).__name__}: {exc}"[:2000]
        manifest.write(str(out))
        (out / "run_info.json").write_text(json.dumps({"status": "failed", "error": manifest.error,
                                                       "config_name": opts.config_name}, indent=2), encoding="utf-8")
        return manifest
    manifest.arrivals = info.pop("arrivals", [])
    manifest.tracer_config = info.pop("tracer_config", {})
    manifest.effective_engine_config = info.get("effective_engine_config", {})
    manifest.engine_version = info.get("engine_version")
    manifest.model_revision = info.get("model_revision")
    manifest.steps, manifest.wall_s, manifest.finished, manifest.health = info["steps"], info["wall_s"], info["finished"], info["health"]
    manifest.expected_requests = len(specs) + int(info.get("settle_requests", 0))  # settle-* are traced too
    problems: List[str] = []
    h = info["health"]["instrumentation"]
    if h.get("instrumentation_errors"):
        problems.append(f"{h['instrumentation_errors']} instrumentation errors")
    if h.get("active_requests"):
        problems.append(f"{h['active_requests']} requests still active at stop")
    if info["finished"] != len(specs):
        problems.append(f"{info['finished']} of {len(specs)} workload requests finished")
    info["problems"] = problems
    manifest.extra = {k: v for k, v in info.items() if k not in ("health", "effective_engine_config")}
    manifest.write(str(out))
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return manifest


def _run_fake(spec: WorkloadSpec, specs: List[RequestSpec], opts: RunOptions) -> Dict[str, Any]:
    from llmtrace import LLMTracer, TracerConfig
    from llmtrace.testing.fakes import FakeClock, FakeLLMEngine, FakeNVMLBackend, RequestOutputKind, SamplingParams

    clock = FakeClock()
    kwargs: Dict[str, Any] = dict(_FAKE_DEFAULTS)
    kwargs.update(opts.engine_kwargs)
    kwargs.update(opts.scheduling_change)
    # Synthetic cost model (defaults): 1.5 ms per step + 8 us per scheduled token. Numbers are invented.
    engine = FakeLLMEngine(clock=clock, in_process_scheduler=True, **kwargs)
    tracer = LLMTracer(TracerConfig(output_dir=opts.out_dir, collection_interval_s=min(opts.collection_interval_s, 0.05),
                                    gpu_sampler={"sample_interval_ms": 10}),
                       gpu_backend=FakeNVMLBackend({0: opts.fake_power_w}))
    # The tracer must read the fake clock so traces and batches are consistent with it.
    tracer.vllm_instrumentation._monotonic = clock.monotonic
    tracer.vllm_instrumentation._wall = clock.time
    tracer.instrument_engine(engine)
    res = drive(engine, specs, lambda s: SamplingParams(max_tokens=s.max_tokens, output_kind=RequestOutputKind.CUMULATIVE),
                clock.monotonic, lambda t: clock.advance(max(0.0, t - clock.mono)), spec.vocab_size, spec.seed, spec.min_token_id)
    tracer.stop()
    return {"engine": "fake", "steps": res["steps"], "wall_s": res["wall_s"], "health": tracer.health(),
            "finished": len(res["finished"]), "arrivals": res["arrivals"], "tracer_config": tracer.config.model_dump(),
            "effective_engine_config": {"fake_engine": {
                k: getattr(engine, k, kwargs.get(k)) for k in
                ("step_seconds", "step_seconds_per_token", "max_num_batched_tokens", "long_prefill_token_threshold",
                 "prefill_chunk", "tokens_per_step")}},
            "engine_version": "fake", "synthetic": True, "config_name": opts.config_name,
            "scheduling_change": dict(opts.scheduling_change)}


def _run_vllm(spec: WorkloadSpec, specs: List[RequestSpec], opts: RunOptions) -> Dict[str, Any]:
    """Same protocol as ``experiments/mixed_prompts/run.py::run_vllm`` (GPU-validated there); this generic
    version has not itself been run on hardware yet."""
    from vllm import LLM, SamplingParams  # type: ignore

    from llmtrace import LLMTracer, TracerConfig
    from llmtrace.vllm_helpers import with_cumulative_outputs

    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0":
        print("WARNING: VLLM_ENABLE_V1_MULTIPROCESSING is not 0; batch metadata and GPU spans (needed by the diagnosis) will be missing")
    kwargs: Dict[str, Any] = {"model": opts.model, "max_model_len": 2048, "gpu_memory_utilization": 0.5, "enable_chunked_prefill": True}
    kwargs.update(opts.engine_kwargs)
    kwargs.update(opts.scheduling_change)
    llm = LLM(**kwargs)
    engine = llm.llm_engine
    vocab_cap = int(engine.model_config.get_vocab_size()) - 1000 if hasattr(engine.model_config, "get_vocab_size") else spec.vocab_size
    vocab = min(spec.vocab_size, vocab_cap)

    def params(s: RequestSpec) -> Any:
        return with_cumulative_outputs(SamplingParams(temperature=0.0, max_tokens=s.max_tokens, ignore_eos=opts.ignore_eos))

    sleep_until = lambda t: time.sleep(max(0.0, t - time.monotonic()))  # noqa: E731
    warm: List[RequestSpec] = []
    if opts.warmup:
        warm = [RequestSpec(f"warm-{s.request_id}", s.kind, s.arrival_s, s.prompt_len, s.max_tokens) for s in specs]
        drive(engine, warm, params, time.monotonic, sleep_until, vocab, spec.seed, spec.min_token_id)
    tracer = LLMTracer(TracerConfig(output_dir=opts.out_dir, gpu_sampler={"sample_interval_ms": opts.gpu_sample_interval_ms},
                                    collection_interval_s=opts.collection_interval_s, enable_nvtx=opts.enable_nvtx))
    tracer.instrument_engine(engine)
    settle: List[RequestSpec] = []
    if opts.settle_requests > 0:
        settle_len = min(s.prompt_len for s in specs)
        settle = [RequestSpec(f"settle-{i}", "settle", 0.0, settle_len, 4) for i in range(opts.settle_requests)]
        drive(engine, settle, params, time.monotonic, lambda t: None, vocab, spec.seed, spec.min_token_id)
    res = drive(engine, specs, params, time.monotonic, sleep_until, vocab, spec.seed, spec.min_token_id)
    tracer.stop()
    return {"engine": "vllm", "model": opts.model, "steps": res["steps"], "wall_s": res["wall_s"], "health": tracer.health(),
            "finished": len(res["finished"]), "collection_interval_s": opts.collection_interval_s,
            "warmup_requests": len(warm), "settle_requests": len(settle), "arrivals": res["arrivals"],
            "tracer_config": tracer.config.model_dump(), "ignore_eos": opts.ignore_eos,
            "effective_engine_config": engine_effective_config(engine), "engine_version": __import__("vllm").__version__,
            "model_revision": getattr(engine.model_config, "revision", None), "synthetic": False,
            "config_name": opts.config_name, "scheduling_change": dict(opts.scheduling_change)}


__all__ = ["drive", "RunOptions", "run_workload"]
