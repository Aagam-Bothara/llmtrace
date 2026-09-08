# llmtrace audit (2026-09-08)

Scope: the repository at the commit after the Nsight cross-check, read in
full (package, tests, examples, experiments, docs, evidence directories), plus
the vLLM 0.11.0 tagged source for the upstream comparison. Every "verified"
statement below names where it was verified; everything else is an
assessment. Nothing in this document was produced by running hardware.

## 1. Architecture and execution flow

```
inside the vLLM process (sync LLMEngine, or AsyncLLM)
  VLLMInstrumentation           wraps add_request / step / abort_request,
                                scheduler.schedule (in-process core only),
                                model_executor.execute_model (UniProcExecutor only)
  AsyncLLMInstrumentation       wraps AsyncLLM.generate / abort (request level only)
  LLMTraceStatLogger            vLLM's own stat_loggers hook (SchedulerStats, IterationStats)
  CudaStepTimer                 CUDA events around execute_model; optional NVTX range per step
  GPUSampler (thread)           NVML power/util/mem/clocks/throttle per GPU
  collector (thread)            drains instrumentation buffers -> TraceWriter; records its own drains
  TraceWriter (thread)          bounded queue -> traces_* batches_* gpu_* gpu_steps_* vllm_stats_* collector_*

offline, no vLLM/GPU needed
  io.load_*  -> Correlator (energy ledger) -> Reporter (TraceAnalysis) -> CLI analyze / compare
             -> findings (evidence-backed hypotheses)      -> CLI findings
             -> decision (config comparison vs a target)    -> CLI decide
             -> visualize (Perfetto trace, HTML report)     -> CLI visualize
  workload.WorkloadSpec -> runner.run_workload (fake or vLLM engine) -> run directory + manifest   [added this phase]
  doctor (environment / run-directory signal availability)                                         [added this phase]
```

Package: `llmtrace/` (about 7,400 lines after this phase, including the synthetic fakes), `tests/`
(CPU only, fakes in `llmtrace/testing/fakes.py`), `experiments/mixed_prompts/`
(the one validated diagnosis experiment), `scripts/` (GPU-side helpers),
`docs/gpu_runs/` (84 MB of committed evidence from six GPU sessions).

## 2. Capabilities and limitations

| Area | Exists | Limitation |
|------|--------|------------|
| Request lifecycle timing | Arrival, first visible token, completion, per-step spans; TTFT/TPOT with explicit unavailable reasons; monotonic durations, wall timestamps, clock domain | Step granularity (TTFT includes the step in which the token appeared); `LLM.generate()` forces FINAL_ONLY so TTFT needs the raw engine loop (`vllm_helpers`) |
| Token accounting | Prompt tokens from engine token ids (source recorded), output tokens counted per step | `n > 1` untested on hardware |
| Scheduler metadata | Per-step request ids, scheduled tokens per request, prefill/decode split, KV usage, running/waiting | In-process engine core only (`VLLM_ENABLE_V1_MULTIPROCESSING=0`); never for `AsyncLLM` or the OpenAI server |
| GPU step span | CUDA events around `execute_model`, host overhead per step, NVTX ranges | Upper bound on busy time (Nsight cross-check: 58% busy on decode steps of opt-125m, 84% on long prefill); `UniProcExecutor` only, refused for TP>1 |
| GPU telemetry, energy | NVML samples; per-GPU trapezoid integration; per-request allocation with policy, coverage and conservation ledger | Allocation is a policy, not a measurement; `is_estimate` always true; 50 to 100 ms sampling |
| vLLM's own stats | `stat_loggers` hook records `SchedulerStats`/`IterationStats` per step | Post-hoc attach on the sync engine tested in fakes only |
| Diagnosis | Five evidence-backed hypotheses (`findings.py`) with supporting events, missing evidence and a suggested experiment; threshold screens (`rules_engine.py`) | No explicit assumptions or competing-explanation fields yet; long-prompt interference is the only finding validated on hardware |
| Experiments | Work-identical replay (`ignore_eos`, token-id prompts, seeds), manifests with scheduled vs actual arrivals, `decide` with eligibility rules, repeats | Only one workload shape existed before this phase; no goodput/SLO metric; uncertainty is min/median/max over repeats, no interval estimate |
| Visualization | Perfetto trace, HTML report | Static |
| Serving path | Sync `LLMEngine` (full), `AsyncLLM` (request level + stats) | The OpenAI server process itself has no hook point |
| Overhead | +4% / +9% wall on opt-125m (256 steps); collector drains and warm-up artifacts found and fixed from the traces | Not measured with GPU step timing on vs off; not measured on a 7B model |

Provenance is carried in the records rather than in a single place:
`prompt_length_source`, `ttft_unavailable_reason`, `output_kind`,
`scheduler_visible`, `BatchMetadata.source`, `StepGpuTiming.source`,
`EnergyAttribution.is_estimate` / `membership_source` / `unavailable_reason`,
`Finding.status` (supported / not_supported / insufficient_evidence) with
`missing_evidence`, `RepeatResult.status` with `problems`. `llmtrace doctor`
(this phase) summarizes signal availability per environment and per run.

## 3. Correctness risks and version-fragile integrations

Verified against the vLLM 0.11.0 tagged source (table in `DEVELOPMENT.md`)
and exercised on hardware, but fragile by construction:

* **Attribute-path attachment**: `engine.engine_core.engine_core.scheduler`,
  `engine.engine_core.engine_core.model_executor.execute_model`,
  `engine.logger_manager.per_engine_logger_dict`. Any rename in a later vLLM
  silently downgrades to "unavailable" (by design, with the reason in
  `health()`), so a version bump needs the smoke tests re-run, not just the
  unit suite. `TARGET_VLLM_VERSION` is the only guard; `doctor` now warns
  when the installed version differs.
* **Scheduler mutation order**: prefill/decode classification relies on
  `Scheduler.schedule()` having already advanced `num_computed_tokens`
  (`_update_after_schedule`), so the wrapper subtracts the step's scheduled
  tokens. A change there would misclassify silently. Covered by fakes that
  mimic the order; no hardware assertion checks it directly beyond the
  queue+prefill == TTFT identity.
* **`execute_model` bracket** assumes the blocking `UniProcExecutor` path;
  async scheduling (`non_block=True`) would make the span cover submission
  only. Detected by executor class name, not by behaviour.
* **AsyncLLM cancellation**: relies on `AsyncLLM.generate` aborting on
  `CancelledError`/`GeneratorExit` before the wrapper's handler runs; the
  wrapper tolerates either order but the abort cause annotation depends on it.
* **`LLM.generate()` FINAL_ONLY**: the helper that restores cumulative outputs
  reproduces `LLM._run_engine`; a change to `LLM` internals would not break
  llmtrace but would leave TTFT unavailable again.
* **Energy ledger** depends on monotonic timestamps shared between traces and
  samples (same `clock_domain`); mixing runs from different sessions falls
  back to wall clock and is reported.
* **Evidence in git**: 84 MB of JSONL under `docs/gpu_runs/`; fine today,
  but every new GPU session adds 5 to 40 MB. Nsight profiles are excluded
  because they embed the pod environment (a RunPod API key was caught by
  GitHub's secret scanning); any other binary evidence needs the same check.

Not risks, but worth stating: no `TODO`/`FIXME` markers exist; every engine
attribute read is `getattr(..., None)`-guarded; instrumentation errors are
counted and never propagate into inference unless `strict_instrumentation`.

## 4. Test coverage and gaps

`tests/`: 209 tests after this phase (174 before), all CPU, all against
fakes shaped like the verified vLLM 0.11.0 surface; one conditional skip
(parquet round trip without pyarrow). Covered: wrapper semantics and
restoration, timing under an injectable clock, scheduler knobs in the fake,
CUDA timer without synchronizing, async wrapper cancellation paths, energy
integration and conservation, findings and decision eligibility, CLI exit
codes, visualize outputs, nsys join, and now workload generation/validation,
runner directories and manifests, doctor probes.

Gaps:

* No test exercises real vLLM; compatibility rests on the smoke tests run
  manually on GPU (`examples/vllm_smoke_test.py`, `vllm_async_smoke_test.py`)
  whose pass/fail is recorded in `docs/gpu_runs/*`. There is no marker-based
  GPU integration suite that a self-hosted runner could execute.
* `_run_vllm` in the new runner mirrors the GPU-validated experiment driver
  line by line but has not itself run on hardware.
* Findings are tested on synthetic traces but there is no benchmark suite of
  "known bottleneck -> expected finding" scenarios with accuracy accounting.
* `utils/batch_analyzer.py` and `utils/latency_explainer.py` (older
  "Feature 1/2" helpers) are exercised only through examples, not tests, and
  overlap with `findings`/`visualize`.

## 5. Claims supported by hardware results

All from 2026-09-08 sessions, evidence under `docs/gpu_runs/`:

| Claim | Evidence |
|-------|----------|
| Sync engine instrumented and restored on real vLLM 0.11.0, both core modes; batch metadata with real ids in-process | RTX A5000 smoke (`2026-09-08-rtx-a5000`) |
| NVML telemetry fields populated; energy integration matches an independent `nvidia-smi` stream of the same sensor within 0.15% over equal boundaries | same |
| Overhead +4% (`generate`) / +9% (engine loop) on opt-125m, 0.13 to 0.29 ms per step | same, `long/overhead.json` |
| Mixed-prompt diagnosis: short-request tail attributed to steps carrying 1536-token chunks; `long_prefill_token_threshold=256` improved short TTFT p95 by 62 to 72% in every repeat, at +64 to +116% long TTFT | RTX A4500 (4 run sets), RTX 4000 Ada, A100 TP=1 (3/3) and TP=2 (2/2) with Qwen2.5-7B |
| GPU span per step never above host time; long-chunk steps are GPU compute (7.9 vs 1.7 ms opt-125m; 103 vs 11 ms 7B) | RTX 4000 Ada, A100 |
| Span vs Nsight busy time: every step range matched; span >= busy on all 626 + 654 steps; 58% busy on decode, 84% on long prefill | RTX A5000 nsys (`*_compare.json`) |
| AsyncLLM: 6 concurrent streams traced, client cancellation recorded as aborted, `stat_loggers` records over the multiprocess core | RTX 4000 Ada asyncllm |
| TP=2 costs +45% J/token for the same work on the 7B model; `decide` selects the capped config for a 50 ms target | A100 |

Not supported by any run: preemption, speculative decoding, `n > 1`, models
above 7B, the OpenAI server process, the CUDA-event recording cost itself,
this phase's `llmtrace run --engine vllm` path, the `findings` other than
long-prompt interference (the others report not_supported or insufficient_evidence
on the recorded runs, which is correct behaviour, not validation).

## 6. Overlap with upstream and adjacent tooling

Verified from the vLLM 0.11.0 source (`vllm/v1/engine/output_processor.py`,
`vllm/v1/metrics/loggers.py`, `vllm/v1/metrics/stats.py`,
`vllm/benchmarks/serve.py`, `vllm/entrypoints/llm.py`):

| Capability | vLLM 0.11.0 already has | llmtrace adds | Verdict |
|------------|------------------------|---------------|---------|
| Per-request latency | OpenTelemetry spans per request (`do_tracing`: time in queue, TTFT, e2e, prefill/decode/inference time, token counts) when an OTLP endpoint is configured; Prometheus histograms `vllm:request_queue_time_seconds`, `request_prefill_time_seconds`, `request_decode_time_seconds`, `time_to_first_token_seconds`, `inter_token_latency_seconds` | The same quantities without an OTLP collector, plus per-step spans, batch ids and status per request in files | Engineering integration, not novel. llmtrace should say so and consume `FinishedRequestStats` (queued/prefill/decode time) through the stat hook instead of re-deriving where possible |
| Engine-level stats | `SchedulerStats` (running, waiting, KV usage, prefix-cache hits, preemptions), `IterationStats` | Recorded per step alongside llmtrace's own step records; llmtrace already ingests them | Overlap by design |
| Batch composition per step | Not exported: no metric or span says which requests shared a step or how many tokens each was scheduled | `batches_*`: request ids, scheduled tokens per request, prefill/decode split per step | Distinct capability; it is what lets a tail be attributed to specific steps |
| GPU time per step | `LLM.start_profile()/stop_profile()` (torch profiler) for deep dives; no continuous per-step GPU timing | CUDA-event span per step with host overhead, cheap enough to leave on; cross-checked against Nsight | Distinct, with a documented bound |
| Energy | None | Per-GPU integration, per-request allocation with conservation | Distinct; allocation policy is an explicit convention |
| Load generation | `vllm bench serve`: request rate, `--burstiness` (gamma), max concurrency, ramp-up, random/dataset prompts, seed, `--goodput` SLOs, percentiles, JSON results | Token-id prompts with exact lengths, per-class arrival processes, manifests with scheduled vs actual arrival, work-identical replay, and the run feeds the diagnosis directly | Overlap in arrival modelling (the gamma parameterization is the same); llmtrace's runner exists to keep workload, engine config and traces in one manifest. Goodput is missing on the llmtrace side |
| Diagnosis, experiments, decision | None | `findings`, `decide` | The intended contribution; only one finding validated |

Adjacent tools (from documentation, not re-verified against source in this
session): SGLang has OpenTelemetry request tracing with instrumentation in
its tokenizer and scheduler threads, `sglang:` Prometheus metrics, and
`bench_serving` with trace-timestamp replay, so the same "request timing and
engine stats exist upstream" caveat applies there. Nsight Systems and the
PyTorch profiler give kernel-level truth at profiling cost; llmtrace's role is
the always-on step-level layer that says where to point them (the NVTX
ranges and `scripts/nsys_step_compare.py` are that hand-off). Research on
prefill/decode interference and scheduling (Sarathi-Serve's chunked prefill
and stall-free batching, Vidur's simulator-driven configuration search,
DistServe/Splitwise disaggregation, Llumnix rescheduling) motivates the
experiment shape but none of it ships a measurement-to-experiment loop for a
deployed vLLM; that loop, with its evidence discipline, is the claim llmtrace
can make, and only to the extent the benchmark suite in Phase 5 backs it.

## 7. Prioritized implementation plan

Done in this phase (Phase 1 increment and the Phase 2 vertical slice):

1. `llmtrace/workload.py`: `WorkloadSpec` (classes with length distributions,
   arrival processes, seed) generating deterministic `RequestSpec` lists;
   template equals the validated experiment workload.
2. `llmtrace/runner.py`: the serving-loop driver moved out of the experiment
   unchanged; `run_workload()` for the fake and vLLM engines writing raw data,
   `workload.json`, `manifest.json`, `run_info.json`; failed runs recorded.
3. `llmtrace/doctor.py` and `llmtrace doctor`: environment and run-directory
   signal availability with reasons; injectable probes.
4. CLI: `run`, `workload template|preview`, `doctor`; fakes moved to
   `llmtrace/testing/` so the synthetic engine ships with the package.
5. `scripts/gpu_overhead.py --gpu-step-timing both` (not yet run).
6. Stale statements corrected in docs and docstrings.

Done in the second phase (Phases 3 and 4, CPU only):

7. Findings carry `assumptions`, `competing_explanations` and
   `confidence_limits` per hypothesis; the absent-data status is
   `insufficient_evidence`; queue overload evaluates vLLM's own `queued_time`
   when scheduler spans are missing (`control_plane/findings.py`).
8. `decide` reports goodput under per-class SLOs (`--slo`), a seeded bootstrap
   interval of the target statistic over pooled requests, and marginal
   candidates (`control_plane/decision.py`).
9. `control_plane/experiments.py` and `llmtrace plan` / `run --plan`: bounded
   configuration experiments from supported findings, executed as fresh
   engines, never against a running server.

Next, in order (files named):

| # | Item | Files | Why first |
|---|------|-------|-----------|
| 1 | Run `llmtrace run --engine vllm` on a GPU for baseline and capped, confirm it reproduces the experiment's numbers, then make `experiments/mixed_prompts/run.py` a thin wrapper | `llmtrace/runner.py`, `experiments/mixed_prompts/run.py`, `docs/GPU_VALIDATION.md` | Removes the last duplicated driver only after the generic one is validated |
| 2 | Overhead matrix with GPU step timing on/off and on the 7B model | `scripts/gpu_overhead.py` (done), evidence | The remaining measurement-foundation gap |
| 3 | Done (second phase): findings context fields and `insufficient_evidence` | `control_plane/findings.py` | |
| 4 | Done (second phase): goodput under SLOs, bootstrap intervals, marginal candidates | `control_plane/decision.py`, `cli.py` | |
| 5 | Done (second phase): experiment planner and `run --plan` | `control_plane/experiments.py`, `cli.py` | Validated on the synthetic engine only; the candidates' effects on real vLLM are what the GPU session must show |
| 6 | Benchmark suite: synthetic scenarios with planted bottlenecks (queueing, prefill interference, KV pressure with preemption, host overhead) and expected findings; accuracy table produced by a test | new `benchmarks/`, `tests/test_benchmark_suite.py` | Phase 5; makes diagnostic accuracy a measured number |
| 7 | Machine-readable + human report combining findings, tested configs, deltas, regressions per class, limitations | `control_plane/report.py` (new), `cli.py` (`report`) | Phase 5 |
| 8 | GPU integration tests behind a `gpu` marker, runnable on a self-hosted runner | `tests/gpu/`, `pyproject.toml`, workflow | Turns the manual smoke tests into repeatable checks |
| 9 | Retire or fold `utils/batch_analyzer.py` and `utils/latency_explainer.py` into `findings`/`visualize` | `llmtrace/utils/` | Reduces overlap; low priority |

Explicitly not planned: an LLM-based recommender, a dashboard, DCGM, AMD,
a new scheduler.
