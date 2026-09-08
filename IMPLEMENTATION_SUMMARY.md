# Implementation Status

Honest inventory of what exists, what is verified, and what is not.
Last updated for version 0.2.0 after the first GPU run (2026-09-08, one RTX
A5000, vLLM 0.11.0, `facebook/opt-125m`; details in `docs/GPU_VALIDATION.md`).

## Verified locally (CPU, fakes)

The `tests/` suite (no GPU, NVML or vLLM required) covers:

* Wrappers preserve synchronous calling convention, return values and
  exceptions; coroutine engines are rejected.
* Method restoration to the exact original bound methods; repeated
  instrument/uninstrument; double-wrap detection; rollback on partial failure.
* Completion without deadlock; abort; incomplete requests returned at stop;
  bookkeeping errors counted without altering inference results; strict mode.
* Timing: exact TTFT/TPOT with a fake clock; zero-token and one-token
  outputs; several tokens in one step; DELTA vs CUMULATIVE parity; FINAL_ONLY
  and pooling reported unavailable; queue/prefill spans only with the
  scheduler; monotonic durations unaffected by wall-clock jumps.
* Token counts from token ids (engine first, caller fallback), never words.
* Sampler thread progresses while the caller blocks; atomic drain; bounded
  buffers with drop counts; unavailable NVML reported, or fatal with
  `require_gpu`; missing power is `null`.
* Writer: every accepted batch written once; full-queue drops counted;
  stop/flush ordering; parquet part files.
* Energy: single-GPU and multi-GPU known totals with unaligned timestamps;
  duplicates; gaps; overlapping requests with conservation; idle energy;
  proportional-token weights; `window_only`; batch-membership allocation;
  sparse telemetry reported unavailable; cost.
* CLI: improvements never regress; regressions fail only with the flag;
  missing metrics and zero baselines reported; GPU samples matched per run;
  `monitor` exits 3.

## Verified on hardware (one run, one GPU, one tiny model)

* Patching and restoration of the real `vllm.v1.engine.llm_engine.LLMEngine`
  (0.11.0) under both engine-core modes.
* In-process scheduler discovery via `engine.engine_core.engine_core.scheduler`;
  batch records with real request ids; prefill/decode classification.
* NVML backend field mapping (all fields populated on an A5000; throttle bits
  observed only as `none`).
* TTFT/TPOT through the raw engine loop; FINAL_ONLY handling under `generate()`.
* Sampling/integration consistent with a separately collected `nvidia-smi`
  stream of the same NVML sensor (within 0.15% over identical boundaries).
* Overhead on opt-125m only: +4% (`generate()`) / +9% (cumulative loop) wall
  time over 256 recorded engine steps.
* CUDA-event GPU span per step (RTX 4000 Ada, A100): one span per step, never
  above host time, resolved without synchronizing; long-prefill interference
  shown to be GPU compute (7.9 vs 1.7 ms on opt-125m; 103 vs 11 ms on
  Qwen2.5-7B); host share 9 to 13% on opt-125m, 2% on the 7B model; refused
  for TP>1 executors.
* `AsyncLLM` request-level tracing (RTX 4000 Ada): concurrent streams, client
  cancellation, restoration, and vLLM per-step stats over the multiprocess core.
* Nsight Systems cross-check of the step spans (RTX A5000, opt-125m): every
  NVTX step range matched a span; the span was never below Nsight's busy time
  (kernels plus CUDA-graph executions); 58% busy on decode steps of this
  launch-bound model, 84% on 1536-token prefill steps
  (`scripts/nsys_step_compare.py`).
* Qwen2.5-7B on A100 at TP=1 and TP=2: the mixed-prompt experiment reproduces
  (improved 3/3 and 2/2), with `decide` selecting the capped configs for a
  50 ms short-TTFT target and reporting the energy-per-token cost of TP=2.
* The CPU suite on the pod ran 83 tests: `tests/test_collection.py` was
  skipped whole by a module-level pyarrow skip (fixed afterwards; not re-run
  on hardware).

* Diagnosis experiment (`experiments/mixed_prompts`): on one GPU the traces
  attribute the short-request tail to long-prefill steps, and one scheduling
  change improved it in 3/3 repeats with the cost quantified. Also found and
  fixed a tracer-induced stall (collector interval) using the same traces.

## Implemented but unverified on hardware

* Behaviour under preemption, speculative decoding, `n > 1`, abort under load,
  pipeline parallelism, models above 7B. The span-vs-busy gap is characterized
  on opt-125m only; the event-recording overhead itself is not measured.
* Throttle-reason bits other than `none`.
* Overhead on models where a step takes longer than a few milliseconds
  (expected to be smaller in relative terms; not measured).

## Not implemented

* `llmtrace monitor` (attach to an external process).
* Instrumenting the OpenAI-compatible server process itself (only a directly
  constructed `AsyncLLM` is supported; the server would need a hook point to
  call `instrument_async_engine` inside its process).
* Distributed / multi-node coordination (multi-GPU telemetry on one host is
  supported by the ledger; tensor-parallel workers are not instrumented).
* DCGM, dashboards, OTLP export, ML-based diagnosis.
* Per-step token attribution for `proportional_tokens` (it weights by each
  request's total tokens, an explicit approximation).

## Known limitations

* All timing is at engine-step granularity; TTFT is an upper bound on the
  true first-token time and includes queue wait.
* Energy allocation is a policy, not a measurement. Idle and unattributable
  energy are reported separately and never assigned to requests.
* Power sampling at 100 ms cannot resolve sub-step power changes; short
  requests may have too few samples for a figure (reported as unavailable).
* Without the in-process scheduler, membership is by request window, which
  includes queue wait.

## Breaking changes vs 0.1.0

* `LLMTracer.stop()`, `analyze()`, `Correlator.correlate_traces()` and
  `RulesEngine.diagnose_request()` are synchronous.
* Config keys removed: `use_dcgm`, `collect_tensor_utilization`,
  `enable_kv_cache_tracking`, `async_write`, `buffer_size`,
  `distributed_mode`, `rank`, `world_size`. Unknown keys now raise.
* `EnergyAttribution` fields renamed (`attributed_joules`,
  `window_device_joules`, `joules_per_output_token`); `confidence` removed;
  `exact` method removed. `DiagnosisResult.confidence` renamed to `score`.
* `GPUSample` numeric fields are optional; `BatchMetadata` reworked.
* Dependencies: vLLM and NVML moved to extras.
