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
* Workload specs (`llmtrace/workload.py`): deterministic generation, seed
  sensitivity, length distributions, arrival processes (constant, Poisson,
  gamma burstiness, bursts, at once), validation, equivalence with the
  GPU-validated experiment workload.
* Runner (`llmtrace/runner.py`, `llmtrace run`): run directories with raw
  data, `workload.json` and manifests; scheduling changes applied and
  recorded; failed runs recorded; repeats; the outputs feed `decide`.
* `llmtrace doctor`: environment probes (vLLM version, engine-core process
  mode, CUDA, NVML, extras) and per-run signal availability with reasons.
* Findings carry assumptions, competing explanations and confidence limits;
  `insufficient_evidence` status; queue overload from vLLM's queued_time.
* `decide`: goodput under per-class SLOs, seeded bootstrap intervals, marginal
  candidates; recommendation by goodput when SLOs are given.
* Experiment planner (`llmtrace plan`, `llmtrace run --plan`): bounded
  candidates from supported findings; plan -> run -> decide loop exercised on
  the synthetic engine; the baseline reproduces the source run's effective
  settings (through an explicit reproducible-settings map, explicit kwargs
  applied on top) and flags settings it cannot reproduce.
* `health.assess_health()`: one reading of the full tracer health record for
  the runner, `doctor` and `decide` (writer and collector failures make a
  repeat ineligible; lossy telemetry only voids the dependent metrics).
* Run directories holding a previous run are refused (`--overwrite` to
  replace); the async cancellation tests rendezvous on an event instead of a
  sleep.

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
* Generic runner (`llmtrace run --engine vllm`) on RTX A5000: reproduces the
  experiment driver's verdicts on the same workload; `decide` treats runner
  and driver runs as repeats of one work-identical configuration.
* Plan loop on real vLLM (RTX A5000): plan from a baseline run, six engines in
  separate processes, `decide` selects the 512-token cap; monotone
  dose-response across caps 1024/512/256.
* vLLM per-step stats through the post-hoc `stat_loggers` attach on the sync
  engine (RTX A5000), with `disable_log_stats=False`.
* Overhead matrix with GPU step timing on/off on opt-125m (RTX A5000): CUDA
  events cost 0.11 ms per step (+3.0%); on Qwen2.5-7B (A100 80GB) the whole
  tracer is +1.2% / +1.7% and CUDA events 0.08 ms per step (+0.6%).
* Clean-commit validation set on Qwen2.5-7B (A100, session 4): queue overload
  with four independent repeats and KV-cache pressure with three, every
  manifest carrying the source fingerprint of commit b3973ed; run-to-run
  spreads under 40 ms (queue) and under 0.4 s (KV, on 30 to 38 s tails).
* Queue overload and KV-cache pressure induced on real vLLM (RTX A5000;
  queue also on the 7B model on A100): found by `findings`, candidates from
  `plan`, replayed by `run --plan`, compared by `decide`; a planner ranking
  defect found and fixed in the process (session 3 evidence).
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
  on opt-125m only; the event-recording overhead is measured on opt-125m
  and Qwen2.5-7B (0.11 and 0.08 ms per step); nothing above 7B.
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
