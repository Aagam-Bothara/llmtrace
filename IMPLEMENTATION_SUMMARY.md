# Implementation Status

Honest inventory of what exists, what is verified, and what is not.
Last updated for version 0.2.0. No GPU run has been performed.

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

## Implemented but unverified on hardware

* Patching of the real `vllm.v1.engine.llm_engine.LLMEngine` (0.11.0).
* In-process scheduler discovery via `engine.engine_core.engine_core.scheduler`.
* NVML backend (`NVMLBackend`) field mapping and throttle bits.
* Behaviour under chunked prefill, preemption, speculative decoding, `n > 1`.
* Overhead of instrumentation and sampling.

## Not implemented

* `llmtrace monitor` (attach to an external process).
* `AsyncLLM` / OpenAI-compatible server instrumentation.
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
