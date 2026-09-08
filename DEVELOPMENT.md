# Development Guide

## Architecture

```
vLLM process
┌───────────────────────────────────────────────────────────────────┐
│ caller thread: vllm.LLM.generate() -> LLMEngine.step() loop        │
│   VLLMInstrumentation wraps add_request / step / abort_request     │
│   (and scheduler.schedule when in-process)                         │
│                                                                    │
│ threads: GPUSampler (NVML) │ collector (drains buffers) │ writer   │
└───────────────────────────────────────────────────────────────────┘
                 traces_*.jsonl  batches_*.jsonl  gpu_*.jsonl
                                     │
             offline: io.load_* -> Correlator -> RulesEngine -> Reporter / CLI
```

Data plane (`llmtrace/data_plane`) collects; control plane
(`llmtrace/control_plane`) analyses offline and never needs vLLM or NVML.
`llmtrace/workload.py` and `llmtrace/runner.py` replay a configuration-driven
workload on the fake or the real engine and write a run directory with its
manifest; `llmtrace/doctor.py` reports which signals an environment or a
recorded run has. `docs/AUDIT.md` is the current architecture and gap audit.

## Workloads and runs (`workload.py`, `runner.py`)

A `WorkloadSpec` (JSON; `llmtrace workload template`) lists request classes,
each with a count, prompt-length and `max_tokens` distributions (fixed,
uniform, choice, clipped lognormal) and an arrival process (at once,
constant, Poisson, gamma with a burstiness shape, bursts). `generate()` is a
pure function of the spec and seed; request ids are `<class>-<index>` and
`class_of()` recovers the class (`decide` and the experiment analysis rely on
that). Prompts are token-id lists so lengths are exact without a tokenizer.

`run_workload(spec, RunOptions)` replays the list on the synthetic engine
(CPU) or on vLLM (GPU; the same warm-up, settle and `ignore_eos` protocol the
experiment validated) under `LLMTracer`, and writes raw data only:
`traces_*`, `batches_*`, `gpu_*`, `gpu_steps_*`, `vllm_stats_*`,
`collector_*`, `workload.json`, `manifest.json`, `run_info.json`. Failures
leave `status: failed` in the manifest. Derived summaries are produced by
`analyze` / `findings` / `decide` / `visualize` and never written into the
run directory by the runner.

## Verified vLLM target

llmtrace targets exactly **vLLM 0.11.0**, V1 engine, synchronous
`vllm.v1.engine.llm_engine.LLMEngine` as used by `vllm.LLM`. The following
was read from the tagged source (not guessed):

| Interface | Location | Shape |
|-----------|----------|-------|
| `LLMEngine.add_request` | `vllm/v1/engine/llm_engine.py` | `(request_id, prompt, params, arrival_time=None, lora_request=None, tokenization_kwargs=None, trace_headers=None, priority=0) -> None`, sync |
| `LLMEngine.step` | same | `() -> list[RequestOutput] | list[PoolingRequestOutput]`, sync |
| `LLMEngine.abort_request` | same | `(request_ids: list[str]) -> None` |
| `LLM._add_request` | `vllm/entrypoints/llm.py` | calls `add_request(request_id, prompt, params, lora_request=..., ...)` positionally |
| `LLM._run_engine` | same | `while has_unfinished_requests(): step()` |
| `LLM._validate_and_add_requests` | same | sets `sp.output_kind = RequestOutputKind.FINAL_ONLY` for every `generate()` call, so first-token timing is not observable through `generate()` (use `llmtrace.vllm_helpers.run_engine_with_timing`) |
| Engine core client | `vllm/v1/engine/core_client.py` | `InprocClient.engine_core.scheduler` in-process; `SyncMPClient` otherwise |
| Multiprocessing default | `vllm/envs.py` | `VLLM_ENABLE_V1_MULTIPROCESSING` defaults to `1` |
| `SchedulerOutput` | `vllm/v1/core/sched/output.py` | `scheduled_new_reqs: list[NewRequestData]`, `num_scheduled_tokens: dict[str,int]`, `total_num_scheduled_tokens`, `finished_req_ids` |
| `Scheduler` | `vllm/v1/core/sched/scheduler.py` | `schedule() -> SchedulerOutput`, `.requests: dict[str, Request]`, `.running`, `.waiting`, `kv_cache_manager.usage: float` |
| `Scheduler._update_after_schedule` | same | called at the end of `schedule()` before it returns; advances `Request.num_computed_tokens` by the step's scheduled tokens (prefill/decode classification subtracts them back) |
| `Request` | `vllm/v1/request.py` | `num_prompt_tokens`, `num_computed_tokens` |
| `RequestOutput` | `vllm/outputs.py` | `request_id`, `prompt_token_ids`, `outputs: list[CompletionOutput]`, `finished`, `metrics` (never set by V1), `num_cached_tokens` |
| `CompletionOutput` | same | `index`, `token_ids`, `finish_reason` |
| Output kinds | `vllm/sampling_params.py` | `RequestOutputKind.CUMULATIVE` (default) / `DELTA` / `FINAL_ONLY` |
| Output construction | `vllm/v1/engine/output_processor.py` | `token_ids` cumulative unless DELTA |
| Finish reasons | `vllm/v1/engine/__init__.py` | `stop`, `length`, `abort` |
| `EngineCore.model_executor` / `Executor.execute_model` | `vllm/v1/engine/core.py`, `vllm/v1/executor/abstract.py`, `vllm/executor/uniproc_executor.py` | `EngineCore.step()` calls `self.model_executor.execute_model(scheduler_output)` synchronously; with `UniProcExecutor` (world size 1) the worker runs on the same thread, so CUDA events recorded before/after the call on the current stream bracket the step's model execution |
| `AsyncLLM.generate` | `vllm/v1/engine/async_llm.py` | `async def generate(prompt, sampling_params, request_id, lora_request=None, trace_headers=None, priority=0, data_parallel_rank=None) -> AsyncGenerator[RequestOutput, None]`; aborts the request itself on `asyncio.CancelledError` / `GeneratorExit` |
| `AsyncLLM.abort`, `from_engine_args`, `logger_manager` | same | `async def abort(request_id: str | Iterable[str])`; `from_engine_args(engine_args, start_engine_loop=True, usage_context=..., stat_loggers=None)`; `logger_manager` is a `StatLoggerManager` when log stats are on |
| `StatLoggerBase` / `StatLoggerFactory` | `vllm/v1/metrics/loggers.py` | `__init__(vllm_config, engine_index)`, `record(scheduler_stats, iteration_stats, engine_idx)`, `log_engine_initialized()`, `log()`; factories are called as `factory(vllm_config, engine_idx)`; vLLM notes the stats classes "are not considered stable interfaces" |
| `StatLoggerManager` | same | `per_engine_logger_dict: dict[int, list[StatLoggerBase]]`; `record()` iterates that list, so a logger can be appended post-hoc (`LLMEngine.logger_manager`, `None` with `disable_log_stats`) |
| `SchedulerStats` / `IterationStats` / `FinishedRequestStats` | `vllm/v1/metrics/stats.py` | running/waiting counts, `kv_cache_usage`, prefix-cache stats; per-step tokens, `num_preempted_reqs`, `time_to_first_tokens_iter`, `inter_token_latencies_iter`, finished-request timings (queued/prefill/decode/e2e) **without request ids** |

Other versions are not supported. `VLLMInstrumentation` warns when the
installed version differs and records both versions in `health()`.

`AsyncLLM` (`data_plane/vllm_async_instrumentation.py`) is wrapped at
`generate` (an async-generator wrapper that forwards outputs, closes the inner
generator on cancellation or close, and records completion, client
cancellation and errors) and `abort`. It reuses the sync accounting for token
counts, TTFT and TPOT, observed when outputs become visible to the consumer.
Because `AsyncLLM.generate` aborts the request in its own exception handler
before the wrapper sees the cancellation, cancelled streams end with
`finish_reason == "abort"` and `metadata["abort_cause"] == "client_cancelled"`.
The engine core is always out of process there, so scheduler and executor
evidence is unavailable and reported as such; vLLM's per-step stats still
arrive through the `stat_loggers` hook.

## Instrumentation semantics

* Wrappers are plain synchronous functions. The engine's return value is
  returned unchanged and its exceptions propagate unchanged. llmtrace
  bookkeeping runs after the engine call inside `_guard`, which counts and logs
  failures (`health()["instrumentation_errors"]`, `last_error`);
  `strict_instrumentation=True` re-raises them after the engine call.
* Patches are stored as `(target object, name, original instance attribute)`.
  Restoration deletes the instance override (restoring the class method) or
  puts back the instance attribute. Double-wrapping is detected via a marker
  attribute; instrumenting twice with the same engine is a no-op; uninstrument
  is idempotent and returns still-active requests marked `incomplete`.
* One `threading.Lock` guards all instrumentation state. Completion is
  finalised inline under that lock (the previous nested-lock deadlock is gone).
* Abort: `abort_request` marks requests `aborted`; so does a `finish_reason`
  of `abort` in step outputs.
* Token counts: prompt length from `RequestOutput.prompt_token_ids` (engine),
  falling back to caller-supplied `prompt_token_ids`; otherwise `null`.
  Output length is the sum of `token_ids` across completions, accumulated for
  DELTA outputs and taken as-is for CUMULATIVE/FINAL_ONLY.
* First token: the first step after which the request's accumulated output
  token count becomes positive (not "exactly one token"). TTFT is not observable
  for FINAL_ONLY and is reported as unavailable.
* TTFT = first-token step end - arrival (monotonic). TPOT = (last-token step
  end - first-token step end) / (tokens - tokens at first observation); `null`
  if the denominator is 0.
* Spans: with the in-process scheduler, `queue` = arrival to the start of the
  first step that scheduled the request, `prefill` = that step start to first
  token step end (includes the first decode step; step granularity).
  Without it, a single `time_to_first_token` span; no boundary is inferred.
* Prefill vs decode per batch: `computed_before = num_computed_tokens -
  num_scheduled_tokens[req]` (because `schedule()` has already advanced the
  counter); prefill iff `computed_before < num_prompt_tokens`. Prefix-cache
  hits start the counter above 0; preemption resets it, so a resumed request
  counts as prefill again.
* Batches: `batch_id` is llmtrace's own (`<session>-s<step>-b<seq>`), because
  `SchedulerOutput` has no identifier; `request_ids` are the engine's.
  `step_end_monotonic` is stamped when the step returns so batches are real
  execution intervals for energy membership.

## vLLM's own stats (`data_plane/vllm_stats.py`)

Besides its own hooks, llmtrace records what vLLM reports through the
supported `stat_loggers` mechanism, once per engine step: KV-cache usage,
queue depth, preemptions, prefix-cache stats, vLLM's own TTFT and inter-token
latency samples, and per-finished-request timings. These work with the default
multiprocess engine core. Two ways to enable it:

* construction time (preferred): `LLMEngine.from_engine_args(args, stat_loggers=[tracer.stat_logger_factory()])`;
* post-hoc: `tracer.instrument_engine(engine)` appends a logger to
  `engine.logger_manager.per_engine_logger_dict` when log stats are enabled,
  and removes it on `stop()`. `health()["vllm_stats"]["unavailable_reason"]`
  says why when it could not.

Records go to `vllm_stats_<session>.jsonl`; `llmtrace analyze` and
`llmtrace visualize` summarize and chart them. vLLM does not attach request
ids to finished-request stats, so they are run-level evidence (e.g. for the
KV-pressure/preemption hypothesis), not per-request attribution.

## Two layers of diagnosis

`RulesEngine` (`control_plane/rules_engine.py`) is a set of threshold
*screens* over one request: they flag a symptom (long queue span, throttled
samples, high memory use) with the evidence value and threshold, and carry a
ranking `score`. They do not assert causes: a high TTFT alone never produces a
diagnosis, and rules whose inputs are missing return nothing rather than a
guess. `findings.py` is the evidence layer: it joins requests, scheduler steps
and vLLM stats, names what is missing, and proposes the experiment that would
test the hypothesis. New diagnosis work belongs in findings.

## Findings and decisions (`control_plane/findings.py`, `control_plane/decision.py`)

A `Finding` is a hypothesis with a status (`supported`, `not_supported`,
`insufficient_evidence`), the affected request ids, supporting events (each
naming the file and field it came from), the evidence that is missing, the
check's `assumptions`, the `competing_explanations` the recorded events are
also consistent with, its `confidence_limits`, and a suggested experiment.
The three context lists are fixed per hypothesis (`_CONTEXT`) and attached by
the `_check` decorator on every return path, so they describe the check
itself, not the outcome. Findings never claim a root cause; the suggested
experiment is the causal test. `queue_overload` uses vLLM's own
`queued_time` from `FinishedRequestStats` when the in-process queue spans are
absent (then with no request ids).

`decision.evaluate()` scores configurations against a parsed `Target`
(`<class|*> <ttft|ttft_sched|tpot|e2e>_<pNN|max> <= <ms>`), per repeat, and
only reports. Uncertainty: per-repeat median/min/max plus a seeded percentile
bootstrap (default 1000 resamples) of the target statistic over the
per-request values pooled across eligible repeats; a candidate whose interval
upper bound misses the target is listed as `marginal`. Optional `Slo`s
(`<class|*>: <metric> <= <ms>, ...`) give goodput: the share of selected
requests meeting every bound, with a request lacking a bounded metric counted
as not meeting it and the coverage reported. The recommendation prefers the
highest goodput when SLOs are given, otherwise the highest median throughput,
among candidates meeting the target in every repeat; always labeled advisory.

## Experiment planner (`control_plane/experiments.py`)

`plan_experiments(findings, manifest, batches)` maps supported findings to a
bounded list of `Candidate`s (name, `scheduling_change`, source finding,
rationale, expected effect, expected cost): `long_prompt_interference` gives
`long_prefill_token_threshold` values below the largest observed chunk (and a
halved `max_num_batched_tokens` when that would cap it); `queue_overload`
doubles `max_num_seqs` when the running count reached it and doubles the
token budget; `kv_cache_pressure` raises `gpu_memory_utilization` by 0.1 (at
most 0.95), halves `max_num_seqs`, enables prefix caching if off;
`host_overhead` doubles `max_num_seqs`. Baseline knobs come from the
manifest's effective config (real sections or the fake engine's). Candidates
are deduplicated and capped (`--max-candidates`), skipped items are listed
with the reason, and the plan is JSON that `llmtrace run --plan` executes as
fresh engines under the same workload (`<out>/<config>/r<i>`), warning if the
workload hash differs from the plan's source run. Nothing is executed by the
planner and no running server is modified.

## GPU span per step (`data_plane/cuda_timing.py`)

With the in-process engine core, llmtrace wraps `model_executor.execute_model`
and records a `torch.cuda.Event(enable_timing=True)` before and after each
call on the engine thread's current stream. The elapsed time is the step's
**GPU span**: an upper bound on GPU busy time (it includes launch gaps on that
stream) that excludes work on vLLM's other streams (async output copy,
communication). `host_overhead_ms = host_step_ms - gpu_span_ms`. Events are
resolved lazily with `query()` at later steps and at collection time, never
by synchronizing; pending events are bounded and drops counted. Records go to
`gpu_steps_*.jsonl`, feed the `host_overhead` finding, the experiment's
per-step GPU/host split, and Perfetto counter tracks. `enable_nvtx` adds an
NVTX range per step for Nsight Systems; `scripts/nsys_step_compare.py` joins
an `nsys export --type sqlite` database with `gpu_steps_*` by step index and
reports Nsight's busy time (union of `CUPTI_ACTIVITY_KIND_KERNEL` and
`CUPTI_ACTIVITY_KIND_GRAPH_TRACE`; vLLM's decode steps are CUDA graphs and
appear only in the latter under the default `--cuda-graph-trace=graph`)
against the span. Scope: vLLM's default blocking path
(`UniProcExecutor.collective_rpc` runs the worker method on the calling
thread); with async scheduling (`non_block=True`) `execute_model` returns a
future and the bracket would cover only submission, so spans are not
meaningful there and that mode is unsupported. Tensor-parallel executors run
workers in other processes and are out of reach. Status: validated on RTX
4000 Ada and A100 (Qwen2.5-7B) and cross-checked against Nsight Systems on
RTX A5000; see `docs/GPU_VALIDATION.md`.

## Clocks

Every record has wall-clock (`*_time`, `timestamp`) and monotonic
(`*_monotonic`, `monotonic`) fields plus a `clock_domain` (the tracer session).
Durations use monotonic. The correlator uses monotonic when traces and samples
share a domain, otherwise wall clock, and records which (`ledger.clock`).

## Energy ledger (`control_plane/correlator.py`)

1. Group samples by `gpu_id`; drop samples with `power_draw_watts == null`
   (counted); dedupe identical timestamps (last wins); sort.
2. Build a cumulative trapezoid curve per GPU. Segments longer than
   `max_sample_gap_s` contribute nothing and are "uncovered".
3. Run window = [earliest arrival, latest completion]. Device energy = sum over
   GPUs of curve energy over the run window.
4. Membership intervals per request: batch intervals (`batch_metadata`) when
   every batch has start/end and the monotonic clock is in use; otherwise the
   request window.
5. Sweep elementary intervals between all interval boundaries. Energy in an
   interval with no active request is `idle`; otherwise it is split by policy
   (`equal_share`; `proportional_tokens` weights by prompt+output tokens).
   Span edges are boundaries too, so each elementary interval is wholly inside
   or outside every span and phase energy is the integrated curve over the
   span, not a time fraction of the request total.
6. Requests whose window has < 2 power samples or coverage below
   `min_coverage_fraction` get no figure; their share goes to
   `unattributable`. Under `window_only` all active-interval energy is
   unattributable and only `window_device_joules` is reported.
7. Conservation check: `|attributed + idle + unattributable - device| <= tol`,
   else an error is logged and noted in the ledger.

No "exact" method exists; `is_estimate` is always true; there are no
confidence probabilities. Diagnosis rules carry a `score` used for ranking.

## Collection

* `GPUSampler` runs in a thread with a bounded deque; `drain()` swaps the
  buffer atomically (no read-then-clear window). Unavailable NVML is reported
  in `stats()`; `require_gpu=True` turns it into a start failure.
* `TraceWriter` writes from a thread fed by a bounded queue; full queue drops
  are counted; `stop()` drains and closes. Parquet writes one part file per
  flush (no read-modify-write append).
* `LLMTracer.stop()` order: stop collector, restore engine (collect leftovers),
  stop sampler, final drain, write incomplete requests, stop writer, log health.
* `LLMTracer.start()` is transactional: if any component fails to start (e.g.
  `require_gpu=True` without NVML) the engine is restored, threads are stopped,
  NVML is released, the tracer is left `stopped`, and the original error is
  re-raised.

## Tests

`python -m pytest`. Fakes live in `llmtrace/testing/fakes.py` (re-exported by
`tests/fakes.py`) and mirror the verified vLLM 0.11.0 shapes with an
injectable clock. They validate llmtrace's logic, not vLLM compatibility. The
same fake engine is what `llmtrace run --engine fake` drives.

## Adding a diagnosis rule

1. Add a `DiagnosisCategory`.
2. Implement `_check_*` in `RulesEngine` returning a `DiagnosisResult` with
   evidence and a `score` in [0, 1]; skip when inputs are missing (`None`).
3. Add a test with a synthetic trace.
