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
| `StatLoggerBase` / `StatLoggerFactory` | `vllm/v1/metrics/loggers.py` | `__init__(vllm_config, engine_index)`, `record(scheduler_stats, iteration_stats, engine_idx)`, `log_engine_initialized()`, `log()`; factories are called as `factory(vllm_config, engine_idx)`; vLLM notes the stats classes "are not considered stable interfaces" |
| `StatLoggerManager` | same | `per_engine_logger_dict: dict[int, list[StatLoggerBase]]`; `record()` iterates that list, so a logger can be appended post-hoc (`LLMEngine.logger_manager`, `None` with `disable_log_stats`) |
| `SchedulerStats` / `IterationStats` / `FinishedRequestStats` | `vllm/v1/metrics/stats.py` | running/waiting counts, `kv_cache_usage`, prefix-cache stats; per-step tokens, `num_preempted_reqs`, `time_to_first_tokens_iter`, `inter_token_latencies_iter`, finished-request timings (queued/prefill/decode/e2e) **without request ids** |

Other versions are not supported. `VLLMInstrumentation` warns when the
installed version differs and records both versions in `health()`.

`AsyncLLM` has a different surface (async generator `generate`, no `step`)
and is out of scope; wrapping a coroutine function raises `InstrumentationError`.

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

## Findings and decisions (`control_plane/findings.py`, `control_plane/decision.py`)

A `Finding` is a hypothesis with a status (`supported`, `not_supported`,
`not_evaluable`), the affected request ids, supporting events (each naming the
file and field it came from), the evidence that is missing, and a suggested
experiment. Findings never claim a root cause; the suggested experiment is the
causal test. `decision.evaluate()` scores configurations against a parsed
`Target` (`<class|*> <ttft|tpot|e2e>_<pNN|max> <= <ms>`), per repeat, and only
reports; it recommends the candidate with the highest median throughput among
those meeting the target in every repeat, labeled advisory.

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

`python -m pytest`. Fakes live in `tests/fakes.py` and mirror the verified
vLLM 0.11.0 shapes with an injectable clock. They validate llmtrace's logic,
not vLLM compatibility.

## Adding a diagnosis rule

1. Add a `DiagnosisCategory`.
2. Implement `_check_*` in `RulesEngine` returning a `DiagnosisResult` with
   evidence and a `score` in [0, 1]; skip when inputs are missing (`None`).
3. Add a test with a synthetic trace.
