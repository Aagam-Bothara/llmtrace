# Development guide

Start with the [quickstart](QUICKSTART.md) to run the tool. This guide explains
the implementation and the meaning of its records.

## Architecture

```text
vLLM engine -> request/scheduler hooks -> buffers -> collector -> writer
NVML        -> GPU sampler            -> buffer  -> collector -> writer

Saved files -> correlator -> reports
            -> findings  -> experiment plan -> new runs -> decision
```

`data_plane/` collects records inside the engine process. `control_plane/`
analyzes them offline, without vLLM or NVML. `workload.py` defines request
workloads; `runner.py` replays them; `doctor.py` reports available signals.

## Workloads and runs

A `WorkloadSpec` lists request classes, prompt and output lengths, arrival
patterns and a seed. Generation is deterministic. Supported length patterns
include fixed, uniform, choice and clipped lognormal; arrival patterns
include simultaneous, constant, Poisson, gamma and bursts.

Request IDs use `<class>-<index>`. Prompts are token-ID lists, so requested
lengths do not depend on a tokenizer. Keep IDs and seeds stable for replay.

The runner writes traces, batches, GPU samples, CUDA spans, vLLM stats,
collector timings, `workload.json`, `manifest.json` and `run_info.json`.
Analysis commands produce the derived reports separately.

Real engines run in fresh child processes to avoid GPU-memory conflicts
between consecutive engines. Start failures, crashes and timeouts leave a
failed manifest. The runner enables vLLM stats with `disable_log_stats=False`.
Existing run directories are refused unless `--overwrite` is set.

## Source provenance

The manifest records a source fingerprint, Git commit and dirty-tree state.
Dirty runs also save a patch and an archive of untracked files where possible.
Files above 1 MB are listed but not archived. Binary changes, oversized files,
archive failures or a missing Git tree make `llmtrace_snapshot_complete` false,
with details in `llmtrace_snapshot_gaps`.

A fingerprint identifies code; it cannot restore it. Reproduction needs the
commit and a complete patch/archive snapshot. `doctor` compares the recorded
fingerprint with the installed code and reports missing source evidence.

## Health

`assess_health()` is shared by the runner, `doctor` and `decide`.
Instrumentation failures, unfinished requests, lost traces and writer or
collector errors make a repeat ineligible. `decide` also rejects unknown
health for recommendations.

Missing or lossy telemetry disables the dependent measurement. It does not
by itself invalidate request latency. Energy reporting additionally needs
sufficient integrated coverage and available allocations.

## Verified vLLM target


llmtrace targets exactly **vLLM 0.11.0**, V1 engine, synchronous
`vllm.v1.engine.llm_engine.LLMEngine` as used by `vllm.LLM`. The reference below was checked against the tagged source:

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

## Instrumentation and timing

Wrappers preserve engine return values and exceptions. Bookkeeping failures
are counted and logged; `strict_instrumentation=True` raises them. Stopping
restores original methods and records requests still active as incomplete.
Abort calls and abort finish reasons produce aborted traces.

Prompt length comes from engine token IDs, with caller token IDs as a fallback.
Output tokens come from completion token IDs. Delta outputs are accumulated;
cumulative and final-only outputs use their reported totals.

| Measurement | Definition |
|-------------|------------|
| TTFT | First-token step end minus request arrival |
| TPOT | Time from first to last token observation divided by tokens after the first observation |
| Queue span | Arrival to the first scheduled step's start |
| Prefill span | That step's start to the first-token step's end |

Durations use monotonic time. TTFT includes queue wait and the full step that
first exposes a token. TPOT is unavailable without enough observations.
Final-only and pooling outputs cannot provide first-token timing.

Queue and prefill boundaries require the in-process scheduler. Otherwise,
records use one `time_to_first_token` span. Batch classification accounts for
`schedule()` already advancing `num_computed_tokens`: subtract scheduled
tokens before deciding whether a request is still in prefill. Prefix-cache
hits and preemption affect that counter.

Batch IDs are generated by llmtrace; request IDs come from the engine.
`step_end_monotonic` makes a batch an execution interval for allocation.

## vLLM stats

`data_plane/vllm_stats.py` records vLLM's `stat_loggers` output: queue depth,
KV usage, preemptions, prefix-cache stats and latency samples. These signals
can work with the default multiprocess core.

Attach at construction with
`LLMEngine.from_engine_args(args, stat_loggers=[tracer.stat_logger_factory()])`,
or let `instrument_engine()` attach when engine stats logging is enabled.
The logger is removed on stop; health records explain attachment failures.
Finished-request stats do not include request IDs, so they support run-level
findings rather than per-request attribution.

## Findings and decisions

`RulesEngine` screens individual requests for symptoms. `findings.py` joins
requests, steps and engine stats to test hypotheses. Add new diagnosis work
to findings when it needs evidence across records.

Each finding includes its status, affected requests, source files and fields,
missing evidence, assumptions, competing explanations and confidence limits.
Statuses are `supported`, `not_supported` and `insufficient_evidence`.
A supported finding suggests an experiment; it does not prove a root cause.

`decision.evaluate()` compares configurations against a target such as
`short ttft_p95 <= 20ms`. It accepts `<` and `<=` without changing the bound.
Before ranking, it checks:

- Compatible workload definition/hash, seed, model/revision, engine/version and intended arrivals.
- Matching per-request IDs, model names, prompt lengths and output lengths.
- Clean health, completed requests and sufficient target-metric coverage.
- Independent runs: duplicate paths or tracer session IDs cannot count twice.

Missing comparison evidence stays exploratory. A configuration needs at
least two eligible repeats by default (`--min-repeats`); three or more are
preferable. A workload mismatch withholds ranking for the comparison.

The report shows the range across runs separately from a seeded request
bootstrap interval (1,000 resamples by default). Requests sharing steps are
related, so the bootstrap understates run-to-run uncertainty. A candidate
whose interval's upper bound misses the target is marked marginal.

SLOs define per-request limits. Goodput is the share of selected requests
meeting every bound; missing metrics count as a failure. Recommendations
prefer median goodput when SLOs are supplied, then throughput; otherwise they
prefer median throughput. They never modify a running server.

## Experiment planner

`plan_experiments()` maps supported findings to bounded candidate changes.
Examples include smaller prefill chunks, a higher sequence cap or more KV
memory. It ranks candidates by affected-request count, removes duplicates
and applies `--max-candidates`. `--finding` restricts the hypotheses.

The plan records the source engine, model, workload hash and settings.
Reproducible settings come from an explicit map of effective configuration
fields, with recorded engine kwargs and scheduling changes applied on top.
Settings without a known input argument are listed as `unreproduced`; derived
values such as `num_gpu_blocks` are excluded from that list.

`run --plan` starts a fresh engine for each configuration and repeat under
`<out>/<config>/r<i>`. The baseline uses the reconstructed source settings;
candidates add their change. Review warnings about a changed workload or
settings the plan cannot reproduce.

## GPU spans

`CudaStepTimer` places CUDA events around `execute_model` on the current
stream. The elapsed span includes launch gaps. It excludes work on other
streams and cannot by itself distinguish kernel execution from host stalls.
`host_overhead_ms` is the difference between host step duration and GPU span.

Events are queried later without synchronization; pending events are bounded
and drops are counted. This supports the blocking `UniProcExecutor` path.
Async submission and workers in other processes are outside its scope.

`enable_nvtx` adds step ranges for Nsight Systems. The comparison script
`scripts/nsys_step_compare.py` joins these ranges to kernel and CUDA-graph
activity. See [GPU validation](docs/GPU_VALIDATION.md) for the observed
span-versus-busy-time gap.

## Clocks

Records carry wall time, monotonic time and a tracer-session `clock_domain`.
The correlator uses monotonic time when traces and samples share a domain;
otherwise it uses wall time. The choice is recorded in `ledger.clock`.

## Energy ledger

Device energy is estimated from power samples. Request energy is an
allocation of that total:

```text
request share = interval energy * request weight / total active weight
```

The correlator works in this order:

1. Select participating physical NVML GPU indices. Ambiguous multi-GPU input or missing selected-device telemetry yields no energy estimate.
2. Group power readings per GPU, ignore missing/nonfinite power, sort timestamps and keep the last duplicate.
3. Integrate each GPU's power by trapezoid. Gaps above `max_sample_gap_s` (default 1 s) are uncovered.
4. Use the run window from earliest arrival to latest completion. Coverage is the minimum covered fraction across participating GPUs.
5. Find active requests from usable batch intervals, or request windows when batch timing is unavailable.
6. Split each interval's energy by policy. `equal_share` uses equal weights; `proportional_tokens` uses total prompt plus output tokens. `window_only` does not allocate request energy.
7. Check conservation: attributed + idle + unattributable energy equals integrated device energy within tolerance.

Request windows need positive coverage, at least two in-window power samples
and the configured coverage fraction (default 50%). Otherwise their share
is unattributable. With no integration interval, energy is `None`, not zero.
`decide` also requires sufficient run coverage and an allocation for every
request before reporting J/token. It records `energy_unavailable_reason`
and keeps latency eligibility separate.

Use `gpu_sampler.gpu_ids` for collection or `EnergyConfig(gpu_ids=[...])`
for direct offline correlation. CLI `run`, `analyze` and `decide` accept
repeatable `--gpu-id`. Physical indices and UUIDs are saved in
`manifest.gpu_selection`; old records listing visible devices alone do not
prove which GPUs participated. Other processes on selected devices still
contribute to whole-device power.

## Collection and shutdown

- `GPUSampler` uses a bounded buffer and an atomic drain. Unavailable telemetry is reported; `require_gpu=True` makes it a start failure.
- `TraceWriter` uses a bounded queue. Full-queue and post-stop submissions count as drops. Acceptance and shutdown-sentinel insertion share a lifecycle lock; the worker is joined after releasing it.
- `LLMTracer.stop()` joins the collector, restores hooks, stops sampling, drains remaining records, writes incomplete requests and stops the writer.
- Failed startup restores the engine, stops started threads and releases NVML before raising the original error.

Parquet output writes separate part files rather than appending to an
existing file.

## Tests and contributions

```bash
pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

Fakes in `llmtrace/testing/fakes.py` use an injectable clock and model the
verified vLLM interfaces. They check local logic; use the
[GPU procedure](docs/GPU_VALIDATION.md) to check engine compatibility.

For a new finding, define the required evidence, missing-data behavior,
assumptions and competing explanations. Test both supported and unsupported
cases. For a simple request-level screen, add a `DiagnosisCategory` and a
`RulesEngine._check_*` method with evidence and a ranking score, then test it
with a synthetic trace. Scores are not probabilities.
