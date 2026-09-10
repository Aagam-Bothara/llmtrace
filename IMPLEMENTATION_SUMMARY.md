# Implementation summary

llmtrace collects traces inside vLLM and analyzes the saved files offline.
It targets vLLM 0.11.0. For measured results, use the
[status page](docs/STATUS.md) and [GPU validation record](docs/GPU_VALIDATION.md).

## Main pieces

| Component | Job |
|-----------|-----|
| Instrumentation | Track request submission, token observations, completion and aborts; restore engine methods on stop |
| Sampler and writer | Collect selected GPUs' telemetry and save records in bounded buffers and queues |
| Correlator | Integrate device power and allocate energy using a stated policy |
| Findings | Connect slow requests to supporting evidence and suggest experiments |
| Runner and planner | Replay a workload under a baseline and candidate settings |
| Decision report | Check compatible, healthy runs against a latency target |
| Doctor and reports | Explain signal availability; produce text, JSON, HTML and Perfetto output |

## Tested on CPU

Tests cover normal operation and failure paths: method restoration,
cancellation, incomplete requests, timing, missing signals, energy coverage,
GPU selection, concurrent writer shutdown, workload replay and CLI behavior.

Comparison tests cover mismatched workloads and models, missing health,
duplicate runs, metric coverage, strict inequalities, SLOs and uncertainty.
Synthetic GPU tests check known energy totals and prevent unrelated devices
or missing telemetry from appearing as request consumption.

These tests check the implementation. Hardware compatibility needs separate
GPU runs. The latest energy-selection and writer-lifecycle fixes were tested
on CPU; GPU inference was not rerun for those fixes.

## Tested on GPUs

Recorded sessions cover opt-125m and Qwen2.5-7B, synchronous and async request
tracing, scheduler metadata, NVML readings, CUDA-event spans and overhead.
Three findings have been exercised through the full experiment loop:
long-prompt interference, queue overload and KV-cache pressure.

The [GPU record](docs/GPU_VALIDATION.md) keeps the measurements, repeat counts
and limitations. The [raw evidence index](docs/gpu_runs/README.md) explains
how to download the traces and verify checksums.

## Main limits

- Full scheduler tracing needs the in-process core. CUDA spans support the blocking single-process executor.
- Step boundaries limit timing precision. `LLM.generate()` does not expose first-token timing.
- Energy is an allocation of whole-device readings. It requires enough telemetry and an unambiguous GPU selection.
- Models above 7B, speculative decoding and multiple outputs per request lack hardware validation.
- External-process attachment, server-process integration, distributed tracing, DCGM and dashboards are not implemented.

## Migrating from 0.1.0

- `LLMTracer.stop()`, `analyze()`, `Correlator.correlate_traces()` and `RulesEngine.diagnose_request()` are synchronous.
- Removed configuration keys: `use_dcgm`, `collect_tensor_utilization`, `enable_kv_cache_tracking`, `async_write`, `buffer_size`, `distributed_mode`, `rank`, `world_size`. Unknown keys raise an error.
- Energy fields are `attributed_joules`, `window_device_joules` and `joules_per_output_token`. The old `confidence` field and `exact` method were removed.
- `DiagnosisResult.confidence` became `score`. It ranks findings; it is not a probability.
- GPU sample fields can be missing. `BatchMetadata` changed, and vLLM/NVML dependencies moved to optional extras.

## Current comparison requirements

Older runs may now remain exploratory if they lack compatible manifests or
clean health records. Multi-GPU telemetry needs a recorded selection or an
explicit `--gpu-id` list. Insufficient energy coverage or allocations produce
`None` with a reason, while latency can remain eligible.
