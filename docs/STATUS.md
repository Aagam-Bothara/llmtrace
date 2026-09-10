# Status and limitations

llmtrace targets **vLLM 0.11.0**. This page separates recorded hardware
results from CPU tests and features that are still missing.

## What works

| Area | Status and limits |
|------|-------------------|
| Synchronous request tracing | Tested on real `LLMEngine`: submission, completion, aborts and method restoration |
| Scheduler metadata | Tested with the in-process core; unavailable with the default multiprocess core |
| Async request tracing | Tested on directly constructed `AsyncLLM`: concurrent streams, cancellation and restoration |
| TTFT and TPOT | Tested through cumulative engine outputs; unavailable through `LLM.generate()` |
| NVML telemetry | Tested on GPUs; missing fields and errors are reported |
| CUDA-event step spans | Tested on RTX 4000 Ada and A100, with an Nsight cross-check on A5000; blocking single-process executor only |
| Energy accounting | Known totals, gaps and allocation policies tested on CPU; a recorded integration matched a separate `nvidia-smi` stream within 0.15% |
| Findings | Long-prompt interference, queue overload and KV pressure tested through the full experiment loop |
| Planner and runner | Baselines and candidate settings replayed on real vLLM in fresh processes |
| Goodput and uncertainty | CPU-tested SLO checks, request bootstrap intervals and variation across runs |
| Comparison validation | CPU-tested manifest compatibility, health requirements, duplicates, missing metrics and strict bounds |
| GPU selection and energy availability | CPU-tested physical-device selection, missing telemetry and incomplete allocations; no new GPU inference run for these fixes |
| Writer shutdown | CPU-tested concurrent submission and shutdown for inline and background writes |
| Doctor, reports and visualization | CPU-tested environment/run checks, text and JSON reports, HTML and Perfetto output |

A separate `nvidia-smi` stream reads the same NVML sensor. Agreement checks
the sampling and integration, not the sensor's absolute accuracy.

## Recorded results

These numbers describe specific experiments. They are not performance
promises for other workloads.

| Experiment | Recorded result |
|------------|-----------------|
| Queue overload, Qwen2.5-7B on A100 | Four repeats: doubling the sequence cap reduced burst TTFT p95 from 6.43 s to 2.18 s, with 47% lower energy per output token |
| KV-cache pressure, Qwen2.5-7B on A100 | Three repeats: raising memory utilization removed 22 preemptions and reduced end-to-end p95 from 38.0 s to 29.9 s |
| Mixed prompts, opt-125m and Qwen2.5-7B | A 256-token prefill cap reduced short-request TTFT p95 by 62?72%, while long-request TTFT rose by 64?116% |
| CUDA span vs Nsight busy time, opt-125m | The span was never below busy time across 1,280 steps; busy time was about 58% of the span on decode steps and 84% on long-prefill steps |
| Tracing overhead with step timing, opt-125m | +7.7% for `generate()` and +14.4% for the engine loop |
| Tracing overhead, Qwen2.5-7B | +1.2% for `generate()` and +1.7% for the engine loop; CUDA-event recording cost 0.08 ms per step |

Full tables, environments, earlier attempts and caveats are in
[GPU validation](GPU_VALIDATION.md). Download the underlying records from
the [evidence index](gpu_runs/README.md). Historical reports describe the
code used in those sessions; current comparison checks may withhold results
when older records lack required evidence.

## How to interpret the limits

**Timing:** observations are tied to engine steps. CUDA spans include launch
gaps and exclude work outside the measured stream; they do not directly
measure GPU busy time.

**Energy:** select the engine's physical GPUs. Readings include other work
sharing those GPUs, and request energy is an allocation. Missing coverage or
allocations make energy unavailable without invalidating latency.

**Findings:** `host_overhead` and `tracer_observer_effect` have returned
`not_supported` on recorded runs. That does not validate their ability to
detect a real positive case.

**Repeats:** the largest recorded comparison has four independent repeats
per configuration. A bootstrap over requests does not replace more runs.

## Not validated on hardware

Models above 7B, speculative decoding, multiple outputs per request (`n > 1`),
pipeline parallelism, aborts under sustained load and nontrivial throttle
reason bits still need coverage. The OpenAI-compatible server process itself
has not been instrumented in the recorded tests.

## Not implemented

- Attaching to an existing process: `llmtrace monitor` exits with code 3.
- Automatic integration inside the OpenAI-compatible server process.
- Distributed or multi-node tracing and tensor-parallel worker instrumentation.
- DCGM, dashboards, OTLP export and ML-based diagnosis.

See the [design review](AUDIT.md) for remaining risks and next steps.
