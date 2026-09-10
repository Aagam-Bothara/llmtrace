# Design review

This is a review of the implementation, recorded GPU evidence and integration
risks. It updates the earlier September 2026 audit; it does not report a new
hardware run. Use [status](STATUS.md) for coverage and
[GPU validation](GPU_VALIDATION.md) for measurements.

## Execution flow

```text
Inside the vLLM process
  Request and scheduler hooks -> bounded buffers
  NVML sampler                -> GPU telemetry buffer
  Collector                   -> writer queue -> run files

Offline
  Run files -> analysis, findings, decision tables and visual reports
  Findings  -> experiment plan -> fresh engines -> new run files
```

The main modules are in `llmtrace/`. CPU tests use fakes in
`llmtrace/testing/fakes.py`. GPU scripts, manifests and reports live under
`docs/gpu_runs/`; raw JSONL records are downloadable release assets.

## What llmtrace adds

The upstream comparison was made against the vLLM 0.11.0 source, including
`output_processor.py`, `metrics/loggers.py`, `metrics/stats.py` and
`benchmarks/serve.py`.

| Capability | vLLM 0.11.0 provides | llmtrace adds |
|------------|---------------------|---------------|
| Request latency | OpenTelemetry spans and Prometheus latency histograms | Local request files linked to engine steps |
| Engine stats | Running/waiting counts, KV usage, preemptions and iteration stats | Saved stats alongside request and step records |
| Batch composition | Scheduler state inside the engine | Request IDs and scheduled tokens for each recorded step |
| Workload benchmarking | Arrival generation, latency metrics and goodput | Findings linked to configuration experiments and repeated comparisons |
| GPU timing and energy | Inputs available through GPU tooling | CUDA-event spans, selected-device power integration and an explicit allocation policy |

The useful connection is from a slow request to the work it shared, then to
a configuration change that can be tested. Existing vLLM metrics remain part
of that evidence.

## Integration risks

| Dependency | Why it needs care |
|------------|-------------------|
| Internal vLLM attribute paths | Scheduler, executor or logger changes can make hooks unavailable. A version upgrade needs source review and GPU smoke tests. |
| Scheduler update order | Prefill classification subtracts the current step's scheduled tokens from an already-updated counter. A change in that order can misclassify work. |
| Blocking executor behavior | CUDA brackets need model execution on the measured stream. Submission-only or remote execution does not provide a meaningful span. |
| Async cancellation order | Wrappers must preserve engine cancellation and abort behavior while recording the outcome. |
| Final-only outputs | `LLM.generate()` hides first-token observations; the cumulative-output helper depends on the engine loop contract. |
| Clock domains | Monotonic clocks are comparable within a tracer session. Mixed sessions fall back to wall time. |
| Shared hardware | Physical GPU selection excludes unrelated devices, but other processes on a selected device still affect power readings. |

Missing hooks are reported with reasons. Instrumentation errors are counted;
`strict_instrumentation` can raise them. These checks help expose failures,
but CPU fakes cannot prove compatibility with a changed engine.

## Correctness checks

Comparisons require compatible manifests and clean tracer health. Duplicate
runs cannot count as independent repeats. Missing metrics or health records
remain visible and cannot qualify that run for a recommendation. Targets and
SLOs preserve strict and inclusive bounds.

Energy reporting checks selected devices, telemetry coverage and allocation
availability. A sample without an integration interval cannot become a zero
consumption claim. Latency eligibility is separate from energy availability.

Writer submission and shutdown-sentinel insertion share a lifecycle lock.
The worker is joined after releasing that lock, so an accepted submission
cannot be left behind the sentinel. CPU tests force the competing operations
for inline and background writers.

## What still needs evidence

- A repeatable GPU integration suite, rather than only manual session scripts.
- Positive hardware cases for host overhead and tracer observer effects.
- Broader workloads, more repeats and models above 7B.
- A benchmark set with planted bottlenecks and measured finding accuracy.
- Hardware validation of the latest GPU-selection and energy-availability fixes.

The original experiment driver and the generic runner also maintain similar
replay protocols. Keeping them aligned remains a maintenance task.

## Next steps

1. Turn the recorded GPU procedures into opt-in integration tests.
2. Build small, repeatable workloads with known bottlenecks and expected findings.
3. Add a report combining findings, tested changes, per-class regressions and limits.
4. Consolidate overlapping helpers in `utils/batch_analyzer.py` and `utils/latency_explainer.py`.

These are proposed improvements, not implemented features. Current scope
excludes a new scheduler, an LLM-based recommender, dashboards, DCGM and AMD support.
