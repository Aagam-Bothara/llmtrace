# llmtrace

**Flight recorder, attribution, and autopsy for vLLM inference.**

llmtrace instruments a vLLM `LLMEngine`, samples GPU telemetry alongside it,
and turns the two into per-request lifecycle traces, an energy ledger, and
rule-based diagnoses.

## Status: pre-GPU-validation

This is an early implementation that has **not yet been run against real vLLM
or a real GPU**. What exists today:

| Area | Status |
|------|--------|
| Instrumentation of vLLM 0.11.0 `LLMEngine` (`add_request`/`step`/`abort_request`) | Implemented against interfaces verified from the vLLM 0.11.0 source; exercised only with a fake engine (CPU tests) |
| Scheduler batch metadata | Implemented for the in-process scheduler (`VLLM_ENABLE_V1_MULTIPROCESSING=0`); reported as unavailable otherwise |
| GPU telemetry (NVML, background thread) | Implemented; exercised only with a fake backend |
| Energy ledger (per-GPU integration, allocation policies, conservation) | Implemented and unit-tested with known totals |
| Rules-based diagnosis, CLI `analyze` / `compare`, offline analysis | Implemented and CPU-tested |
| `llmtrace monitor` (attach to a running process) | Not implemented; exits with status 3 |
| AsyncLLM / OpenAI-compatible server | Not supported; instrumenting it raises `InstrumentationError` |
| Multi-node / distributed tracing, DCGM, dashboards | Not implemented |
| Overhead measurements | None taken yet; see `docs/GPU_VALIDATION.md` |

See [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md) for what the first GPU
run must check, and [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
for a precise list of what is and is not verified.

## What it records

**Per request** (`traces_*.jsonl`): arrival, completion, status
(completed/aborted/incomplete), prompt and output token counts from engine
token ids, spans at engine-step granularity, TTFT and TPOT with explicit
"unavailable" reasons, batch ids, and an energy allocation with its policy and
telemetry coverage. Durations come from the monotonic clock; wall-clock
timestamps are kept as metadata.

**Per scheduler step** (`batches_*.jsonl`, in-process scheduler only): real
request ids scheduled, tokens scheduled per request, prefill/decode counts,
KV-cache usage fraction.

**GPU telemetry** (`gpu_*.jsonl`): power, utilization, memory, clocks,
throttle reasons per GPU. Fields the driver does not report are `null`, never 0.

## Energy accounting, precisely

* **Telemetry** is measured: NVML power readings.
* **Device energy** is an estimate: each GPU's power is integrated on its own
  timestamps (trapezoid); gaps longer than `max_sample_gap_s` are not
  integrated; device energies are then summed.
* **Attributed energy** is an allocation estimate: in every elementary time
  interval the device energy is split among the requests active in it
  (`equal_share` or `proportional_tokens`). Membership comes from scheduler
  batch metadata when available, otherwise from request windows (which include
  queue wait). `window_only` skips allocation and only reports device energy
  during each request window, labeled as shared.
* **Conservation**: `device = attributed + idle + unattributable` over the run
  window, checked to floating-point tolerance. Idle energy (no request active)
  is reported, not attributed. Energy outside the run window is not counted.
* **Insufficient telemetry** yields `null` with a reason, not zero.

## Install

```bash
pip install -e .                 # offline analysis + CLI, no GPU deps
pip install -e ".[nvml]"         # + NVML telemetry
pip install -e ".[parquet]"      # + parquet output
pip install -e ".[vllm]"         # + vllm==0.11.0 (Linux, NVIDIA GPU)
pip install -e ".[dev]"          # + pytest, ruff
```

## Use with vLLM 0.11.0 (offline `LLM` API)

```python
from vllm import LLM, SamplingParams
from llmtrace import LLMTracer

tracer = LLMTracer(output_dir="./traces", gpu_sample_interval_ms=100)
llm = LLM(model="facebook/opt-125m")
tracer.instrument_engine(llm.llm_engine)     # patches the engine, starts collection threads

outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=32))

tracer.stop()                                # restores the engine, drains, flushes
print(tracer.health())                       # errors, drops, telemetry availability
tracer.print_analysis(tracer.analyze())
```

Set `VLLM_ENABLE_V1_MULTIPROCESSING=0` before creating the `LLM` to keep the
engine core in-process; that is the only configuration in which the scheduler
is reachable and batch metadata plus queue/prefill spans are recorded.

## CPU-only synthetic example

```bash
python examples/synthetic_replay.py
```

Everything in it is fabricated (fake engine, fake NVML). It demonstrates the
pipeline and the `compare` exit codes; its numbers mean nothing about hardware.

## CLI

```bash
llmtrace analyze ./traces                       # analyze a run directory
llmtrace analyze ./traces --output report.json  # machine-readable report
llmtrace compare --baseline ./traces/baseline --current ./traces/current \
    --ttft-threshold 5 --energy-threshold 10 --fail-on-regression
llmtrace init-config --output llmtrace_config.json
```

`compare` sign convention: change = (current - baseline) / baseline. All
compared metrics are higher-is-worse, so only a positive change above the
threshold is a regression; improvements never fail. Missing metrics and zero
baselines are reported as unavailable (`--fail-on-missing` makes them fail).

## Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

The suite runs without a GPU, NVML or vLLM. It validates llmtrace's own logic
against fakes shaped like the vLLM 0.11.0 interfaces; it does not prove
compatibility with real vLLM.

## Documentation

* [QUICKSTART.md](QUICKSTART.md)
* [DEVELOPMENT.md](DEVELOPMENT.md): architecture, verified interfaces, semantics
* [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md): first GPU run checklist
* [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md): implementation status

## License

MIT
