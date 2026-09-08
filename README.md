# llmtrace

**Flight recorder, attribution, and autopsy for vLLM inference.**

llmtrace instruments a vLLM `LLMEngine`, samples GPU telemetry alongside it,
and turns the two into per-request lifecycle traces, an energy ledger, and
rule-based diagnoses.

## Status: smoke-tested on one GPU

Validated on real vLLM 0.11.0 with `facebook/opt-125m` (RTX A5000 smoke run and
RTX A4500 diagnosis experiment, 2026-09-08); see
[docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md) for the exact results.
Larger models, multi-GPU, preemption and speculative decoding are untested.

| Area | Status |
|------|--------|
| Instrumentation of vLLM 0.11.0 `LLMEngine` (`add_request`/`step`/`abort_request`) | Verified on hardware: patched, traced 8/8 and 64/64 requests, restored cleanly |
| Scheduler batch metadata | Verified in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`); correctly reported unavailable with the default multiprocess core |
| GPU telemetry (NVML, background thread) | Verified: all fields populated on an A5000, samples taken while `generate()` blocks |
| Energy ledger (per-GPU integration, allocation policies, conservation) | Unit-tested with known totals; on the GPU run, device energy matched a separately collected `nvidia-smi` stream of the same NVML sensor within 0.15% over identical boundaries |
| Timing (TTFT/TPOT) | Verified through the raw engine loop; `LLM.generate()` forces FINAL_ONLY outputs and yields no first-token timing (documented) |
| Overhead | Small-model benchmark only (opt-125m, 64 x 256 tokens, 256 steps): +4% (`generate()`) and +9% (cumulative engine loop) wall time, 0.13 to 0.29 ms per engine step; larger models not measured |
| Rules-based diagnosis, CLI `analyze` / `compare`, offline analysis | Implemented and CPU-tested |
| Diagnosis experiment (short requests mixed with long prompts) | Run on one GPU: traces attribute the short-request tail to steps carrying 1536-token prefill chunks; `long_prefill_token_threshold=256` cut short TTFT p95 by 63% and worst stall by 54 to 62%, doubling long-request TTFT (`experiments/mixed_prompts/README.md`) |
| `llmtrace monitor` (attach to a running process) | Not implemented; exits with status 3 |
| AsyncLLM / OpenAI-compatible server | Not supported; instrumenting it raises `InstrumentationError` |
| Multi-node / distributed tracing, DCGM, dashboards | Not implemented |

[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) lists precisely what
is and is not verified.

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

**Run manifest** (`manifest.json`, written by the experiment driver): workload
and its hash, seed, model and revision, engine and llmtrace versions and git
commit, effective engine config, GPU and driver, tracer config, per-request
scheduled versus actual arrival, and `status: failed` with the error when a
configuration could not run.

**Collector self-events** (`collector_*.jsonl`): when llmtrace's own drains
ran and how long they took, so `findings` can flag engine steps the tracer
itself may have stalled.

**vLLM's own engine stats** (`vllm_stats_*.jsonl`): per step, via vLLM's
supported `stat_loggers` hook (works with the default multiprocess engine
core): KV-cache usage, running/waiting counts, preemptions, prefix-cache
stats, vLLM's own TTFT and inter-token latency samples, and finished-request
timings (queued/prefill/decode/e2e; vLLM attaches no request ids to these).
Enable at construction with `LLMEngine.from_engine_args(args, stat_loggers=[tracer.stat_logger_factory()])`,
or let `instrument_engine()` attach post-hoc when log stats are on.

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

**`LLM.generate()` cannot expose first-token timing.** In vLLM 0.11.0 it
forces `SamplingParams.output_kind = FINAL_ONLY`, so each request produces one
output at completion. Under `generate()` llmtrace records completion, token
counts, batches and energy, and reports `ttft_ms`/`tpot_ms` as unavailable
with that reason. To measure TTFT/TPOT, drive the engine directly with
cumulative outputs:

```python
from llmtrace.vllm_helpers import run_engine_with_timing

tracer.instrument_engine(llm.llm_engine)
outputs = run_engine_with_timing(llm.llm_engine, prompts, SamplingParams(max_tokens=32))
tracer.stop()
```

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
llmtrace visualize ./traces/run --compare ./traces/other --html-out report.html --trace-out run.perfetto.json
```

```bash
llmtrace findings ./exp/baseline_0                 # hypotheses with affected requests, evidence, missing evidence, next experiment
llmtrace decide --target "short ttft_p95 <= 300ms" \
    --config baseline=./exp/baseline_0,./exp/baseline_1 --config capped=./exp/capped_0,./exp/capped_1
```

`findings` evaluates four hypotheses on a run (queue overload, long-prompt
interference, KV-cache pressure with preemption, and llmtrace's own observer
effect) and reports each as supported, not supported, or not evaluable with
the missing evidence named. `decide` compares configurations (each a set of
repeats) against a stated target: which meet it in every repeat, throughput,
energy per output token with telemetry coverage, run-to-run range, failed
repeats, and whether the work was identical. A repeat counts toward a
candidate only if every selected request has the target metric, every
expected request completed (no aborted or incomplete ones), and the tracer's
health was clean; ineligible repeats are listed with reasons. It is advisory
and changes nothing.

`visualize` writes a self-contained HTML report (request timeline with
queue/prefill/decode phases, step durations over time and versus scheduled
tokens, GPU power, latency tables, optional side-by-side comparison) and a
Chrome/Perfetto trace JSON: open it at https://ui.perfetto.dev to scrub any
slow request against the scheduler steps it shared and the GPU power counter.
Step-level views need batch metadata (in-process scheduler).

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
