# llmtrace

**Flight recorder, attribution, and autopsy for vLLM inference.**

llmtrace instruments a vLLM `LLMEngine`, samples GPU telemetry alongside it,
and turns the two into per-request lifecycle traces, an energy ledger, and
rule-based diagnoses.

## Status: smoke-tested on one GPU

Validated on real vLLM 0.11.0 with `facebook/opt-125m` (RTX A5000, A4500, 4000 Ada)
and `Qwen/Qwen2.5-7B` on A100 at TP=1 and TP=2 (2026-09-08); see
[docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md) for the exact results.
Preemption, speculative decoding, the OpenAI server process itself (only `AsyncLLM` used directly) and models above 7B are untested.
llmtrace does not measure GPU busy time; its per-step GPU span was cross-checked against Nsight Systems on opt-125m (span = busy time plus launch gaps, never below Nsight's busy time; see [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md)).

Status labels: **validated** = exercised on real vLLM 0.11.0 on a GPU with
committed evidence; **implemented** = CPU-tested against fakes shaped like the
verified vLLM interfaces; **not implemented** = absent.

| Area | Status |
|------|--------|
| Instrumentation of vLLM 0.11.0 `LLMEngine` (`add_request`/`step`/`abort_request`) | Validated: patched, traced 8/8 and 64/64 requests, restored cleanly |
| Scheduler batch metadata | Validated in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`); correctly reported unavailable with the default multiprocess core |
| GPU telemetry (NVML, background thread) | Validated: all fields populated on an A5000, samples taken while `generate()` blocks |
| Energy ledger (per-GPU integration, allocation policies, conservation) | Unit-tested with known totals; on the GPU run, device energy matched a separately collected `nvidia-smi` stream of the same NVML sensor within 0.15% over identical boundaries |
| Timing (TTFT/TPOT) | Validated through the raw engine loop; `LLM.generate()` forces FINAL_ONLY outputs and yields no first-token timing (documented) |
| Overhead | Small-model benchmark only (opt-125m, 64 x 256 tokens, 256 steps): +4% (`generate()`) and +9% (cumulative engine loop) wall time, 0.13 to 0.29 ms per engine step; larger models not measured |
| Evidence-based findings (`llmtrace findings`) with assumptions, competing explanations and confidence limits | Implemented; the long-prompt-interference finding validated on the GPU experiment; the others report not supported or insufficient evidence on the recorded runs |
| Experiment planner (`llmtrace plan`, `llmtrace run --plan`) | Implemented; CPU-tested end to end on the synthetic engine (plan from findings, run baseline + candidates, decide); rules cover the five findings with bounded scheduler/cache knobs |
| Goodput under per-class SLOs and bootstrap intervals in `decide` | Implemented and CPU-tested |
| vLLM engine stats via `stat_loggers` hook | Validated on `AsyncLLM` (34 per-step records with KV usage over the multiprocess core); attached post-hoc on the sync engine in fakes only |
| GPU span per step (CUDA events around `execute_model`), `host_overhead` finding | Validated (RTX 4000 Ada, A100 with Qwen2.5-7B): one span per step, never above host time, timer clean; long-prefill interference is GPU compute (7.9 vs 1.7 ms on opt-125m, 103 vs 11 ms on the 7B model); host share 9 to 13% on opt-125m, 2% on the 7B model; refused for TP>1 executors |
| NVTX ranges per step, Nsight Systems cross-check (`scripts/nsys_step_compare.py`) | Validated (RTX A5000, opt-125m, Nsight Systems 2026.1): all 626 and 654 step ranges of two runs matched to llmtrace's spans; the span was never below Nsight's GPU busy time (kernels plus CUDA-graph executions); busy/span 0.58 on decode steps of this launch-bound 125M model, 0.84 on 1536-token prefill steps |
| Threshold screens in the rules engine, CLI `analyze` / `compare` | Implemented and CPU-tested; screens flag symptoms only and never assert a cause |
| Diagnosis experiment (short requests mixed with long prompts) | Run on opt-125m (RTX A4500, RTX 4000 Ada) and on Qwen2.5-7B (A100, TP=1 and TP=2): traces attribute the short-request tail to steps carrying 1536-token prefill chunks, CUDA spans show that cost is GPU prefill compute (103 vs 11 ms steps on the 7B model); `long_prefill_token_threshold=256` cut short TTFT p95 by 62 to 72% and the worst stall by 45 to 72%, raising long-request TTFT by 64 to 116% (`experiments/mixed_prompts/README.md`) |
| Configuration-driven workloads and runs (`llmtrace workload`, `llmtrace run`) | Implemented (synthetic engine CPU-tested; template reproduces the GPU-validated experiment workload); the real-engine path mirrors the validated experiment driver but has not itself been run on hardware |
| `llmtrace doctor` (which signals this environment or a recorded run can provide, and why not) | Implemented; CPU-tested with injected probes |
| `llmtrace monitor` (attach to a running process) | Not implemented; exits with status 3 |
| `AsyncLLM` (the OpenAI-server engine) via `instrument_async_engine()` | Validated (RTX 4000 Ada, opt-125m): 6 concurrent streams traced with TTFT and engine token counts, a client-cancelled stream recorded as aborted after 4 tokens, `generate`/`abort` restored, vLLM per-step stats via `stat_loggers` over the multiprocess core; no batch membership, queue/prefill boundary or GPU spans there, reported as unavailable with the reason |
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

**Run manifest** (`manifest.json`, written by `llmtrace run` and the experiment driver): workload
and its hash, seed, model and revision, engine and llmtrace versions and git
commit, effective engine config, GPU and driver, tracer config, per-request
scheduled versus actual arrival, and `status: failed` with the error when a
configuration could not run.

**GPU span per step** (`gpu_steps_*.jsonl`, in-process engine core with
`torch.cuda`): CUDA events recorded before and after each
`model_executor.execute_model` call give the step's GPU span (an upper bound
on GPU busy time; launch gaps included, other streams excluded) and the host
overhead `host_step_ms - gpu_span_ms`. Read lazily, never by synchronizing.
`enable_nvtx` adds an NVTX range per step for Nsight Systems, and
`scripts/nsys_step_compare.py` joins an `nsys` profile with the spans per
step (on opt-125m the span held as an upper bound on every step and was 58%
busy on decode steps, 84% on long prefill steps; the rest is launch gaps).

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

Per-request energy on a shared GPU is an allocation, not a measurement.
Integrating power over a request's lifetime double-counts everything that ran
alongside it, so llmtrace never reports that number as consumption. The model:

```
E_r = sum over elementary intervals t of  E_t * w_{r,t} / sum_{j in A_t} w_{j,t}
```

* `E_t` is the device energy in interval `t`: each GPU's NVML power integrated
  by trapezoid on that GPU's own timestamps, then summed over GPUs. Gaps
  longer than `max_sample_gap_s` are not integrated (uncovered time).
* `A_t` is the set of requests active in `t`. Membership comes from scheduler
  batch records (which requests were scheduled in the step spanning `t`) when
  the in-process scheduler is visible; otherwise from request windows, which
  include queue wait, and the ledger says so.
* `w_{r,t}` is the weight: 1 for `equal_share`, prompt+output tokens for
  `proportional_tokens`. `window_only` sets no weights and reports only the
  shared device energy over each request's window, labeled as shared.
* Elementary intervals are cut at every request/span boundary, so phase energy
  (queue/prefill/decode) integrates the power curve within each phase rather
  than splitting a total by elapsed time.
* Conservation, checked to floating-point tolerance on every run:
  `device = attributed + idle + unattributable`, where idle is energy in
  intervals with no active request (reported, never attributed) and
  unattributable is energy in intervals whose requests lacked enough telemetry
  coverage for a figure. Energy outside the run window is not counted.
* Insufficient telemetry yields `null` with a reason, never zero. Telemetry
  coverage is reported with every figure.

Validated only in the sense that llmtrace's integrated device energy matched a
separately collected `nvidia-smi` stream of the same sensor within 0.15%
(`docs/GPU_VALIDATION.md`); the allocation weights are a stated policy, not a
physical measurement, and the docs say so wherever a per-request figure appears.

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
llmtrace doctor                                 # which signals this environment can produce, and why not
llmtrace workload template --output w.json      # request classes, length distributions, arrival processes, seed
llmtrace workload preview w.json                # generated request list summary and workload hash
llmtrace run --workload w.json --engine fake --out ./runs/base --repeat 3          # synthetic engine (CPU)
llmtrace run --workload w.json --engine fake --out ./runs/capped --repeat 3 \
    --config-name capped --set long_prefill_token_threshold=256
VLLM_ENABLE_V1_MULTIPROCESSING=0 llmtrace run --workload w.json --engine vllm --model facebook/opt-125m --out ./runs/gpu
llmtrace doctor ./runs/base/r0                  # which signals the recorded run has, and why the others are missing
```

```bash
llmtrace analyze ./traces                       # analyze a run directory
llmtrace analyze ./traces --output report.json  # machine-readable report
llmtrace compare --baseline ./traces/baseline --current ./traces/current \
    --ttft-threshold 5 --energy-threshold 10 --fail-on-regression
llmtrace init-config --output llmtrace_config.json
llmtrace visualize ./traces/run --compare ./traces/other --html-out report.html --trace-out run.perfetto.json
```

```bash
llmtrace findings ./runs/base/r0 --verbose         # hypotheses: evidence, missing evidence, assumptions, competing explanations, limits
llmtrace plan ./runs/base/r0 --json plan.json      # bounded configuration experiments derived from the supported findings
llmtrace run --workload w.json --plan plan.json --engine fake --out ./exp   # baseline + candidates, <out>/<config>/r<i>
llmtrace decide --target "short ttft_p95 <= 300ms" --slo "short: ttft <= 300ms, tpot <= 20ms" \
    --config baseline=./exp/baseline/r0,./exp/baseline/r1 --config cap512=./exp/cap512/r0,./exp/cap512/r1
```

`findings` evaluates five hypotheses on a run (queue overload, long-prompt
interference, KV-cache pressure with preemption, host overhead, and
llmtrace's own observer effect) and reports each as supported, not supported,
or insufficient evidence with the missing evidence named. Every finding also
carries the check's assumptions, the competing explanations the recorded data
cannot rule out, and its confidence limits (`--verbose` prints them); a
supported finding is a consistent pattern in the events, never a root cause,
and names the replay experiment that would establish one.

`plan` turns a run's supported findings into a bounded, reviewable list of
configuration experiments (for example a `long_prefill_token_threshold` sweep
below the largest observed chunk, `max_num_seqs` doubled when the running
count hit it, `gpu_memory_utilization` raised by 0.1 under KV pressure), each
with the effect expected if the finding is the cause and the class expected
to pay for it. `llmtrace run --plan` executes the baseline and every
candidate as fresh engines with the same workload and repeat count; no
running server is touched.

`decide` compares configurations (each a set of repeats) against a stated
target: which meet it in every repeat, a seeded 95% bootstrap interval of the
target statistic over the pooled requests (a candidate whose interval's upper
bound misses the target is flagged marginal), goodput under per-class SLOs
(`--slo "short: ttft <= 50ms, tpot <= 15ms"`), throughput, energy per output
token with telemetry coverage, run-to-run range, failed repeats, and whether
the work was identical. A repeat counts toward a candidate only if every
selected request has the target metric, every expected request completed (no
aborted or incomplete ones), and the tracer's health was clean; ineligible
repeats are listed with reasons. It is advisory and changes nothing.

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

* [docs/AUDIT.md](docs/AUDIT.md): architecture, capabilities, risks, hardware-backed claims, upstream overlap, roadmap
* [QUICKSTART.md](QUICKSTART.md)
* [DEVELOPMENT.md](DEVELOPMENT.md): architecture, verified interfaces, semantics
* [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md): first GPU run checklist
* [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md): implementation status

## License

MIT
