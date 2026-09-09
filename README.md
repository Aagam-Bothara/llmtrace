# llmtrace

**Find out why your vLLM workload is slow, try the fix that the evidence points to, and check that it actually helped.**

On Qwen2.5-7B under bursty load, llmtrace's traces showed requests queueing
behind a sequence cap, its planner proposed doubling the cap, and the replay
cut burst time-to-first-token p95 from 6.4 s to 2.2 s at 47% lower energy
per token, across four independent repeats from a fingerprinted commit
([the evidence](docs/GPU_VALIDATION.md)).

llmtrace sits inside a vLLM 0.11.0 process and records what the scheduler
did to every request: which engine step it waited in, which requests it
shared that step with, how many tokens each of them was scheduled, the
CUDA-event span of each step (including gaps between kernel launches), and
what the card drew in power. From those records it answers three questions
in order. What is the bottleneck, and what is the
evidence for it? Which configuration changes are worth trying? Did a change
help the requests you care about without hurting the others, across
independent repeats?

It is not a dashboard, not a metrics exporter, and not an AI that guesses. It
is a measurement tool with a diagnosis layer that shows its work and a small
experiment runner that replays the same workload under one change at a time.

**Read this before anything else.** The scheduler-level signals that make the
diagnosis work (batch membership, chunk sizes, queue and prefill boundaries,
GPU time per step) exist only with vLLM's engine core running in the same
process, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, which production deployments do
not use. llmtrace is a research and pre-production tool: you reproduce a
workload on a scratch engine, diagnose it there, and take the configuration
decision back to production. With the default multiprocess core, or the
OpenAI server's `AsyncLLM`, you get request-level traces, GPU telemetry and
vLLM's own per-step stats, and `llmtrace doctor` tells you exactly what is
missing. It is pinned to vLLM 0.11.0 and verified against that source.

## The problem it was built for

Picture a serving deployment where most requests are short chats and a few
are long documents. The short ones have a bad tail: p95 time-to-first-token
is several times the median, and nothing in the Prometheus metrics says why.
llmtrace's traces show that the slow short requests all sat in engine steps
that also carried a 1536-token prefill chunk from a long prompt, and that
those steps took several times longer than the rest. The CUDA-event spans locate
the extra elapsed time within the step; they include launch gaps and do not
by themselves separate kernel execution from host stalls. `llmtrace plan` proposes
capping the per-step prefill of long prompts, `llmtrace run --plan` replays
the workload under each cap, and `llmtrace decide` reports that the 256-token
cap cuts short-request p95 by about two thirds while making the long
requests' first token up to twice as slow. Now you know the trade-off and
can pick a side.

That story is not hypothetical. It was run on four GPU models and two model
sizes, and every number is in [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md).
Two more bottlenecks, queue overload and KV-cache pressure, went through the
same loop on a 7B model with three and four independent repeats.

## Try it without a GPU

The synthetic engine has an invented cost model, so its numbers mean nothing
about hardware, but the whole pipeline runs on a laptop in a minute or two:

```bash
pip install -e ".[dev]"
llmtrace doctor                                   # what this machine can and cannot record
llmtrace workload template --output w.json        # short requests at 40/s plus long prompts every 0.2 s
llmtrace run --workload w.json --engine fake --out ./runs/base --repeat 2
llmtrace findings ./runs/base/r0 --verbose        # what the traces support, and what they cannot rule out
llmtrace plan ./runs/base/r0 --repeats 2 --json plan.json
llmtrace run --workload w.json --plan plan.json --engine fake --out ./exp
llmtrace decide --target "short ttft_p95 <= 20ms" --slo "short: ttft <= 20ms" \
    --config baseline=./exp/baseline/r0,./exp/baseline/r1 --config cap512=./exp/cap512/r0,./exp/cap512/r1
```

`findings` will tell you that short requests shared steps with long prefill
chunks, `plan` will propose two caps, and `decide` will show which cap meets
the target in every repeat and what it costs the long requests.

## On a real GPU

```bash
pip install -e ".[vllm]"          # vllm==0.11.0, transformers<5, nvidia-ml-py; Linux + NVIDIA
export VLLM_ENABLE_V1_MULTIPROCESSING=0
llmtrace run --workload w.json --engine vllm --model facebook/opt-125m --out ./runs/gpu --repeat 3
llmtrace doctor ./runs/gpu/r0
```

The environment variable keeps vLLM's engine core in the same process. That
is the only way llmtrace can see the scheduler, and the scheduler is where
the interesting evidence lives: batch membership, chunk sizes, queue and
prefill boundaries, and the CUDA-event span of each step. With the default
multiprocess core (and always with `AsyncLLM`) you still get request-level
traces, GPU telemetry and vLLM's own per-step stats, and `doctor` tells you
exactly which signals are missing and why.

Each real engine runs in its own spawned process, because a second vLLM
engine started in the same process fails on free GPU memory. A directory that
already holds a run is refused unless you pass `--overwrite`, so two runs
never get mixed into one.

## In your own code

```python
from vllm import LLM, SamplingParams
from llmtrace import LLMTracer

tracer = LLMTracer(output_dir="./traces", gpu_sample_interval_ms=100)
llm = LLM(model="facebook/opt-125m", disable_log_stats=False)
tracer.instrument_engine(llm.llm_engine)     # patches the engine, starts the collection threads

outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=32))

tracer.stop()                                # restores the engine, drains, flushes
print(tracer.health())                       # errors, drops, which signals were available
```

One thing to know about `LLM.generate()`: in vLLM 0.11.0 it forces every
request to emit a single output at completion, so nobody can observe the
first token through it. llmtrace records completion, token counts, batches
and energy under `generate()` and marks TTFT and TPOT as unavailable with
that reason. To time tokens, drive the engine with cumulative outputs:

```python
from llmtrace.vllm_helpers import run_engine_with_timing

outputs = run_engine_with_timing(llm.llm_engine, prompts, SamplingParams(max_tokens=32))
```

`AsyncLLM`, the engine behind the OpenAI-compatible server, works through
`tracer.instrument_async_engine(engine)` with request-level traces and vLLM's
per-step stats; see `examples/vllm_async_smoke_test.py`.

## How the diagnosis stays honest

Every finding names its evidence down to the file and field it came from,
lists what evidence is missing, states the assumptions the check rests on,
lists the competing explanations the recorded data cannot rule out, and says
where its confidence stops. A supported finding is a consistent pattern in
the events, never a root cause. The suggested experiment is the causal test,
and `plan` turns it into concrete, bounded configuration candidates: a cap
below the largest observed chunk, `max_num_seqs` doubled when the running
count hit it, memory utilization raised by a tenth under KV pressure. Not
every candidate helps, and that is fine: on the GPU runs, doubling the token
budget under queue overload changed nothing, and `decide` said so.

`decide` is equally careful about what counts. A repeat is an independent
run: the same directory twice, a copied directory, or one run shared between
configurations is reported as a duplicate and ignored. A configuration needs
at least `--min-repeats` eligible repeats (two by default, three or more
recommended) and a clean tracer health record to be a candidate. Missing or
unrecognized health records and missing/incomplete manifests remain exploratory:
metrics are shown, but those runs cannot qualify for recommendations. Before
ranking, manifests must agree on workload definition/hash, seed, model/revision,
engine/version and intended arrivals, and traces must agree on per-request prompt
and output lengths. A mismatch withholds ranking for the comparison. Actual
arrival delays and scheduler settings may differ. Both `<` and `<=` retain
their meaning in targets and SLOs. The run-to-run range across repeats is
reported next to a bootstrap interval over
requests, and the bootstrap is labelled as within-run, because requests in
one run share engine steps and are not independent draws. Goodput under
per-class SLOs, throughput and energy per token come alongside. It is
advisory and changes nothing on any server.

Every run manifest records the workload and its hash, the seed, the effective
engine configuration, the GPU, per-request scheduled versus actual arrival,
and which llmtrace code ran it: a content fingerprint of the source, the git
commit, whether the tree was dirty, and if so the diff and any untracked files
archived next to the manifest, with a flag saying whether that snapshot is
complete.

## What has actually been checked

Everything below was run on real vLLM 0.11.0 with committed evidence under
`docs/gpu_runs/`. The full table with every number is in
[docs/STATUS.md](docs/STATUS.md); the caveats are in
[docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md).

* **Instrumentation and restoration** of the real engine, batch metadata with
  real request ids, NVML telemetry, and an energy integral that matched a
  separately collected `nvidia-smi` stream within 0.15%.
* **Three of the five findings, end to end**: long-prompt interference on
  opt-125m and Qwen2.5-7B, queue overload on both, KV-cache pressure on both.
  Each one was induced, found, planned, replayed and decided. The other two
  (host overhead, the tracer's own observer effect) report not supported on
  every run so far, which is consistent but not a validation.
* **Independent repeats from a clean commit**: the 7B queue and KV loops ran
  with four and three repeats from a fingerprinted commit, with run-to-run
  spreads well below the effects.
* **The GPU span per step** is an upper bound on busy time, and Nsight
  Systems confirmed it never fell below the kernels' busy time on any of 1280
  steps. On a 125M model the span is about 58% busy on decode steps, so the
  bound is loose there and tight on prefill steps.
* **Overhead**: on Qwen2.5-7B the tracer adds 1.2% to `generate()` and 1.7%
  to the engine loop, with CUDA-event recording at 0.08 ms per step. On
  opt-125m, where a step is only a couple of milliseconds, it is 7.7% and
  14.4%.

Not checked: speculative decoding, `n > 1`, models above 7B, the OpenAI
server process itself, and more than four repeats per configuration. llmtrace
does not measure GPU busy time; it measures a span and says so.

## How it relates to what vLLM already gives you

vLLM 0.11.0 already exports per-request queue, prefill and decode times as
OpenTelemetry spans and Prometheus histograms, and `vllm bench serve` already
generates bursty arrivals and computes goodput. llmtrace uses vLLM's own
per-step stats where they exist. What vLLM does not export is which requests
shared an engine step and how many tokens each was scheduled, per-step GPU
time cheap enough to leave on, energy, or any link from those measurements to
a testable configuration change. That link is the part llmtrace adds, and
[docs/AUDIT.md](docs/AUDIT.md) spells out the overlap without flattering it.

## What it writes

A run directory holds raw data and nothing derived: per-request traces
(`traces_*.jsonl`), per-step batch records (`batches_*.jsonl`), GPU samples
(`gpu_*.jsonl`), CUDA-event step spans (`gpu_steps_*.jsonl`), vLLM's own
stats (`vllm_stats_*.jsonl`), the tracer's own drain timings
(`collector_*.jsonl`), the workload, and the manifest. Fields the driver
cannot report are `null`, never zero. `analyze`, `findings`, `visualize` and
`decide` read those files offline and need neither a GPU nor vLLM.
`visualize` writes an HTML report and a Perfetto trace you can scrub at
https://ui.perfetto.dev.

## A word on energy

Per-request energy on a shared GPU is an allocation, not a measurement, and
llmtrace never pretends otherwise. Device power is integrated per GPU by
trapezoid over its own timestamps; each elementary interval's energy is split
among the requests active in it by a stated policy (`equal_share`,
`proportional_tokens`, or `window_only` which allocates nothing); idle energy
and energy with insufficient telemetry are reported separately and never
assigned to anyone; and the ledger checks that attributed plus idle plus
unattributable equals the device total on every run. The formula and the edge
cases are in [DEVELOPMENT.md](DEVELOPMENT.md).

## Install and test

```bash
pip install -e .                 # offline analysis and the CLI, no GPU dependencies
pip install -e ".[nvml]"         # GPU telemetry
pip install -e ".[parquet]"      # parquet output
pip install -e ".[vllm]"         # vllm==0.11.0 (Linux, NVIDIA GPU)
pip install -e ".[dev]"          # pytest and ruff
python -m pytest                 # 252 tests, all CPU, against fakes shaped like the verified vLLM interfaces
```

The test suite proves llmtrace's own logic, not vLLM compatibility; that is
what the GPU evidence is for.

## More reading

* [QUICKSTART.md](QUICKSTART.md): the shortest path on CPU and on a GPU
* [docs/STATUS.md](docs/STATUS.md): the detailed validated / implemented / not implemented table
* [docs/gpu_runs/README.md](docs/gpu_runs/README.md): the evidence directories; raw trace files are release assets, summaries and manifests are in git
* [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md): every GPU session, its numbers and its caveats
* [docs/AUDIT.md](docs/AUDIT.md): architecture, risks, overlap with upstream tooling, roadmap
* [DEVELOPMENT.md](DEVELOPMENT.md): how the pieces work and which vLLM interfaces were verified from source
* [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md): what is and is not verified, in one list
* [experiments/mixed_prompts/README.md](experiments/mixed_prompts/README.md): the original interference experiment

## License

MIT
