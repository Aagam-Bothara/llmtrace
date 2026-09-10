# llmtrace

**Find out what slowed down your vLLM requests, test a change, and compare the results.**

llmtrace records requests, scheduler steps and GPU telemetry. It helps you
see where requests waited, what work they shared, and whether a configuration
change helped across repeated runs.

Use it for research and testing before production. It targets **vLLM 0.11.0**.
Full scheduler tracing needs the engine core in the same process:
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. With the default multiprocess core or
`AsyncLLM`, fewer signals are available. `llmtrace doctor` explains which ones.

## Try it without a GPU

From a local checkout:

```bash
pip install -e .
llmtrace workload template --output w.json
llmtrace run --workload w.json --engine fake --out ./runs/base --repeat 2
llmtrace findings ./runs/base/r0 --verbose
llmtrace visualize ./runs/base/r0 --html-out report.html
```

The fake engine demonstrates the workflow. Its timings and power values are
invented, so they tell you nothing about real hardware.

Follow the [quickstart](QUICKSTART.md) to test a configuration change or run
on a GPU.

## How it works

1. **Record:** `run` replays a workload and saves traces, settings and health records.
2. **Investigate:** `findings` shows evidence for possible bottlenecks.
3. **Test:** `plan` suggests changes; `run --plan` tries them in fresh engines.
4. **Compare:** `decide` checks your latency target across independent repeats.

For example, a bursty Qwen2.5-7B workload was waiting behind a limit on
concurrent requests. Doubling that limit reduced p95 time to first token
from **6.4 s to 2.2 s**, with **47% lower energy per output token**, across
four repeats. These are results for that workload, not a general speedup
claim. See the [recorded GPU results](docs/GPU_VALIDATION.md).

## Reading a result

| Term | Meaning |
|------|---------|
| TTFT | Time from submitting a request to observing its first output token, including queue wait |
| TPOT | Average time per output token after the first observation |
| p95 | The latency at the 95th percentile; a way to track slower requests |
| Goodput | Share of selected requests that meet every latency limit in their SLO |
| SLO | The latency limits you set for a request class |
| GPU span | Elapsed time between CUDA events around a step; includes gaps between kernel launches |
| J/token | An estimated share of device energy per output token |

A finding names its evidence, assumptions and missing data. A supported
finding is a hypothesis to test, not proof of a cause.

`decide` requires clean tracer health, compatible manifests and at least two
eligible repeats by default. Three or more repeats give a better view of
variation. It checks the workload, model, seed, intended arrivals and
per-request token counts before ranking. Missing evidence stays exploratory;
it cannot qualify a run for a recommendation. `<` and `<=` keep their meaning.

The report separates variation between runs from a bootstrap interval over
requests. Requests sharing engine steps are related, so that interval is not
a substitute for more runs. Recommendations are advisory; they do not change
a running server.

## Timing and energy limits

`LLM.generate()` in vLLM 0.11.0 returns final outputs only, so llmtrace cannot
observe TTFT or TPOT through it. Use the runner or the timing helper in the
[quickstart](QUICKSTART.md#use-it-in-your-code). Engine timings are measured
at step boundaries.

CUDA-event spans include launch gaps. They do not directly measure GPU busy
time or separate kernel execution from host stalls.

On a multi-GPU host, select every physical GPU used by the engine with
`--gpu-id`, repeated for each device. These are **NVML indices**, not remapped
CUDA indices. The manifest records the selection and physical UUIDs.

Energy comes from sampled device power, then a stated allocation policy
splits it among requests. Other processes on a selected GPU can affect the
reading. `decide` withholds J/token when coverage or allocations are
insufficient, and explains why. A lone power sample never means zero energy.
See [the energy method](DEVELOPMENT.md#energy-ledger).

## What has been checked

GPU sessions cover opt-125m and Qwen2.5-7B on four GPU models. They include
long-prompt interference, queue overload, KV-cache pressure, tracing overhead
and a comparison of CUDA spans with Nsight Systems.

Models above 7B, speculative decoding, multiple outputs per request and the
OpenAI-compatible server process itself have not been validated. Distributed
tracing and attaching to an existing process are not implemented.

CPU tests check llmtrace's logic. They do not establish compatibility with a
new vLLM version. See [status and limitations](docs/STATUS.md) for the details.

## Documentation

| Read this | For |
|-----------|-----|
| [Quickstart](QUICKSTART.md) | Install, record a run and compare changes |
| [Status](docs/STATUS.md) | What works, what was tested and what is missing |
| [GPU results](docs/GPU_VALIDATION.md) | Recorded measurements and their limits |
| [Raw evidence](docs/gpu_runs/README.md) | Download traces and verify checksums |
| [Development guide](DEVELOPMENT.md) | Architecture, methods and contributing |
| [Design review](docs/AUDIT.md) | Integration risks and next steps |
| [Implementation summary](IMPLEMENTATION_SUMMARY.md) | A short technical overview and migration notes |
| [Mixed-prompt experiment](experiments/mixed_prompts/README.md) | A worked example of finding and testing a bottleneck |

## License

MIT
