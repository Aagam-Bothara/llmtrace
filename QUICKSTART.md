# Quickstart

Start with the CPU example to learn the workflow. Real inference needs Linux,
an NVIDIA GPU and vLLM 0.11.0. Run the commands from a local checkout.
Multi-line shell commands below use Bash syntax. In PowerShell, put each
command on one line instead of using `\` continuations.

## Try a run on CPU

```bash
pip install -e .
llmtrace doctor
llmtrace workload template --output w.json
llmtrace run --workload w.json --engine fake --out ./runs/base --repeat 2
llmtrace findings ./runs/base/r0 --verbose
llmtrace visualize ./runs/base/r0 --html-out report.html
```

Open `report.html` to explore the run. The fake engine uses invented timings
and power values; this is a demo, not a hardware benchmark.

## Test a configuration change

This example caps long-prompt processing at 256 tokens per step. It uses the
same workload and two fresh runs for each configuration.

```bash
llmtrace run --workload w.json --engine fake --out ./runs/capped --repeat 2 \
    --config-name capped --set long_prefill_token_threshold=256
llmtrace decide --target "short ttft_p95 <= 20ms" --slo "short: ttft <= 20ms" \
    --config baseline=./runs/base/r0,./runs/base/r1 \
    --config capped=./runs/capped/r0,./runs/capped/r1
```

The target asks whether short requests stay within 20 ms at the 95th
percentile in every repeat. Goodput is the share meeting the per-request SLO.
Read the long-request results too: helping short requests can slow long ones.

To try changes suggested by the findings:

```bash
llmtrace plan ./runs/base/r0 --repeats 2 --json plan.json
llmtrace run --workload w.json --plan plan.json --engine fake --out ./exp
```

Review the plan's candidate names. The runner prints a `decide` command with
the actual run paths; replace its target placeholder with your latency limit.

## Run on a GPU

```bash
pip install -e ".[vllm]"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
llmtrace run --workload w.json --engine vllm --model facebook/opt-125m \
    --out ./runs/gpu --repeat 3
llmtrace doctor ./runs/gpu/r0
```

The environment setting exposes scheduler steps. Without it, request-level
tracing and vLLM stats can still work, but batch membership, queue/prefill
boundaries and CUDA-event step spans are unavailable.

If NVML sees multiple GPUs, add `--gpu-id 0` or, for an engine using two
GPUs, `--gpu-id 0 --gpu-id 1`. Choose the **physical NVML indices** used by
your engine; CUDA logical indices may differ. Without an explicit selection
on a multi-GPU host, latency tracing continues without GPU telemetry.

The runner warms up the workload, records settling requests, then records
the measured replay. Use `--exclude-class settle` when comparing real runs.
Each real engine runs in a fresh process. Use a new output directory for each
experiment; `--overwrite` replaces a previous run's llmtrace files.

## Use it in your code

This example uses cumulative outputs so first-token timing is observable.
Enable the in-process core before starting Python if you need scheduler data.

```python
from vllm import LLM, SamplingParams
from llmtrace import LLMTracer, TracerConfig
from llmtrace.vllm_helpers import run_engine_with_timing

llm = LLM(model="facebook/opt-125m", disable_log_stats=False)
tracer = LLMTracer(TracerConfig(
    output_dir="./traces",
    gpu_sampler={"gpu_ids": [0]},  # physical NVML index used by this engine
))
tracer.instrument_engine(llm.llm_engine)
try:
    outputs = run_engine_with_timing(
        llm.llm_engine, ["Hello, world!"], SamplingParams(max_tokens=64)
    )
finally:
    tracer.stop()

print(tracer.health())
tracer.print_analysis(tracer.analyze())
```

`LLM.generate()` can also be traced, but vLLM 0.11.0 forces final-only outputs
there: completion and token counts remain available; TTFT and TPOT do not.
Sampling, collection and writing run in background threads.

For directly constructed `AsyncLLM` engines, use
`tracer.instrument_async_engine(engine)`. See the
[async example](examples/vllm_async_smoke_test.py). It records requests and
vLLM stats, not scheduler membership or CUDA-event spans.

## Read saved runs

Offline analysis needs neither a GPU nor vLLM.

```bash
llmtrace analyze ./runs/base/r0 --output report.json
llmtrace visualize ./runs/base/r0 --html-out report.html
llmtrace visualize ./runs/base/r0 --trace-out run.perfetto.json
```

Open the Perfetto file at [ui.perfetto.dev](https://ui.perfetto.dev).
Use `analyze --gpu-samples <file-or-directory>` for separately stored samples.
`analyze --attribution window_only` shows shared device energy without
allocating it to requests.

`analyze` and `decide` read GPU selection from the manifest. For older
multi-GPU traces, pass `--gpu-id` for every participating device. A missing
energy value includes a reason; it does not mean zero consumption.

## Check for regressions

```bash
llmtrace compare --baseline ./runs/base/r0 --current ./runs/capped/r0 \
    --ttft-threshold 5 --energy-threshold 10 --fail-on-regression
```

Only increases above the threshold count as regressions. Add
`--fail-on-missing` if missing metrics should fail the check too.

This check uses aggregate request metrics. Expect code 1 in this demo:
the cap improves short requests but slows long ones enough to raise overall
TTFT p95. Use the class-specific `decide` result to understand that trade-off.

| Exit code | Meaning |
|-----------|---------|
| 0 | Check passed, or reporting completed without a requested failure |
| 1 | Regression with `--fail-on-regression`, or missing metric with `--fail-on-missing` |
| 2 | Invalid arguments |
| 3 | Command not implemented (`monitor`) |

## Configure and test

```bash
llmtrace init-config --output my_config.json
pip install -e ".[dev]"
python -m pytest
```

Load the file with `LLMTracer.from_config_file("my_config.json")`. Unknown
configuration keys raise an error. Optional extras include `[nvml]` for GPU
telemetry and `[parquet]` for Parquet files.

See [status](docs/STATUS.md) for limitations and the
[development guide](DEVELOPMENT.md) for metric definitions and configuration details.
