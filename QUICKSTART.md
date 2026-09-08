# llmtrace Quickstart

## 1. Without a GPU (synthetic)

```bash
pip install -e ".[dev]"
python -m pytest                      # CPU-only regression tests
llmtrace doctor                       # what this environment can and cannot record
llmtrace workload template --output w.json
llmtrace run --workload w.json --engine fake --out ./runs/base --repeat 2
llmtrace run --workload w.json --engine fake --out ./runs/capped --repeat 2 --config-name capped --set long_prefill_token_threshold=256
llmtrace findings ./runs/base/r0 --verbose
llmtrace plan ./runs/base/r0 --repeats 2 --json plan.json          # experiments derived from the supported findings
llmtrace run --workload w.json --plan plan.json --engine fake --out ./exp
llmtrace decide --target "short ttft_p95 <= 20ms" --slo "short: ttft <= 20ms" \
    --config baseline=./exp/baseline/r0,./exp/baseline/r1 --config cap512=./exp/cap512/r0,./exp/cap512/r1
python examples/synthetic_replay.py   # fake engine + fake NVML, writes ./traces_synthetic
llmtrace analyze ./traces_synthetic/current --baseline ./traces_synthetic/baseline
```

Everything the synthetic engine produces is fabricated (an invented cost
model); it shows the pipeline, the file formats and the decision logic only.

## 2. With vLLM 0.11.0 on an NVIDIA GPU (Linux)

```bash
pip install -e ".[vllm]"
export VLLM_ENABLE_V1_MULTIPROCESSING=0   # optional; exposes the scheduler for batch metadata
python examples/vllm_smoke_test.py --model facebook/opt-125m --out ./traces_smoke
```

Then follow [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md). The same
workload spec runs on the real engine (in-process core for batch metadata and
GPU spans; untraced warm-up replay first, then a traced settle phase, then
the measured replay with `ignore_eos`):

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 llmtrace run --workload w.json --engine vllm --model facebook/opt-125m --out ./runs/gpu_base --repeat 3
llmtrace doctor ./runs/gpu_base/r0
```

Each real engine runs in its own spawned process (a second vLLM engine in one
process fails on free GPU memory); `--in-process` disables that for a single
run. The runner was validated against the experiment driver on an RTX A5000
(see `docs/GPU_VALIDATION.md`).

In your own code:

```python
from vllm import LLM, SamplingParams
from llmtrace import LLMTracer

tracer = LLMTracer(output_dir="./traces")
llm = LLM(model="facebook/opt-125m")
tracer.instrument_engine(llm.llm_engine)
outputs = llm.generate(prompts, SamplingParams(max_tokens=64))
tracer.stop()
analysis = tracer.analyze()
tracer.print_analysis(analysis)
```

`LLMTracer` is synchronous. Its GPU sampler, collector and writer are
background threads, so they keep running while `llm.generate()` blocks.

`llm.generate()` forces FINAL_ONLY outputs in vLLM 0.11.0, so it yields
completion, token and energy data but no TTFT/TPOT. For timing, drive the
engine with cumulative outputs:

```python
from llmtrace.vllm_helpers import run_engine_with_timing
outputs = run_engine_with_timing(llm.llm_engine, prompts, SamplingParams(max_tokens=64))
```

`AsyncLLM` (the OpenAI-server engine) is supported through
`tracer.instrument_async_engine(engine)` with request-level traces and vLLM's
per-step stats only; see `examples/vllm_async_smoke_test.py`.

## 3. Offline analysis

```bash
llmtrace analyze ./traces                          # directory: traces_*, gpu_*, batches_* files
llmtrace analyze ./traces/traces_x.jsonl --gpu-samples ./traces/gpu_x.jsonl
llmtrace analyze ./traces --attribution window_only   # no per-request allocation
llmtrace analyze ./traces --output report.json
```

## 3b. Visualize a run

```bash
llmtrace visualize ./traces --html-out report.html            # self-contained HTML report
llmtrace visualize ./traces --trace-out run.perfetto.json      # open at https://ui.perfetto.dev
llmtrace visualize ./exp/baseline_0 --compare ./exp/capped_0 --html-out compare.html
```

## 4. Regression gate

```bash
llmtrace compare --baseline ./baseline --current ./current \
    --ttft-threshold 5 --energy-threshold 10 --fail-on-regression
```

Exit codes: 0 ok, 1 regression (or missing metric with `--fail-on-missing`),
2 usage error, 3 not implemented (`monitor`).

Only positive changes count as regressions. A run that is 50% faster passes.

## 5. Configuration

```bash
llmtrace init-config --output my_config.json
```

Unknown keys are rejected (no silently ignored options). Load with
`LLMTracer.from_config_file("my_config.json")`. Convenience keyword arguments
on `LLMTracer(...)`: `output_dir`, `gpu_sample_interval_ms`,
`enable_energy_attribution`, `attribution_method`, `energy_price_usd_per_kwh`,
plus any top-level `TracerConfig` field.

## 6. Reading the numbers

* `ttft_ms`: arrival at `add_request` to the end of the engine step in which
  the first output token became visible. Includes queue wait. Step-granular.
  `null` with `ttft_unavailable_reason` for zero-token, FINAL_ONLY (which is
  what `LLM.generate()` uses) or pooling requests.
* `tpot_ms`: (last token step end - first token step end) / (tokens after the
  first observation). `null` when fewer than two token observations exist.
* Spans `queue` and `prefill` exist only when the scheduler was in-process;
  otherwise one `time_to_first_token` span is recorded and no boundary is
  inferred.
* Energy fields: `window_device_joules` is shared device energy during the
  request; `attributed_joules` is the allocated share under `allocation_policy`
  and `membership_source`; `is_allocated=false` plus `unavailable_reason`
  means no per-request figure exists.
