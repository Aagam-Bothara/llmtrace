# llmtrace Quickstart

## 1. Without a GPU (synthetic)

```bash
pip install -e ".[dev]"
python -m pytest                      # CPU-only regression tests
python examples/synthetic_replay.py   # fake engine + fake NVML, writes ./traces_synthetic
llmtrace analyze ./traces_synthetic/current --baseline ./traces_synthetic/baseline
```

Everything the synthetic example prints is fabricated; it shows the pipeline
and file formats only.

## 2. With vLLM 0.11.0 on an NVIDIA GPU (Linux)

```bash
pip install -e ".[vllm]"
export VLLM_ENABLE_V1_MULTIPROCESSING=0   # optional; exposes the scheduler for batch metadata
python examples/vllm_smoke_test.py --model facebook/opt-125m --out ./traces_smoke
```

Then follow [docs/GPU_VALIDATION.md](docs/GPU_VALIDATION.md).

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

Only the synchronous `LLMEngine` used by `vllm.LLM` is supported. `AsyncLLM`
(the OpenAI server) is rejected with `InstrumentationError`.

## 3. Offline analysis

```bash
llmtrace analyze ./traces                          # directory: traces_*, gpu_*, batches_* files
llmtrace analyze ./traces/traces_x.jsonl --gpu-samples ./traces/gpu_x.jsonl
llmtrace analyze ./traces --attribution window_only   # no per-request allocation
llmtrace analyze ./traces --output report.json
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
  `null` with `ttft_unavailable_reason` for zero-token, FINAL_ONLY or pooling
  requests.
* `tpot_ms`: (last token step end - first token step end) / (tokens after the
  first observation). `null` when fewer than two token observations exist.
* Spans `queue` and `prefill` exist only when the scheduler was in-process;
  otherwise one `time_to_first_token` span is recorded and no boundary is
  inferred.
* Energy fields: `window_device_joules` is shared device energy during the
  request; `attributed_joules` is the allocated share under `allocation_policy`
  and `membership_source`; `is_allocated=false` plus `unavailable_reason`
  means no per-request figure exists.
