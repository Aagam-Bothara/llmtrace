# llmtrace Quickstart

Get started with llmtrace in 5 minutes.

## Installation

```bash
pip install llmtrace

# If you want Parquet support
pip install llmtrace[parquet]

# If you want DCGM support (enterprise)
pip install llmtrace[dcgm]
```

## Basic Usage

### 1. Instrument Your vLLM Server

```python
from vllm import LLM
from llmtrace import LLMTracer

# Initialize tracer
tracer = LLMTracer(
    output_dir="./traces",
    gpu_sample_interval_ms=100,
)

# Initialize vLLM
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Instrument the engine
tracer.instrument_engine(llm.llm_engine)

# Your inference as usual
outputs = llm.generate(prompts)

# Stop and analyze
await tracer.stop()
analysis = await tracer.analyze()
tracer.print_analysis(analysis)
```

### 2. CLI Analysis

Analyze existing traces:

```bash
# Analyze traces from a directory
llmtrace analyze ./traces

# Compare against baseline (CI regression check)
llmtrace compare --baseline ./traces/baseline --current ./traces/current \
    --ttft-threshold 5% --energy-threshold 10% --fail-on-regression
```

### 3. Feature Demos

#### Feature 1: Batch/Scheduler Visibility

```python
from llmtrace.utils.batch_analyzer import BatchAnalyzer

analyzer = BatchAnalyzer()
analyzer.print_batch_summary(batches)
```

**What you get**:
- Batch size distribution
- Prefill vs decode ratios
- Prompt length variance
- KV cache pressure indicators

#### Feature 2: Tail Latency Explainer

```python
from llmtrace.utils.latency_explainer import LatencyExplainer

explainer = LatencyExplainer()
explanation = explainer.explain_request(trace)
explainer.print_explanation(explanation)
```

**What you get**:
- Root cause identification (queueing, throttling, etc.)
- Confidence scores
- Evidence with thresholds
- Mitigation suggestions

#### Feature 3: Energy Regression Guardrail

```bash
# In your CI pipeline
llmtrace compare \
    --baseline ./baseline_traces \
    --current ./current_traces \
    --ttft-threshold 5 \
    --energy-threshold 10 \
    --fail-on-regression
```

**What you get**:
- Automated regression detection
- P95 TTFT tracking
- J/token energy tracking
- CI exit codes for failures

## Configuration

Generate a config file:

```bash
llmtrace init-config --output my_config.json
```

Edit the config:

```json
{
  "output_dir": "./traces",
  "gpu_sampler": {
    "sample_interval_ms": 100
  },
  "energy": {
    "enabled": true,
    "energy_price_usd_per_kwh": 0.12
  },
  "autopsy": {
    "enabled": true,
    "queue_overload_threshold_ms": 100.0
  }
}
```

Use the config:

```python
tracer = LLMTracer.from_config_file("my_config.json")
```

## Multi-GPU / Distributed

For tensor parallelism setups:

```python
tracer = LLMTracer(
    distributed_mode=True,
    rank=torch.distributed.get_rank(),
    world_size=torch.distributed.get_world_size(),
)
```

llmtrace automatically:
- Aggregates GPU metrics across ranks
- Sums energy across devices
- Coordinates trace collection

## Output Files

llmtrace generates these files:

```
./traces/
├── traces_20260204_153045.jsonl      # Request traces
├── batches_20260204_153045.jsonl     # Batch metadata
└── gpu_20260204_153045.jsonl         # GPU samples
```

Each file is append-only JSONL for easy parsing and streaming.

## Analysis Workflow

```
1. Collect traces
   └─> llmtrace instruments vLLM + samples GPU

2. Correlate
   └─> Align request windows with GPU telemetry

3. Attribute
   └─> Compute J/request, J/token with phase breakdown

4. Diagnose
   └─> Rules engine identifies issues with evidence

5. Report
   └─> CLI output, exports, regression detection
```

## Common Use Cases

### Use Case 1: Debug Slow Request

```bash
llmtrace analyze ./traces
# Look at "Top Diagnosed Issues"
# Check individual request explanations
```

### Use Case 2: Optimize Energy

```python
analysis = await tracer.analyze()
print(f"Avg J/token: {analysis.avg_joules_per_token:.4f}")

# Optimize your config, re-run
# Compare energy metrics
```

### Use Case 3: CI Performance Gate

```yaml
# .github/workflows/perf.yml
- name: Run benchmark
  run: python benchmark.py --output ./current_traces

- name: Check regressions
  run: |
    llmtrace compare \
      --baseline ./baseline_traces \
      --current ./current_traces \
      --fail-on-regression
```

## Next Steps

- Check out [examples/](examples/) for complete examples
- Read [DEVELOPMENT.md](DEVELOPMENT.md) for architecture details
- Open issues for bugs or feature requests

## FAQ

**Q: Does llmtrace work with other LLM frameworks?**

A: Currently vLLM-specific, but the architecture is extensible. GPU sampling and analysis components are framework-agnostic.

**Q: What's the performance overhead?**

A: <1% in typical setups. GPU sampling is async and low-frequency. vLLM instrumentation is lightweight.

**Q: Can I use this in production?**

A: Yes! Async writing and buffering minimize impact. You can also run in sampling mode (trace subset of requests).

**Q: Is energy attribution exact?**

A: No - it's approximate in batched execution. We make this explicit via `is_approximate` and `confidence` fields. See DEVELOPMENT.md for details.

**Q: How do I get help?**

A: Open an issue on GitHub or check the documentation.
