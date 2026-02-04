# llmtrace

**Flight recorder, attribution, and autopsy for vLLM inference.**

llmtrace is a vLLM-native observability tool that answers the questions you can't answer today:
- Why is this request slow? (with ranked causes + evidence)
- How much energy does each request actually cost?
- What's vLLM doing internally? (batching, scheduling, KV cache pressure)
- Did my optimization regress performance or energy?

## Core Pillars

### Pillar A: Flight Recorder (Truth)
Per-request lifecycle spans from inside vLLM:
- Queue wait, prefill, decode phases
- Scheduler/batch metadata
- GPU telemetry aligned in time: power, clocks, utilization, throttling

### Pillar B: Attribution (Blame)
Per-request energy and cost attribution:
- J/request, J/token
- Cost/request (with optional $/token from energy price)
- Energy breakdown across queue/prefill/decode

### Pillar C: Autopsy (Answers)
Automated diagnosis:
- For a bad request: "why" with ranked causes + evidence
- For a run: top regressions, anomalies, inefficiency flags

## vLLM-Specific Features

### 1. Batch/Scheduler Visibility
See what vLLM is doing internally:
- Batch size over time (prefill vs decode)
- Prompt length distribution per batch
- KV cache pressure proxy
- Queue depth snapshots

### 2. Tail Latency Explainer
Automatic diagnosis with mechanism attribution:
- Queueing overload
- Batch fragmentation
- GPU downclocking/throttling
- Memory pressure
- Host bottlenecks
- Cold path issues

### 3. Energy Regression Guardrail
CI integration for serving performance:
- Baseline comparison
- Automatic regression detection (TTFT, J/token, throttling)
- Fail CI on regressions

## Architecture

### Data Plane
- **vLLM instrumentation plugin**: Hooks into engine lifecycle
- **GPU sampler**: NVML/DCGM polling for GPU telemetry
- **Trace writer**: Append-only JSONL/Parquet output

### Control Plane
- **Correlator**: Aligns request windows with GPU samples
- **Rules engine**: Diagnoses issues with evidence thresholds
- **Reporters**: CLI summaries, notebook exports, OTLP

## Installation

```bash
pip install llmtrace

# With DCGM support for enterprise deployments
pip install llmtrace[dcgm]
```

## Quick Start

### Basic Usage

```python
import asyncio
from llmtrace import LLMTracer
from vllm import LLM

async def main():
    # Initialize tracer
    tracer = LLMTracer(
        output_dir="./traces",
        gpu_sample_interval_ms=100,
        enable_energy_attribution=True,
    )

    # Instrument your vLLM engine
    llm = LLM(model="meta-llama/Llama-2-7b-hf")
    tracer.instrument_engine(llm.llm_engine)

    # Your inference as usual
    outputs = llm.generate("Hello, world!")

    # Stop tracing and analyze
    await tracer.stop()
    analysis = await tracer.analyze()
    tracer.print_analysis(analysis)

asyncio.run(main())
```

### What You Get

**Console Output:**
```
llmtrace Analysis Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overview
┌─────────────┬───────────┐
│ Metric      │ Value     │
├─────────────┼───────────┤
│ Requests    │ 4         │
│ Duration    │ 2.34s     │
│ Throughput  │ 1.71 req/s│
└─────────────┴───────────┘

Latency Metrics
┌───────────┬──────────┬──────────┬──────────┬──────────┐
│ Metric    │ Avg      │ P50      │ P95      │ P99      │
├───────────┼──────────┼──────────┼──────────┼──────────┤
│ TTFT (ms) │ 45.23    │ 43.12    │ 52.34    │ 53.21    │
│ TPOT (ms) │ 12.45    │ 11.23    │ 15.67    │ 16.12    │
└───────────┴──────────┴──────────┴──────────┴──────────┘

Energy & Efficiency
┌──────────────────────┬────────────┐
│ Metric               │ Value      │
├──────────────────────┼────────────┤
│ Total Energy         │ 456.78 J   │
│ Avg per Request      │ 114.20 J   │
│ Avg per Token        │ 2.2840 J   │
│ Avg GPU Utilization  │ 87.3%      │
│ Avg Power Draw       │ 245.6 W    │
│ Throttle Incidents   │ 0          │
└──────────────────────┴────────────┘
```

**Trace Files:**
```
./traces/
├── traces_20260204_153045.jsonl      # Request lifecycle data
├── batches_20260204_153045.jsonl     # Batch metadata
└── gpu_20260204_153045.jsonl         # GPU telemetry
```

See [QUICKSTART.md](QUICKSTART.md) for detailed getting started guide.

## CLI Usage

```bash
# Live monitoring
llmtrace monitor --pid <vllm_process_pid>

# Analyze traces
llmtrace analyze traces/run_*.jsonl

# CI regression check
llmtrace compare --baseline traces/baseline.jsonl --current traces/current.jsonl \
    --fail-on-regression --ttft-threshold 5% --energy-threshold 10%
```

## Distributed/Multi-GPU Support

llmtrace automatically detects tensor parallelism and pipeline parallelism:
- Aggregates GPU metrics across all ranks
- Attributes energy proportionally by device
- Correlates across distributed workers

## Examples

Check out [examples/](examples/) for complete working examples:

- **[basic_usage.py](examples/basic_usage.py)**: Basic integration with vLLM
- **[batch_analysis.py](examples/batch_analysis.py)**: Feature 1 - Batch/Scheduler visibility
- **[latency_diagnosis.py](examples/latency_diagnosis.py)**: Feature 2 - Tail latency explainer
- **[ci_regression.py](examples/ci_regression.py)**: Feature 3 - Energy regression guardrail for CI

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: Get started in 5 minutes
- **[DEVELOPMENT.md](DEVELOPMENT.md)**: Architecture details and contribution guide
- **[examples/](examples/)**: Complete working examples

## Why llmtrace?

### The Problem

LLM serving is a black box:
- "Why is this request slow?" → No visibility into vLLM internals
- "How much does this cost in energy?" → No per-request attribution
- "Did my optimization help?" → No regression detection

### The Solution

llmtrace provides:
1. **Truth**: Per-request lifecycle traces from inside vLLM + GPU telemetry
2. **Blame**: Energy attribution (J/request, J/token) with phase breakdown
3. **Answers**: Automated diagnosis with ranked causes and evidence

### Key Differentiators

- **vLLM-native**: Not a generic tracer - built specifically for vLLM internals
- **Energy-focused**: First-class energy attribution and cost tracking
- **Actionable**: Diagnoses point to mechanisms (not just symptoms) with mitigations
- **Production-ready**: <1% overhead, async I/O, distributed support
- **CI-integrated**: Regression guardrails for your performance pipeline

## Performance

**Overhead Targets:**
- GPU sampling: <1% CPU
- vLLM instrumentation: <1% latency increase
- Trace writing: Non-blocking async I/O

Measured in production workloads.

## Roadmap

**v0.2 (Near-term)**
- [ ] DCGM support for enterprise deployments
- [ ] Improved multi-GPU attribution
- [ ] Streaming analysis mode
- [ ] Web UI for trace exploration

**v0.3 (Long-term)**
- [ ] ML-based anomaly detection
- [ ] Real-time monitoring dashboard
- [ ] OTLP/OpenTelemetry integration
- [ ] Support for other LLM frameworks

## Contributing

Contributions welcome! See [DEVELOPMENT.md](DEVELOPMENT.md) for architecture details.

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit PR

## Citation

If you use llmtrace in your research, please cite:

```bibtex
@software{llmtrace2026,
  title={llmtrace: Flight Recorder, Attribution, and Autopsy for vLLM Inference},
  author={llmtrace contributors},
  year={2026},
  url={https://github.com/yourusername/llmtrace}
}
```

## License

MIT
