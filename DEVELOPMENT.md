# Development Guide

## Architecture Overview

llmtrace follows a clean separation between **data plane** (collection) and **control plane** (analysis):

```
┌─────────────────────────────────────────────────────────────┐
│                        User Code                            │
│  (vLLM inference with llmtrace instrumentation)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    DATA PLANE                               │
│                                                             │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ vLLM             │  │ GPU Sampler  │  │ Trace Writer │ │
│  │ Instrumentation  │  │ (NVML/DCGM)  │  │(JSONL/Parquet│ │
│  └──────────────────┘  └──────────────┘  └──────────────┘ │
│         │                      │                 │         │
│         └──────────────────────┼─────────────────┘         │
│                                │                           │
└────────────────────────────────┼───────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────┐
│                   CONTROL PLANE                            │
│                                                            │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Correlator │  │ Rules Engine │  │ Reporter         │  │
│  │ (Energy)   │  │ (Autopsy)    │  │ (CLI/Exports)    │  │
│  └────────────┘  └──────────────┘  └──────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Core Components

### Data Plane

#### 1. VLLMInstrumentation (`data_plane/vllm_instrumentation.py`)

**Purpose**: Hooks into vLLM's LLMEngine to capture request lifecycle events.

**Key Methods**:
- `instrument_engine(engine)`: Patches vLLM methods to inject tracing
- `_wrap_add_request()`: Captures request arrival (queue entry)
- `_wrap_step()`: Captures execution steps (prefill, decode)
- `_wrap_schedule()`: Captures batch formation

**Design Notes**:
- Uses method patching (monkey patching) with care - stores original methods
- Minimal overhead via async event collection
- Thread-safe with asyncio locks

#### 2. GPUSampler (`data_plane/gpu_sampler.py`)

**Purpose**: Continuously samples GPU telemetry using NVML.

**Metrics Collected**:
- GPU/memory utilization
- Power draw
- Clocks (SM, memory)
- Temperature
- Throttling reasons

**Design Notes**:
- Async sampling loop runs in background
- Configurable interval (default 100ms)
- Low overhead (<1% CPU)

#### 3. TraceWriter (`data_plane/trace_writer.py`)

**Purpose**: Persists traces to disk in JSONL or Parquet format.

**Features**:
- Async writing (non-blocking)
- Buffering for efficiency
- Automatic file rotation by session

### Control Plane

#### 4. Correlator (`control_plane/correlator.py`)

**Purpose**: Aligns request traces with GPU samples and computes energy attribution.

**Energy Attribution**:
- `_integrate_power()`: Trapezoidal integration of power over time
- `_compute_phase_energy()`: Energy breakdown by phase (queue/prefill/decode)
- `_compute_attribution_factor()`: Handles batched execution (approximate)

**Key Insight**:
In batched execution, multiple requests share GPU resources. Energy attribution is **approximate** and we make this explicit via confidence scores.

#### 5. RulesEngine (`control_plane/rules_engine.py`)

**Purpose**: Diagnoses performance issues via rules-based analysis.

**Diagnosis Categories**:
- `QUEUEING_OVERLOAD`: Queue wait exceeds threshold
- `GPU_THROTTLING`: GPU was throttled during execution
- `MEMORY_PRESSURE`: High memory utilization
- `HOST_BOTTLENECK`: Low GPU util but high latency (CPU bound)
- `COLD_PATH`: First-run overhead
- `BATCH_FRAGMENTATION`: Mixed prompt lengths hurt efficiency

**Design**:
- Each rule returns `DiagnosisResult` with confidence and evidence
- Best diagnosis (highest confidence) is selected
- Rules can evolve to ML-based detection

#### 6. Reporter (`control_plane/reporter.py`)

**Purpose**: Generates analysis reports and CLI output.

**Features**:
- Rich CLI output (using `rich` library)
- Regression detection vs baseline
- Export to various formats

### Utilities

#### 7. BatchAnalyzer (`utils/batch_analyzer.py`)

**Feature 1: Batch/Scheduler Visibility**

Analyzes batch metadata to provide insights into vLLM's internal batching:
- Batch size distribution
- Prefill vs decode ratios
- Prompt length variance
- KV cache pressure

#### 8. LatencyExplainer (`utils/latency_explainer.py`)

**Feature 2: Tail Latency Explainer**

Explains tail latency with ranked causes:
- Identifies tail requests (P95/P99)
- Ranks root causes
- Provides mitigation suggestions

## vLLM Integration Points

### Where We Hook

1. **LLMEngine.add_request**: Request enters queue
2. **LLMEngine.step**: Batch execution (prefill/decode)
3. **Scheduler.schedule**: Batch formation

### Version Compatibility

This implementation targets vLLM >= 0.3.0. vLLM's internal API evolves rapidly, so:
- Keep instrumentation defensive (try/except)
- Log warnings when vLLM structures change
- Update hooks when vLLM releases major versions

## Energy Attribution Details

### The Challenge

In batched execution, GPU processes multiple requests concurrently. How do we attribute energy to individual requests?

### Our Approach

We use **proportional attribution with explicit confidence**:

```python
# Total GPU energy during request (measured exactly)
total_energy = integrate_power(gpu_samples)

# Attribution factor (approximate)
attribution_factor, confidence = compute_attribution_factor(trace)

# Attributed energy (approximate)
attributed_energy = total_energy * attribution_factor
```

**Attribution Methods**:
1. `proportional_time`: Energy ∝ request duration
2. `proportional_tokens`: Energy ∝ tokens processed
3. `exact`: Direct measurement (requires isolation, rare)

**Why Approximate**:
- Can't perfectly isolate requests in batched execution
- GPU work is shared across batch members
- We make approximations explicit via `is_approximate` and `confidence` fields

## Diagnosis Rules

### Rule: Queueing Overload

**Symptom**: Queue duration > threshold

**Evidence**: Queue wait time

**Confidence**: Higher for longer waits

**Mitigation**: Reduce request rate, increase batch size, add capacity

### Rule: GPU Throttling

**Symptom**: Multiple throttled GPU samples during request

**Evidence**: Throttle reason counts

**Confidence**: Higher for more throttle incidents

**Mitigation**: Check cooling, power limits, thermals

### Rule: Memory Pressure

**Symptom**: High memory utilization

**Evidence**: Avg memory util > threshold

**Confidence**: Higher for higher utilization

**Mitigation**: Reduce batch size, enable PagedAttention

### Rule: Host Bottleneck

**Symptom**: Low GPU util but high latency

**Evidence**: GPU util < threshold, total time > threshold

**Confidence**: Moderate (can have false positives)

**Mitigation**: Profile CPU, optimize tokenization

### Rule: Batch Fragmentation

**Symptom**: High prompt length variance in batch

**Evidence**: Coefficient of variation

**Confidence**: Moderate to high

**Mitigation**: Bucket requests by length

## Performance Considerations

### Overhead Targets

- GPU sampling: <1% CPU
- vLLM instrumentation: <1% latency increase
- Trace writing: Non-blocking via async I/O

### Memory Management

- Periodic flushing of samples/traces to disk
- Configurable buffer sizes
- Clear old samples after correlation

## Testing Strategy

### Unit Tests

Test individual components in isolation:
- GPU sampler with mock NVML
- Correlator with synthetic traces
- Rules engine with known patterns

### Integration Tests

Test full pipeline:
- vLLM instrumentation with real engine
- End-to-end trace collection and analysis

### Regression Tests

CI integration:
- Baseline vs current comparison
- Automated regression detection

## Extending llmtrace

### Adding New Diagnosis Rules

1. Add new `DiagnosisCategory` to `models/trace.py`
2. Implement rule method in `RulesEngine`
3. Update `diagnose_request()` to call new rule
4. Add tests

### Supporting New Metrics

1. Update `GPUSample` model if GPU metric
2. Update `RequestTrace` model if request metric
3. Update sampling/instrumentation code
4. Update correlation/analysis logic

### Adding Export Formats

1. Update `TraceWriter` to support new format
2. Update `Reporter` to export in new format
3. Update CLI to accept new format option

## Distributed Tracing

For multi-GPU setups (tensor/pipeline parallelism):

1. **Rank Identification**: Each GPU worker needs unique rank
2. **Sample Aggregation**: Aggregate GPU samples across ranks
3. **Energy Summation**: Sum energy across all devices
4. **Trace Coordination**: Central coordinator collects from all ranks

Current implementation has basic support via `Correlator.aggregate_gpu_samples_multi_gpu()`.

## Future Enhancements

### Near-term
- DCGM support for enterprise deployments
- Tensor core utilization tracking
- More sophisticated attribution (account for batch composition)
- ML-based anomaly detection

### Long-term
- Web UI for trace exploration
- Real-time monitoring dashboard
- Distributed tracing coordinator
- Integration with OTLP/OpenTelemetry ecosystems

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit PR with description

## Code Style

- Black formatting (line length 100)
- Ruff linting
- Type hints (mypy)
- Docstrings for public APIs
