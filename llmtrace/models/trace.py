"""Core data models for traces and telemetry.

Clock conventions
-----------------
Every timed record carries two clocks:

* ``*_time`` / ``timestamp`` fields are wall-clock seconds (``time.time()``).
  They are metadata: human readable, comparable across processes, but not
  guaranteed monotonic.
* ``*_monotonic`` fields are seconds from ``time.monotonic()``. They are only
  comparable within one process, identified by ``clock_domain``. All durations
  (``duration_ms``, ``ttft_ms``, ``tpot_ms``) are computed from the monotonic
  clock when it is available.

Unavailable measurements are represented as ``None`` plus, where useful, a
``*_unavailable_reason`` string. They are never substituted with zero.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SpanPhase(str, Enum):
    """Request lifecycle phases.

    ``QUEUE`` and ``PREFILL`` are only emitted when the scheduler is visible to
    the instrumentation (in-process engine core). Otherwise the engine does not
    expose the queue/prefill boundary and a single ``TIME_TO_FIRST_TOKEN`` span
    covering arrival -> first visible output token is emitted instead.
    """

    QUEUE = "queue"
    PREFILL = "prefill"
    TIME_TO_FIRST_TOKEN = "time_to_first_token"
    DECODE = "decode"


class RequestStatus(str, Enum):
    """Terminal status of a traced request."""

    COMPLETED = "completed"  # engine reported finished=True
    ABORTED = "aborted"  # engine abort_request() or finish_reason == "abort"
    INCOMPLETE = "incomplete"  # tracer stopped / uninstrumented while active


class ThrottleReason(str, Enum):
    """GPU throttling reasons from NVML."""

    NONE = "none"
    GPU_IDLE = "gpu_idle"
    APPLICATIONS_CLOCKS = "applications_clocks"
    SW_POWER_CAP = "sw_power_cap"
    HW_SLOWDOWN = "hw_slowdown"
    SYNC_BOOST = "sync_boost"
    SW_THERMAL = "sw_thermal"
    HW_THERMAL = "hw_thermal"
    HW_POWER_BRAKE = "hw_power_brake"
    DISPLAY_CLOCK = "display_clock"
    UNKNOWN = "unknown"  # NVML query failed; throttle state not observed


class DiagnosisCategory(str, Enum):
    """Root cause categories for tail latency."""

    QUEUEING_OVERLOAD = "queueing_overload"
    BATCH_FRAGMENTATION = "batch_fragmentation"
    GPU_THROTTLING = "gpu_throttling"
    MEMORY_PRESSURE = "memory_pressure"
    HOST_BOTTLENECK = "host_bottleneck"
    COLD_PATH = "cold_path"
    UNKNOWN = "unknown"


class RequestSpan(BaseModel):
    """A span within a request lifecycle.

    Span boundaries observed through ``LLMEngine.step()`` are only known at
    step granularity; ``metadata["granularity"] == "engine_step"`` marks this.
    """

    phase: SpanPhase
    start_time: float  # wall clock, seconds
    end_time: float  # wall clock, seconds
    duration_ms: float  # from monotonic clock when available
    start_monotonic: Optional[float] = None
    end_monotonic: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0


class BatchMetadata(BaseModel):
    """Metadata about one scheduler decision (one engine step).

    Only available when the vLLM scheduler runs in-process
    (``VLLM_ENABLE_V1_MULTIPROCESSING=0``). ``request_ids`` are the real engine
    request ids scheduled in this step; ``batch_id`` is assigned by llmtrace
    (vLLM's ``SchedulerOutput`` carries no identifier).
    """

    batch_id: str
    step_index: int
    timestamp: float  # wall clock at schedule() return
    monotonic: Optional[float] = None  # monotonic clock at schedule() return
    step_start_monotonic: Optional[float] = None
    step_end_monotonic: Optional[float] = None
    num_requests: int
    num_prefill: int
    num_decode: int
    total_scheduled_tokens: int
    request_ids: List[str] = Field(default_factory=list)
    scheduled_tokens: Dict[str, int] = Field(default_factory=dict)
    prompt_lengths: List[int] = Field(default_factory=list)  # newly scheduled requests only
    kv_cache_usage_fraction: Optional[float] = None  # 0..1, from KVCacheManager.usage
    num_running: Optional[int] = None
    num_waiting: Optional[int] = None
    source: str = "unknown"  # e.g. "vllm_v1_in_process_scheduler", "synthetic"

    @property
    def kv_cache_utilization(self) -> Optional[float]:
        return self.kv_cache_usage_fraction


class GPUSample(BaseModel):
    """A single GPU telemetry sample. Fields the driver could not report are ``None``."""

    timestamp: float  # wall clock, seconds
    monotonic: Optional[float] = None
    clock_domain: Optional[str] = None
    gpu_id: int
    device_name: str

    gpu_utilization_pct: Optional[float] = None  # 0-100
    memory_utilization_pct: Optional[float] = None  # 0-100
    memory_used_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None

    power_draw_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None
    temperature_c: Optional[float] = None

    sm_clock_mhz: Optional[int] = None
    memory_clock_mhz: Optional[int] = None

    throttle_reasons: List[ThrottleReason] = Field(default_factory=list)

    @property
    def is_throttled(self) -> bool:
        """True only when a throttle reason other than none/unknown was observed."""
        return any(
            r not in (ThrottleReason.NONE, ThrottleReason.UNKNOWN) for r in self.throttle_reasons
        )


class EnergyCoverage(BaseModel):
    """How well telemetry covers a time window."""

    window_s: float
    covered_s: float  # window time bracketed by consecutive samples within max_gap
    coverage_fraction: float  # covered_s / window_s (0 when window_s == 0)
    num_samples: int  # samples with a power reading inside the window (all GPUs)
    gpu_ids: List[int] = Field(default_factory=list)
    max_gap_s: Optional[float] = None
    clock: str = "wall"  # "monotonic" or "wall"


class EnergyAttribution(BaseModel):
    """Energy accounting for one request.

    Three distinct quantities are kept apart:

    * telemetry: the raw ``GPUSample`` power readings (measured);
    * ``window_device_joules``: power integrated over the request window for
      every monitored GPU (an integration estimate of what the devices consumed
      while the request was alive, shared with everything else running);
    * ``attributed_joules``: the share allocated to this request under
      ``allocation_policy`` (an allocation estimate, never a measurement).
    """

    request_id: str

    window_device_joules: Optional[float] = None
    attributed_joules: Optional[float] = None
    joules_per_output_token: Optional[float] = None

    queue_joules: Optional[float] = None
    prefill_joules: Optional[float] = None
    time_to_first_token_joules: Optional[float] = None
    decode_joules: Optional[float] = None

    cost_usd: Optional[float] = None
    energy_price_usd_per_kwh: Optional[float] = None

    allocation_policy: str = "equal_share"
    membership_source: str = "unavailable"
    is_allocated: bool = False
    is_estimate: bool = True
    unavailable_reason: Optional[str] = None
    coverage: Optional[EnergyCoverage] = None

    @property
    def total_joules(self) -> Optional[float]:
        """Alias for ``attributed_joules`` kept for readability in reports."""
        return self.attributed_joules


class RunEnergyLedger(BaseModel):
    """Run-level conservation ledger.

    ``device_joules == attributed_joules + idle_joules + unattributable_joules``
    within ``conservation_error_joules``. Energy outside the run window
    (before the first request arrives / after the last completes) is not
    counted anywhere. Idle energy is device energy inside the run window during
    which no traced request was active; it is reported, not attributed.
    """

    window_start: float  # in the ledger clock
    window_end: float
    clock: str = "wall"
    device_joules: Optional[float] = None
    per_gpu_joules: Dict[int, float] = Field(default_factory=dict)
    attributed_joules: float = 0.0
    idle_joules: float = 0.0
    unattributable_joules: float = 0.0
    conservation_error_joules: Optional[float] = None
    allocation_policy: str = "equal_share"
    membership_source: str = "unavailable"
    num_requests: int = 0
    num_requests_allocated: int = 0
    num_requests_without_telemetry: int = 0
    coverage: Optional[EnergyCoverage] = None
    samples_without_power: int = 0
    notes: List[str] = Field(default_factory=list)


class DiagnosisEvidence(BaseModel):
    metric: str
    value: float
    threshold: float
    comparison: str  # "exceeds", "below"
    severity: float  # 0-1


class DiagnosisResult(BaseModel):
    """Diagnosis result for a request or run.

    ``score`` is a rule-derived ranking value in [0, 1], not a probability.
    """

    request_id: Optional[str] = None
    category: DiagnosisCategory
    score: float
    evidence: List[DiagnosisEvidence] = Field(default_factory=list)
    description: str
    mitigation: Optional[str] = None


class RequestTrace(BaseModel):
    """Complete trace for a single request."""

    request_id: str
    start_time: float  # wall clock, arrival at LLMEngine.add_request
    end_time: float  # wall clock, last observed step end (or abort/stop time)
    start_monotonic: Optional[float] = None
    end_monotonic: Optional[float] = None
    clock_domain: Optional[str] = None

    status: RequestStatus = RequestStatus.COMPLETED
    finish_reason: Optional[str] = None  # vLLM finish_reason string when available

    # Token counts. prompt_length is None until verified token ids are seen.
    prompt_length: Optional[int] = None
    prompt_length_source: str = "unavailable"  # engine_prompt_token_ids | caller_token_ids | unavailable
    output_length: int = 0  # tokens across all sampled sequences
    num_sequences: int = 1
    output_kind: str = "unknown"  # cumulative | delta | final_only | pooling | unknown
    model_name: str = "unknown"

    spans: List[RequestSpan] = Field(default_factory=list)
    batch_ids: List[str] = Field(default_factory=list)
    scheduler_visible: bool = False

    gpu_samples: List[GPUSample] = Field(default_factory=list)
    energy: Optional[EnergyAttribution] = None
    diagnosis: Optional[DiagnosisResult] = None

    # Latency metrics; None with a reason when not measurable.
    ttft_ms: Optional[float] = None
    ttft_unavailable_reason: Optional[str] = None
    tpot_ms: Optional[float] = None
    tpot_unavailable_reason: Optional[str] = None
    first_token_monotonic: Optional[float] = None
    tokens_at_first_observation: int = 0

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def total_duration_ms(self) -> float:
        if self.start_monotonic is not None and self.end_monotonic is not None:
            return (self.end_monotonic - self.start_monotonic) * 1000.0
        return (self.end_time - self.start_time) * 1000.0

    def _phase_ms(self, phase: SpanPhase) -> float:
        return sum(s.duration_ms for s in self.spans if s.phase == phase)

    @property
    def queue_duration_ms(self) -> float:
        return self._phase_ms(SpanPhase.QUEUE)

    @property
    def prefill_duration_ms(self) -> float:
        return self._phase_ms(SpanPhase.PREFILL)

    @property
    def time_to_first_token_span_ms(self) -> float:
        return self._phase_ms(SpanPhase.TIME_TO_FIRST_TOKEN)

    @property
    def decode_duration_ms(self) -> float:
        return self._phase_ms(SpanPhase.DECODE)


class MetricComparison(BaseModel):
    """Baseline vs current comparison for one metric.

    Sign convention: ``pct_change`` is ``(current - baseline) / baseline * 100``.
    For every metric compared here (latency, energy per token, throttling)
    higher is worse, so a positive value is a potential regression and a
    negative value is an improvement.
    """

    metric: str
    baseline: Optional[float] = None
    current: Optional[float] = None
    pct_change: Optional[float] = None
    status: str = "ok"  # ok | missing_baseline | missing_current | zero_baseline
    higher_is_worse: bool = True


class TraceAnalysis(BaseModel):
    """Analysis results for a collection of traces."""

    num_requests: int
    start_time: float
    end_time: float
    duration_s: float

    num_with_ttft: int = 0
    avg_ttft_ms: Optional[float] = None
    p50_ttft_ms: Optional[float] = None
    p95_ttft_ms: Optional[float] = None
    p99_ttft_ms: Optional[float] = None

    num_with_tpot: int = 0
    avg_tpot_ms: Optional[float] = None
    p50_tpot_ms: Optional[float] = None
    p95_tpot_ms: Optional[float] = None
    p99_tpot_ms: Optional[float] = None

    # Energy: device totals come from the ledger, per-request numbers from allocations.
    energy_ledger: Optional[RunEnergyLedger] = None
    num_with_energy: int = 0
    total_device_joules: Optional[float] = None
    attributed_joules: Optional[float] = None
    unallocated_joules: Optional[float] = None
    avg_attributed_joules_per_request: Optional[float] = None
    avg_joules_per_output_token: Optional[float] = None

    diagnoses: List[DiagnosisResult] = Field(default_factory=list)
    top_issues: List[DiagnosisCategory] = Field(default_factory=list)

    regressions: Dict[str, MetricComparison] = Field(default_factory=dict)

    num_gpu_samples: int = 0
    avg_gpu_utilization_pct: Optional[float] = None
    avg_power_draw_watts: Optional[float] = None
    throttle_incidents: int = 0

    status_counts: Dict[str, int] = Field(default_factory=dict)

    def summary(self) -> str:
        def fmt(v: Optional[float], unit: str = "", digits: int = 2) -> str:
            return "n/a" if v is None else f"{v:.{digits}f}{unit}"

        lines = [
            "Trace Analysis Summary",
            "=" * 50,
            f"Requests: {self.num_requests} {self.status_counts}",
            f"Duration: {self.duration_s:.2f}s",
            "",
            f"Latency (TTFT measured on {self.num_with_ttft}, TPOT on {self.num_with_tpot} requests):",
            f"  TTFT: avg={fmt(self.avg_ttft_ms, 'ms')} p95={fmt(self.p95_ttft_ms, 'ms')} "
            f"p99={fmt(self.p99_ttft_ms, 'ms')}",
            f"  TPOT: avg={fmt(self.avg_tpot_ms, 'ms')} p95={fmt(self.p95_tpot_ms, 'ms')} "
            f"p99={fmt(self.p99_tpot_ms, 'ms')}",
            "",
            "Energy (integration estimates from sampled power; allocations are estimates):",
            f"  Device energy in run window: {fmt(self.total_device_joules, 'J')}",
            f"  Attributed to requests: {fmt(self.attributed_joules, 'J')}",
            f"  Unallocated (idle/unattributable): {fmt(self.unallocated_joules, 'J')}",
            f"  Avg attributed per request: {fmt(self.avg_attributed_joules_per_request, 'J')}",
            f"  Avg per output token: {fmt(self.avg_joules_per_output_token, 'J', 4)}",
            "",
            f"GPU ({self.num_gpu_samples} samples):",
            f"  Avg utilization: {fmt(self.avg_gpu_utilization_pct, '%', 1)}",
            f"  Avg power: {fmt(self.avg_power_draw_watts, 'W', 1)}",
            f"  Throttle incidents: {self.throttle_incidents}",
        ]

        if self.top_issues:
            lines.append("")
            lines.append("Top Issues:")
            for issue in self.top_issues[:5]:
                count = sum(1 for d in self.diagnoses if d.category == issue)
                lines.append(f"  - {issue.value}: {count} occurrences")

        if self.regressions:
            lines.append("")
            lines.append("Comparison vs Baseline (positive = worse):")
            for metric, cmp in self.regressions.items():
                if cmp.pct_change is None:
                    lines.append(f"  - {metric}: {cmp.status}")
                else:
                    lines.append(f"  - {metric}: {cmp.pct_change:+.2f}%")

        return "\n".join(lines)
