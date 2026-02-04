"""Core data models for traces and telemetry."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SpanPhase(str, Enum):
    """Request lifecycle phases."""

    QUEUE = "queue"
    PREFILL = "prefill"
    DECODE = "decode"


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
    """A span within a request lifecycle."""

    phase: SpanPhase
    start_time: float  # Unix timestamp in seconds
    end_time: float
    duration_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.duration_ms / 1000.0


class BatchMetadata(BaseModel):
    """Metadata about a vLLM batch."""

    batch_id: str
    timestamp: float
    num_requests: int
    num_prefill: int
    num_decode: int
    total_tokens: int
    prompt_lengths: List[int] = Field(default_factory=list)
    kv_cache_usage_bytes: Optional[int] = None
    kv_cache_capacity_bytes: Optional[int] = None

    @property
    def kv_cache_utilization(self) -> Optional[float]:
        """KV cache utilization as a fraction [0, 1]."""
        if self.kv_cache_usage_bytes is not None and self.kv_cache_capacity_bytes is not None:
            return self.kv_cache_usage_bytes / max(self.kv_cache_capacity_bytes, 1)
        return None


class GPUSample(BaseModel):
    """A single GPU telemetry sample."""

    timestamp: float  # Unix timestamp in seconds
    gpu_id: int
    device_name: str

    # Utilization
    gpu_utilization_pct: float  # 0-100
    memory_utilization_pct: float  # 0-100
    memory_used_mb: float
    memory_total_mb: float

    # Power and thermal
    power_draw_watts: float
    power_limit_watts: float
    temperature_c: float

    # Clocks
    sm_clock_mhz: int
    memory_clock_mhz: int

    # Throttling
    throttle_reasons: List[ThrottleReason] = Field(default_factory=list)

    # Optional: tensor core utilization if available
    tensor_utilization_pct: Optional[float] = None

    @property
    def is_throttled(self) -> bool:
        """Check if GPU is being throttled."""
        return len(self.throttle_reasons) > 0 and ThrottleReason.NONE not in self.throttle_reasons


class EnergyAttribution(BaseModel):
    """Energy attribution for a request."""

    request_id: str

    # Total energy
    total_joules: float
    joules_per_token: float

    # Phase breakdown (approximate if batched)
    queue_joules: float = 0.0
    prefill_joules: float = 0.0
    decode_joules: float = 0.0

    # Cost (optional)
    cost_usd: Optional[float] = None
    energy_price_usd_per_kwh: Optional[float] = None

    # Attribution method metadata
    attribution_method: str = "proportional_time"  # or "proportional_tokens", "exact"
    is_approximate: bool = True
    confidence: float = 1.0  # 0-1 confidence in attribution


class DiagnosisEvidence(BaseModel):
    """Evidence supporting a diagnosis."""

    metric: str
    value: float
    threshold: float
    comparison: str  # "exceeds", "below", etc.
    severity: float  # 0-1, how much it exceeds threshold


class DiagnosisResult(BaseModel):
    """Diagnosis result for a request or run."""

    request_id: Optional[str] = None
    category: DiagnosisCategory
    confidence: float  # 0-1
    evidence: List[DiagnosisEvidence] = Field(default_factory=list)
    description: str
    mitigation: Optional[str] = None  # Suggested fix

    class Config:
        use_enum_values = True


class RequestTrace(BaseModel):
    """Complete trace for a single request."""

    request_id: str
    start_time: float
    end_time: float

    # Request metadata
    prompt_length: int
    output_length: int
    model_name: str

    # Lifecycle spans
    spans: List[RequestSpan] = Field(default_factory=list)

    # Batch context (which batches did this request participate in?)
    batch_ids: List[str] = Field(default_factory=list)

    # Correlated GPU samples (samples during request lifetime)
    gpu_samples: List[GPUSample] = Field(default_factory=list)

    # Attribution
    energy: Optional[EnergyAttribution] = None

    # Diagnosis
    diagnosis: Optional[DiagnosisResult] = None

    # Metrics
    ttft_ms: Optional[float] = None  # Time to first token
    tpot_ms: Optional[float] = None  # Time per output token (average)

    @property
    def total_duration_ms(self) -> float:
        """Total request duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def queue_duration_ms(self) -> float:
        """Queue wait duration in milliseconds."""
        queue_spans = [s for s in self.spans if s.phase == SpanPhase.QUEUE]
        return sum(s.duration_ms for s in queue_spans)

    @property
    def prefill_duration_ms(self) -> float:
        """Prefill duration in milliseconds."""
        prefill_spans = [s for s in self.spans if s.phase == SpanPhase.PREFILL]
        return sum(s.duration_ms for s in prefill_spans)

    @property
    def decode_duration_ms(self) -> float:
        """Decode duration in milliseconds."""
        decode_spans = [s for s in self.spans if s.phase == SpanPhase.DECODE]
        return sum(s.duration_ms for s in decode_spans)


class TraceAnalysis(BaseModel):
    """Analysis results for a collection of traces."""

    num_requests: int
    start_time: float
    end_time: float
    duration_s: float

    # Aggregate metrics
    avg_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float

    avg_tpot_ms: float
    p50_tpot_ms: float
    p95_tpot_ms: float
    p99_tpot_ms: float

    # Energy
    total_joules: float
    avg_joules_per_request: float
    avg_joules_per_token: float

    # Diagnoses
    diagnoses: List[DiagnosisResult] = Field(default_factory=list)
    top_issues: List[DiagnosisCategory] = Field(default_factory=list)

    # Regressions (if baseline provided)
    regressions: Dict[str, float] = Field(default_factory=dict)  # metric -> % change

    # GPU stats
    avg_gpu_utilization_pct: float = 0.0
    avg_power_draw_watts: float = 0.0
    throttle_incidents: int = 0

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Trace Analysis Summary",
            f"=" * 50,
            f"Requests: {self.num_requests}",
            f"Duration: {self.duration_s:.2f}s",
            f"",
            f"Latency:",
            f"  TTFT: avg={self.avg_ttft_ms:.2f}ms p95={self.p95_ttft_ms:.2f}ms p99={self.p99_ttft_ms:.2f}ms",
            f"  TPOT: avg={self.avg_tpot_ms:.2f}ms p95={self.p95_tpot_ms:.2f}ms p99={self.p99_tpot_ms:.2f}ms",
            f"",
            f"Energy:",
            f"  Total: {self.total_joules:.2f}J",
            f"  Per request: {self.avg_joules_per_request:.2f}J",
            f"  Per token: {self.avg_joules_per_token:.4f}J",
            f"",
            f"GPU:",
            f"  Avg utilization: {self.avg_gpu_utilization_pct:.1f}%",
            f"  Avg power: {self.avg_power_draw_watts:.1f}W",
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
            lines.append("Regressions vs Baseline:")
            for metric, pct_change in self.regressions.items():
                sign = "+" if pct_change > 0 else ""
                lines.append(f"  - {metric}: {sign}{pct_change:.2f}%")

        return "\n".join(lines)
