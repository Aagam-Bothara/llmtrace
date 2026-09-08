"""Configuration models for llmtrace.

All models forbid unknown fields so that a misspelled or unsupported option
fails loudly instead of being silently ignored.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GPUSamplerConfig(_StrictModel):
    """Configuration for GPU sampling (NVML)."""

    sample_interval_ms: int = Field(
        default=100, ge=10, le=10000, description="GPU sampling interval in milliseconds"
    )
    gpu_ids: Optional[List[int]] = Field(
        default=None, description="Specific GPU indices to monitor (None = all)"
    )
    max_buffered_samples: int = Field(
        default=100_000,
        ge=100,
        description="Samples kept in memory before the oldest are dropped (drops are counted)",
    )
    require_gpu: bool = Field(
        default=False,
        description="Fail start() if NVML cannot be initialised. Default: continue without telemetry.",
    )


AttributionMethod = Literal["equal_share", "proportional_tokens", "window_only"]


class EnergyConfig(_StrictModel):
    """Configuration for energy integration and allocation."""

    enabled: bool = Field(default=True, description="Enable energy accounting")
    attribution_method: AttributionMethod = Field(
        default="equal_share",
        description=(
            "Allocation policy for device energy while several requests are active: "
            "equal_share (split each interval equally among active requests), "
            "proportional_tokens (split by prompt+output tokens of active requests), "
            "window_only (no allocation; only report device energy over each request window)"
        ),
    )
    max_sample_gap_s: float = Field(
        default=1.0,
        gt=0,
        description="Consecutive samples further apart than this are treated as a telemetry gap "
        "and no energy is integrated across it",
    )
    min_coverage_fraction: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Request windows with less telemetry coverage than this get no energy figure",
    )
    energy_price_usd_per_kwh: Optional[float] = Field(
        default=None, ge=0, description="Energy price for cost calculation (USD per kWh)"
    )


class AutopsyConfig(_StrictModel):
    """Configuration for the rules-based diagnosis engine."""

    enabled: bool = Field(default=True, description="Enable automatic diagnosis")
    queue_overload_threshold_ms: float = Field(
        default=100.0, description="Queue wait threshold for overload diagnosis"
    )
    throttle_severity_threshold: int = Field(
        default=3, description="Number of throttled samples to trigger throttling diagnosis"
    )
    memory_pressure_threshold_pct: float = Field(
        default=90.0, ge=0, le=100, description="Memory utilization % for pressure diagnosis"
    )
    low_gpu_util_threshold_pct: float = Field(
        default=50.0, ge=0, le=100, description="GPU utilization % for host bottleneck diagnosis"
    )
    batch_fragmentation_threshold: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Prompt length coefficient of variation for fragmentation diagnosis",
    )


class ReporterConfig(_StrictModel):
    """Configuration for reporters."""

    cli_rich_output: bool = Field(default=True, description="Use rich formatting when available")


class TracerConfig(_StrictModel):
    """Main configuration for LLMTracer."""

    output_dir: str = Field(default="./traces", description="Output directory for traces")
    output_format: Literal["jsonl", "parquet"] = Field(
        default="jsonl", description="Output format: jsonl or parquet (parquet needs pyarrow)"
    )

    gpu_sampler: GPUSamplerConfig = Field(default_factory=GPUSamplerConfig)
    energy: EnergyConfig = Field(default_factory=EnergyConfig)
    autopsy: AutopsyConfig = Field(default_factory=AutopsyConfig)
    reporter: ReporterConfig = Field(default_factory=ReporterConfig)

    enable_batch_metadata: bool = Field(
        default=True,
        description="Record scheduler batches when the vLLM scheduler is reachable in-process",
    )

    collection_interval_s: float = Field(
        default=1.0, gt=0, description="How often the collector thread drains buffers to disk"
    )
    max_buffered_events: int = Field(
        default=10_000,
        ge=10,
        description="Completed traces / batches kept in memory before the oldest are dropped",
    )
    background_writes: bool = Field(
        default=True, description="Write files from a background thread (bounded queue)"
    )
    max_write_queue: int = Field(
        default=10_000, ge=10, description="Bounded write queue size for background writes"
    )
    strict_instrumentation: bool = Field(
        default=False,
        description="Re-raise instrumentation bookkeeping errors instead of counting them. "
        "Never affects the engine's own return values or exceptions.",
    )
    write_incomplete_requests: bool = Field(
        default=True,
        description="On stop, write still-active requests with status=incomplete",
    )
