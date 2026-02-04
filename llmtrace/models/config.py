"""Configuration models for llmtrace."""

from typing import List, Optional

from pydantic import BaseModel, Field


class GPUSamplerConfig(BaseModel):
    """Configuration for GPU sampling."""

    sample_interval_ms: int = Field(
        default=100, ge=10, le=10000, description="GPU sampling interval in milliseconds"
    )
    gpu_ids: Optional[List[int]] = Field(
        default=None, description="Specific GPU IDs to monitor (None = all)"
    )
    use_dcgm: bool = Field(
        default=False, description="Use DCGM instead of NVML (requires dcgm-exporter)"
    )
    collect_tensor_utilization: bool = Field(
        default=False, description="Collect tensor core utilization (may have overhead)"
    )


class EnergyConfig(BaseModel):
    """Configuration for energy attribution."""

    enabled: bool = Field(default=True, description="Enable energy attribution")
    attribution_method: str = Field(
        default="proportional_time",
        description="Attribution method: proportional_time, proportional_tokens, or exact",
    )
    energy_price_usd_per_kwh: Optional[float] = Field(
        default=None, ge=0, description="Energy price for cost calculation (USD per kWh)"
    )


class AutopsyConfig(BaseModel):
    """Configuration for autopsy/diagnosis engine."""

    enabled: bool = Field(default=True, description="Enable automatic diagnosis")

    # Thresholds for diagnosis rules
    queue_overload_threshold_ms: float = Field(
        default=100.0, description="Queue wait threshold for overload diagnosis"
    )
    throttle_severity_threshold: int = Field(
        default=3, description="Number of throttle samples to trigger throttling diagnosis"
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
        description="Prompt length variance threshold for fragmentation diagnosis",
    )


class TracerConfig(BaseModel):
    """Main configuration for LLMTracer."""

    output_dir: str = Field(default="./traces", description="Output directory for traces")
    output_format: str = Field(
        default="jsonl", description="Output format: jsonl or parquet"
    )

    # Component configs
    gpu_sampler: GPUSamplerConfig = Field(default_factory=GPUSamplerConfig)
    energy: EnergyConfig = Field(default_factory=EnergyConfig)
    autopsy: AutopsyConfig = Field(default_factory=AutopsyConfig)

    # Instrumentation
    enable_batch_metadata: bool = Field(
        default=True, description="Collect batch/scheduler metadata from vLLM"
    )
    enable_kv_cache_tracking: bool = Field(
        default=True, description="Track KV cache usage (may require vLLM patching)"
    )

    # Performance
    async_write: bool = Field(
        default=True, description="Write traces asynchronously to avoid blocking"
    )
    buffer_size: int = Field(
        default=1000, ge=1, description="Number of traces to buffer before flushing"
    )

    # Distributed
    distributed_mode: bool = Field(
        default=False, description="Enable distributed tracing for multi-GPU/multi-node"
    )
    rank: Optional[int] = Field(default=None, description="Rank ID in distributed setup")
    world_size: Optional[int] = Field(default=None, description="Total number of ranks")


class ReporterConfig(BaseModel):
    """Configuration for reporters."""

    cli_rich_output: bool = Field(default=True, description="Use rich formatting for CLI")
    export_formats: List[str] = Field(
        default_factory=lambda: ["jsonl"], description="Export formats: jsonl, parquet, otlp"
    )
    otlp_endpoint: Optional[str] = Field(
        default=None, description="OTLP endpoint for OpenTelemetry export"
    )
