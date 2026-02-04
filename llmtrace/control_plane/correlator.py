"""Correlator for aligning request traces with GPU telemetry and computing energy attribution."""

import logging
from typing import Dict, List, Optional, Tuple

from llmtrace.models.config import EnergyConfig
from llmtrace.models.trace import (
    RequestTrace,
    GPUSample,
    EnergyAttribution,
    SpanPhase,
)

logger = logging.getLogger(__name__)


class Correlator:
    """
    Correlates request traces with GPU telemetry and computes energy attribution.

    Energy attribution challenges:
    - Multiple requests may execute concurrently (batched)
    - GPU energy consumption is shared across all requests in a batch
    - Need to attribute proportionally with explicit approximations

    Attribution methods:
    - proportional_time: Energy proportional to request duration
    - proportional_tokens: Energy proportional to tokens processed
    - exact: Direct measurement (requires isolated execution, rarely possible)
    """

    def __init__(self, energy_config: EnergyConfig):
        self.config = energy_config

    async def correlate_traces(
        self,
        traces: List[RequestTrace],
        gpu_samples: List[GPUSample],
    ) -> List[RequestTrace]:
        """
        Correlate request traces with GPU samples and compute energy.

        Args:
            traces: List of RequestTrace objects
            gpu_samples: List of GPUSample objects

        Returns:
            List of RequestTrace objects with GPU samples and energy attribution
        """
        if not self.config.enabled:
            return traces

        logger.info(f"Correlating {len(traces)} traces with {len(gpu_samples)} GPU samples")

        # Sort samples by timestamp for efficient lookup
        sorted_samples = sorted(gpu_samples, key=lambda s: s.timestamp)

        # Process each trace
        correlated_traces = []
        for trace in traces:
            # Find GPU samples during request lifetime
            trace.gpu_samples = self._find_samples_in_window(
                sorted_samples, trace.start_time, trace.end_time
            )

            # Compute energy attribution
            if trace.gpu_samples:
                trace.energy = self._compute_energy_attribution(trace)

            correlated_traces.append(trace)

        logger.info(f"Correlation complete. {len(correlated_traces)} traces attributed.")
        return correlated_traces

    def _find_samples_in_window(
        self, sorted_samples: List[GPUSample], start_time: float, end_time: float
    ) -> List[GPUSample]:
        """
        Find GPU samples within a time window.

        Uses binary search for efficiency with sorted samples.
        """
        # Simple linear scan for now - could optimize with bisect
        samples = []
        for sample in sorted_samples:
            if start_time <= sample.timestamp <= end_time:
                samples.append(sample)
            elif sample.timestamp > end_time:
                break  # Since sorted, we're done

        return samples

    def _compute_energy_attribution(self, trace: RequestTrace) -> EnergyAttribution:
        """
        Compute energy attribution for a request.

        Energy = integral of power over time = sum(power_i * delta_t_i)

        For batched execution:
        - Total GPU energy during request is measured exactly
        - Attribution to individual request is approximate
        - We use proportional methods with explicit confidence
        """
        if not trace.gpu_samples:
            return EnergyAttribution(
                request_id=trace.request_id,
                total_joules=0.0,
                joules_per_token=0.0,
                attribution_method=self.config.attribution_method,
                is_approximate=True,
                confidence=0.0,
            )

        # Compute total energy consumed by GPU during request
        total_energy_joules = self._integrate_power(trace.gpu_samples)

        # Compute phase-specific energy
        queue_joules = self._compute_phase_energy(trace, SpanPhase.QUEUE)
        prefill_joules = self._compute_phase_energy(trace, SpanPhase.PREFILL)
        decode_joules = self._compute_phase_energy(trace, SpanPhase.DECODE)

        # Attribution adjustment based on method
        attribution_factor, confidence = self._compute_attribution_factor(trace)
        attributed_energy = total_energy_joules * attribution_factor

        # Normalize phase energies to match total
        phase_total = queue_joules + prefill_joules + decode_joules
        if phase_total > 0:
            scale = attributed_energy / phase_total
            queue_joules *= scale
            prefill_joules *= scale
            decode_joules *= scale
        else:
            # Fallback: distribute proportionally by time
            total_duration = trace.total_duration_ms
            if total_duration > 0:
                queue_joules = attributed_energy * (trace.queue_duration_ms / total_duration)
                prefill_joules = attributed_energy * (trace.prefill_duration_ms / total_duration)
                decode_joules = attributed_energy * (trace.decode_duration_ms / total_duration)

        # Cost calculation (optional)
        cost_usd = None
        if self.config.energy_price_usd_per_kwh is not None:
            kwh = attributed_energy / 3600.0 / 1000.0  # J -> kWh
            cost_usd = kwh * self.config.energy_price_usd_per_kwh

        # Joules per token
        joules_per_token = 0.0
        if trace.output_length > 0:
            joules_per_token = attributed_energy / trace.output_length

        return EnergyAttribution(
            request_id=trace.request_id,
            total_joules=attributed_energy,
            joules_per_token=joules_per_token,
            queue_joules=queue_joules,
            prefill_joules=prefill_joules,
            decode_joules=decode_joules,
            cost_usd=cost_usd,
            energy_price_usd_per_kwh=self.config.energy_price_usd_per_kwh,
            attribution_method=self.config.attribution_method,
            is_approximate=True,  # Always approximate in batched execution
            confidence=confidence,
        )

    def _integrate_power(self, samples: List[GPUSample]) -> float:
        """
        Integrate power over time to compute energy.

        Energy (J) = integral(Power (W) dt)

        Uses trapezoidal rule for integration.
        """
        if len(samples) < 2:
            return 0.0

        total_energy = 0.0

        for i in range(len(samples) - 1):
            # Time delta in seconds
            dt = samples[i + 1].timestamp - samples[i].timestamp

            # Average power across all GPUs
            avg_power_i = samples[i].power_draw_watts
            avg_power_i_plus_1 = samples[i + 1].power_draw_watts

            # Trapezoidal integration
            energy_segment = 0.5 * (avg_power_i + avg_power_i_plus_1) * dt
            total_energy += energy_segment

        return total_energy

    def _compute_phase_energy(self, trace: RequestTrace, phase: SpanPhase) -> float:
        """Compute energy for a specific request phase."""
        # Find spans for this phase
        phase_spans = [s for s in trace.spans if s.phase == phase]
        if not phase_spans:
            return 0.0

        # Find samples during phase spans
        phase_energy = 0.0
        for span in phase_spans:
            span_samples = self._find_samples_in_window(
                trace.gpu_samples, span.start_time, span.end_time
            )
            if span_samples:
                phase_energy += self._integrate_power(span_samples)

        return phase_energy

    def _compute_attribution_factor(self, trace: RequestTrace) -> Tuple[float, float]:
        """
        Compute attribution factor and confidence for energy attribution.

        Returns:
            (attribution_factor, confidence)
            - attribution_factor: multiplier for total energy (0-1)
            - confidence: confidence in attribution (0-1)

        In batched execution, we need to estimate what fraction of GPU energy
        should be attributed to this specific request.
        """
        method = self.config.attribution_method

        if method == "exact":
            # Exact measurement - only valid if request was executed in isolation
            # For now, we assume not isolated, so return with low confidence
            return 1.0, 0.5

        elif method == "proportional_tokens":
            # Attribute proportionally by tokens processed
            # Need to estimate total tokens across all concurrent requests
            # This is approximate - we don't have perfect visibility into concurrent work
            #
            # For MVP: assume single request or equal sharing (conservative)
            return 1.0, 0.7

        elif method == "proportional_time":
            # Attribute proportionally by request duration
            # Similar challenge - don't know total concurrent request time
            #
            # For MVP: assume single request or equal sharing
            return 1.0, 0.7

        else:
            logger.warning(f"Unknown attribution method: {method}, using proportional_time")
            return 1.0, 0.5

    async def aggregate_gpu_samples_multi_gpu(
        self, samples_by_gpu: Dict[int, List[GPUSample]]
    ) -> List[GPUSample]:
        """
        Aggregate GPU samples across multiple GPUs.

        For multi-GPU setups (tensor parallelism), we need to aggregate energy
        across all devices.

        Args:
            samples_by_gpu: Dict mapping gpu_id -> list of samples

        Returns:
            Aggregated samples (one per timestamp)
        """
        if not samples_by_gpu:
            return []

        # Group samples by timestamp
        timestamp_groups: Dict[float, List[GPUSample]] = {}

        for samples in samples_by_gpu.values():
            for sample in samples:
                if sample.timestamp not in timestamp_groups:
                    timestamp_groups[sample.timestamp] = []
                timestamp_groups[sample.timestamp].append(sample)

        # Aggregate each timestamp group
        aggregated = []
        for timestamp, group in sorted(timestamp_groups.items()):
            # Sum power across GPUs
            total_power = sum(s.power_draw_watts for s in group)

            # Average other metrics
            num_gpus = len(group)
            avg_util = sum(s.gpu_utilization_pct for s in group) / num_gpus
            avg_mem_util = sum(s.memory_utilization_pct for s in group) / num_gpus
            total_mem_used = sum(s.memory_used_mb for s in group)
            total_mem = sum(s.memory_total_mb for s in group)
            avg_temp = sum(s.temperature_c for s in group) / num_gpus

            # Aggregate throttle reasons (union)
            throttle_reasons = []
            for s in group:
                throttle_reasons.extend(s.throttle_reasons)
            unique_throttles = list(set(throttle_reasons))

            # Create aggregated sample
            agg_sample = GPUSample(
                timestamp=timestamp,
                gpu_id=-1,  # Indicates aggregated
                device_name=f"Aggregated ({num_gpus} GPUs)",
                gpu_utilization_pct=avg_util,
                memory_utilization_pct=avg_mem_util,
                memory_used_mb=total_mem_used,
                memory_total_mb=total_mem,
                power_draw_watts=total_power,  # Sum of all GPUs
                power_limit_watts=sum(s.power_limit_watts for s in group),
                temperature_c=avg_temp,
                sm_clock_mhz=int(sum(s.sm_clock_mhz for s in group) / num_gpus),
                memory_clock_mhz=int(sum(s.memory_clock_mhz for s in group) / num_gpus),
                throttle_reasons=unique_throttles,
            )

            aggregated.append(agg_sample)

        logger.info(f"Aggregated {len(aggregated)} samples across {len(samples_by_gpu)} GPUs")
        return aggregated
