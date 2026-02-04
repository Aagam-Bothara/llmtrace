"""Rules engine for diagnosing performance issues (Autopsy)."""

import logging
from typing import List, Optional
import statistics

from llmtrace.models.config import AutopsyConfig
from llmtrace.models.trace import (
    RequestTrace,
    DiagnosisResult,
    DiagnosisCategory,
    DiagnosisEvidence,
    ThrottleReason,
)

logger = logging.getLogger(__name__)


class RulesEngine:
    """
    Rules-based diagnosis engine for identifying performance issues.

    Implements "Pillar C: Autopsy" - answers "why is this request slow?"

    Diagnosis categories:
    1. Queueing overload: Request spent too long in queue
    2. Batch fragmentation: Inefficient batching due to prompt length variance
    3. GPU throttling: GPU was throttled during execution
    4. Memory pressure: High memory utilization, KV cache pressure
    5. Host bottleneck: GPU underutilized but latency high (CPU bound)
    6. Cold path: First-run overhead (JIT, loading, etc.)
    """

    def __init__(self, config: AutopsyConfig):
        self.config = config

    async def diagnose_request(self, trace: RequestTrace) -> Optional[DiagnosisResult]:
        """
        Diagnose a single request trace.

        Returns the highest-confidence diagnosis, or None if no issues detected.
        """
        if not self.config.enabled:
            return None

        # Run all diagnosis rules
        diagnoses = []

        # Rule 1: Queueing overload
        if diag := self._check_queueing_overload(trace):
            diagnoses.append(diag)

        # Rule 2: GPU throttling
        if diag := self._check_throttling(trace):
            diagnoses.append(diag)

        # Rule 3: Memory pressure
        if diag := self._check_memory_pressure(trace):
            diagnoses.append(diag)

        # Rule 4: Host bottleneck
        if diag := self._check_host_bottleneck(trace):
            diagnoses.append(diag)

        # Rule 5: Cold path
        if diag := self._check_cold_path(trace):
            diagnoses.append(diag)

        # Return highest confidence diagnosis
        if diagnoses:
            best = max(diagnoses, key=lambda d: d.confidence)
            return best

        return None

    async def diagnose_batch(self, traces: List[RequestTrace]) -> List[DiagnosisResult]:
        """
        Diagnose a batch of traces.

        Returns list of high-level issues affecting the batch (e.g., batch fragmentation).
        """
        if not self.config.enabled or not traces:
            return []

        diagnoses = []

        # Batch-level rule: Batch fragmentation
        if diag := self._check_batch_fragmentation(traces):
            diagnoses.append(diag)

        return diagnoses

    def _check_queueing_overload(self, trace: RequestTrace) -> Optional[DiagnosisResult]:
        """Check if request suffered from queue overload."""
        queue_time_ms = trace.queue_duration_ms

        if queue_time_ms < self.config.queue_overload_threshold_ms:
            return None

        # Calculate severity
        severity = min(
            queue_time_ms / self.config.queue_overload_threshold_ms - 1.0,
            1.0
        )

        evidence = [
            DiagnosisEvidence(
                metric="queue_wait_ms",
                value=queue_time_ms,
                threshold=self.config.queue_overload_threshold_ms,
                comparison="exceeds",
                severity=severity,
            )
        ]

        # Calculate confidence
        confidence = min(0.5 + severity * 0.5, 0.95)

        return DiagnosisResult(
            request_id=trace.request_id,
            category=DiagnosisCategory.QUEUEING_OVERLOAD,
            confidence=confidence,
            evidence=evidence,
            description=f"Request spent {queue_time_ms:.1f}ms in queue, "
            f"exceeding threshold of {self.config.queue_overload_threshold_ms:.1f}ms. "
            f"System is likely overloaded.",
            mitigation="Reduce request rate, increase batch size, or add more GPU capacity.",
        )

    def _check_throttling(self, trace: RequestTrace) -> Optional[DiagnosisResult]:
        """Check if GPU was throttled during request."""
        if not trace.gpu_samples:
            return None

        # Count throttled samples
        throttled_samples = [s for s in trace.gpu_samples if s.is_throttled]

        if len(throttled_samples) < self.config.throttle_severity_threshold:
            return None

        # Identify primary throttle reason
        throttle_counts = {}
        for sample in throttled_samples:
            for reason in sample.throttle_reasons:
                if reason != ThrottleReason.NONE and reason != ThrottleReason.GPU_IDLE:
                    throttle_counts[reason] = throttle_counts.get(reason, 0) + 1

        if not throttle_counts:
            return None

        primary_reason = max(throttle_counts.items(), key=lambda x: x[1])
        reason_name, reason_count = primary_reason

        severity = min(reason_count / len(trace.gpu_samples), 1.0)

        evidence = [
            DiagnosisEvidence(
                metric="throttled_samples",
                value=len(throttled_samples),
                threshold=self.config.throttle_severity_threshold,
                comparison="exceeds",
                severity=severity,
            )
        ]

        confidence = 0.7 + severity * 0.2

        # Reason-specific descriptions
        reason_descriptions = {
            ThrottleReason.SW_POWER_CAP: "software power cap",
            ThrottleReason.HW_SLOWDOWN: "hardware slowdown",
            ThrottleReason.SW_THERMAL: "software thermal limit",
            ThrottleReason.HW_THERMAL: "hardware thermal limit",
            ThrottleReason.HW_POWER_BRAKE: "hardware power brake",
        }

        reason_desc = reason_descriptions.get(reason_name, str(reason_name))

        return DiagnosisResult(
            request_id=trace.request_id,
            category=DiagnosisCategory.GPU_THROTTLING,
            confidence=confidence,
            evidence=evidence,
            description=f"GPU was throttled during {len(throttled_samples)} samples "
            f"({len(throttled_samples) / len(trace.gpu_samples) * 100:.1f}% of request time). "
            f"Primary reason: {reason_desc}.",
            mitigation=f"Check GPU cooling, power limits, and ambient temperature. "
            f"For {reason_desc}, consider increasing power/thermal limits if safe.",
        )

    def _check_memory_pressure(self, trace: RequestTrace) -> Optional[DiagnosisResult]:
        """Check if request suffered from memory pressure."""
        if not trace.gpu_samples:
            return None

        # Calculate average memory utilization during request
        avg_mem_util = statistics.mean(s.memory_utilization_pct for s in trace.gpu_samples)

        if avg_mem_util < self.config.memory_pressure_threshold_pct:
            return None

        severity = (avg_mem_util - self.config.memory_pressure_threshold_pct) / (
            100 - self.config.memory_pressure_threshold_pct
        )

        evidence = [
            DiagnosisEvidence(
                metric="avg_memory_utilization_pct",
                value=avg_mem_util,
                threshold=self.config.memory_pressure_threshold_pct,
                comparison="exceeds",
                severity=severity,
            )
        ]

        confidence = 0.6 + severity * 0.3

        return DiagnosisResult(
            request_id=trace.request_id,
            category=DiagnosisCategory.MEMORY_PRESSURE,
            confidence=confidence,
            evidence=evidence,
            description=f"High memory utilization ({avg_mem_util:.1f}%) during request. "
            f"KV cache pressure may be causing slowdowns.",
            mitigation="Reduce batch size, enable PagedAttention if not already, "
            "or reduce max_num_seqs to limit concurrent requests.",
        )

    def _check_host_bottleneck(self, trace: RequestTrace) -> Optional[DiagnosisResult]:
        """Check if request was bottlenecked by host (CPU/Python overhead)."""
        if not trace.gpu_samples:
            return None

        # Low GPU utilization but high latency suggests host bottleneck
        avg_gpu_util = statistics.mean(s.gpu_utilization_pct for s in trace.gpu_samples)

        # Only diagnose if GPU util is low
        if avg_gpu_util >= self.config.low_gpu_util_threshold_pct:
            return None

        # Check if latency is high
        # High latency with low GPU util = host bottleneck
        total_time_ms = trace.total_duration_ms

        # Heuristic: if total time > 100ms and GPU util < threshold, likely host bottleneck
        if total_time_ms < 100:
            return None

        severity = (self.config.low_gpu_util_threshold_pct - avg_gpu_util) / self.config.low_gpu_util_threshold_pct

        evidence = [
            DiagnosisEvidence(
                metric="avg_gpu_utilization_pct",
                value=avg_gpu_util,
                threshold=self.config.low_gpu_util_threshold_pct,
                comparison="below",
                severity=severity,
            )
        ]

        confidence = 0.5 + severity * 0.3

        return DiagnosisResult(
            request_id=trace.request_id,
            category=DiagnosisCategory.HOST_BOTTLENECK,
            confidence=confidence,
            evidence=evidence,
            description=f"Low GPU utilization ({avg_gpu_util:.1f}%) but high latency "
            f"({total_time_ms:.1f}ms) suggests host-side bottleneck (CPU, Python overhead).",
            mitigation="Profile CPU usage, check for GIL contention, optimize tokenization, "
            "or reduce Python overhead in preprocessing.",
        )

    def _check_cold_path(self, trace: RequestTrace) -> Optional[DiagnosisResult]:
        """Check if request hit cold path (first-run overhead)."""
        # Heuristic: very long prefill time relative to prompt length
        # This is speculative - real cold path detection would need historical data

        if trace.prefill_duration_ms == 0 or trace.prompt_length == 0:
            return None

        # Rough heuristic: prefill should be ~1-10ms per token on warm path
        # If much longer, might be cold
        time_per_token = trace.prefill_duration_ms / trace.prompt_length

        # Threshold: > 50ms per token is suspiciously slow (might be cold)
        cold_threshold = 50.0

        if time_per_token < cold_threshold:
            return None

        severity = min((time_per_token - cold_threshold) / cold_threshold, 1.0)

        evidence = [
            DiagnosisEvidence(
                metric="prefill_ms_per_token",
                value=time_per_token,
                threshold=cold_threshold,
                comparison="exceeds",
                severity=severity,
            )
        ]

        confidence = 0.4 + severity * 0.2  # Lower confidence - this is speculative

        return DiagnosisResult(
            request_id=trace.request_id,
            category=DiagnosisCategory.COLD_PATH,
            confidence=confidence,
            evidence=evidence,
            description=f"Unusually slow prefill ({time_per_token:.1f}ms/token). "
            f"May indicate cold path (JIT compilation, model loading, etc.).",
            mitigation="Warm up model with dummy requests before serving, "
            "or investigate first-run initialization overhead.",
        )

    def _check_batch_fragmentation(self, traces: List[RequestTrace]) -> Optional[DiagnosisResult]:
        """Check if batch suffered from fragmentation (mixed prompt lengths)."""
        if len(traces) < 2:
            return None

        # Calculate prompt length variance
        prompt_lengths = [t.prompt_length for t in traces]
        avg_length = statistics.mean(prompt_lengths)

        if avg_length == 0:
            return None

        stdev = statistics.stdev(prompt_lengths) if len(prompt_lengths) > 1 else 0
        coefficient_of_variation = stdev / avg_length

        if coefficient_of_variation < self.config.batch_fragmentation_threshold:
            return None

        severity = min(
            coefficient_of_variation / self.config.batch_fragmentation_threshold - 1.0,
            1.0
        )

        evidence = [
            DiagnosisEvidence(
                metric="prompt_length_coefficient_of_variation",
                value=coefficient_of_variation,
                threshold=self.config.batch_fragmentation_threshold,
                comparison="exceeds",
                severity=severity,
            )
        ]

        confidence = 0.6 + severity * 0.2

        return DiagnosisResult(
            request_id=None,  # Batch-level diagnosis
            category=DiagnosisCategory.BATCH_FRAGMENTATION,
            confidence=confidence,
            evidence=evidence,
            description=f"Batch has high prompt length variance (CoV={coefficient_of_variation:.2f}). "
            f"Mixed lengths ({min(prompt_lengths)}-{max(prompt_lengths)} tokens) "
            f"can reduce batching efficiency due to padding overhead.",
            mitigation="Consider bucketing requests by prompt length before batching, "
            "or adjust batching strategy to group similar-length prompts.",
        )
