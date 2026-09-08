"""Rules engine for diagnosing performance issues (Autopsy).

Each rule returns a ``DiagnosisResult`` whose ``score`` is a rule-specific
ranking value in [0, 1] derived from how far a metric exceeds its threshold.
It is used to pick the most prominent diagnosis; it is not a probability.
"""

from __future__ import annotations

import logging
import statistics
from typing import List, Optional

from llmtrace.models.config import AutopsyConfig
from llmtrace.models.trace import (
    DiagnosisCategory,
    DiagnosisEvidence,
    DiagnosisResult,
    RequestTrace,
    ThrottleReason,
)

logger = logging.getLogger(__name__)


class RulesEngine:
    def __init__(self, config: AutopsyConfig):
        self.config = config

    def diagnose_request(self, trace: RequestTrace) -> Optional[DiagnosisResult]:
        if not self.config.enabled:
            return None
        diagnoses = [
            d
            for d in (
                self._check_queueing_overload(trace),
                self._check_throttling(trace),
                self._check_memory_pressure(trace),
                self._check_host_bottleneck(trace),
                self._check_cold_path(trace),
            )
            if d is not None
        ]
        return max(diagnoses, key=lambda d: d.score) if diagnoses else None

    def diagnose_batch(self, traces: List[RequestTrace]) -> List[DiagnosisResult]:
        if not self.config.enabled or not traces:
            return []
        diag = self._check_batch_fragmentation(traces)
        return [diag] if diag else []

    # ------------------------------------------------------------------ rules

    def _check_queueing_overload(self, trace: RequestTrace) -> Optional[DiagnosisResult]:
        # Only a QUEUE span (scheduler visible) is a queue measurement. A
        # TIME_TO_FIRST_TOKEN span mixes queue and prefill and is not used.
        queue_ms = trace.queue_duration_ms
        if queue_ms <= 0 or queue_ms < self.config.queue_overload_threshold_ms:
            return None
        severity = min(queue_ms / self.config.queue_overload_threshold_ms - 1.0, 1.0)
        return DiagnosisResult(
            request_id=trace.request_id,
            category=DiagnosisCategory.QUEUEING_OVERLOAD,
            score=min(0.5 + severity * 0.5, 0.95),
            evidence=[
                DiagnosisEvidence(
                    metric="queue_wait_ms",
                    value=queue_ms,
                    threshold=self.config.queue_overload_threshold_ms,
                    comparison="exceeds",
                    severity=severity,
                )
            ],
            description=f"Request waited {queue_ms:.1f}ms before first being scheduled "
            f"(threshold {self.config.queue_overload_threshold_ms:.1f}ms).",
            mitigation="Reduce request rate, raise max_num_seqs, or add GPU capacity.",
        )

    def _check_throttling(self, trace: RequestTrace) -> Optional[DiagnosisResult]:
        if not trace.gpu_samples:
            return None
        throttled = [s for s in trace.gpu_samples if s.is_throttled]
        if len(throttled) < self.config.throttle_severity_threshold:
            return None
        counts: dict = {}
        for s in throttled:
            for r in s.throttle_reasons:
                if r not in (ThrottleReason.NONE, ThrottleReason.GPU_IDLE, ThrottleReason.UNKNOWN):
                    counts[r] = counts.get(r, 0) + 1
        if not counts:
            return None
        reason, count = max(counts.items(), key=lambda x: x[1])
        severity = min(count / len(trace.gpu_samples), 1.0)
        names = {
            ThrottleReason.SW_POWER_CAP: "software power cap",
            ThrottleReason.HW_SLOWDOWN: "hardware slowdown",
            ThrottleReason.SW_THERMAL: "software thermal limit",
            ThrottleReason.HW_THERMAL: "hardware thermal limit",
            ThrottleReason.HW_POWER_BRAKE: "hardware power brake",
        }
        desc = names.get(reason, str(reason.value))
        return DiagnosisResult(
            request_id=trace.request_id,
            category=DiagnosisCategory.GPU_THROTTLING,
            score=0.7 + severity * 0.2,
            evidence=[
                DiagnosisEvidence(
                    metric="throttled_samples",
                    value=len(throttled),
                    threshold=self.config.throttle_severity_threshold,
                    comparison="exceeds",
                    severity=severity,
                )
            ],
            description=f"GPU throttled in {len(throttled)} of {len(trace.gpu_samples)} samples "
            f"during the request. Primary reason: {desc}.",
            mitigation="Check cooling, power limits and ambient temperature.",
        )

    def _check_memory_pressure(self, trace: RequestTrace) -> Optional[DiagnosisResult]:
        # Capacity pressure uses used/total memory. NVML's "memory utilization"
        # is memory *bandwidth* activity and is not a capacity signal.
        ratios = [
            s.memory_used_mb / s.memory_total_mb * 100.0
            for s in trace.gpu_samples
            if s.memory_used_mb is not None and s.memory_total_mb
        ]
        if not ratios:
            return None
        avg = statistics.mean(ratios)
        thr = self.config.memory_pressure_threshold_pct
        if avg < thr:
            return None
        severity = (avg - thr) / max(100.0 - thr, 1e-9)
        return DiagnosisResult(
            request_id=trace.request_id,
            category=DiagnosisCategory.MEMORY_PRESSURE,
            score=min(0.6 + severity * 0.3, 0.9),
            evidence=[
                DiagnosisEvidence(
                    metric="avg_memory_used_pct",
                    value=avg,
                    threshold=thr,
                    comparison="exceeds",
                    severity=severity,
                )
            ],
            description=f"GPU memory {avg:.1f}% used on average during the request. "
            "Note: vLLM pre-allocates KV cache, so high usage alone does not prove pressure.",
            mitigation="Check KV cache usage in batch metadata; reduce max_num_seqs or max_model_len.",
        )

    def _check_host_bottleneck(self, trace: RequestTrace) -> Optional[DiagnosisResult]:
        utils = [s.gpu_utilization_pct for s in trace.gpu_samples if s.gpu_utilization_pct is not None]
        if not utils:
            return None
        avg = statistics.mean(utils)
        thr = self.config.low_gpu_util_threshold_pct
        if avg >= thr or trace.total_duration_ms < 100:
            return None
        severity = (thr - avg) / max(thr, 1e-9)
        return DiagnosisResult(
            request_id=trace.request_id,
            category=DiagnosisCategory.HOST_BOTTLENECK,
            score=0.5 + severity * 0.3,
            evidence=[
                DiagnosisEvidence(
                    metric="avg_gpu_utilization_pct",
                    value=avg,
                    threshold=thr,
                    comparison="below",
                    severity=severity,
                )
            ],
            description=f"Low GPU utilization ({avg:.1f}%) over a {trace.total_duration_ms:.1f}ms request "
            "suggests host-side (CPU/Python) limits.",
            mitigation="Profile the host process; check tokenization and scheduling overhead.",
        )

    def _check_cold_path(self, trace: RequestTrace) -> Optional[DiagnosisResult]:
        # Requires a real PREFILL span (scheduler visible) and a verified prompt length.
        if trace.prefill_duration_ms <= 0 or not trace.prompt_length:
            return None
        per_token = trace.prefill_duration_ms / trace.prompt_length
        cold_threshold = 50.0
        if per_token < cold_threshold:
            return None
        severity = min((per_token - cold_threshold) / cold_threshold, 1.0)
        return DiagnosisResult(
            request_id=trace.request_id,
            category=DiagnosisCategory.COLD_PATH,
            score=0.4 + severity * 0.2,
            evidence=[
                DiagnosisEvidence(
                    metric="prefill_ms_per_prompt_token",
                    value=per_token,
                    threshold=cold_threshold,
                    comparison="exceeds",
                    severity=severity,
                )
            ],
            description=f"Unusually slow prefill ({per_token:.1f}ms per prompt token).",
            mitigation="Warm up the model before serving; check for compilation on first requests.",
        )

    def _check_batch_fragmentation(self, traces: List[RequestTrace]) -> Optional[DiagnosisResult]:
        lengths = [t.prompt_length for t in traces if t.prompt_length]
        if len(lengths) < 2:
            return None
        avg = statistics.mean(lengths)
        if avg == 0:
            return None
        cv = statistics.stdev(lengths) / avg
        if cv < self.config.batch_fragmentation_threshold:
            return None
        severity = min(cv / self.config.batch_fragmentation_threshold - 1.0, 1.0)
        return DiagnosisResult(
            request_id=None,
            category=DiagnosisCategory.BATCH_FRAGMENTATION,
            score=0.6 + severity * 0.2,
            evidence=[
                DiagnosisEvidence(
                    metric="prompt_length_coefficient_of_variation",
                    value=cv,
                    threshold=self.config.batch_fragmentation_threshold,
                    comparison="exceeds",
                    severity=severity,
                )
            ],
            description=f"High prompt length variance (CoV={cv:.2f}, {min(lengths)}-{max(lengths)} tokens).",
            mitigation="Bucket requests by prompt length or adjust chunked prefill settings.",
        )
