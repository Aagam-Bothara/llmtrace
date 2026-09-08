"""Tail latency explainer (Feature 2)."""

import logging
from typing import List, Dict, Any
import statistics

from llmtrace.models.trace import RequestTrace, DiagnosisCategory

logger = logging.getLogger(__name__)


class LatencyExplainer:
    """
    Explains tail latency with mechanism attribution.

    Feature 2: Tail Latency Explainer

    For slow requests, provides ranked causes with evidence:
    - Queueing overload
    - Batch fragmentation
    - GPU downclocking/throttling
    - Memory pressure
    - Host bottlenecks
    - Cold path issues
    """

    def explain_request(self, trace: RequestTrace) -> Dict[str, Any]:
        """
        Explain a single request's latency.

        Args:
            trace: RequestTrace with diagnosis

        Returns:
            Dictionary with explanation
        """
        if not trace.diagnosis:
            return {
                "request_id": trace.request_id,
                "total_latency_ms": trace.total_duration_ms,
                "is_tail": False,
                "explanation": "No performance issues detected.",
            }

        # Phase breakdown
        phase_breakdown = {
            "queue_ms": trace.queue_duration_ms,
            "queue_pct": (
                trace.queue_duration_ms / trace.total_duration_ms * 100
                if trace.total_duration_ms > 0
                else 0
            ),
            "prefill_ms": trace.prefill_duration_ms,
            "prefill_pct": (
                trace.prefill_duration_ms / trace.total_duration_ms * 100
                if trace.total_duration_ms > 0
                else 0
            ),
            "decode_ms": trace.decode_duration_ms,
            "decode_pct": (
                trace.decode_duration_ms / trace.total_duration_ms * 100
                if trace.total_duration_ms > 0
                else 0
            ),
        }

        explanation = {
            "request_id": trace.request_id,
            "total_latency_ms": trace.total_duration_ms,
            "is_tail": True,  # If we have a diagnosis, it's a tail latency case
            "phase_breakdown": phase_breakdown,
            "root_cause": {
                "category": DiagnosisCategory(trace.diagnosis.category).value,
                "score": trace.diagnosis.score,
                "description": trace.diagnosis.description,
                "mitigation": trace.diagnosis.mitigation,
                "evidence": [
                    {
                        "metric": e.metric,
                        "value": e.value,
                        "threshold": e.threshold,
                        "severity": e.severity,
                    }
                    for e in trace.diagnosis.evidence
                ],
            },
        }

        return explanation

    def explain_tail_latency_batch(
        self, traces: List[RequestTrace], tail_percentile: float = 95
    ) -> Dict[str, Any]:
        """
        Explain tail latency across a batch of requests.

        Args:
            traces: List of RequestTrace objects
            tail_percentile: Percentile to consider as "tail" (default 95)

        Returns:
            Dictionary with batch tail latency explanation
        """
        if not traces:
            return {}

        # Calculate tail threshold
        latencies = [t.total_duration_ms for t in traces]
        tail_threshold = self._percentile(latencies, tail_percentile)

        # Identify tail requests
        tail_requests = [t for t in traces if t.total_duration_ms >= tail_threshold]

        # Count diagnoses in tail
        tail_diagnoses = [t.diagnosis for t in tail_requests if t.diagnosis]
        cause_counts: Dict[DiagnosisCategory, int] = {}

        for diag in tail_diagnoses:
            cat = DiagnosisCategory(diag.category)
            cause_counts[cat] = cause_counts.get(cat, 0) + 1

        # Rank causes
        ranked_causes = sorted(
            cause_counts.items(), key=lambda x: x[1], reverse=True
        )

        # Generate summary
        summary = {
            "total_requests": len(traces),
            "tail_percentile": tail_percentile,
            "tail_threshold_ms": tail_threshold,
            "tail_count": len(tail_requests),
            "avg_latency_ms": statistics.mean(latencies),
            "p50_latency_ms": statistics.median(latencies),
            f"p{int(tail_percentile)}_latency_ms": tail_threshold,
            "ranked_causes": [
                {
                    "category": cause.value,
                    "count": count,
                    "percentage": count / len(tail_requests) * 100,
                }
                for cause, count in ranked_causes
            ],
            "example_tail_requests": [
                self.explain_request(t) for t in tail_requests[:5]
            ],
        }

        return summary

    def print_explanation(self, explanation: Dict[str, Any]) -> None:
        """Print a human-readable latency explanation."""
        print("\n" + "=" * 60)
        print("Tail Latency Explanation (Feature 2)")
        print("=" * 60)

        print(f"\nRequest ID: {explanation['request_id']}")
        print(f"Total Latency: {explanation['total_latency_ms']:.2f}ms")

        if not explanation.get("is_tail"):
            print("Status: Normal (no issues detected)")
            return

        print("Status: TAIL LATENCY DETECTED")

        # Phase breakdown
        breakdown = explanation["phase_breakdown"]
        print("\nPhase Breakdown:")
        print(f"  Queue:   {breakdown['queue_ms']:.2f}ms ({breakdown['queue_pct']:.1f}%)")
        print(f"  Prefill: {breakdown['prefill_ms']:.2f}ms ({breakdown['prefill_pct']:.1f}%)")
        print(f"  Decode:  {breakdown['decode_ms']:.2f}ms ({breakdown['decode_pct']:.1f}%)")

        # Root cause
        root_cause = explanation["root_cause"]
        print(f"\nRoot Cause: {root_cause['category']}")
        print(f"Rule score: {root_cause['score']:.2f} (ranking value, not a probability)")
        print("\nDescription:")
        print(f"  {root_cause['description']}")

        if root_cause.get("mitigation"):
            print("\nMitigation:")
            print(f"  {root_cause['mitigation']}")

        # Evidence
        if root_cause.get("evidence"):
            print("\nEvidence:")
            for i, ev in enumerate(root_cause["evidence"], 1):
                print(
                    f"  {i}. {ev['metric']}: {ev['value']:.2f} "
                    f"(threshold: {ev['threshold']:.2f}, severity: {ev['severity']:.2f})"
                )

        print("\n" + "=" * 60)

    def print_batch_explanation(self, batch_summary: Dict[str, Any]) -> None:
        """Print batch tail latency summary."""
        print("\n" + "=" * 60)
        print("Batch Tail Latency Analysis")
        print("=" * 60)

        print(f"\nTotal Requests: {batch_summary['total_requests']}")
        print(f"Tail Percentile: P{int(batch_summary['tail_percentile'])}")
        print(f"Tail Threshold: {batch_summary['tail_threshold_ms']:.2f}ms")
        print(f"Tail Count: {batch_summary['tail_count']}")

        print("\nLatency Distribution:")
        print(f"  Avg: {batch_summary['avg_latency_ms']:.2f}ms")
        print(f"  P50: {batch_summary['p50_latency_ms']:.2f}ms")
        p_key = f"p{int(batch_summary['tail_percentile'])}_latency_ms"
        print(f"  P{int(batch_summary['tail_percentile'])}: {batch_summary[p_key]:.2f}ms")

        print("\nRanked Root Causes (in tail):")
        for i, cause in enumerate(batch_summary["ranked_causes"], 1):
            print(
                f"  {i}. {cause['category']}: {cause['count']} occurrences "
                f"({cause['percentage']:.1f}% of tail)"
            )

        print("\n" + "=" * 60)

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def generate_diagnosis_report(
        self, traces: List[RequestTrace]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive diagnosis report.

        Args:
            traces: List of traces with diagnoses

        Returns:
            Diagnosis report dictionary
        """
        # Overall statistics
        total = len(traces)
        diagnosed = sum(1 for t in traces if t.diagnosis)

        # Category breakdown
        category_stats: Dict[DiagnosisCategory, Dict[str, Any]] = {}

        for trace in traces:
            if not trace.diagnosis:
                continue

            cat = DiagnosisCategory(trace.diagnosis.category)

            if cat not in category_stats:
                category_stats[cat] = {
                    "count": 0,
                    "total_latency": 0.0,
                    "avg_score": 0.0,
                    "scores": [],
                }

            category_stats[cat]["count"] += 1
            category_stats[cat]["total_latency"] += trace.total_duration_ms
            category_stats[cat]["scores"].append(trace.diagnosis.score)

        # Compute averages
        for cat, stats in category_stats.items():
            stats["avg_latency"] = stats["total_latency"] / stats["count"]
            stats["avg_score"] = statistics.mean(stats["scores"])

        report = {
            "total_requests": total,
            "diagnosed_requests": diagnosed,
            "diagnosis_rate": diagnosed / total * 100 if total > 0 else 0,
            "category_breakdown": {
                cat.value: {
                    "count": stats["count"],
                    "percentage": stats["count"] / diagnosed * 100 if diagnosed > 0 else 0,
                    "avg_latency_ms": stats["avg_latency"],
                    "avg_score": stats["avg_score"],
                }
                for cat, stats in category_stats.items()
            },
        }

        return report
