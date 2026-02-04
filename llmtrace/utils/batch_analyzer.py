"""Batch/Scheduler visibility analyzer (Feature 1)."""

import logging
from typing import List, Dict, Any
import statistics

from llmtrace.models.trace import BatchMetadata, RequestTrace

logger = logging.getLogger(__name__)


class BatchAnalyzer:
    """
    Analyzes batch and scheduler behavior for vLLM.

    Feature 1: Batch/Scheduler Visibility

    Provides insights into:
    - Batch size over time (prefill vs decode)
    - Prompt length distribution per batch
    - KV cache pressure
    - Queue depth snapshots
    """

    def analyze_batches(self, batches: List[BatchMetadata]) -> Dict[str, Any]:
        """
        Analyze batch metadata to extract insights.

        Args:
            batches: List of BatchMetadata objects

        Returns:
            Dictionary of batch analytics
        """
        if not batches:
            return {}

        # Batch size statistics
        batch_sizes = [b.num_requests for b in batches]
        prefill_counts = [b.num_prefill for b in batches]
        decode_counts = [b.num_decode for b in batches]

        # Prompt length statistics
        all_prompt_lengths = []
        for batch in batches:
            all_prompt_lengths.extend(batch.prompt_lengths)

        # KV cache statistics
        kv_utilizations = [
            b.kv_cache_utilization
            for b in batches
            if b.kv_cache_utilization is not None
        ]

        analytics = {
            "num_batches": len(batches),
            "batch_size": {
                "mean": statistics.mean(batch_sizes) if batch_sizes else 0,
                "median": statistics.median(batch_sizes) if batch_sizes else 0,
                "min": min(batch_sizes) if batch_sizes else 0,
                "max": max(batch_sizes) if batch_sizes else 0,
            },
            "prefill_decode_ratio": {
                "mean_prefill": statistics.mean(prefill_counts) if prefill_counts else 0,
                "mean_decode": statistics.mean(decode_counts) if decode_counts else 0,
                "total_prefill_batches": sum(1 for b in batches if b.num_prefill > 0),
                "total_decode_batches": sum(1 for b in batches if b.num_decode > 0),
                "mixed_batches": sum(
                    1 for b in batches if b.num_prefill > 0 and b.num_decode > 0
                ),
            },
            "prompt_lengths": {
                "mean": statistics.mean(all_prompt_lengths) if all_prompt_lengths else 0,
                "median": statistics.median(all_prompt_lengths) if all_prompt_lengths else 0,
                "min": min(all_prompt_lengths) if all_prompt_lengths else 0,
                "max": max(all_prompt_lengths) if all_prompt_lengths else 0,
                "stdev": statistics.stdev(all_prompt_lengths)
                if len(all_prompt_lengths) > 1
                else 0,
            },
            "kv_cache": {
                "mean_utilization": statistics.mean(kv_utilizations)
                if kv_utilizations
                else None,
                "max_utilization": max(kv_utilizations) if kv_utilizations else None,
                "pressure_incidents": sum(1 for u in kv_utilizations if u > 0.9)
                if kv_utilizations
                else 0,
            },
        }

        return analytics

    def generate_batch_timeline(
        self, batches: List[BatchMetadata]
    ) -> List[Dict[str, Any]]:
        """
        Generate timeline of batch behavior.

        Returns:
            List of timeline entries showing batch evolution
        """
        timeline = []

        for batch in sorted(batches, key=lambda b: b.timestamp):
            entry = {
                "timestamp": batch.timestamp,
                "batch_id": batch.batch_id,
                "total_requests": batch.num_requests,
                "prefill": batch.num_prefill,
                "decode": batch.num_decode,
                "total_tokens": batch.total_tokens,
                "kv_utilization": batch.kv_cache_utilization,
            }
            timeline.append(entry)

        return timeline

    def detect_batching_inefficiencies(
        self, batches: List[BatchMetadata]
    ) -> List[Dict[str, Any]]:
        """
        Detect batching inefficiencies.

        Returns:
            List of detected inefficiencies with descriptions
        """
        inefficiencies = []

        # Check for underutilized batches
        small_batches = [b for b in batches if b.num_requests == 1]
        if len(small_batches) > len(batches) * 0.5:
            inefficiencies.append(
                {
                    "type": "underutilized_batching",
                    "severity": "high",
                    "description": f"{len(small_batches)}/{len(batches)} batches "
                    f"({len(small_batches)/len(batches)*100:.1f}%) contain only 1 request. "
                    f"System may be underutilizing batching capabilities.",
                    "mitigation": "Increase batch wait time or adjust scheduling policy "
                    "to accumulate more requests before execution.",
                }
            )

        # Check for high prompt length variance (fragmentation)
        for batch in batches:
            if len(batch.prompt_lengths) > 1:
                mean_len = statistics.mean(batch.prompt_lengths)
                if mean_len > 0:
                    stdev = statistics.stdev(batch.prompt_lengths)
                    cv = stdev / mean_len  # Coefficient of variation

                    if cv > 0.5:  # High variance
                        inefficiencies.append(
                            {
                                "type": "batch_fragmentation",
                                "severity": "medium",
                                "description": f"Batch {batch.batch_id} has high prompt length "
                                f"variance (CV={cv:.2f}, lengths={min(batch.prompt_lengths)}-"
                                f"{max(batch.prompt_lengths)}). Padding overhead may reduce efficiency.",
                                "mitigation": "Consider bucketing requests by prompt length.",
                            }
                        )

        # Check for KV cache pressure
        for batch in batches:
            if (
                batch.kv_cache_utilization is not None
                and batch.kv_cache_utilization > 0.95
            ):
                inefficiencies.append(
                    {
                        "type": "kv_cache_pressure",
                        "severity": "high",
                        "description": f"Batch {batch.batch_id} has very high KV cache "
                        f"utilization ({batch.kv_cache_utilization*100:.1f}%). "
                        f"May cause evictions and recomputation.",
                        "mitigation": "Reduce max_num_seqs, enable better cache management, "
                        "or reduce max sequence length.",
                    }
                )

        return inefficiencies

    def print_batch_summary(self, batches: List[BatchMetadata]) -> None:
        """Print a human-readable batch summary."""
        analytics = self.analyze_batches(batches)

        print("\n" + "=" * 60)
        print("Batch/Scheduler Analysis (Feature 1)")
        print("=" * 60)

        print(f"\nTotal Batches: {analytics['num_batches']}")

        print("\nBatch Size:")
        print(f"  Mean: {analytics['batch_size']['mean']:.2f}")
        print(f"  Median: {analytics['batch_size']['median']:.0f}")
        print(f"  Range: {analytics['batch_size']['min']}-{analytics['batch_size']['max']}")

        print("\nPrefill vs Decode:")
        print(f"  Mean prefill/batch: {analytics['prefill_decode_ratio']['mean_prefill']:.2f}")
        print(f"  Mean decode/batch: {analytics['prefill_decode_ratio']['mean_decode']:.2f}")
        print(f"  Mixed batches: {analytics['prefill_decode_ratio']['mixed_batches']}")

        print("\nPrompt Lengths:")
        print(f"  Mean: {analytics['prompt_lengths']['mean']:.2f}")
        print(f"  Median: {analytics['prompt_lengths']['median']:.0f}")
        print(f"  Range: {analytics['prompt_lengths']['min']}-{analytics['prompt_lengths']['max']}")
        print(f"  StdDev: {analytics['prompt_lengths']['stdev']:.2f}")

        if analytics["kv_cache"]["mean_utilization"] is not None:
            print("\nKV Cache:")
            print(
                f"  Mean utilization: {analytics['kv_cache']['mean_utilization']*100:.1f}%"
            )
            print(
                f"  Max utilization: {analytics['kv_cache']['max_utilization']*100:.1f}%"
            )
            print(f"  Pressure incidents (>90%): {analytics['kv_cache']['pressure_incidents']}")

        # Inefficiencies
        inefficiencies = self.detect_batching_inefficiencies(batches)
        if inefficiencies:
            print("\nDetected Inefficiencies:")
            for i, ineff in enumerate(inefficiencies[:5], 1):
                print(f"\n  {i}. [{ineff['severity'].upper()}] {ineff['type']}")
                print(f"     {ineff['description']}")
                print(f"     Mitigation: {ineff['mitigation']}")

        print("\n" + "=" * 60)
