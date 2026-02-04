"""Example: Batch/Scheduler visibility (Feature 1)."""

import asyncio
import json
from pathlib import Path
from llmtrace.models.trace import BatchMetadata
from llmtrace.utils.batch_analyzer import BatchAnalyzer


async def main():
    # Load batch metadata from traces
    trace_dir = Path("./traces")  # Update to your trace directory

    batches = []
    for batch_file in trace_dir.glob("batches_*.jsonl"):
        with open(batch_file) as f:
            for line in f:
                if line.strip():
                    batch = BatchMetadata.model_validate(json.loads(line))
                    batches.append(batch)

    if not batches:
        print("No batch metadata found. Run basic_usage.py first.")
        return

    print(f"Loaded {len(batches)} batch metadata entries")

    # Analyze batches
    analyzer = BatchAnalyzer()

    # Print summary
    analyzer.print_batch_summary(batches)

    # Get timeline
    timeline = analyzer.generate_batch_timeline(batches)
    print(f"\nGenerated timeline with {len(timeline)} entries")

    # Detect inefficiencies
    inefficiencies = analyzer.detect_batching_inefficiencies(batches)
    if inefficiencies:
        print(f"\nFound {len(inefficiencies)} batching inefficiencies")
    else:
        print("\nNo batching inefficiencies detected!")


if __name__ == "__main__":
    asyncio.run(main())
