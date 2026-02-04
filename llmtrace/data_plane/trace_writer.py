"""Trace writer for persisting traces to disk."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

from llmtrace.models.trace import RequestTrace, BatchMetadata, GPUSample

logger = logging.getLogger(__name__)


class TraceWriter:
    """
    Writes traces to disk in JSONL or Parquet format.

    Features:
    - Async writing to avoid blocking
    - Buffering for efficiency
    - Automatic file rotation
    """

    def __init__(
        self,
        output_dir: str,
        output_format: str = "jsonl",
        async_write: bool = True,
        buffer_size: int = 1000,
    ):
        self.output_dir = Path(output_dir)
        self.output_format = output_format.lower()
        self.async_write = async_write
        self.buffer_size = buffer_size

        if self.output_format == "parquet" and not PARQUET_AVAILABLE:
            raise RuntimeError(
                "Parquet format requires pyarrow. Install with: pip install pyarrow"
            )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Buffers
        self._trace_buffer: List[RequestTrace] = []
        self._batch_buffer: List[BatchMetadata] = []
        self._gpu_buffer: List[GPUSample] = []
        self._buffer_lock = asyncio.Lock()

        # Write task
        self._write_task: Optional[asyncio.Task] = None
        self._running = False

        # File handles
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._trace_file: Optional[object] = None
        self._batch_file: Optional[object] = None
        self._gpu_file: Optional[object] = None

        logger.info(
            f"TraceWriter initialized: output_dir={output_dir}, "
            f"format={output_format}, session={self._session_id}"
        )

    def start(self) -> None:
        """Start the async write loop."""
        if self.async_write:
            self._running = True
            self._write_task = asyncio.create_task(self._write_loop())
            logger.info("TraceWriter async write loop started")

    async def stop(self) -> None:
        """Stop writing and flush remaining data."""
        logger.info("Stopping TraceWriter")

        if self.async_write and self._running:
            self._running = False
            if self._write_task:
                await self._write_task

        # Final flush
        await self._flush_all()

        # Close files
        self._close_files()

        logger.info("TraceWriter stopped")

    async def write_traces(self, traces: List[RequestTrace]) -> None:
        """
        Write request traces.

        Args:
            traces: List of RequestTrace objects to write
        """
        if not traces:
            return

        async with self._buffer_lock:
            self._trace_buffer.extend(traces)

            # Flush if buffer full or not async
            if not self.async_write or len(self._trace_buffer) >= self.buffer_size:
                await self._flush_traces()

    async def write_batch_metadata(self, batches: List[BatchMetadata]) -> None:
        """Write batch metadata."""
        if not batches:
            return

        async with self._buffer_lock:
            self._batch_buffer.extend(batches)

            if not self.async_write or len(self._batch_buffer) >= self.buffer_size:
                await self._flush_batches()

    async def write_gpu_samples(self, samples: List[GPUSample]) -> None:
        """Write GPU samples."""
        if not samples:
            return

        async with self._buffer_lock:
            self._gpu_buffer.extend(samples)

            if not self.async_write or len(self._gpu_buffer) >= self.buffer_size:
                await self._flush_gpu_samples()

    async def _write_loop(self) -> None:
        """Async write loop - periodically flushes buffers."""
        while self._running:
            await asyncio.sleep(1.0)  # Flush every second

            async with self._buffer_lock:
                await self._flush_all()

    async def _flush_all(self) -> None:
        """Flush all buffers."""
        await self._flush_traces()
        await self._flush_batches()
        await self._flush_gpu_samples()

    async def _flush_traces(self) -> None:
        """Flush trace buffer to disk."""
        if not self._trace_buffer:
            return

        traces = list(self._trace_buffer)
        self._trace_buffer.clear()

        if self.output_format == "jsonl":
            await self._write_jsonl(traces, "traces")
        elif self.output_format == "parquet":
            await self._write_parquet(traces, "traces")

        logger.debug(f"Flushed {len(traces)} traces")

    async def _flush_batches(self) -> None:
        """Flush batch metadata buffer to disk."""
        if not self._batch_buffer:
            return

        batches = list(self._batch_buffer)
        self._batch_buffer.clear()

        if self.output_format == "jsonl":
            await self._write_jsonl(batches, "batches")
        elif self.output_format == "parquet":
            await self._write_parquet(batches, "batches")

        logger.debug(f"Flushed {len(batches)} batch metadata entries")

    async def _flush_gpu_samples(self) -> None:
        """Flush GPU samples buffer to disk."""
        if not self._gpu_buffer:
            return

        samples = list(self._gpu_buffer)
        self._gpu_buffer.clear()

        if self.output_format == "jsonl":
            await self._write_jsonl(samples, "gpu")
        elif self.output_format == "parquet":
            await self._write_parquet(samples, "gpu")

        logger.debug(f"Flushed {len(samples)} GPU samples")

    async def _write_jsonl(self, items: List, data_type: str) -> None:
        """Write items to JSONL file."""
        filepath = self.output_dir / f"{data_type}_{self._session_id}.jsonl"

        # Convert Pydantic models to dicts
        lines = [json.dumps(item.model_dump()) + "\n" for item in items]

        # Write to file (append mode)
        with open(filepath, "a") as f:
            f.writelines(lines)

    async def _write_parquet(self, items: List, data_type: str) -> None:
        """Write items to Parquet file."""
        filepath = self.output_dir / f"{data_type}_{self._session_id}.parquet"

        # Convert to list of dicts
        dicts = [item.model_dump() for item in items]

        # Convert to PyArrow table
        # This is simplified - in production you'd want to define schema explicitly
        table = pa.Table.from_pylist(dicts)

        # Append to parquet file
        if filepath.exists():
            # Read existing and append
            existing_table = pq.read_table(filepath)
            combined = pa.concat_tables([existing_table, table])
            pq.write_table(combined, filepath)
        else:
            pq.write_table(table, filepath)

    def _close_files(self) -> None:
        """Close any open file handles."""
        # Currently using context managers, so nothing to close
        pass

    def get_output_files(self) -> dict:
        """Get paths to output files for this session."""
        files = {}

        for data_type in ["traces", "batches", "gpu"]:
            if self.output_format == "jsonl":
                filepath = self.output_dir / f"{data_type}_{self._session_id}.jsonl"
            else:
                filepath = self.output_dir / f"{data_type}_{self._session_id}.parquet"

            if filepath.exists():
                files[data_type] = str(filepath)

        return files
