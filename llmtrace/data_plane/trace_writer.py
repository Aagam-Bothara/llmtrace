"""Trace writer for persisting traces to disk (JSONL or Parquet).

Writes happen either inline or from a background thread fed by a bounded
queue. When the queue is full, batches are dropped and counted; nothing blocks
the inference thread indefinitely. ``stop()`` drains the queue before closing
files, so every accepted batch is written exactly once.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_SENTINEL: Tuple[str, Optional[List[Any]]] = ("__stop__", None)


class TraceWriter:
    """Writes RequestTrace / BatchMetadata / GPUSample records to files."""

    DATA_TYPES = ("traces", "batches", "gpu", "vllm_stats", "collector")

    def __init__(
        self,
        output_dir: str,
        output_format: str = "jsonl",
        background: bool = True,
        max_queue: int = 10_000,
        session_id: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_format = output_format.lower()
        if self.output_format not in ("jsonl", "parquet"):
            raise ValueError(f"Unsupported output format: {output_format}")
        if self.output_format == "parquet":
            try:
                import pyarrow  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("Parquet output requires pyarrow (pip install 'llmtrace[parquet]')") from exc
        self.background = background
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._files: Dict[str, IO[str]] = {}
        self._parquet_parts: Dict[str, int] = {}
        self._queue: "queue.Queue[Tuple[str, Optional[List[Any]]]]" = queue.Queue(maxsize=max_queue)
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._stopped = False
        self._lock = threading.Lock()

        self.written: Dict[str, int] = {t: 0 for t in self.DATA_TYPES}
        self.dropped: Dict[str, int] = {t: 0 for t in self.DATA_TYPES}
        self.write_errors = 0
        self.last_error: Optional[str] = None

    # -------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._stopped = False
        if self.background:
            self._thread = threading.Thread(target=self._loop, name="llmtrace-writer", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Drain pending writes, close files. Idempotent."""
        if not self._started or self._stopped:
            return
        self._stopped = True
        if self._thread is not None:
            self._queue.put(_SENTINEL)  # blocks only if the queue is full; the thread is draining
            self._thread.join()
            self._thread = None
        self._close_files()
        logger.info("TraceWriter stopped: written=%s dropped=%s", self.written, self.dropped)

    def flush(self) -> None:
        """Block until everything accepted so far is on disk."""
        if self._thread is not None:
            self._queue.join()
        for f in self._files.values():
            f.flush()

    # ----------------------------------------------------------------- writes

    def write_traces(self, traces: Sequence[BaseModel]) -> None:
        self._submit("traces", list(traces))

    def write_batch_metadata(self, batches: Sequence[BaseModel]) -> None:
        self._submit("batches", list(batches))

    def write_gpu_samples(self, samples: Sequence[BaseModel]) -> None:
        self._submit("gpu", list(samples))

    def write_vllm_stats(self, records: Sequence[BaseModel]) -> None:
        self._submit("vllm_stats", list(records))

    def write_collector_events(self, events: Sequence[BaseModel]) -> None:
        self._submit("collector", list(events))

    def _submit(self, data_type: str, items: List[Any]) -> None:
        if not items:
            return
        if not self._started:
            raise RuntimeError("TraceWriter.start() must be called before writing")
        if self._stopped:
            self.dropped[data_type] += len(items)
            logger.error("TraceWriter is stopped; dropped %d %s", len(items), data_type)
            return
        if self._thread is None:
            self._write(data_type, items)
            return
        try:
            self._queue.put_nowait((data_type, items))
        except queue.Full:
            self.dropped[data_type] += len(items)
            logger.error("TraceWriter queue full; dropped %d %s records", len(items), data_type)

    def _loop(self) -> None:
        while True:
            data_type, items = self._queue.get()
            try:
                if data_type == _SENTINEL[0]:
                    return
                self._write(data_type, items or [])
            finally:
                self._queue.task_done()

    def _write(self, data_type: str, items: List[Any]) -> None:
        try:
            with self._lock:
                if self.output_format == "jsonl":
                    self._write_jsonl(data_type, items)
                else:
                    self._write_parquet(data_type, items)
            self.written[data_type] += len(items)
        except Exception as exc:
            self.write_errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.dropped[data_type] += len(items)
            logger.error("Failed writing %d %s records: %s", len(items), data_type, exc, exc_info=True)

    def _write_jsonl(self, data_type: str, items: List[Any]) -> None:
        f = self._files.get(data_type)
        if f is None:
            path = self.output_dir / f"{data_type}_{self.session_id}.jsonl"
            f = open(path, "a", encoding="utf-8")
            self._files[data_type] = f
        for item in items:
            f.write(item.model_dump_json())
            f.write("\n")
        f.flush()

    def _write_parquet(self, data_type: str, items: List[Any]) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        part = self._parquet_parts.get(data_type, 0)
        self._parquet_parts[data_type] = part + 1
        path = self.output_dir / f"{data_type}_{self.session_id}_part{part:05d}.parquet"
        rows = [json.loads(item.model_dump_json()) for item in items]
        pq.write_table(pa.Table.from_pylist(rows), path)

    def _close_files(self) -> None:
        with self._lock:
            for f in self._files.values():
                try:
                    f.close()
                except Exception:  # pragma: no cover
                    pass
            self._files.clear()

    # ------------------------------------------------------------------ info

    def get_output_files(self) -> Dict[str, List[str]]:
        files: Dict[str, List[str]] = {}
        for data_type in self.DATA_TYPES:
            paths = sorted(self.output_dir.glob(f"{data_type}_{self.session_id}*.{self.output_format}"))
            if paths:
                files[data_type] = [str(p) for p in paths]
        return files

    def stats(self) -> Dict[str, Any]:
        return {
            "written": dict(self.written),
            "dropped": dict(self.dropped),
            "write_errors": self.write_errors,
            "last_error": self.last_error,
            "queued": self._queue.qsize(),
        }
