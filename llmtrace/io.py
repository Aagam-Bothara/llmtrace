"""Loading trace files for offline analysis (no GPU, NVML or vLLM needed)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Type, TypeVar, Union

from pydantic import BaseModel

from llmtrace.data_plane.cuda_timing import StepGpuTiming
from llmtrace.data_plane.vllm_stats import CollectorEvent, VLLMIterationRecord
from llmtrace.models.trace import BatchMetadata, GPUSample, RequestTrace

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
PathLike = Union[str, Path]


def _load_file(path: Path, model: Type[T]) -> List[T]:
    if path.suffix == ".jsonl":
        out: List[T] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(model.model_validate_json(line))
        return out
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("reading parquet requires pyarrow (pip install 'llmtrace[parquet]')") from exc
        return [model.model_validate(row) for row in pq.read_table(path).to_pylist()]
    raise ValueError(f"Unsupported trace file: {path}")


_PREFIXES = ("traces", "batches", "gpu", "vllm_stats", "collector", "gpu_steps")


def _matches_prefix(path: Path, prefix: str) -> bool:
    """``gpu_*`` must not pick up ``gpu_steps_*``: the longest known prefix wins."""
    name = path.name
    best = max((p for p in _PREFIXES if name.startswith(p + "_")), key=len, default=None)
    return best == prefix


def _expand(paths: Iterable[PathLike], prefix: str) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(f for f in sorted(p.glob(f"{prefix}_*.jsonl")) if _matches_prefix(f, prefix))
            files.extend(f for f in sorted(p.glob(f"{prefix}_*.parquet")) if _matches_prefix(f, prefix))
        elif p.exists():
            files.append(p)
        else:
            raise FileNotFoundError(str(p))
    return files


def load_traces(paths: Sequence[PathLike]) -> List[RequestTrace]:
    """Load request traces from files or directories (``traces_*.jsonl|parquet``)."""
    out: List[RequestTrace] = []
    for f in _expand(paths, "traces"):
        out.extend(_load_file(f, RequestTrace))
    return out


def load_gpu_samples(paths: Sequence[PathLike]) -> List[GPUSample]:
    out: List[GPUSample] = []
    for f in _expand(paths, "gpu"):
        out.extend(_load_file(f, GPUSample))
    return out


def load_batches(paths: Sequence[PathLike]) -> List[BatchMetadata]:
    out: List[BatchMetadata] = []
    for f in _expand(paths, "batches"):
        out.extend(_load_file(f, BatchMetadata))
    return out


def load_vllm_stats(paths: Sequence[PathLike]) -> List[VLLMIterationRecord]:
    """Load vLLM's own per-step stats (``vllm_stats_*.jsonl``), if the run recorded them."""
    out: List[VLLMIterationRecord] = []
    for f in _expand(paths, "vllm_stats"):
        out.extend(_load_file(f, VLLMIterationRecord))
    return out


def load_collector_events(paths: Sequence[PathLike]) -> List[CollectorEvent]:
    out: List[CollectorEvent] = []
    for f in _expand(paths, "collector"):
        out.extend(_load_file(f, CollectorEvent))
    return out


def load_gpu_steps(paths: Sequence[PathLike]) -> List[StepGpuTiming]:
    """Per-step GPU spans from CUDA events (``gpu_steps_*.jsonl``), if recorded."""
    out: List[StepGpuTiming] = []
    for f in _expand(paths, "gpu_steps"):
        out.extend(_load_file(f, StepGpuTiming))
    return out


def run_directories_for(trace_paths: Sequence[PathLike]) -> List[Path]:
    """Directories whose gpu_*/batches_* files belong with the given trace paths."""
    dirs: List[Path] = []
    for p in trace_paths:
        d = Path(p) if Path(p).is_dir() else Path(p).parent
        if d not in dirs:
            dirs.append(d)
    return dirs


def write_jsonl(path: PathLike, items: Iterable[BaseModel]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json())
            f.write("\n")


def read_json(path: PathLike) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
