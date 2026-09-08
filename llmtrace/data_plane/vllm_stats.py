"""Collect vLLM's own engine statistics through its supported ``stat_loggers`` hook.

vLLM 0.11.0 (``vllm/v1/metrics/loggers.py``) lets API users register custom
loggers implementing ``StatLoggerBase``::

    class StatLoggerBase(ABC):
        def __init__(self, vllm_config, engine_index: int = 0): ...
        def record(self, scheduler_stats, iteration_stats, engine_idx: int = 0): ...
        def log_engine_initialized(self): ...
        def log(self): pass

    StatLoggerFactory = Callable[[VllmConfig, int], StatLoggerBase]

and ``LLMEngine.from_engine_args(engine_args, stat_loggers=[factory])`` calls
``record()`` once per ``step()`` with the engine core's ``SchedulerStats``
(running/waiting counts, KV-cache usage, prefix-cache stats, corrupted count)
and an ``IterationStats`` (tokens this step, preemptions, vLLM's own
time-to-first-token and inter-token latencies observed this step, and
per-finished-request stats without request ids). This works in the default
multiprocess engine-core mode, unlike scheduler patching.

vLLM's docstring warns that ``SchedulerStats`` and ``IterationStats`` "are not
considered stable interfaces"; this module reads them with ``getattr`` and
records what it finds. Nothing here imports vLLM; the logger is duck-typed so
it can be exercised with fakes.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VLLMFinishedRequestStats(BaseModel):
    """vLLM's FinishedRequestStats (no request id is exposed by vLLM)."""

    finish_reason: Optional[str] = None
    e2e_latency_s: Optional[float] = None
    num_prompt_tokens: Optional[int] = None
    num_generation_tokens: Optional[int] = None
    max_tokens_param: Optional[int] = None
    queued_time_s: Optional[float] = None
    prefill_time_s: Optional[float] = None
    inference_time_s: Optional[float] = None
    decode_time_s: Optional[float] = None
    mean_time_per_output_token_s: Optional[float] = None


class VLLMIterationRecord(BaseModel):
    """One ``record()`` call: what vLLM reported for one engine step."""

    timestamp: float  # wall clock at record()
    monotonic: Optional[float] = None
    clock_domain: Optional[str] = None
    engine_idx: int = 0
    step_seq: int  # llmtrace's count of record() calls for this logger

    # SchedulerStats
    num_running_reqs: Optional[int] = None
    num_waiting_reqs: Optional[int] = None
    kv_cache_usage: Optional[float] = None  # fraction 0..1
    prefix_cache_requests: Optional[int] = None
    prefix_cache_queries: Optional[int] = None
    prefix_cache_hits: Optional[int] = None
    num_corrupted_reqs: Optional[int] = None

    # IterationStats
    num_generation_tokens: Optional[int] = None
    num_prompt_tokens: Optional[int] = None
    num_preempted_reqs: Optional[int] = None
    time_to_first_tokens_s: List[float] = Field(default_factory=list)
    inter_token_latencies_s: List[float] = Field(default_factory=list)
    finished_requests: List[VLLMFinishedRequestStats] = Field(default_factory=list)

    source: str = "vllm_stat_logger"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def record_from_stats(scheduler_stats: Any, iteration_stats: Any, engine_idx: int, step_seq: int,
                      clock_domain: Optional[str]) -> VLLMIterationRecord:
    """Build a record from vLLM's stats objects (duck-typed; missing fields stay None)."""
    pcs = _get(scheduler_stats, "prefix_cache_stats")
    finished = []
    for f in _get(iteration_stats, "finished_requests", []) or []:
        fr = _get(f, "finish_reason")
        finished.append(VLLMFinishedRequestStats(
            finish_reason=str(fr) if fr is not None else None,
            e2e_latency_s=_get(f, "e2e_latency"), num_prompt_tokens=_get(f, "num_prompt_tokens"),
            num_generation_tokens=_get(f, "num_generation_tokens"), max_tokens_param=_get(f, "max_tokens_param"),
            queued_time_s=_get(f, "queued_time"), prefill_time_s=_get(f, "prefill_time"),
            inference_time_s=_get(f, "inference_time"), decode_time_s=_get(f, "decode_time"),
            mean_time_per_output_token_s=_get(f, "mean_time_per_output_token"),
        ))
    return VLLMIterationRecord(
        timestamp=time.time(), monotonic=time.monotonic(), clock_domain=clock_domain, engine_idx=engine_idx, step_seq=step_seq,
        num_running_reqs=_get(scheduler_stats, "num_running_reqs"), num_waiting_reqs=_get(scheduler_stats, "num_waiting_reqs"),
        kv_cache_usage=_get(scheduler_stats, "kv_cache_usage"),
        prefix_cache_requests=_get(pcs, "requests"), prefix_cache_queries=_get(pcs, "queries"), prefix_cache_hits=_get(pcs, "hits"),
        num_corrupted_reqs=_get(scheduler_stats, "num_corrupted_reqs"),
        num_generation_tokens=_get(iteration_stats, "num_generation_tokens"), num_prompt_tokens=_get(iteration_stats, "num_prompt_tokens"),
        num_preempted_reqs=_get(iteration_stats, "num_preempted_reqs"),
        time_to_first_tokens_s=list(_get(iteration_stats, "time_to_first_tokens_iter", []) or []),
        inter_token_latencies_s=list(_get(iteration_stats, "inter_token_latencies_iter", []) or []),
        finished_requests=finished,
    )


class VLLMStatsSink:
    """Thread-safe bounded buffer shared by the tracer and the loggers it creates."""

    def __init__(self, max_buffered: int = 10_000, clock_domain: Optional[str] = None):
        self.max_buffered = max_buffered
        self.clock_domain = clock_domain or uuid.uuid4().hex[:12]
        self._lock = threading.Lock()
        self._records: List[VLLMIterationRecord] = []
        self.dropped = 0
        self.errors = 0
        self.last_error: Optional[str] = None
        self.loggers_created = 0
        self.engine_initialized: List[int] = []

    def push(self, rec: VLLMIterationRecord) -> None:
        with self._lock:
            if len(self._records) >= self.max_buffered:
                self._records.pop(0)
                self.dropped += 1
            self._records.append(rec)

    def drain(self) -> List[VLLMIterationRecord]:
        with self._lock:
            out = self._records
            self._records = []
        return out

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            buffered = len(self._records)
        return {"loggers_created": self.loggers_created, "engine_initialized": list(self.engine_initialized),
                "buffered": buffered, "dropped": self.dropped, "errors": self.errors, "last_error": self.last_error}


class LLMTraceStatLogger:
    """A ``StatLoggerBase``-shaped logger that forwards vLLM stats to a ``VLLMStatsSink``.

    Bookkeeping errors are counted on the sink and never raised into the engine.
    """

    def __init__(self, sink: VLLMStatsSink, vllm_config: Any = None, engine_index: int = 0):
        self.sink = sink
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        self._seq = 0
        sink.loggers_created += 1

    def record(self, scheduler_stats: Any, iteration_stats: Any, engine_idx: int = 0) -> None:
        try:
            self._seq += 1
            self.sink.push(record_from_stats(scheduler_stats, iteration_stats, engine_idx, self._seq, self.sink.clock_domain))
        except Exception as exc:
            self.sink.errors += 1
            self.sink.last_error = f"{type(exc).__name__}: {exc}"
            logger.error("vLLM stat logger error: %s", exc, exc_info=True)

    def log_engine_initialized(self) -> None:
        self.sink.engine_initialized.append(self.engine_index)

    def log(self) -> None:
        pass


def make_stat_logger_factory(sink: VLLMStatsSink) -> Callable[[Any, int], LLMTraceStatLogger]:
    """A ``StatLoggerFactory`` for ``LLMEngine.from_engine_args(..., stat_loggers=[factory])``."""

    def factory(vllm_config: Any, engine_idx: int = 0) -> LLMTraceStatLogger:
        return LLMTraceStatLogger(sink, vllm_config, engine_idx)

    return factory


def attach_to_engine(engine: Any, sink: VLLMStatsSink) -> Optional[str]:
    """Best-effort post-hoc attach to an already constructed ``LLMEngine``.

    Verified structure (vLLM 0.11.0): ``engine.logger_manager`` is a
    ``StatLoggerManager`` (``None`` when ``disable_log_stats=True``) whose
    ``per_engine_logger_dict`` maps engine index -> list of loggers, and whose
    ``record()`` iterates that list. Returns None on success, else a reason.
    Prefer passing ``make_stat_logger_factory(sink)`` at engine construction.
    """
    manager = getattr(engine, "logger_manager", None)
    if manager is None:
        return "engine has no logger_manager (vLLM stats logging disabled: disable_log_stats=True?)"
    per_engine = getattr(manager, "per_engine_logger_dict", None)
    if not isinstance(per_engine, dict):
        return f"unsupported StatLoggerManager layout ({type(manager).__name__} has no per_engine_logger_dict)"
    vllm_config = getattr(engine, "vllm_config", None)
    attached = 0
    for idx, loggers in per_engine.items():
        if isinstance(loggers, list):
            if any(isinstance(lg, LLMTraceStatLogger) for lg in loggers):
                return "an llmtrace stat logger is already attached"
            loggers.append(LLMTraceStatLogger(sink, vllm_config, int(idx)))
            attached += 1
    return None if attached else "no per-engine logger lists found"


def detach_from_engine(engine: Any) -> int:
    """Remove llmtrace loggers attached post-hoc. Returns how many were removed."""
    manager = getattr(engine, "logger_manager", None)
    per_engine = getattr(manager, "per_engine_logger_dict", None) if manager is not None else None
    removed = 0
    if isinstance(per_engine, dict):
        for loggers in per_engine.values():
            if isinstance(loggers, list):
                keep = [lg for lg in loggers if not isinstance(lg, LLMTraceStatLogger)]
                removed += len(loggers) - len(keep)
                loggers[:] = keep
    return removed
