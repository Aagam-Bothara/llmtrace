"""Per-step GPU span timing with CUDA events, and optional NVTX ranges.

What is measured
----------------
Around every ``model_executor.execute_model()`` call (vLLM 0.11.0, in-process
engine core only) two CUDA events are recorded on the calling thread's current
stream. Their elapsed time is the **GPU span** of the step: the time between
the two event records as observed on that stream. It includes any idle gaps on
the stream between kernel launches, so it is an upper bound on GPU busy time,
not a busy-time measurement (that needs CUPTI). Work enqueued on other streams
(vLLM's async output copy stream, the communication stream) is not bracketed.

``host_overhead_ms = host_step_ms - gpu_span_ms`` is the part of the engine
step the GPU was not spanned by: scheduling, input preparation before the
first launch, output processing after the last sync, and llmtrace itself.

Reading elapsed times never synchronizes: events are polled with ``query()``
on later steps and at collection time, and resolved records are emitted then.
Pending events are bounded; drops are counted.

NVTX
----
With ``enable_nvtx`` an ``llmtrace step <n> tokens=<t>`` range brackets each
engine step so Nsight Systems shows llmtrace's steps next to the kernels. It is
off by default because it only matters under a profiler.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class StepGpuTiming(BaseModel):
    """GPU span of one engine step, resolved from CUDA events after the fact."""

    step_index: int
    timestamp: float  # wall clock at step start
    monotonic: Optional[float] = None  # step start (engine thread monotonic clock)
    clock_domain: Optional[str] = None
    host_step_ms: float  # wrapper-measured step wall time (schedule + execute + output processing)
    gpu_span_ms: Optional[float] = None  # elapsed between start/end events on the executor stream
    host_overhead_ms: Optional[float] = None  # host_step_ms - gpu_span_ms
    executor_calls: int = 1  # execute_model calls seen in the step
    resolved_after_steps: int = 0  # how many later steps passed before the events were ready
    source: str = "cuda_events"


class CudaEventBackend(Protocol):
    def available(self) -> Optional[str]:
        """None if usable, else a reason."""

    def record(self) -> Any:
        """Record an event on the current stream and return a handle."""

    def elapsed_ms(self, start: Any, end: Any) -> Optional[float]:
        """Elapsed ms if both events completed (never blocks), else None."""

    def nvtx_push(self, label: str) -> None: ...

    def nvtx_pop(self) -> None: ...


class TorchCudaBackend:
    """Backend on ``torch.cuda.Event(enable_timing=True)``; imports torch lazily."""

    def __init__(self) -> None:
        self._torch: Any = None
        self._reason: Optional[str] = None

    def available(self) -> Optional[str]:
        if self._torch is not None:
            return None
        try:
            import torch  # type: ignore
        except ImportError:
            self._reason = "torch not installed"
            return self._reason
        if not torch.cuda.is_available():
            self._reason = "torch.cuda not available"
            return self._reason
        self._torch = torch
        return None

    def record(self) -> Any:
        ev = self._torch.cuda.Event(enable_timing=True)
        ev.record()  # current stream of the calling thread
        return ev

    def elapsed_ms(self, start: Any, end: Any) -> Optional[float]:
        if not end.query() or not start.query():
            return None
        return float(start.elapsed_time(end))

    def nvtx_push(self, label: str) -> None:
        try:
            self._torch.cuda.nvtx.range_push(label)
        except Exception:
            pass

    def nvtx_pop(self) -> None:
        try:
            self._torch.cuda.nvtx.range_pop()
        except Exception:
            pass


class _Pending:
    __slots__ = ("step_index", "timestamp", "monotonic", "host_step_ms", "starts", "ends", "seen_steps")

    def __init__(self, step_index: int, timestamp: float, monotonic: Optional[float]) -> None:
        self.step_index = step_index
        self.timestamp = timestamp
        self.monotonic = monotonic
        self.host_step_ms: Optional[float] = None
        self.starts: List[Any] = []
        self.ends: List[Any] = []
        self.seen_steps = 0


class CudaStepTimer:
    """Collects per-step GPU spans without ever synchronizing the engine thread."""

    def __init__(self, backend: Optional[CudaEventBackend] = None, max_pending: int = 1024,
                 enable_nvtx: bool = False, clock_domain: Optional[str] = None):
        self.backend: CudaEventBackend = backend or TorchCudaBackend()
        self.max_pending = max_pending
        self.enable_nvtx = enable_nvtx
        self.clock_domain = clock_domain
        self.unavailable_reason: Optional[str] = "not started"
        self._lock = threading.Lock()
        self._pending: List[_Pending] = []
        self._current: Optional[_Pending] = None
        self._resolved: List[StepGpuTiming] = []
        self.dropped = 0
        self.errors = 0
        self.last_error: Optional[str] = None
        self.resolved_count = 0

    def start(self) -> bool:
        self.unavailable_reason = self.backend.available()
        return self.unavailable_reason is None

    # --- called by the instrumentation on the engine thread

    def begin_step(self, step_index: int, timestamp: float, monotonic: Optional[float], label: str = "") -> None:
        if self.unavailable_reason:
            return
        self._current = _Pending(step_index, timestamp, monotonic)
        if self.enable_nvtx:
            self.backend.nvtx_push(label or f"llmtrace step {step_index}")

    def before_execute(self) -> None:
        if self.unavailable_reason or self._current is None:
            return
        try:
            self._current.starts.append(self.backend.record())
        except Exception as exc:
            self._fail(exc)

    def after_execute(self) -> None:
        if self.unavailable_reason or self._current is None or len(self._current.ends) >= len(self._current.starts):
            return
        try:
            self._current.ends.append(self.backend.record())
        except Exception as exc:
            self._fail(exc)

    def end_step(self, host_step_ms: float) -> None:
        if self.unavailable_reason:
            return
        if self.enable_nvtx:
            self.backend.nvtx_pop()
        cur, self._current = self._current, None
        if cur is None:
            return
        cur.host_step_ms = host_step_ms
        with self._lock:
            for p in self._pending:  # steps that were already waiting have now seen one more step
                p.seen_steps += 1
            if cur.starts and len(cur.starts) == len(cur.ends):
                if len(self._pending) >= self.max_pending:
                    self._pending.pop(0)
                    self.dropped += 1
                self._pending.append(cur)
            else:
                # step without an executor call (e.g. dummy batch) or unmatched events: host-only record
                self._resolved.append(StepGpuTiming(step_index=cur.step_index, timestamp=cur.timestamp, monotonic=cur.monotonic,
                                                    clock_domain=self.clock_domain, host_step_ms=host_step_ms,
                                                    executor_calls=len(cur.starts), source="no_executor_call"))
        self.poll()

    # --- resolution (engine thread at step end, or collector thread)

    def poll(self) -> int:
        """Resolve pending steps whose events completed. Never blocks. Returns how many resolved."""
        if self.unavailable_reason:
            return 0
        n = 0
        with self._lock:
            keep: List[_Pending] = []
            for p in self._pending:
                try:
                    spans = [self.backend.elapsed_ms(s, e) for s, e in zip(p.starts, p.ends)]
                except Exception as exc:
                    self._fail(exc)
                    continue
                if any(v is None for v in spans):
                    keep.append(p)
                    continue
                total = float(sum(spans))  # type: ignore[arg-type]
                self._resolved.append(StepGpuTiming(
                    step_index=p.step_index, timestamp=p.timestamp, monotonic=p.monotonic, clock_domain=self.clock_domain,
                    host_step_ms=p.host_step_ms or 0.0, gpu_span_ms=total,
                    host_overhead_ms=(p.host_step_ms - total) if p.host_step_ms is not None else None,
                    executor_calls=len(p.starts), resolved_after_steps=p.seen_steps))
                n += 1
            self._pending = keep
            self.resolved_count += n
        return n

    def drain(self) -> List[StepGpuTiming]:
        self.poll()
        with self._lock:
            out = self._resolved
            self._resolved = []
        return out

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"available": self.unavailable_reason is None, "unavailable_reason": self.unavailable_reason,
                    "resolved": self.resolved_count, "pending": len(self._pending), "buffered": len(self._resolved),
                    "dropped": self.dropped, "errors": self.errors, "last_error": self.last_error, "nvtx": self.enable_nvtx}

    def _fail(self, exc: Exception) -> None:
        self.errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        logger.error("CUDA step timing error: %s", exc, exc_info=True)


class FakeCudaBackend:
    """Test backend: elapsed = ms_per_call (+ ms_per_token * tokens set by the test); events become
    ready after ``ready_after_polls`` queries so lazy resolution is exercised."""

    def __init__(self, ms_per_call: float = 1.0, ready_after_polls: int = 0, fail_available: Optional[str] = None):
        self.ms_per_call = ms_per_call
        self.ready_after_polls = ready_after_polls
        self.fail_available = fail_available
        self._t = 0.0
        self.nvtx: List[str] = []
        self.next_span_ms: Optional[float] = None

    def available(self) -> Optional[str]:
        return self.fail_available

    def record(self) -> Any:
        self._t += (self.next_span_ms if self.next_span_ms is not None else self.ms_per_call)
        return {"t": self._t, "polls": 0, "wall": time.monotonic()}

    def elapsed_ms(self, start: Any, end: Any) -> Optional[float]:
        end["polls"] += 1
        if end["polls"] <= self.ready_after_polls:
            return None
        return end["t"] - start["t"]

    def nvtx_push(self, label: str) -> None:
        self.nvtx.append("push:" + label)

    def nvtx_pop(self) -> None:
        self.nvtx.append("pop")
