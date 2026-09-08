"""Instrumentation for vLLM's ``AsyncLLM`` (the engine behind the OpenAI-compatible server).

Target (vLLM 0.11.0, ``vllm/v1/engine/async_llm.py``, verified from source):

* ``AsyncLLM.generate(prompt, sampling_params, request_id, lora_request=None,
  trace_headers=None, priority=0, data_parallel_rank=None) -> AsyncGenerator[RequestOutput, None]``
  which adds the request and yields outputs; on ``asyncio.CancelledError`` or
  ``GeneratorExit`` it aborts the request itself.
* ``AsyncLLM.abort(request_id: str | Iterable[str]) -> None`` (coroutine).
* ``AsyncLLM.from_engine_args(..., stat_loggers=...)`` and ``AsyncLLM.logger_manager``
  for vLLM's own per-step stats (see ``vllm_stats.py``).

What is and is not observable here: the engine core always runs out of
process for ``AsyncLLM``, so there is no scheduler or executor access: no batch
membership, no queue/prefill boundary, no GPU spans. Request-level traces
(arrival at ``generate()``, first visible token, token counts, completion,
client cancellation, abort) and vLLM's per-step stats are recorded, and every
missing quantity is reported as unavailable with that reason.

The wrapper is itself an async generator, so ``aclose()``/cancellation and
exceptions reach the original generator unchanged, and the original's return
values are forwarded as-is.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any, Callable, Dict, Optional

from llmtrace.data_plane.vllm_instrumentation import (
    InstrumentationError,
    VLLMInstrumentation,
    _Active,
    _output_kind_from_params,
    _prompt_token_ids,
    _StepContext,
)
from llmtrace.models.trace import RequestStatus, RequestTrace

logger = logging.getLogger(__name__)

ASYNC_UNAVAILABLE = "AsyncLLM: engine core runs in another process; no scheduler or executor access"


class AsyncLLMInstrumentation(VLLMInstrumentation):
    """Wraps ``AsyncLLM.generate`` and ``AsyncLLM.abort``; reuses the sync accounting."""

    def instrument_engine(self, engine: Any) -> None:
        if self._engine is not None:
            if self._engine is engine:
                logger.warning("AsyncLLM already instrumented; ignoring repeated call")
                return
            raise InstrumentationError("Already instrumenting another engine; call uninstrument_engine() first")
        self._check_version()
        gen = getattr(engine, "generate", None)
        abort = getattr(engine, "abort", None)
        if gen is None or not inspect.isasyncgenfunction(gen):
            raise InstrumentationError("Engine 'generate' is not an async generator function; expected vLLM AsyncLLM")
        if abort is None or not inspect.iscoroutinefunction(abort):
            raise InstrumentationError("Engine 'abort' is not a coroutine function; expected vLLM AsyncLLM")
        for attr in (gen, abort):
            if getattr(attr, "__llmtrace_wrapped__", False):
                raise InstrumentationError("AsyncLLM is already wrapped by another llmtrace instance")
        self._engine = engine
        self._model_name = self._resolve_model_name(engine)
        try:
            self._patch(engine, "generate", self._wrap_generate)
            self._patch(engine, "abort", self._wrap_async_abort)
        except Exception:
            self._restore_all()
            self._engine = None
            raise
        self.scheduler_visible = False
        self.scheduler_unavailable_reason = ASYNC_UNAVAILABLE
        self.executor_visible = False
        self.executor_unavailable_reason = ASYNC_UNAVAILABLE
        logger.info("Instrumented vLLM AsyncLLM (model=%s); request-level tracing only", self._model_name)

    # ------------------------------------------------------------------ wrappers

    def _wrap_generate(self, original: Callable) -> Callable:
        instr = self

        @functools.wraps(original)
        async def wrapper(*args: Any, **kwargs: Any):
            arrival_m, arrival_w = instr._monotonic(), instr._wall()
            request_id = instr._arg(args, kwargs, 2, "request_id")
            prompt = instr._arg(args, kwargs, 0, "prompt")
            params = instr._arg(args, kwargs, 1, "sampling_params")
            rid = str(request_id) if request_id is not None else None
            if rid is not None:
                instr._guard(instr._on_async_added, rid, prompt, params, arrival_m, arrival_w)
            agen = original(*args, **kwargs)
            exhausted = False
            try:
                async for out in agen:
                    if rid is not None:
                        instr._guard(instr._on_async_output, rid, out)
                    yield out
                exhausted = True
                if rid is not None:
                    instr._guard(instr._finish_async, rid, RequestStatus.COMPLETED, None)
            except (asyncio.CancelledError, GeneratorExit):
                # The consumer cancelled or closed the stream. AsyncLLM aborts the request itself, usually
                # before this handler runs (its own except runs first), so the trace may already be final.
                if rid is not None:
                    instr._guard(instr._finish_async, rid, RequestStatus.ABORTED, "abort", "client_cancelled")
                raise
            except BaseException as exc:
                if rid is not None:
                    instr._guard(instr._finish_async, rid, RequestStatus.INCOMPLETE, f"error:{type(exc).__name__}")
                raise
            finally:
                if not exhausted:
                    try:
                        await agen.aclose()
                    except BaseException:  # pragma: no cover - the original's own cleanup errors are not ours
                        pass

        return wrapper

    def _wrap_async_abort(self, original: Callable) -> Callable:
        @functools.wraps(original)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await original(*args, **kwargs)
            request_ids = args[0] if args else kwargs.get("request_id")
            self._guard(self._on_abort, request_ids)
            return result

        return wrapper

    @staticmethod
    def _arg(args: tuple, kwargs: Dict[str, Any], index: int, name: str) -> Any:
        return args[index] if len(args) > index else kwargs.get(name)

    # --------------------------------------------------------------- bookkeeping

    def _on_async_added(self, request_id: str, prompt: Any, params: Any, arrival_m: float, arrival_w: float) -> None:
        caller_ids = _prompt_token_ids(prompt)
        trace = RequestTrace(
            request_id=request_id, start_time=arrival_w, end_time=arrival_w, start_monotonic=arrival_m, end_monotonic=arrival_m,
            clock_domain=self.clock_domain,
            prompt_length=len(caller_ids) if caller_ids is not None else None,
            prompt_length_source="caller_token_ids" if caller_ids is not None else "unavailable",
            output_kind=_output_kind_from_params(params), num_sequences=int(getattr(params, "n", 1) or 1),
            model_name=self._model_name, scheduler_visible=False,
            metadata={"engine": "AsyncLLM", "granularity": "output_visible_to_consumer"},
        )
        with self._lock:
            self._active[request_id] = _Active(trace, arrival_m, arrival_w, trace.output_kind)

    def _on_async_output(self, request_id: str, output: Any) -> None:
        now_m, now_w = self._monotonic(), self._wall()
        with self._lock:
            active = self._active.get(request_id)
            if active is None:
                return
            # No engine steps are visible: each yielded output is the observation point.
            self._apply_output(active, output, _StepContext(0, now_m, now_w), now_m)
            if getattr(output, "finished", False):
                reason = self._finish_reason(output)
                status = RequestStatus.ABORTED if reason == "abort" else RequestStatus.COMPLETED
                self._push_completed(self._finalize(active, now_m, now_w, status, reason))
                del self._active[request_id]

    def _finish_async(self, request_id: str, status: RequestStatus, reason: Optional[str],
                      abort_cause: Optional[str] = None) -> None:
        now_m, now_w = self._monotonic(), self._wall()
        with self._lock:
            active = self._active.pop(request_id, None)
            if active is not None:
                t = self._finalize(active, now_m, now_w, status, reason)
                if abort_cause:
                    t.metadata["abort_cause"] = abort_cause
                self._push_completed(t)
            elif abort_cause:
                # Already finalized by the abort wrapper during the engine's own cancellation cleanup:
                # annotate the cause on that record (search from the newest).
                for t in reversed(self._completed):
                    if t.request_id == request_id:
                        if t.status == RequestStatus.ABORTED:
                            t.metadata.setdefault("abort_cause", abort_cause)
                        break

    def health(self) -> Dict[str, Any]:
        h = super().health()
        h["engine_kind"] = "AsyncLLM"
        return h


def is_async_llm(engine: Any) -> bool:
    return inspect.isasyncgenfunction(getattr(engine, "generate", None)) and inspect.iscoroutinefunction(getattr(engine, "abort", None))


__all__ = ["AsyncLLMInstrumentation", "is_async_llm", "ASYNC_UNAVAILABLE"]
