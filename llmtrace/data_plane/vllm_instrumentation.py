"""vLLM instrumentation for request lifecycle tracing.

Target
------
This module targets exactly one engine API, verified against the vLLM source
at tag ``v0.11.0`` (``vllm/v1/engine/llm_engine.py``):

* ``LLMEngine.add_request(request_id, prompt, params, arrival_time=None, ...) -> None``
* ``LLMEngine.step() -> list[RequestOutput] | list[PoolingRequestOutput]``
* ``LLMEngine.abort_request(request_ids: list[str]) -> None``

All three are synchronous. This is the engine used by ``vllm.LLM`` (offline
inference). ``AsyncLLM`` / the OpenAI server expose a different API and are
not supported in this phase; instrumenting a coroutine function raises
``InstrumentationError``.

Scheduler visibility
--------------------
``LLMEngine`` has no ``scheduler`` attribute. The V1 scheduler lives in the
engine core, which by default runs in a separate process
(``VLLM_ENABLE_V1_MULTIPROCESSING=1``). It is only reachable in-process, via
``engine.engine_core.engine_core.scheduler`` (``InprocClient``), when the
engine is started with ``VLLM_ENABLE_V1_MULTIPROCESSING=0``. When reachable,
``Scheduler.schedule() -> SchedulerOutput`` is wrapped to record batch
membership (``SchedulerOutput.num_scheduled_tokens: dict[str, int]``) and
``KVCacheManager.usage`` (a fraction). When not reachable, batch metadata and
queue/prefill boundaries are reported as unavailable rather than inferred.

Output semantics (verified in ``vllm/v1/engine/output_processor.py``):
``CompletionOutput.token_ids`` is cumulative unless the request's
``SamplingParams.output_kind`` is ``DELTA`` (only new tokens) or ``FINAL_ONLY``
(a single output at completion). ``RequestOutput.metrics`` is not populated by
the V1 engine, so timing comes from step boundaries observed here.

Safety
------
Wrappers never change the wrapped method's return value or exceptions. Any
failure inside llmtrace bookkeeping is counted, logged, and (optionally, with
``strict=True``) re-raised *after* the engine call has completed and its
result captured; the engine result is still returned.
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from llmtrace.models.trace import (
    BatchMetadata,
    RequestSpan,
    RequestStatus,
    RequestTrace,
    SpanPhase,
)

logger = logging.getLogger(__name__)

TARGET_VLLM_VERSION = "0.11.0"


class InstrumentationError(RuntimeError):
    """Raised when the engine cannot be instrumented safely."""


@dataclass
class _Patch:
    target: Any
    name: str
    original: Any  # value found in the instance __dict__, or None if inherited
    had_instance_attr: bool


@dataclass
class _StepContext:
    index: int
    start_monotonic: float
    start_wall: float
    batches: List[BatchMetadata] = field(default_factory=list)


@dataclass
class _Active:
    trace: RequestTrace
    arrival_monotonic: float
    arrival_wall: float
    output_kind: str
    tokens_per_seq: Dict[int, int] = field(default_factory=dict)
    first_scheduled_monotonic: Optional[float] = None
    first_token_monotonic: Optional[float] = None
    tokens_at_first_observation: int = 0
    last_token_monotonic: Optional[float] = None
    last_seen_monotonic: Optional[float] = None


def _output_kind_from_params(params: Any) -> str:
    kind = getattr(params, "output_kind", None)
    if kind is None:
        # PoolingParams has no output_kind.
        return "pooling" if type(params).__name__ == "PoolingParams" else "unknown"
    name = getattr(kind, "name", str(kind)).lower()
    if name in ("cumulative", "delta", "final_only"):
        return name
    return "unknown"


def _prompt_token_ids(prompt: Any) -> Optional[List[int]]:
    if isinstance(prompt, dict):
        ids = prompt.get("prompt_token_ids")
        if isinstance(ids, (list, tuple)):
            return list(ids)
    return None


class VLLMInstrumentation:
    """Instruments a vLLM ``LLMEngine`` (V1, vLLM 0.11.0) to collect request traces."""

    def __init__(
        self,
        enable_batch_metadata: bool = True,
        max_buffered: int = 10_000,
        strict: bool = False,
        clock_domain: Optional[str] = None,
        monotonic: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ):
        self.enable_batch_metadata = enable_batch_metadata
        self.max_buffered = max_buffered
        self.strict = strict
        self.clock_domain = clock_domain or uuid.uuid4().hex[:12]
        self._monotonic = monotonic
        self._wall = wall

        self._lock = threading.Lock()
        self._active: Dict[str, _Active] = {}
        self._completed: List[RequestTrace] = []
        self._batches: List[BatchMetadata] = []
        self._dropped_traces = 0
        self._dropped_batches = 0
        self._errors = 0
        self._last_error: Optional[str] = None

        self._engine: Optional[Any] = None
        self._patches: List[_Patch] = []
        self._model_name = "unknown"
        self._step_index = 0
        self._batch_seq = 0
        self._current_step: Optional[_StepContext] = None
        self._scheduler: Optional[Any] = None
        self.scheduler_visible = False
        self.scheduler_unavailable_reason: Optional[str] = "not instrumented"
        self.vllm_version: Optional[str] = None

    # ------------------------------------------------------------------ setup

    @property
    def is_instrumented(self) -> bool:
        return self._engine is not None

    def instrument_engine(self, engine: Any) -> None:
        """Patch ``add_request``, ``step`` and ``abort_request`` on ``engine``.

        Calling twice with the same engine is a no-op; with a different engine
        while instrumented raises ``InstrumentationError``.
        """
        if self._engine is not None:
            if self._engine is engine:
                logger.warning("Engine already instrumented; ignoring repeated call")
                return
            raise InstrumentationError(
                "Already instrumenting another engine; call uninstrument_engine() first"
            )

        self._check_version()
        for name in ("add_request", "step", "abort_request"):
            attr = getattr(engine, name, None)
            if attr is None or not callable(attr):
                raise InstrumentationError(f"Engine has no callable '{name}'; expected vLLM LLMEngine")
            if inspect.iscoroutinefunction(attr):
                raise InstrumentationError(
                    f"Engine '{name}' is a coroutine function. Only the synchronous vLLM "
                    f"LLMEngine (vllm.LLM) is supported; AsyncLLM is not."
                )
            if getattr(attr, "__llmtrace_wrapped__", False):
                raise InstrumentationError(
                    f"Engine '{name}' is already wrapped by another llmtrace instance"
                )

        self._engine = engine
        self._model_name = self._resolve_model_name(engine)

        try:
            self._patch(engine, "add_request", self._wrap_add_request)
            self._patch(engine, "step", self._wrap_step)
            self._patch(engine, "abort_request", self._wrap_abort)
            if self.enable_batch_metadata:
                self._attach_scheduler(engine)
            else:
                self.scheduler_unavailable_reason = "batch metadata disabled by config"
        except Exception:
            self._restore_all()
            self._engine = None
            raise

        logger.info(
            "Instrumented vLLM LLMEngine (model=%s, scheduler_visible=%s%s)",
            self._model_name,
            self.scheduler_visible,
            "" if self.scheduler_visible else f": {self.scheduler_unavailable_reason}",
        )

    def uninstrument_engine(self) -> List[RequestTrace]:
        """Restore original methods. Idempotent.

        Returns traces for requests still active at this moment, marked
        ``INCOMPLETE``; they are removed from the active set so nothing leaks.
        """
        if self._engine is None:
            return []
        self._restore_all()
        self._engine = None
        self._scheduler = None
        self._current_step = None
        now_m, now_w = self._monotonic(), self._wall()
        with self._lock:
            leftovers = [
                self._finalize(a, now_m, now_w, RequestStatus.INCOMPLETE, "tracer_stopped")
                for a in list(self._active.values())
            ]
            self._active.clear()
        logger.info("Restored vLLM engine methods (%d incomplete requests)", len(leftovers))
        return leftovers

    def _check_version(self) -> None:
        try:
            import vllm  # type: ignore

            self.vllm_version = getattr(vllm, "__version__", None)
        except Exception:
            self.vllm_version = None
            return
        if self.vllm_version != TARGET_VLLM_VERSION:
            logger.warning(
                "llmtrace was verified against vLLM %s but found %s; interfaces are unverified",
                TARGET_VLLM_VERSION,
                self.vllm_version,
            )

    @staticmethod
    def _resolve_model_name(engine: Any) -> str:
        for path in (("model_config", "model"), ("vllm_config", "model_config", "model")):
            obj = engine
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if isinstance(obj, str):
                return obj
        return "unknown"

    def _attach_scheduler(self, engine: Any) -> None:
        # Verified path for vLLM 0.11.0 with VLLM_ENABLE_V1_MULTIPROCESSING=0:
        # LLMEngine.engine_core is an InprocClient holding EngineCore in .engine_core,
        # and EngineCore.scheduler is the V1 Scheduler.
        client = getattr(engine, "engine_core", None)
        core = getattr(client, "engine_core", None) if client is not None else None
        scheduler = getattr(core, "scheduler", None) if core is not None else None
        if scheduler is None:
            client_name = type(client).__name__ if client is not None else "None"
            self.scheduler_unavailable_reason = (
                f"scheduler not reachable in-process (engine_core client is {client_name}; "
                "start vLLM with VLLM_ENABLE_V1_MULTIPROCESSING=0 for batch metadata)"
            )
            return
        schedule = getattr(scheduler, "schedule", None)
        if schedule is None or not callable(schedule):
            self.scheduler_unavailable_reason = "scheduler has no callable schedule()"
            return
        if getattr(schedule, "__llmtrace_wrapped__", False):
            raise InstrumentationError("scheduler.schedule is already wrapped by llmtrace")
        self._patch(scheduler, "schedule", self._wrap_schedule)
        self._scheduler = scheduler
        self.scheduler_visible = True
        self.scheduler_unavailable_reason = None

    def _patch(self, target: Any, name: str, factory: Callable[[Callable], Callable]) -> None:
        had_instance_attr = name in getattr(target, "__dict__", {})
        original_instance_value = target.__dict__[name] if had_instance_attr else None
        bound_original = getattr(target, name)
        wrapper = factory(bound_original)
        wrapper.__llmtrace_wrapped__ = True  # type: ignore[attr-defined]
        setattr(target, name, wrapper)
        self._patches.append(_Patch(target, name, original_instance_value, had_instance_attr))

    def _restore_all(self) -> None:
        for p in reversed(self._patches):
            try:
                if p.had_instance_attr:
                    setattr(p.target, p.name, p.original)
                else:
                    # Original was a class attribute; removing the instance override
                    # restores the bound method exactly.
                    try:
                        delattr(p.target, p.name)
                    except AttributeError:
                        pass
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to restore %s.%s: %s", type(p.target).__name__, p.name, exc)
        self._patches.clear()
        self.scheduler_visible = False
        self.scheduler_unavailable_reason = "not instrumented"

    # --------------------------------------------------------------- wrappers

    def _guard(self, fn: Callable, *args: Any) -> None:
        """Run bookkeeping; never let it alter the engine call. Re-raise only if strict."""
        try:
            fn(*args)
        except Exception as exc:
            self._errors += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.error("llmtrace instrumentation error in %s: %s", fn.__name__, exc, exc_info=True)
            if self.strict:
                raise

    def _wrap_add_request(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            arrival_m, arrival_w = self._monotonic(), self._wall()
            result = original(*args, **kwargs)  # exceptions propagate untouched
            self._guard(self._on_request_added, args, kwargs, arrival_m, arrival_w)
            return result

        return wrapper

    def _wrap_step(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self._step_index += 1
            ctx = _StepContext(self._step_index, self._monotonic(), self._wall())
            self._current_step = ctx
            try:
                outputs = original(*args, **kwargs)
            except BaseException:
                self._current_step = None
                with self._lock:
                    self._publish_batches(ctx, None)  # no end time: the step did not complete
                raise  # engine failure surfaces unchanged; requests stay active until abort/stop
            end_m = self._monotonic()
            self._current_step = None
            self._guard(self._on_step_completed, outputs, ctx, end_m)
            return outputs

        return wrapper

    def _wrap_abort(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            request_ids = args[0] if args else kwargs.get("request_ids")
            self._guard(self._on_abort, request_ids)
            return result

        return wrapper

    def _wrap_schedule(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self._guard(self._on_batch_scheduled, result, self._monotonic(), self._wall())
            return result

        return wrapper

    # ------------------------------------------------------------- bookkeeping

    def _on_request_added(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any], arrival_m: float, arrival_w: float
    ) -> None:
        request_id = args[0] if len(args) > 0 else kwargs.get("request_id")
        prompt = args[1] if len(args) > 1 else kwargs.get("prompt")
        params = args[2] if len(args) > 2 else kwargs.get("params")
        if request_id is None:
            raise ValueError("add_request called without request_id")
        request_id = str(request_id)

        caller_ids = _prompt_token_ids(prompt)
        trace = RequestTrace(
            request_id=request_id,
            start_time=arrival_w,
            end_time=arrival_w,
            start_monotonic=arrival_m,
            end_monotonic=arrival_m,
            clock_domain=self.clock_domain,
            prompt_length=len(caller_ids) if caller_ids is not None else None,
            prompt_length_source="caller_token_ids" if caller_ids is not None else "unavailable",
            output_kind=_output_kind_from_params(params),
            num_sequences=int(getattr(params, "n", 1) or 1),
            model_name=self._model_name,
            scheduler_visible=self.scheduler_visible,
        )
        active = _Active(trace, arrival_m, arrival_w, trace.output_kind)
        with self._lock:
            if request_id in self._active:
                logger.warning("Duplicate request_id %s; replacing active entry", request_id)
            self._active[request_id] = active

    def _publish_batches(self, ctx: _StepContext, end_m: Optional[float]) -> None:
        """Stamp step end and make the step's batches drainable. Caller holds the lock."""
        for meta in ctx.batches:
            meta.step_end_monotonic = end_m
            if len(self._batches) >= self.max_buffered:
                self._batches.pop(0)
                self._dropped_batches += 1
            self._batches.append(meta)

    def _on_step_completed(self, outputs: Any, ctx: _StepContext, end_m: float) -> None:
        if not outputs:
            with self._lock:
                self._publish_batches(ctx, end_m)
            return
        end_w = ctx.start_wall + (end_m - ctx.start_monotonic)
        finished: List[RequestTrace] = []
        with self._lock:
            self._publish_batches(ctx, end_m)
            for output in outputs:
                request_id = str(getattr(output, "request_id", ""))
                active = self._active.get(request_id)
                if active is None:
                    continue
                self._apply_output(active, output, ctx, end_m)
                if getattr(output, "finished", False):
                    reason = self._finish_reason(output)
                    status = RequestStatus.ABORTED if reason == "abort" else RequestStatus.COMPLETED
                    finished.append(self._finalize(active, end_m, end_w, status, reason))
                    del self._active[request_id]
            for t in finished:
                self._push_completed(t)

    def _apply_output(self, active: _Active, output: Any, ctx: _StepContext, end_m: float) -> None:
        trace = active.trace
        active.last_seen_monotonic = end_m
        for meta in ctx.batches:
            if meta.batch_id not in trace.batch_ids:
                trace.batch_ids.append(meta.batch_id)

        engine_prompt_ids = getattr(output, "prompt_token_ids", None)
        if isinstance(engine_prompt_ids, (list, tuple)) and trace.prompt_length_source != "engine_prompt_token_ids":
            trace.prompt_length = len(engine_prompt_ids)
            trace.prompt_length_source = "engine_prompt_token_ids"

        completions = getattr(output, "outputs", None) or []
        if not isinstance(completions, (list, tuple)):
            completions = []  # PoolingRequestOutput.outputs is a single PoolingOutput: no tokens
        before_total = sum(active.tokens_per_seq.values())
        for idx, comp in enumerate(completions):
            token_ids = getattr(comp, "token_ids", None)
            if token_ids is None:
                # PoolingOutput has no token ids.
                continue
            seq_index = int(getattr(comp, "index", idx))
            n = len(token_ids)
            if active.output_kind == "delta":
                active.tokens_per_seq[seq_index] = active.tokens_per_seq.get(seq_index, 0) + n
            else:
                # cumulative / final_only / unknown: token_ids is the full sequence so far
                active.tokens_per_seq[seq_index] = max(active.tokens_per_seq.get(seq_index, 0), n)
        after_total = sum(active.tokens_per_seq.values())

        if after_total > before_total:
            # FINAL_ONLY emits one output at completion: the first-token time is not observable.
            if active.first_token_monotonic is None and active.output_kind != "final_only":
                active.first_token_monotonic = end_m
                active.tokens_at_first_observation = after_total
            active.last_token_monotonic = end_m
        trace.output_length = after_total

    @staticmethod
    def _finish_reason(output: Any) -> Optional[str]:
        completions = getattr(output, "outputs", None) or []
        if not isinstance(completions, (list, tuple)):
            return None  # pooling output
        for comp in completions:
            reason = getattr(comp, "finish_reason", None)
            if reason is not None:
                return str(reason)
        return None

    def _on_abort(self, request_ids: Any) -> None:
        if request_ids is None:
            return
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        now_m, now_w = self._monotonic(), self._wall()
        with self._lock:
            for rid in request_ids:
                active = self._active.pop(str(rid), None)
                if active is not None:
                    self._push_completed(
                        self._finalize(active, now_m, now_w, RequestStatus.ABORTED, "abort")
                    )

    def _on_batch_scheduled(self, sched_out: Any, now_m: float, now_w: float) -> None:
        scheduled_tokens = getattr(sched_out, "num_scheduled_tokens", None)
        if not isinstance(scheduled_tokens, dict):
            raise TypeError("SchedulerOutput.num_scheduled_tokens is not a dict; unsupported scheduler")
        if not scheduled_tokens:
            return
        req_ids = [str(r) for r in scheduled_tokens.keys()]
        new_reqs = getattr(sched_out, "scheduled_new_reqs", None) or []
        new_ids = {str(getattr(r, "req_id", "")) for r in new_reqs}
        prompt_lengths = [
            len(r.prompt_token_ids)
            for r in new_reqs
            if isinstance(getattr(r, "prompt_token_ids", None), (list, tuple))
        ]

        # Prefill/decode classification from the scheduler's own request state.
        # In vLLM 0.11.0, Scheduler.schedule() calls _update_after_schedule() before
        # returning, which has already advanced Request.num_computed_tokens by this
        # step's scheduled tokens. Subtract them to recover the pre-execution state
        # (prefix-cache hits set the initial value; preemption resets it to 0, so a
        # resumed request correctly counts as prefill again). A request is in
        # prefill for this step if it had not yet computed its whole prompt.
        requests = getattr(self._scheduler, "requests", None)
        num_prefill = 0
        for rid in req_ids:
            req = requests.get(rid) if isinstance(requests, dict) else None
            if req is not None and hasattr(req, "num_computed_tokens") and hasattr(req, "num_prompt_tokens"):
                computed_before = int(req.num_computed_tokens) - int(scheduled_tokens.get(rid, 0))
                if computed_before < int(req.num_prompt_tokens):
                    num_prefill += 1
            elif rid in new_ids:
                num_prefill += 1

        kv_usage: Optional[float] = None
        kv_mgr = getattr(self._scheduler, "kv_cache_manager", None)
        usage = getattr(kv_mgr, "usage", None) if kv_mgr is not None else None
        if isinstance(usage, (int, float)):
            kv_usage = float(usage)

        running = getattr(self._scheduler, "running", None)
        waiting = getattr(self._scheduler, "waiting", None)

        self._batch_seq += 1
        ctx = self._current_step
        step_index = ctx.index if ctx else self._step_index
        batch_id = f"{self.clock_domain}-s{step_index}-b{self._batch_seq}"
        meta = BatchMetadata(
            batch_id=batch_id,
            step_index=step_index,
            timestamp=now_w,
            monotonic=now_m,
            step_start_monotonic=ctx.start_monotonic if ctx else None,
            num_requests=len(req_ids),
            num_prefill=num_prefill,
            num_decode=len(req_ids) - num_prefill,
            total_scheduled_tokens=int(
                getattr(sched_out, "total_num_scheduled_tokens", sum(scheduled_tokens.values()))
            ),
            request_ids=req_ids,
            scheduled_tokens={str(k): int(v) for k, v in scheduled_tokens.items()},
            prompt_lengths=prompt_lengths,
            kv_cache_usage_fraction=kv_usage,
            num_running=len(running) if running is not None and hasattr(running, "__len__") else None,
            num_waiting=len(waiting) if waiting is not None and hasattr(waiting, "__len__") else None,
            source="vllm_v1_in_process_scheduler",
        )
        with self._lock:
            for rid in req_ids:
                active = self._active.get(rid)
                if active is None:
                    continue
                if active.first_scheduled_monotonic is None:
                    active.first_scheduled_monotonic = ctx.start_monotonic if ctx else now_m
                if batch_id not in active.trace.batch_ids:
                    active.trace.batch_ids.append(batch_id)
            if ctx is not None:
                # Published (with its end time) when the step returns, so a concurrent
                # drain can never observe a batch whose execution interval is still open.
                ctx.batches.append(meta)
            else:
                if len(self._batches) >= self.max_buffered:
                    self._batches.pop(0)
                    self._dropped_batches += 1
                self._batches.append(meta)

    def _push_completed(self, trace: RequestTrace) -> None:
        if len(self._completed) >= self.max_buffered:
            self._completed.pop(0)
            self._dropped_traces += 1
        self._completed.append(trace)

    # --------------------------------------------------------------- finalize

    def _finalize(
        self, active: _Active, end_m: float, end_w: float, status: RequestStatus, reason: Optional[str]
    ) -> RequestTrace:
        """Build spans and latency metrics. Caller holds the lock."""
        t = active.trace
        t.end_monotonic = end_m
        t.end_time = end_w
        t.status = status
        t.finish_reason = reason
        t.first_token_monotonic = active.first_token_monotonic
        t.tokens_at_first_observation = active.tokens_at_first_observation

        def wall(m: float) -> float:
            return active.arrival_wall + (m - active.arrival_monotonic)

        def span(phase: SpanPhase, a: float, b: float, **meta: Any) -> RequestSpan:
            return RequestSpan(
                phase=phase,
                start_time=wall(a),
                end_time=wall(b),
                duration_ms=(b - a) * 1000.0,
                start_monotonic=a,
                end_monotonic=b,
                metadata={"granularity": "engine_step", **meta},
            )

        spans: List[RequestSpan] = []
        ft = active.first_token_monotonic
        arrival = active.arrival_monotonic

        # TTFT: arrival at add_request -> end of the step in which the first output token
        # became visible to the caller. Includes queueing. Step-granular upper bound on the
        # true first-token time.
        if ft is not None:
            t.ttft_ms = (ft - arrival) * 1000.0
            if active.first_scheduled_monotonic is not None:
                fs = min(max(active.first_scheduled_monotonic, arrival), ft)
                spans.append(span(SpanPhase.QUEUE, arrival, fs))
                spans.append(span(SpanPhase.PREFILL, fs, ft, includes_first_decode_step=True))
            else:
                spans.append(
                    span(SpanPhase.TIME_TO_FIRST_TOKEN, arrival, ft, boundary="not_exposed_by_engine")
                )
        else:
            if active.output_kind == "final_only":
                t.ttft_unavailable_reason = "output_kind=FINAL_ONLY: intermediate outputs not emitted"
            elif active.output_kind == "pooling":
                t.ttft_unavailable_reason = "pooling request: no output tokens"
            elif t.output_length == 0:
                t.ttft_unavailable_reason = "no output tokens observed"
            else:  # pragma: no cover - defensive
                t.ttft_unavailable_reason = "first token time not observed"
            if active.first_scheduled_monotonic is not None:
                fs = max(active.first_scheduled_monotonic, arrival)
                spans.append(span(SpanPhase.QUEUE, arrival, min(fs, end_m)))

        # TPOT: average interval between token observations after the first one.
        n_total = t.output_length
        n_first = active.tokens_at_first_observation
        lt = active.last_token_monotonic
        if ft is not None and lt is not None and n_total - n_first > 0 and lt > ft:
            t.tpot_ms = (lt - ft) * 1000.0 / (n_total - n_first)
            spans.append(span(SpanPhase.DECODE, ft, lt))
        elif ft is not None and n_total - n_first > 0 and lt is not None:
            t.tpot_unavailable_reason = "tokens after the first observation arrived in the same step"
        elif ft is not None:
            t.tpot_unavailable_reason = (
                f"all {n_total} output token(s) arrived in the first observed step"
                if n_total > 0
                else "no output tokens observed"
            )
        else:
            t.tpot_unavailable_reason = t.ttft_unavailable_reason

        t.spans = spans
        return t

    # ---------------------------------------------------------------- drains

    def drain_completed_traces(self) -> List[RequestTrace]:
        with self._lock:
            out = self._completed
            self._completed = []
        return out

    def drain_batch_metadata(self) -> List[BatchMetadata]:
        with self._lock:
            out = self._batches
            self._batches = []
        return out

    def active_request_count(self) -> int:
        with self._lock:
            return len(self._active)

    def health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "instrumented": self._engine is not None,
                "target_vllm_version": TARGET_VLLM_VERSION,
                "detected_vllm_version": self.vllm_version,
                "scheduler_visible": self.scheduler_visible,
                "scheduler_unavailable_reason": self.scheduler_unavailable_reason,
                "active_requests": len(self._active),
                "buffered_traces": len(self._completed),
                "buffered_batches": len(self._batches),
                "dropped_traces": self._dropped_traces,
                "dropped_batches": self._dropped_batches,
                "instrumentation_errors": self._errors,
                "last_error": self._last_error,
                "steps_observed": self._step_index,
            }
