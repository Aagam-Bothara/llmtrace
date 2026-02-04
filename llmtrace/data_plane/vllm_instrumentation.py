"""vLLM instrumentation plugin for lifecycle tracing."""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional
from contextlib import contextmanager
from functools import wraps

from llmtrace.models.trace import (
    RequestSpan,
    RequestTrace,
    SpanPhase,
    BatchMetadata,
)

logger = logging.getLogger(__name__)


class VLLMInstrumentation:
    """
    Instruments a vLLM LLMEngine to collect request lifecycle traces.

    This hooks into vLLM's internals to track:
    - Request queueing, prefill, and decode phases
    - Batch formation and execution
    - KV cache usage
    - Scheduler state

    Design:
    - Minimal overhead via async event collection
    - Non-intrusive: uses callbacks and lightweight wrappers
    - Handles distributed setups (tensor/pipeline parallelism)
    """

    def __init__(self, enable_batch_metadata: bool = True, enable_kv_cache: bool = True):
        self.enable_batch_metadata = enable_batch_metadata
        self.enable_kv_cache = enable_kv_cache

        # Active requests: request_id -> RequestTrace
        self._active_requests: Dict[str, RequestTrace] = {}
        self._request_lock = asyncio.Lock()

        # Completed traces
        self._completed_traces: List[RequestTrace] = []
        self._completed_lock = asyncio.Lock()

        # Batch metadata
        self._batch_metadata: List[BatchMetadata] = []
        self._batch_lock = asyncio.Lock()

        # Engine reference (set during instrumentation)
        self._engine: Optional[Any] = None
        self._original_methods: Dict[str, Callable] = {}

    def instrument_engine(self, engine: Any) -> None:
        """
        Instrument a vLLM LLMEngine.

        Args:
            engine: vLLM LLMEngine instance

        This patches engine methods to inject tracing hooks.
        """
        logger.info("Instrumenting vLLM LLMEngine")
        self._engine = engine

        # Hook into request addition
        self._patch_method(engine, "add_request", self._wrap_add_request)

        # Hook into step (where batches are processed)
        self._patch_method(engine, "step", self._wrap_step)

        # Hook into scheduler if we need batch metadata
        if self.enable_batch_metadata:
            try:
                scheduler = engine.scheduler
                self._patch_method(scheduler, "schedule", self._wrap_schedule)
            except AttributeError:
                logger.warning("Could not access engine.scheduler for batch metadata")

        logger.info("vLLM LLMEngine instrumented successfully")

    def uninstrument_engine(self) -> None:
        """Remove instrumentation from engine."""
        if not self._engine:
            return

        logger.info("Removing instrumentation from vLLM LLMEngine")

        # Restore original methods
        for obj_method, original in self._original_methods.items():
            obj, method_name = obj_method.rsplit(".", 1)
            target = self._engine if obj == "engine" else getattr(self._engine, obj)
            setattr(target, method_name, original)

        self._original_methods.clear()
        self._engine = None

    def _patch_method(self, obj: Any, method_name: str, wrapper: Callable) -> None:
        """Patch a method with instrumentation wrapper."""
        original = getattr(obj, method_name)
        obj_key = f"{id(obj)}.{method_name}"
        self._original_methods[obj_key] = original
        setattr(obj, method_name, wrapper(original))

    def _wrap_add_request(self, original: Callable) -> Callable:
        """Wrap add_request to track request arrival."""

        @wraps(original)
        async def wrapper(*args, **kwargs):
            # Extract request_id (typically first arg after self)
            request_id = args[0] if args else kwargs.get("request_id")

            if request_id:
                await self._on_request_added(
                    request_id=request_id,
                    prompt=kwargs.get("prompt", ""),
                    params=kwargs.get("params", {}),
                )

            return await original(*args, **kwargs)

        return wrapper

    def _wrap_step(self, original: Callable) -> Callable:
        """Wrap step to track request lifecycle phases."""

        @wraps(original)
        async def wrapper(*args, **kwargs):
            step_start = time.time()

            # Call original step
            outputs = await original(*args, **kwargs)

            # Process outputs to update request states
            if outputs:
                await self._on_step_completed(outputs, step_start, time.time())

            return outputs

        return wrapper

    def _wrap_schedule(self, original: Callable) -> Callable:
        """Wrap scheduler.schedule to capture batch metadata."""

        @wraps(original)
        def wrapper(*args, **kwargs):
            schedule_start = time.time()

            # Call original schedule
            result = original(*args, **kwargs)

            # Extract batch metadata
            asyncio.create_task(
                self._on_batch_scheduled(result, schedule_start)
            )

            return result

        return wrapper

    async def _on_request_added(
        self, request_id: str, prompt: str, params: Dict[str, Any]
    ) -> None:
        """Handle request addition (enters queue)."""
        now = time.time()

        # Get model name from engine
        model_name = "unknown"
        if self._engine and hasattr(self._engine, "model_config"):
            model_name = getattr(self._engine.model_config, "model", "unknown")

        trace = RequestTrace(
            request_id=request_id,
            start_time=now,
            end_time=now,  # Will update on completion
            prompt_length=len(prompt.split()),  # Rough estimate
            output_length=0,  # Will update
            model_name=model_name,
            spans=[],
            batch_ids=[],
        )

        # Create queue span (starts immediately)
        queue_span = RequestSpan(
            phase=SpanPhase.QUEUE,
            start_time=now,
            end_time=now,  # Will update when prefill starts
            duration_ms=0,
        )
        trace.spans.append(queue_span)

        async with self._request_lock:
            self._active_requests[request_id] = trace

        logger.debug(f"Request {request_id} added to queue")

    async def _on_step_completed(
        self, outputs: List[Any], step_start: float, step_end: float
    ) -> None:
        """Handle step completion - update request phases."""
        async with self._request_lock:
            for output in outputs:
                request_id = output.request_id

                if request_id not in self._active_requests:
                    continue

                trace = self._active_requests[request_id]

                # Check if this is first token (end of prefill, start of decode)
                if output.outputs and len(output.outputs[0].token_ids) == 1:
                    # End queue span
                    if trace.spans and trace.spans[-1].phase == SpanPhase.QUEUE:
                        trace.spans[-1].end_time = step_start
                        trace.spans[-1].duration_ms = (
                            step_start - trace.spans[-1].start_time
                        ) * 1000

                    # Create prefill span
                    prefill_span = RequestSpan(
                        phase=SpanPhase.PREFILL,
                        start_time=step_start,
                        end_time=step_end,
                        duration_ms=(step_end - step_start) * 1000,
                    )
                    trace.spans.append(prefill_span)
                    trace.ttft_ms = (step_end - trace.start_time) * 1000

                # Decode phase
                elif output.outputs and len(output.outputs[0].token_ids) > 1:
                    # Check if we already have a decode span
                    decode_spans = [s for s in trace.spans if s.phase == SpanPhase.DECODE]
                    if not decode_spans:
                        # First decode token - create decode span
                        decode_span = RequestSpan(
                            phase=SpanPhase.DECODE,
                            start_time=step_start,
                            end_time=step_end,
                            duration_ms=(step_end - step_start) * 1000,
                        )
                        trace.spans.append(decode_span)
                    else:
                        # Extend existing decode span
                        decode_spans[-1].end_time = step_end
                        decode_spans[-1].duration_ms = (
                            step_end - decode_spans[-1].start_time
                        ) * 1000

                # Update output length
                if output.outputs:
                    trace.output_length = len(output.outputs[0].token_ids)

                # Check if request is finished
                if output.finished:
                    await self._on_request_completed(request_id, step_end)

    async def _on_request_completed(self, request_id: str, end_time: float) -> None:
        """Handle request completion."""
        async with self._request_lock:
            if request_id not in self._active_requests:
                return

            trace = self._active_requests.pop(request_id)
            trace.end_time = end_time

            # Calculate TPOT (time per output token)
            if trace.output_length > 0:
                decode_time = trace.decode_duration_ms
                trace.tpot_ms = decode_time / trace.output_length

        async with self._completed_lock:
            self._completed_traces.append(trace)

        logger.debug(
            f"Request {request_id} completed: "
            f"queue={trace.queue_duration_ms:.2f}ms "
            f"prefill={trace.prefill_duration_ms:.2f}ms "
            f"decode={trace.decode_duration_ms:.2f}ms"
        )

    async def _on_batch_scheduled(self, schedule_output: Any, timestamp: float) -> None:
        """Handle batch scheduling - capture batch metadata."""
        if not self.enable_batch_metadata:
            return

        try:
            # Extract batch info from schedule output
            # This is vLLM-version specific, may need adjustment
            scheduled_requests = getattr(schedule_output, "scheduled_seq_groups", [])
            ignored_requests = getattr(schedule_output, "ignored_seq_groups", [])

            if not scheduled_requests:
                return

            num_prefill = sum(
                1 for req in scheduled_requests if not req.is_prefill_complete
            )
            num_decode = len(scheduled_requests) - num_prefill

            prompt_lengths = []
            total_tokens = 0
            for req in scheduled_requests:
                prompt_len = getattr(req, "get_seqs", lambda: [None])[0]
                if prompt_len:
                    prompt_len = len(prompt_len.get_token_ids())
                    prompt_lengths.append(prompt_len)
                    total_tokens += prompt_len

            # Try to get KV cache usage
            kv_usage = None
            kv_capacity = None
            if self.enable_kv_cache and hasattr(schedule_output, "num_batched_tokens"):
                # This is approximate - actual KV cache tracking may require vLLM patching
                kv_usage = getattr(schedule_output, "num_batched_tokens", None)

            batch_meta = BatchMetadata(
                batch_id=f"batch_{timestamp}",
                timestamp=timestamp,
                num_requests=len(scheduled_requests),
                num_prefill=num_prefill,
                num_decode=num_decode,
                total_tokens=total_tokens,
                prompt_lengths=prompt_lengths,
                kv_cache_usage_bytes=kv_usage,
                kv_cache_capacity_bytes=kv_capacity,
            )

            async with self._batch_lock:
                self._batch_metadata.append(batch_meta)

            logger.debug(
                f"Batch scheduled: {len(scheduled_requests)} requests "
                f"({num_prefill} prefill, {num_decode} decode)"
            )

        except Exception as e:
            logger.warning(f"Failed to extract batch metadata: {e}")

    async def get_completed_traces(
        self, clear: bool = True
    ) -> List[RequestTrace]:
        """
        Get completed request traces.

        Args:
            clear: If True, clear the completed traces buffer

        Returns:
            List of completed RequestTrace objects
        """
        async with self._completed_lock:
            traces = list(self._completed_traces)
            if clear:
                self._completed_traces.clear()
            return traces

    async def get_batch_metadata(
        self, clear: bool = True
    ) -> List[BatchMetadata]:
        """
        Get batch metadata.

        Args:
            clear: If True, clear the batch metadata buffer

        Returns:
            List of BatchMetadata objects
        """
        async with self._batch_lock:
            metadata = list(self._batch_metadata)
            if clear:
                self._batch_metadata.clear()
            return metadata

    async def get_active_request_count(self) -> int:
        """Get number of currently active requests."""
        async with self._request_lock:
            return len(self._active_requests)

    @contextmanager
    def trace_context(self, request_id: str):
        """
        Context manager for explicit request tracing.

        Useful for manual instrumentation if auto-instrumentation isn't available.
        """
        start = time.time()
        try:
            yield
        finally:
            # Mark request as completed
            asyncio.create_task(self._on_request_completed(request_id, time.time()))
