"""AsyncLLM instrumentation: async-generator wrapper semantics, cancellation, abort, restoration, tracer wiring."""

from __future__ import annotations

import asyncio
import inspect

import pytest
from fakes import FakeAsyncLLM, FakeClock, FakeNVMLBackend, RequestOutputKind, SamplingParams

from llmtrace import LLMTracer, TracerConfig, io
from llmtrace.data_plane.vllm_async_instrumentation import ASYNC_UNAVAILABLE, AsyncLLMInstrumentation, is_async_llm
from llmtrace.data_plane.vllm_instrumentation import InstrumentationError
from llmtrace.models.trace import RequestStatus, SpanPhase


def make(clock: FakeClock, **kw):
    engine = FakeAsyncLLM(clock=clock, **kw)
    instr = AsyncLLMInstrumentation(clock_domain="a", monotonic=clock.monotonic, wall=clock.time)
    instr.instrument_engine(engine)
    return engine, instr


async def consume(engine, rid, max_tokens=4, kind=RequestOutputKind.CUMULATIVE, prompt=None):
    outs = []
    async for out in engine.generate(prompt or {"prompt_token_ids": [1, 2, 3]}, SamplingParams(max_tokens=max_tokens, output_kind=kind), rid):
        outs.append(out)
    return outs


class TestWrapper:
    def test_generate_stays_async_generator_and_forwards_outputs(self, clock):
        engine, instr = make(clock, step_seconds=0.1)
        assert inspect.isasyncgenfunction(engine.generate) and inspect.iscoroutinefunction(engine.abort)
        outs = asyncio.run(consume(engine, "r1", max_tokens=3))
        assert [len(o.outputs[0].token_ids) for o in outs] == [1, 2, 3] and outs[-1].finished
        (t,) = instr.drain_completed_traces()
        assert t.status == RequestStatus.COMPLETED and t.output_length == 3 and t.finish_reason == "length"
        assert t.ttft_ms == pytest.approx(100.0) and t.tpot_ms == pytest.approx(100.0)
        assert t.prompt_length == 3 and t.prompt_length_source == "engine_prompt_token_ids"
        assert [s.phase for s in t.spans] == [SpanPhase.TIME_TO_FIRST_TOKEN, SpanPhase.DECODE]
        assert t.metadata["engine"] == "AsyncLLM" and not t.scheduler_visible
        assert instr.active_request_count() == 0

    def test_delta_outputs_count_tokens(self, clock):
        engine, instr = make(clock, step_seconds=0.05)
        asyncio.run(consume(engine, "r", max_tokens=5, kind=RequestOutputKind.DELTA))
        (t,) = instr.drain_completed_traces()
        assert t.output_length == 5 and t.output_kind == "delta" and t.ttft_ms == pytest.approx(50.0)

    def test_final_only_has_no_ttft(self, clock):
        engine, instr = make(clock)
        asyncio.run(consume(engine, "r", max_tokens=3, kind=RequestOutputKind.FINAL_ONLY))
        (t,) = instr.drain_completed_traces()
        assert t.output_length == 3 and t.ttft_ms is None and "FINAL_ONLY" in t.ttft_unavailable_reason

    def test_client_stops_reading_is_recorded_as_aborted(self, clock):
        engine, instr = make(clock)

        async def partial():
            async for out in engine.generate({"prompt_token_ids": [1]}, SamplingParams(max_tokens=10), "r"):
                if len(out.outputs[0].token_ids) == 3:
                    break  # closes the generator: GeneratorExit -> AsyncLLM aborts

        asyncio.run(partial())
        (t,) = instr.drain_completed_traces()
        assert t.status == RequestStatus.ABORTED and t.finish_reason == "abort" and t.output_length == 3
        assert t.metadata["abort_cause"] == "client_cancelled"
        assert engine.aborted == ["r"] and instr.active_request_count() == 0

    def test_task_cancellation_is_recorded_as_aborted(self, clock):
        engine, instr = make(clock, block_at_token=3)  # the stream parks after 3 tokens; no timing race

        async def run():
            task = asyncio.create_task(consume(engine, "r", max_tokens=1000))
            await engine.wait_blocked()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        (t,) = instr.drain_completed_traces()
        assert t.status == RequestStatus.ABORTED and t.finish_reason == "abort" and t.output_length == 3
        assert t.metadata["abort_cause"] == "client_cancelled"  # engine abort ran first; cause annotated afterwards
        assert engine.aborted == ["r"]

    def test_engine_error_propagates_and_is_recorded_incomplete(self, clock):
        engine, instr = make(clock, fail_at_token=2)
        with pytest.raises(RuntimeError, match="injected"):
            asyncio.run(consume(engine, "r", max_tokens=5))
        (t,) = instr.drain_completed_traces()
        assert t.status == RequestStatus.INCOMPLETE and t.finish_reason == "error:RuntimeError" and t.output_length == 1
        assert instr.health()["instrumentation_errors"] == 0

    def test_explicit_abort_marks_aborted(self, clock):
        engine, instr = make(clock, block_at_token=2)

        async def run():
            task = asyncio.create_task(consume(engine, "r", max_tokens=1000))
            await engine.wait_blocked()
            await engine.abort("r")  # engine-side abort; the fake's stream keeps running, so cancel the consumer
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run())
        traces = instr.drain_completed_traces()
        assert len(traces) == 1 and traces[0].status == RequestStatus.ABORTED and traces[0].finish_reason == "abort"
        assert engine.calls.count("abort:r") >= 1

    def test_concurrent_streams(self, clock):
        engine, instr = make(clock, step_seconds=0.01)

        async def run():
            return await asyncio.gather(*(consume(engine, f"r{i}", max_tokens=3 + i) for i in range(5)))

        asyncio.run(run())
        traces = {t.request_id: t for t in instr.drain_completed_traces()}
        assert set(traces) == {f"r{i}" for i in range(5)} and all(traces[f"r{i}"].output_length == 3 + i for i in range(5))


class TestSetup:
    def test_rejects_sync_engine_and_restores(self, clock):
        from fakes import FakeLLMEngine
        instr = AsyncLLMInstrumentation(monotonic=clock.monotonic, wall=clock.time)
        with pytest.raises(InstrumentationError, match="async generator"):
            instr.instrument_engine(FakeLLMEngine(clock=clock))
        engine = FakeAsyncLLM(clock=clock)
        original_gen, original_abort = engine.generate, engine.abort
        instr.instrument_engine(engine)
        assert "generate" in engine.__dict__ and instr.scheduler_unavailable_reason == ASYNC_UNAVAILABLE
        assert instr.executor_unavailable_reason == ASYNC_UNAVAILABLE and instr.health()["engine_kind"] == "AsyncLLM"
        assert instr.uninstrument_engine() == []
        assert engine.generate == original_gen and engine.abort == original_abort
        assert is_async_llm(engine) and not is_async_llm(FakeLLMEngine(clock=clock))

    def test_uninstrument_returns_incomplete_streams(self, clock):
        engine, instr = make(clock)

        async def run():
            agen = engine.generate({"prompt_token_ids": [1]}, SamplingParams(max_tokens=100), "r")
            await agen.__anext__()
            left = instr.uninstrument_engine()
            await agen.aclose()
            return left

        left = asyncio.run(run())
        assert [t.request_id for t in left] == ["r"] and left[0].status == RequestStatus.INCOMPLETE


class TestTracer:
    def test_instrument_async_engine_end_to_end(self, tmp_path):
        clock = FakeClock()
        tracer = LLMTracer(TracerConfig(output_dir=str(tmp_path), collection_interval_s=0.02), gpu_backend=FakeNVMLBackend({0: 5.0}))
        engine = FakeAsyncLLM(clock=clock)
        tracer.instrument_async_engine(engine)
        tracer.vllm_instrumentation._monotonic = clock.monotonic
        tracer.vllm_instrumentation._wall = clock.time

        async def run():
            await asyncio.gather(consume(engine, "a", 3), consume(engine, "b", 2))
            # vLLM stats: the fake AsyncLLM has a logger_manager; the tracer attached its logger post-hoc
            from fakes import FakeIterationStats, FakeSchedulerStats
            engine.logger_manager.record(FakeSchedulerStats(num_running_reqs=2, kv_cache_usage=0.2), FakeIterationStats(num_generation_tokens=2))

        asyncio.run(run())
        tracer.stop()
        h = tracer.health()
        assert h["instrumentation"]["engine_kind"] == "AsyncLLM" and h["scheduler_unavailable_reason_during_run"] == ASYNC_UNAVAILABLE
        assert h["executor_unavailable_reason_during_run"] == ASYNC_UNAVAILABLE and h["vllm_stats"]["unavailable_reason"] is None
        files = tracer.get_output_files()
        traces = {t.request_id: t for t in io.load_traces(files["traces"])}
        assert traces["a"].output_length == 3 and traces["b"].output_length == 2 and "batches" not in files
        assert len(io.load_vllm_stats(files["vllm_stats"])) == 1
        assert "generate" not in engine.__dict__
        with pytest.raises(RuntimeError):
            tracer.instrument_async_engine(engine)  # stopped
