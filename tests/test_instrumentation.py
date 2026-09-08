"""Wrapper semantics, lifecycle, restoration and failure isolation for VLLMInstrumentation."""

from __future__ import annotations

import inspect
import threading

import pytest
from fakes import FakeLLMEngine, SamplingParams, run_to_completion

from llmtrace.data_plane.vllm_instrumentation import InstrumentationError, VLLMInstrumentation
from llmtrace.models.trace import RequestStatus, SpanPhase


def _add(engine: FakeLLMEngine, rid: str, max_tokens: int = 4, **kw) -> None:
    engine.add_request(rid, {"prompt_token_ids": [1, 2, 3]}, SamplingParams(max_tokens=max_tokens, **kw))


class TestWrapperSemantics:
    def test_methods_stay_synchronous_and_return_engine_values(self, engine, instr):
        instr.instrument_engine(engine)
        assert not inspect.iscoroutinefunction(engine.step)
        assert not inspect.iscoroutinefunction(engine.add_request)
        assert engine.add_request("r1", {"prompt_token_ids": [1, 2]}, SamplingParams(max_tokens=2)) is None
        outputs = engine.step()
        assert isinstance(outputs, list) and outputs[0].request_id == "r1"
        assert not inspect.isawaitable(outputs)
        assert engine.abort_request(["r1"]) is None

    def test_positional_and_keyword_add_request(self, engine, instr):
        instr.instrument_engine(engine)
        engine.add_request("a", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=1))
        engine.add_request(request_id="b", prompt={"prompt_token_ids": [1, 2]}, params=SamplingParams(max_tokens=1))
        assert instr.active_request_count() == 2
        run_to_completion(engine)
        ids = {t.request_id for t in instr.drain_completed_traces()}
        assert ids == {"a", "b"}

    def test_engine_exceptions_propagate_unchanged(self, clock, instr):
        engine = FakeLLMEngine(clock=clock, fail_step_at=2)
        instr.instrument_engine(engine)
        _add(engine, "r1", max_tokens=5)
        engine.step()
        with pytest.raises(RuntimeError, match="engine failure injected"):
            engine.step()
        # Request is still active (engine did not finish it); nothing leaked or fabricated.
        assert instr.active_request_count() == 1
        assert instr.health()["instrumentation_errors"] == 0
        run_to_completion(engine)  # engine recovers on later steps
        (t,) = instr.drain_completed_traces()
        assert t.status == RequestStatus.COMPLETED

    def test_add_request_failure_is_not_tracked(self, engine, instr):
        instr.instrument_engine(engine)
        with pytest.raises(TypeError):
            engine.add_request("bad", 12345, SamplingParams())
        assert instr.active_request_count() == 0

    def test_async_engine_is_rejected(self, instr):
        class AsyncEngine:
            async def add_request(self, *a, **k): ...
            async def step(self): ...
            async def abort_request(self, ids): ...

        with pytest.raises(InstrumentationError, match="coroutine"):
            instr.instrument_engine(AsyncEngine())
        assert not instr.is_instrumented

    def test_missing_method_is_rejected(self, instr):
        class NotAnEngine:
            def step(self): ...

        with pytest.raises(InstrumentationError):
            instr.instrument_engine(NotAnEngine())


class TestRestoration:
    def test_uninstrument_restores_exact_bound_methods(self, engine, instr):
        original_step = engine.step
        original_add = engine.add_request
        original_schedule = engine.engine_core.engine_core.scheduler.schedule
        instr.instrument_engine(engine)
        assert engine.step is not original_step
        assert "step" in engine.__dict__
        instr.uninstrument_engine()
        assert "step" not in engine.__dict__ and "add_request" not in engine.__dict__
        assert engine.step == original_step and engine.add_request == original_add
        assert engine.engine_core.engine_core.scheduler.schedule == original_schedule
        assert not getattr(engine.step, "__llmtrace_wrapped__", False)

    def test_instance_attribute_originals_are_restored(self, engine, instr):
        marker = lambda *a, **k: []  # noqa: E731
        engine.step = marker  # instance-level override present before instrumentation
        instr.instrument_engine(engine)
        instr.uninstrument_engine()
        assert engine.step is marker

    def test_repeated_instrument_and_uninstrument_are_safe(self, engine, instr):
        instr.instrument_engine(engine)
        instr.instrument_engine(engine)  # no-op
        assert sum(1 for p in instr._patches) == 4  # add/step/abort/schedule, not doubled
        _add(engine, "r1")
        run_to_completion(engine)
        assert instr.uninstrument_engine() == []
        assert instr.uninstrument_engine() == []
        assert engine.step.__func__ is FakeLLMEngine.step
        # Can instrument again
        instr.instrument_engine(engine)
        _add(engine, "r2")
        run_to_completion(engine)
        assert {t.request_id for t in instr.drain_completed_traces()} == {"r1", "r2"}

    def test_two_instances_cannot_double_wrap(self, engine, instr, clock):
        instr.instrument_engine(engine)
        other = VLLMInstrumentation(monotonic=clock.monotonic, wall=clock.time)
        with pytest.raises(InstrumentationError, match="already wrapped"):
            other.instrument_engine(engine)
        assert "step" in engine.__dict__  # first instance intact
        other_engine = FakeLLMEngine(clock=clock)
        with pytest.raises(InstrumentationError, match="another engine"):
            instr.instrument_engine(other_engine)

    def test_partial_patch_failure_rolls_back(self, engine, clock):
        instr = VLLMInstrumentation(monotonic=clock.monotonic, wall=clock.time)
        engine.engine_core.engine_core.scheduler.schedule.__func__.__llmtrace_wrapped__ = True  # type: ignore
        try:
            with pytest.raises(InstrumentationError):
                instr.instrument_engine(engine)
            assert "step" not in engine.__dict__ and not instr.is_instrumented
        finally:
            del engine.engine_core.engine_core.scheduler.schedule.__func__.__llmtrace_wrapped__  # type: ignore


class TestLifecycle:
    def test_completion_does_not_deadlock(self, engine, instr):
        instr.instrument_engine(engine)
        _add(engine, "r1", max_tokens=3)
        _add(engine, "r2", max_tokens=1)
        done = threading.Event()

        def run():
            run_to_completion(engine)
            done.set()

        threading.Thread(target=run, daemon=True).start()
        assert done.wait(5.0), "step()/completion deadlocked"
        traces = instr.drain_completed_traces()
        assert {t.request_id for t in traces} == {"r1", "r2"}
        assert all(t.status == RequestStatus.COMPLETED for t in traces)
        assert instr.active_request_count() == 0
        assert instr.drain_completed_traces() == []  # drained exactly once

    def test_abort_marks_aborted_and_releases(self, engine, instr):
        instr.instrument_engine(engine)
        _add(engine, "r1", max_tokens=10)
        engine.step()
        engine.abort_request(["r1"])
        assert instr.active_request_count() == 0
        (t,) = instr.drain_completed_traces()
        assert t.status == RequestStatus.ABORTED and t.finish_reason == "abort"
        assert t.output_length == 1
        assert t.end_monotonic is not None and t.end_monotonic >= t.start_monotonic

    def test_uninstrument_returns_incomplete_requests(self, engine, instr):
        instr.instrument_engine(engine)
        _add(engine, "r1", max_tokens=10)
        engine.step()
        leftovers = instr.uninstrument_engine()
        assert [t.request_id for t in leftovers] == ["r1"]
        assert leftovers[0].status == RequestStatus.INCOMPLETE
        assert instr.active_request_count() == 0

    def test_unknown_request_ids_in_outputs_are_ignored(self, engine, instr):
        _add(engine, "before", max_tokens=1)  # added before instrumentation
        instr.instrument_engine(engine)
        run_to_completion(engine)
        assert instr.drain_completed_traces() == []
        assert instr.health()["instrumentation_errors"] == 0


class TestFailureIsolation:
    def test_bookkeeping_error_does_not_change_inference_result(self, engine, instr, monkeypatch):
        instr.instrument_engine(engine)
        _add(engine, "r1", max_tokens=2)

        def boom(*a, **k):
            raise ValueError("bookkeeping bug")

        monkeypatch.setattr(instr, "_on_step_completed", boom)
        out1 = engine.step()
        assert out1 and out1[0].request_id == "r1" and out1[0].outputs[0].token_ids == [0]
        health = instr.health()
        assert health["instrumentation_errors"] == 1
        assert "bookkeeping bug" in health["last_error"]

    def test_strict_mode_raises_after_engine_call(self, engine, clock, monkeypatch):
        instr = VLLMInstrumentation(strict=True, monotonic=clock.monotonic, wall=clock.time)
        instr.instrument_engine(engine)
        _add(engine, "r1", max_tokens=2)
        monkeypatch.setattr(instr, "_on_step_completed", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
        with pytest.raises(ValueError):
            engine.step()
        assert engine._requests["r1"].generated == [0]  # engine work still happened

    def test_malformed_scheduler_output_is_counted_not_fatal(self, engine, instr):
        instr.instrument_engine(engine)
        sched = engine.engine_core.engine_core.scheduler
        orig = sched.schedule.__wrapped__  # type: ignore[attr-defined]

        class Weird:
            num_scheduled_tokens = "not a dict"

        sched.schedule = instr._wrap_schedule(lambda: (engine._do_schedule(), Weird())[1])  # bookkeeping still runs
        _add(engine, "r1", max_tokens=1)
        outputs = engine.step()
        assert outputs[0].finished
        assert instr.health()["instrumentation_errors"] == 1
        sched.schedule = orig


class TestSchedulerVisibility:
    def test_batches_link_requests_with_real_ids(self, engine, instr):
        instr.instrument_engine(engine)
        assert instr.scheduler_visible
        _add(engine, "r1", max_tokens=2)
        _add(engine, "r2", max_tokens=1)
        run_to_completion(engine)
        batches = instr.drain_batch_metadata()
        assert len(batches) == 2
        assert batches[0].request_ids == ["r1", "r2"]
        assert batches[0].num_prefill == 2 and batches[0].num_decode == 0
        assert batches[1].request_ids == ["r1"] and batches[1].num_decode == 1
        assert batches[0].kv_cache_usage_fraction == pytest.approx(0.2)
        assert batches[0].step_end_monotonic is not None and batches[0].step_end_monotonic > batches[0].monotonic
        assert batches[0].source == "vllm_v1_in_process_scheduler"
        traces = {t.request_id: t for t in instr.drain_completed_traces()}
        assert traces["r1"].batch_ids == [b.batch_id for b in batches]
        assert traces["r2"].batch_ids == [batches[0].batch_id]
        assert traces["r1"].scheduler_visible

    def test_prefill_classified_from_pre_execution_state(self, clock):
        # Real order: schedule() returns with num_computed_tokens already advanced.
        engine = FakeLLMEngine(clock=clock, prefill_chunk=4)
        instr = VLLMInstrumentation(monotonic=clock.monotonic, wall=clock.time)
        instr.instrument_engine(engine)
        sched = engine.engine_core.engine_core.scheduler
        engine.add_request("r", {"prompt_token_ids": list(range(10))}, SamplingParams(max_tokens=2))
        observed = []
        wrapped = sched.schedule

        def outer():
            out = wrapped()
            observed.append((sched.requests["r"].num_computed_tokens, out.num_scheduled_tokens["r"]))
            return out

        sched.schedule = outer
        run_to_completion(engine)
        sched.schedule = wrapped
        # After each schedule() the counter already includes that step's tokens (4, 8, 10, 11, ...).
        assert observed[:3] == [(4, 4), (8, 4), (10, 2)]
        batches = instr.drain_batch_metadata()
        kinds = [(b.num_prefill, b.num_decode) for b in batches]
        assert kinds == [(1, 0), (1, 0), (1, 0), (0, 1)]  # 3 prefill chunks (last one yields token 1), 1 decode

    def test_batches_are_not_drainable_before_the_step_ends(self, engine, instr):
        instr.instrument_engine(engine)
        sched = engine.engine_core.engine_core.scheduler
        wrapped = sched.schedule
        seen_mid_step = []

        def outer():
            out = wrapped()
            seen_mid_step.append(instr.drain_batch_metadata())  # a collector draining mid-step
            return out

        sched.schedule = outer
        _add(engine, "r1", max_tokens=1)
        engine.step()
        assert seen_mid_step == [[]]
        (b,) = instr.drain_batch_metadata()
        assert b.step_end_monotonic is not None and b.request_ids == ["r1"]
        sched.schedule = wrapped

    def test_failed_step_publishes_batches_without_end_time(self, clock):
        engine = FakeLLMEngine(clock=clock)
        instr = VLLMInstrumentation(monotonic=clock.monotonic, wall=clock.time)
        instr.instrument_engine(engine)
        _add(engine, "r1", max_tokens=1)

        def explode(req):  # fails after schedule() ran, before step() returns
            raise RuntimeError("model execution failed")

        engine._make_output = explode
        with pytest.raises(RuntimeError, match="model execution failed"):
            engine.step()
        (b,) = instr.drain_batch_metadata()
        assert b.step_end_monotonic is None and b.request_ids == ["r1"]
        assert instr.active_request_count() == 1

    def test_no_in_process_scheduler_reports_reason(self, clock):
        engine = FakeLLMEngine(clock=clock, in_process_scheduler=False)
        instr = VLLMInstrumentation(monotonic=clock.monotonic, wall=clock.time)
        instr.instrument_engine(engine)
        assert not instr.scheduler_visible
        assert "VLLM_ENABLE_V1_MULTIPROCESSING=0" in instr.scheduler_unavailable_reason
        _add(engine, "r1", max_tokens=2)
        run_to_completion(engine)
        assert instr.drain_batch_metadata() == []
        (t,) = instr.drain_completed_traces()
        assert t.batch_ids == [] and not t.scheduler_visible
        assert {s.phase for s in t.spans} == {SpanPhase.TIME_TO_FIRST_TOKEN, SpanPhase.DECODE}

    def test_batch_metadata_can_be_disabled(self, engine, clock):
        instr = VLLMInstrumentation(enable_batch_metadata=False, monotonic=clock.monotonic, wall=clock.time)
        instr.instrument_engine(engine)
        assert not instr.scheduler_visible and "disabled" in instr.scheduler_unavailable_reason
        assert "schedule" not in engine.engine_core.engine_core.scheduler.__dict__


class TestBuffers:
    def test_completed_buffer_is_bounded_and_drops_counted(self, engine, clock):
        instr = VLLMInstrumentation(max_buffered=2, monotonic=clock.monotonic, wall=clock.time)
        instr.instrument_engine(engine)
        for i in range(5):
            _add(engine, f"r{i}", max_tokens=1)
        run_to_completion(engine)
        h = instr.health()
        assert h["buffered_traces"] == 2 and h["dropped_traces"] == 3
        assert len(instr.drain_completed_traces()) == 2

    def test_health_reports_target_version(self, instr):
        assert instr.health()["target_vllm_version"] == "0.11.0"
