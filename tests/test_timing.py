"""Timing and token semantics: TTFT, TPOT, spans, output kinds, prompt lengths."""

from __future__ import annotations

import pytest
from fakes import FakeClock, FakeLLMEngine, PoolingParams, RequestOutputKind, SamplingParams, run_to_completion

from llmtrace.data_plane.vllm_instrumentation import VLLMInstrumentation
from llmtrace.models.trace import SpanPhase


def make(clock: FakeClock, **engine_kw):
    engine = FakeLLMEngine(clock=clock, **engine_kw)
    instr = VLLMInstrumentation(clock_domain="t", monotonic=clock.monotonic, wall=clock.time)
    instr.instrument_engine(engine)
    return engine, instr


def one(instr):
    traces = instr.drain_completed_traces()
    assert len(traces) == 1
    return traces[0]


class TestFirstTokenAndTPOT:
    def test_exact_ttft_and_tpot_with_fake_clock(self, clock):
        engine, instr = make(clock, step_seconds=0.1)
        engine.add_request("r", {"prompt_token_ids": [1, 2, 3, 4]}, SamplingParams(max_tokens=4))
        clock.advance(0.5)  # request waits before the loop starts stepping
        run_to_completion(engine)
        t = one(instr)
        assert t.ttft_ms == pytest.approx(600.0)  # 0.5 wait + 0.1 first step
        assert t.tokens_at_first_observation == 1
        assert t.tpot_ms == pytest.approx(100.0)  # 3 more tokens over 3 steps of 0.1s
        assert t.output_length == 4
        assert t.prompt_length == 4 and t.prompt_length_source == "engine_prompt_token_ids"
        assert t.total_duration_ms == pytest.approx(900.0)
        assert t.end_time - t.start_time == pytest.approx(0.9)

    def test_zero_output_tokens(self, clock):
        engine, instr = make(clock)
        engine.add_request("r", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=0))
        run_to_completion(engine)
        t = one(instr)
        assert t.output_length == 0
        assert t.ttft_ms is None and "no output tokens" in t.ttft_unavailable_reason
        assert t.tpot_ms is None
        assert not any(s.phase == SpanPhase.DECODE for s in t.spans)

    def test_single_output_token(self, clock):
        engine, instr = make(clock, step_seconds=0.05)
        engine.add_request("r", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=1))
        run_to_completion(engine)
        t = one(instr)
        assert t.ttft_ms == pytest.approx(50.0)
        assert t.tpot_ms is None and "first observed step" in t.tpot_unavailable_reason
        assert not any(s.phase == SpanPhase.DECODE for s in t.spans)

    def test_several_tokens_in_first_step_do_not_manufacture_tpot(self, clock):
        # e.g. speculative decoding: all 3 tokens arrive in the first step.
        engine, instr = make(clock, tokens_per_step=3, step_seconds=0.2)
        engine.add_request("r", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=3))
        run_to_completion(engine)
        t = one(instr)
        assert t.ttft_ms == pytest.approx(200.0) and t.tokens_at_first_observation == 3
        assert t.tpot_ms is None

    def test_tpot_uses_tokens_after_first_observation(self, clock):
        # 3 tokens per step, 6 total: first observation has 3; the other 3 arrive one step later.
        engine, instr = make(clock, tokens_per_step=3, step_seconds=0.3)
        engine.add_request("r", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=6))
        run_to_completion(engine)
        t = one(instr)
        assert t.tpot_ms == pytest.approx(100.0)

    def test_first_token_detection_is_not_len_equals_one(self, clock):
        # DELTA outputs carry exactly one token every step; only the first must count as TTFT.
        engine, instr = make(clock, step_seconds=0.1)
        engine.add_request(
            "r", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=3, output_kind=RequestOutputKind.DELTA)
        )
        run_to_completion(engine)
        t = one(instr)
        assert t.output_kind == "delta"
        assert t.ttft_ms == pytest.approx(100.0)
        assert t.output_length == 3
        assert t.tpot_ms == pytest.approx(100.0)

    def test_cumulative_and_delta_agree(self, clock):
        results = {}
        for kind in (RequestOutputKind.CUMULATIVE, RequestOutputKind.DELTA):
            c = FakeClock()
            engine, instr = make(c, step_seconds=0.1)
            engine.add_request("r", {"prompt_token_ids": [1, 2]}, SamplingParams(max_tokens=5, output_kind=kind))
            run_to_completion(engine)
            t = one(instr)
            results[kind] = (t.output_length, t.ttft_ms, t.tpot_ms)
        assert results[RequestOutputKind.CUMULATIVE] == results[RequestOutputKind.DELTA]

    def test_final_only_has_no_ttft(self, clock):
        engine, instr = make(clock, step_seconds=0.1)
        engine.add_request(
            "r", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=3, output_kind=RequestOutputKind.FINAL_ONLY)
        )
        run_to_completion(engine)
        t = one(instr)
        assert t.output_length == 3
        assert t.ttft_ms is None and "FINAL_ONLY" in t.ttft_unavailable_reason
        assert t.tpot_ms is None

    def test_pooling_request(self, clock):
        engine, instr = make(clock)
        engine.add_request("r", {"prompt_token_ids": [1, 2, 3]}, PoolingParams())
        run_to_completion(engine)
        t = one(instr)
        assert t.output_kind == "pooling" and t.output_length == 0
        assert t.ttft_ms is None and "pooling" in t.ttft_unavailable_reason
        assert t.prompt_length == 3


class TestPromptLength:
    def test_string_prompt_has_no_length_until_engine_reports_tokens(self, clock):
        engine, instr = make(clock)
        engine.add_request("r", "hello brave new world", SamplingParams(max_tokens=1))
        assert instr._active["r"].trace.prompt_length is None
        assert instr._active["r"].trace.prompt_length_source == "unavailable"
        run_to_completion(engine)
        t = one(instr)
        assert t.prompt_length == 4 and t.prompt_length_source == "engine_prompt_token_ids"

    def test_caller_token_ids_used_before_engine_confirms(self, clock):
        engine, instr = make(clock)
        engine.add_request("r", {"prompt_token_ids": [7, 8, 9]}, SamplingParams(max_tokens=1))
        assert instr._active["r"].trace.prompt_length == 3
        assert instr._active["r"].trace.prompt_length_source == "caller_token_ids"


class TestSpans:
    def test_queue_and_prefill_spans_only_with_scheduler(self, clock):
        # Chunked prefill: 10-token prompt, 4 tokens per step -> 3 prefill steps before the first token.
        engine, instr = make(clock, prefill_chunk=4, step_seconds=0.1)
        engine.add_request("r", {"prompt_token_ids": list(range(10))}, SamplingParams(max_tokens=2))
        clock.advance(0.3)
        run_to_completion(engine)
        t = one(instr)
        phases = [s.phase for s in t.spans]
        assert phases == [SpanPhase.QUEUE, SpanPhase.PREFILL, SpanPhase.DECODE]
        q, p, d = t.spans
        assert q.duration_ms == pytest.approx(300.0)  # arrival -> start of first scheduled step
        assert p.duration_ms == pytest.approx(300.0)  # 3 steps until first token visible
        assert p.metadata["granularity"] == "engine_step"
        assert d.duration_ms == pytest.approx(100.0)
        assert t.ttft_ms == pytest.approx(600.0)
        assert t.queue_duration_ms + t.prefill_duration_ms == pytest.approx(t.ttft_ms)
        assert q.end_time - q.start_time == pytest.approx(0.3)  # wall metadata consistent

    def test_without_scheduler_only_ttft_span(self, clock):
        engine, instr = make(clock, in_process_scheduler=False, prefill_chunk=4, step_seconds=0.1)
        engine.add_request("r", {"prompt_token_ids": list(range(10))}, SamplingParams(max_tokens=2))
        run_to_completion(engine)
        t = one(instr)
        phases = [s.phase for s in t.spans]
        assert phases == [SpanPhase.TIME_TO_FIRST_TOKEN, SpanPhase.DECODE]
        assert t.spans[0].metadata["boundary"] == "not_exposed_by_engine"
        assert t.queue_duration_ms == 0 and t.prefill_duration_ms == 0  # not inferred
        assert t.time_to_first_token_span_ms == pytest.approx(t.ttft_ms)

    def test_monotonic_used_even_if_wall_clock_jumps(self, clock):
        engine, instr = make(clock, step_seconds=0.1)
        engine.add_request("r", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=2))
        clock.wall -= 3600  # NTP step backwards between arrival and completion
        run_to_completion(engine)
        t = one(instr)
        assert t.ttft_ms == pytest.approx(100.0)
        assert t.total_duration_ms == pytest.approx(200.0)
        assert all(s.duration_ms >= 0 for s in t.spans)
