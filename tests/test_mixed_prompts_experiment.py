"""CPU tests for the mixed-prompt experiment: fake scheduler knobs, driver, diagnosis and comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fakes import FakeClock, FakeLLMEngine, RequestOutputKind, SamplingParams, run_to_completion

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "mixed_prompts"))
from analyze import analyze_run, compare, explain  # noqa: E402
from run import drive  # noqa: E402
from workload import WorkloadConfig, build_workload, make_prompt  # noqa: E402

from llmtrace.data_plane.vllm_instrumentation import VLLMInstrumentation
from llmtrace.models.trace import BatchMetadata, RequestTrace


class TestFakeSchedulerKnobs:
    def test_long_prefill_threshold_caps_chunk_and_budget_limits_step(self):
        clock = FakeClock()
        engine = FakeLLMEngine(clock=clock, long_prefill_token_threshold=256, max_num_batched_tokens=300,
                               step_seconds=0.001, step_seconds_per_token=1e-5)
        instr = VLLMInstrumentation(monotonic=clock.monotonic, wall=clock.time)
        instr.instrument_engine(engine)
        engine.add_request("long", {"prompt_token_ids": list(range(1000))}, SamplingParams(max_tokens=1))
        engine.add_request("short", {"prompt_token_ids": list(range(10))}, SamplingParams(max_tokens=2))
        run_to_completion(engine)
        batches = instr.drain_batch_metadata()
        first = batches[0]
        assert first.scheduled_tokens["long"] == 256  # capped by the threshold
        assert first.scheduled_tokens["short"] == 10  # fits in the remaining budget (300 - 256)
        assert max(b.scheduled_tokens.get("long", 0) for b in batches) == 256
        # step time grows with scheduled tokens
        durations = [(b.step_end_monotonic - b.monotonic) for b in batches]
        assert durations[0] == pytest.approx(0.001 + 1e-5 * 266)

    def test_budget_starves_waiting_request_until_budget_frees(self):
        clock = FakeClock()
        engine = FakeLLMEngine(clock=clock, max_num_batched_tokens=100)
        instr = VLLMInstrumentation(monotonic=clock.monotonic, wall=clock.time)
        instr.instrument_engine(engine)
        engine.add_request("a", {"prompt_token_ids": list(range(100))}, SamplingParams(max_tokens=2))
        engine.add_request("b", {"prompt_token_ids": list(range(5))}, SamplingParams(max_tokens=1))
        run_to_completion(engine)
        batches = instr.drain_batch_metadata()
        assert batches[0].request_ids == ["a"]  # b did not fit in step 1
        assert "b" in batches[1].request_ids
        traces = {t.request_id: t for t in instr.drain_completed_traces()}
        assert traces["b"].output_length == 1 and traces["a"].output_length == 2


class TestWorkloadAndDriver:
    def test_workload_is_deterministic_and_sorted(self):
        cfg = WorkloadConfig(num_short=10, short_rate_per_s=10, num_long=2, seed=3)
        a, b = build_workload(cfg), build_workload(cfg)
        assert a == b
        assert [s.arrival_s for s in a] == sorted(s.arrival_s for s in a)
        assert make_prompt(a[0], 1000, 3) == make_prompt(a[0], 1000, 3)
        assert len(make_prompt(a[0], 1000, 3)["prompt_token_ids"]) == cfg.short_prompt_len

    def test_driver_respects_arrival_times(self):
        clock = FakeClock()
        engine = FakeLLMEngine(clock=clock, step_seconds=0.01)
        instr = VLLMInstrumentation(monotonic=clock.monotonic, wall=clock.time)
        instr.instrument_engine(engine)
        specs = build_workload(WorkloadConfig(num_short=3, short_rate_per_s=2, num_long=1, first_long_at_s=0.75,
                                              short_max_tokens=2, long_max_tokens=1, long_prompt_len=64))
        res = drive(engine, specs, lambda s: SamplingParams(max_tokens=s.max_tokens, output_kind=RequestOutputKind.CUMULATIVE),
                    clock.monotonic, lambda t: clock.advance(max(0.0, t - clock.mono)), 1000, 0)
        assert len(res["finished"]) == 4
        traces = {t.request_id: t for t in instr.drain_completed_traces()}
        arrivals = {s.request_id: s.arrival_s for s in specs}
        for rid, t in traces.items():
            assert t.start_monotonic - 1000.0 == pytest.approx(arrivals[rid], abs=0.011)  # added at (or just after) arrival


def _batch(i: int, ids, tokens, dur: float, start: float = None) -> BatchMetadata:
    t0 = float(i) if start is None else start
    return BatchMetadata(batch_id=f"b{i}", step_index=i, timestamp=t0, monotonic=t0,
                         step_end_monotonic=t0 + dur, num_requests=len(ids), num_prefill=0, num_decode=len(ids),
                         total_scheduled_tokens=sum(tokens.values()), request_ids=list(ids), scheduled_tokens=tokens)


def _trace(rid: str, batch_ids, tpot: float, ttft: float = 5.0, first_token: float = None) -> RequestTrace:
    return RequestTrace(request_id=rid, start_time=0, end_time=10, prompt_length=8, output_length=4, model_name="m",
                        batch_ids=batch_ids, tpot_ms=tpot, ttft_ms=ttft, first_token_monotonic=first_token)


class TestDiagnosis:
    def test_interference_attribution_and_step_model(self):
        # Steps 0,1: short-only, 2 ms. Step 2: long chunk (1000 tokens) shared with s1, 10 ms. Step 3: short-only.
        batches = [
            _batch(0, ["short-0", "short-1"], {"short-0": 1, "short-1": 1}, 0.002),
            _batch(1, ["short-0", "short-1"], {"short-0": 1, "short-1": 1}, 0.002),
            _batch(2, ["short-1", "long-0"], {"short-1": 1, "long-0": 1000}, 0.010),
            _batch(3, ["short-1", "long-0"], {"short-1": 1, "long-0": 1}, 0.002),
        ]
        traces = [_trace("short-0", ["b0", "b1"], tpot=2.0), _trace("short-1", ["b0", "b1", "b2", "b3"], tpot=4.0),
                  RequestTrace(request_id="long-0", start_time=0, end_time=10, prompt_length=1000, output_length=2,
                               model_name="m", batch_ids=["b2", "b3"], tpot_ms=2.0, ttft_ms=12.0)]
        a = analyze_run(traces, batches, chunk_threshold=128)
        assert a["steps"]["steps"] == 4 and a["steps"]["steps_with_long_chunk"] == 1
        i = a["interference"]
        assert i["short_requests_sharing_a_long_chunk_step"] == 1
        # s0: 4 ms in normal steps; s1: 2+2+10+2 = 16 ms of which 10 in the long-chunk step -> 10/20 = 50%
        assert i["share_of_short_step_time_in_long_chunk_steps"] == pytest.approx(0.5)
        assert i["tpot_ms_affected"]["p95"] == 4.0 and i["tpot_ms_unaffected"]["p95"] == 2.0
        assert a["latency"]["short"]["step_ms"]["max"] == pytest.approx(10.0)  # the shared long-chunk step (compute proxy)
        assert a["latency"]["short"]["step_ms"]["p50"] == pytest.approx(2.0)
        assert a["latency"]["short"]["itl_ms"]["n"] == 0  # no first-token time on these synthetic traces: no real ITL
        m = a["step_time_model"]
        assert m["us_per_token"] == pytest.approx(8.0, rel=0.01)  # (10-2) ms over 998 extra tokens
        assert "shared at least one step with a long prefill chunk" in explain(a)

    def test_unexplained_stall_is_reported(self):
        # 20 uniform 2 ms decode steps, then one 30 ms step with the same token count: not explained by tokens.
        batches = [_batch(i, ["short-0"], {"short-0": 1}, 0.002) for i in range(20)]
        batches.append(_batch(20, ["short-0"], {"short-0": 1}, 0.030))
        batches.append(_batch(21, ["short-0", "long-0"], {"short-0": 1, "long-0": 1000}, 0.010))
        tr = [_trace("short-0", [b.batch_id for b in batches], tpot=2.0)]
        a = analyze_run(tr, batches, 128)
        u = a["unexplained_stalls"]
        assert u["count"] == 1 and u["top"][0]["step_index"] == 20
        assert u["top"][0]["residual_ms"] > 20
        assert "unexplained stalls" in explain(a) and "step 20 at 20.000s" in explain(a)

    def test_no_batches_reports_unavailable(self):
        a = analyze_run([_trace("short-0", [], tpot=2.0)], [], 128)
        assert not a["batch_metadata_available"] and a["step_time_model"] is None
        assert "unavailable" in explain(a)

    def test_compare_verdicts(self):
        base = analyze_run([_trace(f"short-{i}", [], tpot=10.0) for i in range(5)] + [_trace("long-0", [], tpot=1.0, ttft=50.0)], [], 128)
        better = analyze_run([_trace(f"short-{i}", [], tpot=5.0) for i in range(5)] + [_trace("long-0", [], tpot=1.0, ttft=80.0)], [], 128)
        same = analyze_run([_trace(f"short-{i}", [], tpot=9.5) for i in range(5)] + [_trace("long-0", [], tpot=1.0, ttft=50.0)], [], 128)
        c = compare(base, better)
        assert set(c["verdict_metrics"]) == {"short_ttft_ms_p95", "short_tpot_ms_p95"}  # no batch data: TPOT fallback
        assert c["verdict_metrics"]["short_tpot_ms_p95"] == pytest.approx(-50.0)
        assert c["verdict"] == "no_meaningful_change"  # TTFT did not move: both metrics must improve
        assert c["long_ttft_change_pct"] == pytest.approx(60.0)  # cost is reported, not hidden
        assert compare(base, same)["verdict"] == "no_meaningful_change"
        assert compare(better, base)["verdict"] == "worse"  # any metric regressing beyond threshold is "worse"
        faster_all = analyze_run([_trace(f"short-{i}", [], tpot=5.0, ttft=2.0) for i in range(5)]
                                 + [_trace("long-0", [], tpot=1.0, ttft=80.0)], [], 128)
        assert compare(base, faster_all)["verdict"] == "improved"

    def test_compare_uses_real_itl_when_batches_exist(self):
        # Contiguous steps; ITL = intervals between the short request's successive step ends after its first token.
        slow = [_batch(0, ["short-0"], {"short-0": 1}, 0.002, start=0.0),
                _batch(1, ["short-0", "long-0"], {"short-0": 1, "long-0": 1500}, 0.014, start=0.002),
                _batch(2, ["short-0", "long-0"], {"short-0": 1, "long-0": 1}, 0.002, start=0.016)]
        fast = [_batch(0, ["short-0"], {"short-0": 1}, 0.002, start=0.0),
                _batch(1, ["short-0", "long-0"], {"short-0": 1, "long-0": 256}, 0.0036, start=0.002),
                _batch(2, ["short-0", "long-0"], {"short-0": 1, "long-0": 256}, 0.002, start=0.0056)]
        tr = [_trace("short-0", ["b0", "b1", "b2"], tpot=8.0, ttft=8.0, first_token=0.002), _trace("long-0", ["b1", "b2"], tpot=1.0, ttft=14.0)]
        tr2 = [_trace("short-0", ["b0", "b1", "b2"], tpot=2.8, ttft=2.0, first_token=0.002), _trace("long-0", ["b1", "b2"], tpot=1.0, ttft=22.0)]
        a, b = analyze_run(tr, slow, 128), analyze_run(tr2, fast, 128)
        assert a["latency"]["short"]["itl_ms"]["max"] == pytest.approx(14.0) and a["latency"]["short"]["step_ms"]["max"] == pytest.approx(14.0)
        assert b["latency"]["short"]["itl_ms"]["max"] == pytest.approx(3.6)
        c = compare(a, b)
        assert set(c["verdict_metrics"]) == {"short_ttft_ms_p95", "short_itl_ms_max"} and c["verdict"] == "improved"
        assert c["verdict_metrics"]["short_itl_ms_max"] == pytest.approx((3.6 - 14.0) / 14.0 * 100, rel=1e-3)
        assert c["long_ttft_change_pct"] == pytest.approx(57.14, rel=1e-3)

    def test_itl_includes_unscheduled_gaps(self):
        # short-0 is scheduled in b0 and b2 but not b1: its token interval spans the gap.
        batches = [_batch(0, ["short-0"], {"short-0": 1}, 0.002, start=0.0), _batch(1, ["long-0"], {"long-0": 100}, 0.005, start=0.002),
                   _batch(2, ["short-0"], {"short-0": 1}, 0.002, start=0.007)]
        a = analyze_run([_trace("short-0", ["b0", "b2"], tpot=1.0, first_token=0.002)], batches, 128)
        assert a["latency"]["short"]["itl_ms"]["max"] == pytest.approx(7.0)  # 0.009 - 0.002
        assert a["latency"]["short"]["step_ms"]["max"] == pytest.approx(2.0)

    def test_compare_unavailable_when_candidate_ttft_missing(self):
        base = analyze_run([_trace(f"short-{i}", [], tpot=10.0) for i in range(3)] + [_trace("long-0", [], tpot=1.0, ttft=50.0)], [], 128)
        cand_traces = [_trace(f"short-{i}", [], tpot=5.0) for i in range(3)] + [_trace("long-0", [], tpot=1.0, ttft=50.0)]
        for t in cand_traces:
            t.ttft_ms = None  # e.g. FINAL_ONLY outputs
        c = compare(base, analyze_run(cand_traces, [], 128))
        assert c["verdict"] == "unavailable" and c["missing_metrics"] == ["short_ttft_ms_p95"]

    def test_ttft_from_intended_arrival_uses_delays(self):
        a = analyze_run([_trace("short-0", [], tpot=1.0, ttft=5.0), _trace("short-1", [], tpot=1.0, ttft=5.0)], [], 128,
                        arrival_delays_ms={"short-0": 20.0, "short-1": -0.001})
        assert a["latency"]["short"]["ttft_sched_ms"]["max"] == pytest.approx(25.0)
        assert a["latency"]["short"]["ttft_sched_ms"]["p50"] == pytest.approx(5.0)  # negative delay clamps to 0
        assert a["latency"]["short"]["arrival_delay_ms"]["max"] == pytest.approx(20.0)
