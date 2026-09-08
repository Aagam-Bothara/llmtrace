"""Findings (hypotheses with evidence), decision tables, manifests, collector self-events."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner
from fakes import FakeClock, FakeLLMEngine, FakeNVMLBackend, SamplingParams, run_to_completion

from llmtrace import LLMTracer, TracerConfig, io
from llmtrace.cli import main
from llmtrace.control_plane.decision import Target, evaluate, format_decision
from llmtrace.control_plane.findings import (check_kv_cache_pressure, check_long_prompt_interference, check_queue_overload,
                                             check_tracer_self_effect, evaluate_all, format_findings)
from llmtrace.data_plane.vllm_stats import CollectorEvent, VLLMFinishedRequestStats, VLLMIterationRecord
from llmtrace.manifest import ArrivalRecord, RunManifest, workload_hash
from llmtrace.models.trace import BatchMetadata, RequestSpan, RequestTrace, SpanPhase

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "mixed_prompts"))


def _batch(i, ids, tokens, dur, kv=None):
    return BatchMetadata(batch_id=f"b{i}", step_index=i, timestamp=float(i), monotonic=float(i), step_end_monotonic=float(i) + dur,
                         num_requests=len(ids), num_prefill=0, num_decode=len(ids), total_scheduled_tokens=sum(tokens.values()),
                         request_ids=list(ids), scheduled_tokens=tokens, kv_cache_usage_fraction=kv)


def _trace(rid, batch_ids=(), ttft=5.0, queue_ms=0.0):
    spans = [RequestSpan(phase=SpanPhase.QUEUE, start_time=0, end_time=queue_ms / 1000, duration_ms=queue_ms)] if queue_ms else []
    return RequestTrace(request_id=rid, start_time=0, end_time=1, prompt_length=8, output_length=4, model_name="m",
                        batch_ids=list(batch_ids), ttft_ms=ttft, spans=spans)


def _stat(seq, kv=0.1, waiting=0, preempted=0, queued=None):
    fin = [VLLMFinishedRequestStats(queued_time_s=queued)] if queued is not None else []
    return VLLMIterationRecord(timestamp=float(seq), monotonic=float(seq), step_seq=seq, kv_cache_usage=kv, num_waiting_reqs=waiting,
                               num_preempted_reqs=preempted, finished_requests=fin)


class TestFindings:
    def test_interference_supported_with_victims_and_evidence(self):
        batches = [_batch(0, ["short-0", "short-1"], {"short-0": 1, "short-1": 1}, 0.002),
                   _batch(1, ["short-1", "long-0"], {"short-1": 1, "long-0": 1500}, 0.010),
                   _batch(2, ["short-1", "long-0"], {"short-1": 1, "long-0": 1}, 0.002)]
        traces = [_trace("short-0", ["b0"], ttft=2.0), _trace("short-1", ["b0", "b1", "b2"], ttft=12.0), _trace("long-0", ["b1", "b2"], ttft=10.0)]
        f = check_long_prompt_interference(traces, batches)
        assert f.status == "supported" and f.affected_requests == ["short-1"] and f.parameters["chunk_owners"] == ["long-0"]
        assert any(e.source.startswith("batches_") for e in f.supporting_events)
        assert "long_prefill_token_threshold" in f.suggested_experiment
        assert "waited while the scheduler spent its token budget" in f.summary

    def test_interference_insufficient_evidence_without_batches(self):
        f = check_long_prompt_interference([_trace("short-0")], [])
        assert f.status == "insufficient_evidence" and "VLLM_ENABLE_V1_MULTIPROCESSING=0" in f.missing_evidence[0]

    def test_interference_not_supported_when_steps_not_slower(self):
        batches = [_batch(0, ["short-0", "long-0"], {"short-0": 1, "long-0": 1500}, 0.002), _batch(1, ["short-0"], {"short-0": 1}, 0.002)]
        f = check_long_prompt_interference([_trace("short-0", ["b0", "b1"]), _trace("long-0", ["b0"])], batches)
        assert f.status == "not_supported"

    def test_queue_overload_from_spans_and_vllm_stats(self):
        traces = [_trace("short-0", queue_ms=250.0), _trace("short-1", queue_ms=5.0)]
        f = check_queue_overload(traces, [], [_stat(1, waiting=4, queued=0.3)])
        assert f.status == "supported" and f.affected_requests == ["short-0"]
        srcs = {e.source for e in f.supporting_events}
        assert "traces_*.jsonl:spans[phase=queue]" in srcs and "vllm_stats_*.jsonl:num_waiting_reqs" in srcs
        assert check_queue_overload([_trace("a")], [], []).status == "insufficient_evidence"

    def test_kv_pressure_needs_high_usage_and_preemptions(self):
        traces = [_trace("short-0", ["b0"])]
        f = check_kv_cache_pressure(traces, [_batch(0, ["short-0"], {"short-0": 1}, 0.002, kv=0.97)], [_stat(1, kv=0.97, preempted=2)])
        assert f.status == "supported" and f.affected_requests == ["short-0"]
        assert any("request ids" in m for m in f.missing_evidence)
        f2 = check_kv_cache_pressure(traces, [], [_stat(1, kv=0.97, preempted=0)])
        assert f2.status == "not_supported" and "no preemptions" in f2.summary
        assert check_kv_cache_pressure(traces, [], []).status == "insufficient_evidence"

    def test_tracer_self_effect(self):
        batches = [_batch(i, ["a"], {"a": 1}, 0.002) for i in range(10)] + [_batch(10, ["a"], {"a": 1}, 0.012)]
        ev = [CollectorEvent(timestamp=10.0, monotonic=10.001, duration_ms=9.0)]
        f = check_tracer_self_effect(batches, ev)
        assert f.status == "supported" and f.parameters["steps"] == ["b10"]
        assert check_tracer_self_effect(batches, [CollectorEvent(timestamp=3.0, monotonic=3.0005, duration_ms=1.0)]).status == "not_supported"
        # A 100 ms prefill step overlapping a 0.2 ms drain is not a tracer effect (the drain is immaterial to the excess).
        big = [_batch(i, ["a"], {"a": 1}, 0.011) for i in range(10)] + [_batch(10, ["a", "long-0"], {"a": 1, "long-0": 1536}, 0.100)]
        f2 = check_tracer_self_effect(big, [CollectorEvent(timestamp=10.0, monotonic=10.02, duration_ms=0.2)])
        assert f2.status == "not_supported"

    def test_evaluate_all_and_format(self):
        out = evaluate_all([_trace("short-0")], [], [], [])
        assert [f.hypothesis for f in out] == ["queue_overload", "long_prompt_interference", "kv_cache_pressure", "host_overhead",
                                               "tracer_observer_effect"]
        text = format_findings(out)
        assert "[insufficient_evidence] long_prompt_interference" in text and "missing:" in text


class TestDecision:
    def test_target_parse(self):
        t = Target.parse("short ttft_p95 <= 300ms")
        assert (t.request_class, t.metric, t.stat, t.value_ms) == ("short", "ttft", "p95", 300.0)
        assert Target.parse("* e2e_max < 2000").describe() == "* e2e_max <= 2000 ms"
        with pytest.raises(ValueError):
            Target.parse("fast please")

    def _run(self, tmp_path, name, ttfts, fail=False, tokens=4, status="completed", manifest=None):
        d = tmp_path / name
        d.mkdir()
        if fail:
            (d / "manifest.json").write_text(RunManifest(status="failed", error="CUDA out of memory").model_dump_json())
            return str(d)
        traces = [RequestTrace(request_id=f"short-{i}", start_time=0.0, end_time=1.0, prompt_length=4, output_length=tokens,
                               model_name="m", ttft_ms=v, status=status) for i, v in enumerate(ttfts)]
        io.write_jsonl(d / "traces_x.jsonl", traces)
        if manifest is not None:
            manifest.write(str(d))
        return str(d)

    def test_evaluate_with_failed_config_and_work_check(self, tmp_path):
        base = [self._run(tmp_path, "b0", [100, 120, 400]), self._run(tmp_path, "b1", [110, 130, 380])]
        capped = [self._run(tmp_path, "c0", [90, 95, 100]), self._run(tmp_path, "c1", [92, 96, 110])]
        broken = [self._run(tmp_path, "x0", [], fail=True)]
        uneven = [self._run(tmp_path, "u0", [50, 50, 50]), self._run(tmp_path, "u1", [50, 50, 50], tokens=8)]
        dec = evaluate({"baseline": base, "capped": capped, "big_batch": broken, "uneven": uneven}, Target.parse("short ttft_p95 <= 300ms"))
        by = {c.name: c for c in dec.configs}
        assert by["baseline"].meets_target_all_repeats is False and by["capped"].meets_target_all_repeats is True
        assert by["big_batch"].all_ok is False and by["big_batch"].repeats[0].status == "failed"
        assert by["uneven"].work_identical_across_repeats is False
        assert "capped" in dec.candidates and "uneven" in dec.candidates
        assert any("out of memory" in n for n in dec.notes) and any("not identical" in n for n in dec.notes)
        assert "Advisory" in dec.recommendation
        text = format_decision(dec)
        assert "big_batch" in text and "NO" in text and "candidates meeting the target" in text

    def test_repeat_missing_target_metric_is_ineligible(self, tmp_path):
        good = self._run(tmp_path, "g", [10, 20, 30])
        partial = self._run(tmp_path, "p", [10, None, 30])  # one request without TTFT
        dec = evaluate({"cfg": [good, partial]}, Target.parse("short ttft_p95 <= 100ms"))
        c = dec.configs[0]
        assert c.repeats[1].status == "ineligible" and c.repeats[1].metric_coverage == pytest.approx(2 / 3)
        assert "present for only 67%" in c.repeats[1].problems[0]
        assert c.meets_target_all_repeats is False and dec.candidates == []
        assert any("ineligible" in n for n in dec.notes)
        dec2 = evaluate({"cfg": [good, partial]}, Target.parse("short ttft_p95 <= 100ms"), min_metric_coverage=0.5)
        assert dec2.candidates == ["cfg"]  # only when the user explicitly relaxes coverage

    def test_aborted_requests_make_a_repeat_ineligible(self, tmp_path):
        aborted = self._run(tmp_path, "a", [1.0, 2.0], status="aborted")
        dec = evaluate({"cfg": [aborted]}, Target.parse("short ttft_p95 <= 100ms"))
        r = dec.configs[0].repeats[0]
        assert r.status == "ineligible" and r.aborted == 2 and dec.candidates == []
        assert "aborted" in r.problems[0]

    def test_expected_request_count_and_health_from_manifest(self, tmp_path):
        short = self._run(tmp_path, "s", [1.0, 2.0], manifest=RunManifest(expected_requests=3, health={"instrumentation_errors": 0}))
        sick = self._run(tmp_path, "h", [1.0, 2.0], manifest=RunManifest(expected_requests=2, health={"instrumentation": {"instrumentation_errors": 2}}))
        fine = self._run(tmp_path, "f", [1.0, 2.0], manifest=RunManifest(expected_requests=2, health={"instrumentation_errors": 0}))
        dec = evaluate({"short": [short], "sick": [sick], "fine": [fine]}, Target.parse("short ttft_p95 <= 100ms"))
        by = {c.name: c for c in dec.configs}
        assert "2 traced requests but 3 expected" in by["short"].repeats[0].problems[0]
        assert "health not clean" in by["sick"].repeats[0].problems[0]
        assert dec.candidates == ["fine"]

    def test_exclude_class_reconciles_expected_count(self, tmp_path):
        d = tmp_path / "s"
        d.mkdir()
        traces = [RequestTrace(request_id=f"short-{i}", start_time=0.0, end_time=1.0, prompt_length=4, output_length=4,
                               model_name="m", ttft_ms=10.0) for i in range(3)]
        traces += [RequestTrace(request_id=f"settle-{i}", start_time=0.0, end_time=1.0, prompt_length=4, output_length=4,
                                model_name="m", ttft_ms=10.0) for i in range(2)]
        io.write_jsonl(d / "traces_x.jsonl", traces)
        RunManifest(expected_requests=3).write(str(d))  # old driver counted the workload only
        assert evaluate({"c": [str(d)]}, Target.parse("short ttft_p95 <= 20ms")).candidates == []
        dec = evaluate({"c": [str(d)]}, Target.parse("short ttft_p95 <= 20ms"), exclude_classes=["settle"])
        assert dec.candidates == ["c"] and dec.configs[0].repeats[0].requests == 3
        assert any("open-loop" in n for n in dec.notes)
        r = CliRunner().invoke(main, ["decide", "--target", "short ttft_p95 <= 20ms", "--config", f"c={d}", "--exclude-class", "settle"])
        assert r.exit_code == 0 and "candidates meeting the target in every repeat: c" in r.output

    def test_ttft_sched_target_uses_manifest_delays(self, tmp_path):
        m = RunManifest(expected_requests=2, arrivals=[ArrivalRecord(request_id="short-0", scheduled_s=0, actual_s=0.05, delay_ms=50.0),
                                                       ArrivalRecord(request_id="short-1", scheduled_s=0, actual_s=0.0, delay_ms=0.0)])
        d = self._run(tmp_path, "d", [10.0, 10.0], manifest=m)
        eng = evaluate({"c": [d]}, Target.parse("short ttft_p95 <= 20ms"))
        sched = evaluate({"c": [d]}, Target.parse("short ttft_sched_p95 <= 20ms"))
        assert eng.candidates == ["c"] and sched.candidates == []  # 60 ms from intended arrival
        assert sched.configs[0].repeats[0].target_value_ms == pytest.approx(60.0)
        assert sched.configs[0].repeats[0].arrival_delay_ms_max == pytest.approx(50.0)

    def test_cli_decide_and_findings(self, tmp_path):
        a = self._run(tmp_path, "a0", [10, 20])
        b = self._run(tmp_path, "b0", [5, 6])
        r = CliRunner().invoke(main, ["decide", "--target", "short ttft_p95 <= 15ms", "--config", f"A={a}", "--config", f"B={b}",
                                      "--json", str(tmp_path / "d.json")])
        assert r.exit_code == 0, r.output
        assert "candidates meeting the target in every repeat: B" in r.output
        assert json.loads((tmp_path / "d.json").read_text())["candidates"] == ["B"]
        r = CliRunner().invoke(main, ["findings", a, "--json", str(tmp_path / "f.json")])
        assert r.exit_code == 0 and "queue_overload" in r.output and (tmp_path / "f.json").exists()
        assert CliRunner().invoke(main, ["decide", "--target", "nonsense", "--config", f"A={a}"]).exit_code == 2


class TestManifestAndCollectorEvents:
    def test_manifest_roundtrip_and_arrival_delays(self, tmp_path):
        m = RunManifest(label="r", engine="fake", workload={"n": 1}, workload_hash=workload_hash([{"a": 1}]),
                        arrivals=[ArrivalRecord(request_id="a", scheduled_s=0.0, actual_s=0.004, delay_ms=4.0),
                                  ArrivalRecord(request_id="b", scheduled_s=0.1, actual_s=0.15, delay_ms=50.0)])
        m.write(str(tmp_path))
        back = RunManifest.read(str(tmp_path))
        assert back.arrival_delay_ms_p50 == 50.0 and back.arrival_delay_ms_max == 50.0 and back.workload_hash.startswith("sha256:")
        assert workload_hash([{"a": 1}]) == workload_hash([{"a": 1}]) != workload_hash([{"a": 2}])

    def test_gpu_loader_does_not_pick_up_gpu_steps(self, tmp_path):
        from conftest import mk_sample

        from llmtrace.data_plane.cuda_timing import StepGpuTiming
        io.write_jsonl(tmp_path / "gpu_x.jsonl", [mk_sample(1.0)])
        io.write_jsonl(tmp_path / "gpu_steps_x.jsonl", [StepGpuTiming(step_index=1, timestamp=0, host_step_ms=1.0, gpu_span_ms=0.5)])
        assert len(io.load_gpu_samples([tmp_path])) == 1 and len(io.load_gpu_steps([tmp_path])) == 1

    def test_tracer_writes_collector_events(self, tmp_path):
        tracer = LLMTracer(TracerConfig(output_dir=str(tmp_path), collection_interval_s=0.02), gpu_backend=FakeNVMLBackend({0: 1.0}))
        engine = FakeLLMEngine(clock=FakeClock())
        tracer.instrument_engine(engine)
        engine.add_request("r", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=3))
        run_to_completion(engine)
        tracer.stop()
        ev = io.load_collector_events(tracer.get_output_files()["collector"])
        assert ev and all(e.duration_ms >= 0 and e.clock_domain == tracer.session_id for e in ev)
        assert sum(e.traces for e in ev) == 1 and sum(e.batches for e in ev) == 3

    def test_experiment_driver_writes_manifest_with_arrivals(self, tmp_path):
        import subprocess
        out = tmp_path / "fake_run"
        cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "experiments" / "mixed_prompts" / "run.py"),
               "--engine", "fake", "--config", "capped", "--out", str(out), "--num-short", "6", "--num-long", "1"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
        m = RunManifest.read(str(out))
        assert m.status == "ok" and m.synthetic and len(m.arrivals) == 7 and m.workload_hash
        assert m.scheduling_change == {"long_prefill_token_threshold": 256} and m.tracer_config["collection_interval_s"] > 0
        assert m.expected_requests == 7 and all(a.submit_ms is not None and a.submit_ms >= 0 for a in m.arrivals)  # fake path has no settle phase
        assert all(a.delay_ms is not None and a.delay_ms >= -0.01 for a in m.arrivals)  # driver tolerance is 1 us
