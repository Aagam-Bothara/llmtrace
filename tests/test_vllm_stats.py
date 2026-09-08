"""vLLM stat_loggers integration, exercised with fakes shaped like vLLM 0.11.0."""

from __future__ import annotations

from fakes import (FakeFinishedRequestStats, FakeIterationStats, FakeLLMEngine, FakeNVMLBackend, FakeSchedulerStats,
                   FakeStatLoggerManager, SamplingParams, run_to_completion)

from llmtrace import LLMTracer, TracerConfig, io
from llmtrace.control_plane.reporter import summarize_vllm_stats
from llmtrace.data_plane.vllm_stats import (LLMTraceStatLogger, VLLMStatsSink, attach_to_engine, detach_from_engine,
                                            make_stat_logger_factory, record_from_stats)


def test_factory_logger_records_fields():
    sink = VLLMStatsSink(clock_domain="d")
    mgr = FakeStatLoggerManager([make_stat_logger_factory(sink)])
    assert sink.loggers_created == 1
    mgr.record(FakeSchedulerStats(num_running_reqs=3, num_waiting_reqs=2, kv_cache_usage=0.42),
               FakeIterationStats(num_generation_tokens=3, num_preempted_reqs=1, time_to_first_tokens_iter=[0.01],
                                  inter_token_latencies_iter=[0.002, 0.003],
                                  finished_requests=[FakeFinishedRequestStats("length", 0.5, 10, 20, 20, 0.01, 0.02, 0.4, 0.38, 0.019)]))
    (r,) = sink.drain()
    assert r.num_running_reqs == 3 and r.num_waiting_reqs == 2 and r.kv_cache_usage == 0.42
    assert r.num_preempted_reqs == 1 and r.time_to_first_tokens_s == [0.01] and r.inter_token_latencies_s == [0.002, 0.003]
    assert r.finished_requests[0].queued_time_s == 0.01 and r.finished_requests[0].finish_reason == "length"
    assert r.clock_domain == "d" and r.step_seq == 1 and r.monotonic is not None


def test_missing_fields_become_none_and_errors_do_not_propagate():
    r = record_from_stats(None, None, 0, 1, None)
    assert r.kv_cache_usage is None and r.num_preempted_reqs is None and r.finished_requests == []
    sink = VLLMStatsSink()
    lg = LLMTraceStatLogger(sink)

    class Bad:
        @property
        def finished_requests(self):
            raise RuntimeError("boom")

    lg.record(None, Bad())  # must not raise into the engine
    assert sink.errors == 1 and "boom" in sink.last_error
    lg.log_engine_initialized(); lg.log()
    assert sink.engine_initialized == [0]


def test_attach_and_detach_post_hoc():
    engine = FakeLLMEngine()
    sink = VLLMStatsSink()
    assert attach_to_engine(engine, sink) is None
    assert "already attached" in attach_to_engine(engine, sink)
    engine.add_request("r", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=3))
    run_to_completion(engine)
    recs = sink.drain()
    assert len(recs) == 3 and recs[0].num_running_reqs == 1
    assert detach_from_engine(engine) == 1 and engine.logger_manager.per_engine_logger_dict[0] == []
    engine.logger_manager = None
    assert "disable_log_stats" in attach_to_engine(engine, sink)


def test_tracer_writes_and_summarizes_vllm_stats(tmp_path):
    tracer = LLMTracer(TracerConfig(output_dir=str(tmp_path), collection_interval_s=0.02), gpu_backend=FakeNVMLBackend({0: 1.0}))
    engine = FakeLLMEngine()
    tracer.instrument_engine(engine)
    for i in range(3):
        engine.add_request(f"r{i}", {"prompt_token_ids": [1, 2]}, SamplingParams(max_tokens=4))
    run_to_completion(engine)
    tracer.stop()
    assert tracer.health()["vllm_stats"]["unavailable_reason"] is None
    assert engine.logger_manager.per_engine_logger_dict[0] == []  # detached on stop
    files = tracer.get_output_files()
    recs = io.load_vllm_stats(files["vllm_stats"])
    assert len(recs) == 4 and max(r.num_running_reqs for r in recs) == 3
    summ = summarize_vllm_stats(recs)
    assert summ["steps"] == 4 and summ["num_running_max"] == 3 and summ["preemptions"] == 0
    assert any("vLLM engine stats" in n for n in tracer.analyze().energy_ledger.notes)


def test_tracer_reports_reason_when_stats_disabled(tmp_path):
    tracer = LLMTracer(TracerConfig(output_dir=str(tmp_path), collection_interval_s=0.02), gpu_backend=FakeNVMLBackend({0: 1.0}))
    engine = FakeLLMEngine()
    engine.logger_manager = None
    tracer.instrument_engine(engine)
    tracer.stop()
    assert "disable_log_stats" in tracer.health()["vllm_stats"]["unavailable_reason"]
    assert "vllm_stats" not in tracer.get_output_files()
