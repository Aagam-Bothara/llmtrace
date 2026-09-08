"""CUDA-event step timing: timer semantics, instrumentation hook, tracer wiring, findings, analysis split."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fakes import FakeClock, FakeLLMEngine, FakeNVMLBackend, SamplingParams, run_to_completion

from llmtrace import LLMTracer, TracerConfig, io
from llmtrace.control_plane.findings import check_host_overhead, evaluate_all
from llmtrace.data_plane.cuda_timing import CudaStepTimer, FakeCudaBackend, StepGpuTiming
from llmtrace.data_plane.vllm_instrumentation import VLLMInstrumentation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "mixed_prompts"))


class TestTimer:
    def test_lazy_resolution_never_blocks(self):
        backend = FakeCudaBackend(ms_per_call=2.0, ready_after_polls=2)
        t = CudaStepTimer(backend=backend, clock_domain="d")
        assert t.start()
        t.begin_step(1, 100.0, 10.0)
        t.before_execute()
        t.after_execute()
        t.end_step(host_step_ms=5.0)  # polls once: not ready yet
        assert t.stats()["pending"] == 1 and t.drain() == []  # drain polls again: still not ready (2 polls needed)
        out = t.drain()  # third poll resolves
        assert len(out) == 1
        r = out[0]
        assert r.step_index == 1 and r.gpu_span_ms == pytest.approx(2.0) and r.host_overhead_ms == pytest.approx(3.0)
        assert r.resolved_after_steps == 0 and r.executor_calls == 1 and r.clock_domain == "d"

    def test_step_without_executor_call_is_host_only(self):
        t = CudaStepTimer(backend=FakeCudaBackend())
        t.start()
        t.begin_step(1, 0.0, 0.0)
        t.end_step(host_step_ms=1.5)
        (r,) = t.drain()
        assert r.gpu_span_ms is None and r.source == "no_executor_call" and r.host_step_ms == 1.5

    def test_multiple_executor_calls_sum_and_pending_is_bounded(self):
        t = CudaStepTimer(backend=FakeCudaBackend(ms_per_call=1.0, ready_after_polls=10_000), max_pending=2)
        t.start()
        for i in range(4):
            t.begin_step(i, 0.0, 0.0)
            for _ in range(2):
                t.before_execute()
                t.after_execute()
            t.end_step(4.0)
        s = t.stats()
        assert s["pending"] == 2 and s["dropped"] == 2

    def test_unavailable_backend(self):
        t = CudaStepTimer(backend=FakeCudaBackend(fail_available="torch.cuda not available"))
        assert not t.start() and t.unavailable_reason == "torch.cuda not available"
        t.begin_step(1, 0, 0)
        t.before_execute()
        t.after_execute()
        t.end_step(1.0)
        assert t.drain() == [] and t.stats()["available"] is False

    def test_nvtx_ranges(self):
        backend = FakeCudaBackend()
        t = CudaStepTimer(backend=backend, enable_nvtx=True)
        t.start()
        t.begin_step(7, 0, 0, label="llmtrace step 7")
        t.end_step(1.0)
        assert backend.nvtx == ["push:llmtrace step 7", "pop"]


class TestInstrumentationHook:
    def test_executor_wrapped_and_restored(self, clock):
        engine = FakeLLMEngine(clock=clock, step_seconds=0.004)
        backend = FakeCudaBackend(ms_per_call=1.5)
        timer = CudaStepTimer(backend=backend)
        instr = VLLMInstrumentation(monotonic=clock.monotonic, wall=clock.time, cuda_timer=timer)
        instr.instrument_engine(engine)
        executor = engine.engine_core.engine_core.model_executor
        assert instr.executor_visible and "execute_model" in executor.__dict__
        engine.add_request("r", {"prompt_token_ids": [1, 2]}, SamplingParams(max_tokens=3))
        run_to_completion(engine)
        assert executor.calls == 3
        recs = timer.drain()
        assert [r.step_index for r in recs] == [1, 2, 3]
        assert all(r.gpu_span_ms == pytest.approx(1.5) and r.executor_calls == 1 for r in recs)
        assert all(r.host_step_ms == pytest.approx(4.0) and r.host_overhead_ms == pytest.approx(2.5) for r in recs)
        instr.uninstrument_engine()
        assert "execute_model" not in executor.__dict__ and not instr.executor_visible

    def test_multiprocess_core_reports_reason(self, clock):
        engine = FakeLLMEngine(clock=clock, in_process_scheduler=False)
        instr = VLLMInstrumentation(monotonic=clock.monotonic, wall=clock.time, cuda_timer=CudaStepTimer(backend=FakeCudaBackend()))
        instr.instrument_engine(engine)
        assert not instr.executor_visible and "not reachable in-process" in instr.executor_unavailable_reason
        assert instr.health()["executor_visible"] is False

    def test_out_of_process_executor_is_refused(self, clock):
        engine = FakeLLMEngine(clock=clock)
        core = engine.engine_core.engine_core

        class MultiprocExecutor:  # name matters: vLLM's TP>1 executor forwards to worker processes
            def __init__(self, inner):
                self.inner = inner

            def execute_model(self, so):
                return self.inner.execute_model(so)

        core.model_executor = MultiprocExecutor(core.model_executor)
        timer = CudaStepTimer(backend=FakeCudaBackend())
        instr = VLLMInstrumentation(monotonic=clock.monotonic, wall=clock.time, cuda_timer=timer)
        instr.instrument_engine(engine)
        assert not instr.executor_visible and "MultiprocExecutor" in instr.executor_unavailable_reason
        assert "execute_model" not in core.model_executor.__dict__
        engine.add_request("r", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=2))
        run_to_completion(engine)
        assert timer.drain() == []  # nothing mis-measured

    def test_cuda_unavailable_reports_reason(self, clock):
        engine = FakeLLMEngine(clock=clock)
        instr = VLLMInstrumentation(monotonic=clock.monotonic, wall=clock.time,
                                    cuda_timer=CudaStepTimer(backend=FakeCudaBackend(fail_available="torch not installed")))
        instr.instrument_engine(engine)
        assert not instr.executor_visible and "torch not installed" in instr.executor_unavailable_reason
        assert "execute_model" not in engine.engine_core.engine_core.model_executor.__dict__

    def test_engine_exception_still_closes_step(self, clock):
        engine = FakeLLMEngine(clock=clock)
        timer = CudaStepTimer(backend=FakeCudaBackend())
        instr = VLLMInstrumentation(monotonic=clock.monotonic, wall=clock.time, cuda_timer=timer)
        instr.instrument_engine(engine)
        engine.add_request("r", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=1))
        engine._make_output = lambda req: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            engine.step()
        assert timer._current is None and len(timer.drain()) == 1  # step recorded, no dangling state


class TestTracerAndAnalysis:
    def _run(self, tmp_path, in_process=True):
        clock = FakeClock()
        tracer = LLMTracer(TracerConfig(output_dir=str(tmp_path), collection_interval_s=0.02),
                           gpu_backend=FakeNVMLBackend({0: 1.0}), cuda_backend=FakeCudaBackend(ms_per_call=0.7))
        engine = FakeLLMEngine(clock=clock, in_process_scheduler=in_process, step_seconds=0.002)
        tracer.vllm_instrumentation._monotonic = clock.monotonic
        tracer.vllm_instrumentation._wall = clock.time
        tracer.instrument_engine(engine)
        for i in range(2):
            engine.add_request(f"short-{i}", {"prompt_token_ids": [1, 2]}, SamplingParams(max_tokens=3))
        run_to_completion(engine)
        tracer.stop()
        return tracer

    def test_tracer_writes_gpu_steps_and_health(self, tmp_path):
        tracer = self._run(tmp_path)
        h = tracer.health()
        assert h["executor_visible_during_run"] is True and h["cuda_timing"]["available"] and h["cuda_timing"]["dropped"] == 0
        steps = io.load_gpu_steps(tracer.get_output_files()["gpu_steps"])
        assert len(steps) == 3 and all(s.gpu_span_ms == pytest.approx(0.7) for s in steps)
        assert all(s.gpu_span_ms <= s.host_step_ms for s in steps)

    def test_tracer_without_in_process_core(self, tmp_path):
        tracer = self._run(tmp_path, in_process=False)
        assert tracer.health()["executor_visible_during_run"] is False
        assert "gpu_steps" not in tracer.get_output_files()

    def test_host_overhead_finding(self):
        big_host = [StepGpuTiming(step_index=i, timestamp=0, host_step_ms=4.0, gpu_span_ms=1.0, host_overhead_ms=3.0) for i in range(5)]
        f = check_host_overhead([], big_host)
        assert f.status == "supported" and "75%" in f.summary and "upper bound" in f.summary
        gpu_bound = [StepGpuTiming(step_index=i, timestamp=0, host_step_ms=4.0, gpu_span_ms=3.6, host_overhead_ms=0.4) for i in range(5)]
        assert check_host_overhead([], gpu_bound).status == "not_supported"
        f3 = check_host_overhead([], [])
        assert f3.status == "insufficient_evidence" and "gpu_steps_" in f3.missing_evidence[0]
        assert [x.hypothesis for x in evaluate_all([], [], [], [], gpu_steps=big_host)][3] == "host_overhead"

    def test_analysis_splits_gpu_and_host(self, tmp_path):
        from analyze import analyze_run, explain

        tracer = self._run(tmp_path)
        files = tracer.get_output_files()
        a = analyze_run(io.load_traces(files["traces"]), io.load_batches(files["batches"]), 128, None, io.load_gpu_steps(files["gpu_steps"]))
        g = a["gpu_split"]
        assert g["steps_with_gpu_span"] == 3 and g["gpu_span_ms"]["p50"] == pytest.approx(0.7)
        assert g["host_overhead_ms"]["p50"] == pytest.approx(1.3) and 0.6 < g["host_share_median"] < 0.7
        assert "GPU span (CUDA events) on 3 steps" in explain(a)
        assert "unavailable" in explain(analyze_run(io.load_traces(files["traces"]), io.load_batches(files["batches"]), 128))
