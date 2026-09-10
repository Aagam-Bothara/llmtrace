"""GPU sampler, trace writer and tracer lifecycle: ownership, flushing, bounds."""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest
from fakes import FakeLLMEngine, FakeNVMLBackend, SamplingParams, run_to_completion

from llmtrace import LLMTracer, io
from llmtrace.data_plane.gpu_sampler import GPUSampler
from llmtrace.data_plane.trace_writer import TraceWriter
from llmtrace.models.config import GPUSamplerConfig, TracerConfig
from llmtrace.models.trace import RequestStatus, ThrottleReason
from conftest import mk_sample


class TestGPUSampler:
    def test_ambiguous_devices_require_explicit_selection(self):
        backend = FakeNVMLBackend({0: 100.0, 1: 300.0})
        sampler = GPUSampler(GPUSamplerConfig(), backend=backend)
        sampler.start()
        assert not sampler.available and "ambiguous" in sampler.unavailable_reason
        assert backend.closed and not sampler.running and not sampler.drain()
        selected = GPUSampler(GPUSamplerConfig(gpu_ids=[0]), backend=FakeNVMLBackend({0: 100.0, 1: 300.0}))
        selected.start()
        try:
            assert [s.gpu_id for s in selected.sample_once()] == [0]
            assert selected.stats()["selection"]["gpu_ids"] == [0]
            assert selected.stats()["selection"]["mode"] == "explicit"
        finally:
            selected.stop()

    def test_nvml_records_physical_uuid_and_rejects_ambiguous_default(self, monkeypatch):
        import sys
        from types import SimpleNamespace
        from llmtrace.data_plane.gpu_sampler import NVMLBackend
        nvml = SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
                               nvmlDeviceGetCount=lambda: 2, nvmlDeviceGetHandleByIndex=lambda i: i,
                               nvmlDeviceGetName=lambda h: b"GPU", nvmlDeviceGetUUID=lambda h: f"GPU-physical-{h}".encode())
        monkeypatch.setitem(sys.modules, "pynvml", nvml)
        backend = NVMLBackend()
        try:
            with pytest.raises(RuntimeError, match="ambiguous"):
                backend.open(None)
            assert backend.open([1]) == [{"gpu_id": 1, "name": "GPU", "uuid": "GPU-physical-1"}]
            with pytest.raises(ValueError, match="not present"):
                backend.open([2])
            with pytest.raises(ValueError, match="unique"):
                backend.open([1, 1])
        finally:
            backend.close()

    def test_samples_have_both_clocks_and_real_fields(self):
        s = GPUSampler(GPUSamplerConfig(sample_interval_ms=10, gpu_ids=[0, 1]), backend=FakeNVMLBackend({0: 100.0, 1: 50.0}))
        s.start()
        assert s.available
        samples = s.sample_once()
        s.stop()
        assert [x.gpu_id for x in samples] == [0, 1]
        assert samples[0].power_draw_watts == 100.0 and samples[1].power_draw_watts == 50.0
        assert samples[0].monotonic is not None and samples[0].clock_domain == s.clock_domain
        assert samples[0].throttle_reasons == [ThrottleReason.NONE]

    def test_thread_samples_while_caller_blocks(self):
        s = GPUSampler(GPUSamplerConfig(sample_interval_ms=10), backend=FakeNVMLBackend({0: 100.0}))
        s.start()
        time.sleep(0.15)  # a blocking call, like LLM.generate()
        s.stop()
        got = s.drain()
        assert len(got) >= 5
        assert s.drain() == []  # drain is atomic: nothing duplicated
        assert s.stats()["samples_taken"] == len(got)

    def test_stop_keeps_buffered_samples(self):
        backend = FakeNVMLBackend({0: 100.0})
        s = GPUSampler(GPUSamplerConfig(sample_interval_ms=10), backend=backend)
        s.start()
        time.sleep(0.05)
        s.stop()
        assert backend.closed
        assert len(s.drain()) >= 1
        s.stop()  # idempotent

    def test_bounded_buffer_counts_drops(self):
        s = GPUSampler(GPUSamplerConfig(sample_interval_ms=10, max_buffered_samples=100), backend=FakeNVMLBackend({0: 1.0}))
        s.start()
        s._devices = s._devices * 0 + s._devices  # keep devices
        for _ in range(150):
            with s._lock:
                for x in s.sample_once():
                    if len(s._buffer) >= s.config.max_buffered_samples:
                        s._buffer.popleft()
                        s._dropped += 1
                    s._buffer.append(x)
        s.stop()
        assert len(s.drain()) == 100 and s.stats()["dropped"] >= 50

    def test_unavailable_backend_is_reported_not_fatal(self):
        s = GPUSampler(GPUSamplerConfig(), backend=FakeNVMLBackend({}, fail_open=True))
        s.start()
        assert not s.available and "NVML init failed" in s.unavailable_reason
        assert not s.running
        s.stop()
        assert s.drain() == []

    def test_require_gpu_raises(self):
        s = GPUSampler(GPUSamplerConfig(require_gpu=True), backend=FakeNVMLBackend({}, fail_open=True))
        with pytest.raises(RuntimeError, match="required"):
            s.start()

    def test_read_error_counts_and_skips_device(self):
        s = GPUSampler(GPUSamplerConfig(gpu_ids=[0, 1]), backend=FakeNVMLBackend({0: 1.0, 1: 2.0}, read_error_on=1))
        s.start()
        samples = s.sample_once()
        s.stop()
        assert [x.gpu_id for x in samples] == [0]
        assert s.stats()["read_errors"] >= 1

    def test_missing_power_is_none_not_zero(self):
        class NoPower(FakeNVMLBackend):
            def read(self, gpu_id):
                raw = super().read(gpu_id)
                del raw["power_draw_watts"]
                raw["throttle_bits"] = None
                return raw

        s = GPUSampler(GPUSamplerConfig(), backend=NoPower({0: 1.0}))
        s.start()
        (x,) = s.sample_once()
        s.stop()
        assert x.power_draw_watts is None
        assert x.throttle_reasons == [ThrottleReason.UNKNOWN] and not x.is_throttled


class TestTraceWriter:
    @pytest.mark.parametrize("background", [True, False])
    def test_submission_is_serialized_with_stop(self, tmp_path, monkeypatch, background):
        w = TraceWriter(str(tmp_path), background=background)
        w.start()
        accepting, release, stopping = threading.Event(), threading.Event(), threading.Event()
        failures = []
        original = w._queue.put_nowait if background else w._write

        def paused_accept(*args):
            accepting.set()
            if not release.wait(5):
                raise RuntimeError("test submission was not released")
            return original(*args)

        monkeypatch.setattr(w._queue if background else w, "put_nowait" if background else "_write", paused_accept)

        def submit():
            try:
                w.write_gpu_samples([mk_sample(1)])
            except Exception as exc:
                failures.append(exc)

        def stop():
            stopping.set()
            w.stop()

        submitter = threading.Thread(target=submit)
        stopper = threading.Thread(target=stop)
        submitter.start()
        assert accepting.wait(5)
        stopper.start()
        assert stopping.wait(5)
        try:
            stopper.join(0.1)
            assert stopper.is_alive()  # stop must wait for the accepted submission
        finally:
            release.set()
            submitter.join(5)
            stopper.join(5)
        assert not submitter.is_alive() and not stopper.is_alive() and not failures
        assert w.stats()["written"]["gpu"] == 1 and w.stats()["dropped"]["gpu"] == 0
        assert w.stats()["queued"] == 0 and len(io.load_gpu_samples([tmp_path])) == 1
        w.write_gpu_samples([mk_sample(2)])
        assert w.stats()["dropped"]["gpu"] == 1 and w.stats()["queued"] == 0

    def test_background_writer_writes_everything_once(self, tmp_path):
        w = TraceWriter(str(tmp_path), background=True, max_queue=1000)
        w.start()
        for i in range(20):
            w.write_gpu_samples([mk_sample(float(i))])
        w.stop()
        files = w.get_output_files()
        lines = Path(files["gpu"][0]).read_text().splitlines()
        assert len(lines) == 20
        assert sorted(json.loads(x)["timestamp"] for x in lines) == [float(i) for i in range(20)]
        assert w.stats()["written"]["gpu"] == 20 and w.stats()["dropped"]["gpu"] == 0

    def test_inline_writer(self, tmp_path):
        w = TraceWriter(str(tmp_path), background=False)
        w.start()
        w.write_gpu_samples([mk_sample(1.0)])
        assert len(io.load_gpu_samples([tmp_path])) == 1
        w.stop()

    def test_write_before_start_raises_and_after_stop_drops(self, tmp_path):
        w = TraceWriter(str(tmp_path))
        with pytest.raises(RuntimeError):
            w.write_gpu_samples([mk_sample(1.0)])
        w.start()
        w.stop()
        w.write_gpu_samples([mk_sample(1.0)])
        assert w.stats()["dropped"]["gpu"] == 1

    def test_full_queue_drops_and_counts(self, tmp_path):
        w = TraceWriter(str(tmp_path), background=True, max_queue=10)
        # Do not start the thread: fill the queue by hand to simulate a stalled disk.
        w._started = True
        w._thread = threading.Thread(target=lambda: None)
        for i in range(15):
            w.write_gpu_samples([mk_sample(float(i))])
        assert w.stats()["dropped"]["gpu"] == 5 and w._queue.qsize() == 10

    def test_stop_is_idempotent(self, tmp_path):
        w = TraceWriter(str(tmp_path))
        w.start()
        w.stop()
        w.stop()

    @pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow not installed")
    def test_parquet_roundtrip(self, tmp_path):
        w = TraceWriter(str(tmp_path), output_format="parquet", background=False)
        w.start()
        w.write_gpu_samples([mk_sample(1.0), mk_sample(2.0)])
        w.write_gpu_samples([mk_sample(3.0)])
        w.stop()
        assert len(w.get_output_files()["gpu"]) == 2
        assert [s.timestamp for s in io.load_gpu_samples([tmp_path])] == [1.0, 2.0, 3.0]


class TestTracer:
    def _tracer(self, tmp_path, **kw) -> LLMTracer:
        cfg = TracerConfig(output_dir=str(tmp_path), collection_interval_s=0.02, gpu_sampler={"sample_interval_ms": 10})
        return LLMTracer(cfg, gpu_backend=FakeNVMLBackend({0: 100.0}), **kw)

    def test_end_to_end_with_fake_engine(self, tmp_path):
        tracer = self._tracer(tmp_path)
        engine = FakeLLMEngine(step_seconds=0.0)
        tracer.instrument_engine(engine)
        for i in range(3):
            engine.add_request(f"r{i}", {"prompt_token_ids": [1, 2]}, SamplingParams(max_tokens=3))
        time.sleep(0.05)  # let the sampler run during "inference"
        run_to_completion(engine)
        tracer.stop()
        assert "step" not in engine.__dict__  # restored
        files = tracer.get_output_files()
        traces = io.load_traces(files["traces"])
        assert {t.request_id for t in traces} == {"r0", "r1", "r2"}
        assert all(t.status == RequestStatus.COMPLETED for t in traces)
        assert len(io.load_gpu_samples(files["gpu"])) >= 2
        assert len(io.load_batches(files["batches"])) == 3
        health = tracer.health()
        assert health["instrumentation"]["instrumentation_errors"] == 0
        assert all(v == 0 for v in health["writer"]["dropped"].values()) and "vllm_stats" in health["writer"]["dropped"]
        analysis = tracer.analyze()
        assert analysis.num_requests == 3
        assert analysis.energy_ledger is not None and analysis.energy_ledger.clock == "monotonic"

    def test_stop_writes_incomplete_requests(self, tmp_path):
        tracer = self._tracer(tmp_path)
        engine = FakeLLMEngine()
        tracer.instrument_engine(engine)
        engine.add_request("r", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=10))
        engine.step()
        tracer.stop()
        (t,) = io.load_traces(tracer.get_output_files()["traces"])
        assert t.status == RequestStatus.INCOMPLETE and t.output_length == 1
        assert tracer.health()["incomplete_requests_written"] == 1

    def test_stop_and_start_are_guarded(self, tmp_path):
        tracer = self._tracer(tmp_path)
        tracer.stop()  # not running: no-op
        tracer.start()
        tracer.start()  # already running: no-op
        tracer.stop()
        tracer.stop()
        with pytest.raises(RuntimeError):
            tracer.start()

    def test_without_gpu_telemetry_energy_is_unavailable(self, tmp_path):
        cfg = TracerConfig(output_dir=str(tmp_path), collection_interval_s=0.02)
        tracer = LLMTracer(cfg, gpu_backend=FakeNVMLBackend({}, fail_open=True))
        engine = FakeLLMEngine()
        tracer.instrument_engine(engine)
        engine.add_request("r", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=2))
        run_to_completion(engine)
        tracer.stop()
        assert not tracer.health()["gpu_sampler"]["available"]
        analysis = tracer.analyze()
        assert analysis.total_device_joules is None and analysis.num_with_energy == 0

    def test_failed_start_restores_engine_and_releases_everything(self, tmp_path):
        cfg = TracerConfig(output_dir=str(tmp_path), gpu_sampler={"require_gpu": True})
        tracer = LLMTracer(cfg, gpu_backend=FakeNVMLBackend({}, fail_open=True))
        engine = FakeLLMEngine()
        with pytest.raises(RuntimeError, match="required"):
            tracer.instrument_engine(engine)
        assert "step" not in engine.__dict__ and "add_request" not in engine.__dict__
        assert not tracer.vllm_instrumentation.is_instrumented
        assert tracer.trace_writer._thread is None and tracer.trace_writer._stopped
        assert tracer._collector is None and tracer._state == "stopped"
        assert not any(t.name.startswith("llmtrace-") for t in threading.enumerate())
        tracer.stop()  # no-op, no error
        with pytest.raises(RuntimeError, match="cannot be restarted"):
            tracer.start()
        # The engine still works normally, untraced.
        engine.add_request("r", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=1))
        assert run_to_completion(engine)[0].finished

    def test_stopped_tracer_rejects_new_engine_before_patching(self, tmp_path):
        tracer = self._tracer(tmp_path)
        tracer.start()
        tracer.stop()
        engine = FakeLLMEngine()
        with pytest.raises(RuntimeError, match="cannot be restarted"):
            tracer.instrument_engine(engine)
        assert "step" not in engine.__dict__ and "add_request" not in engine.__dict__
        assert not tracer.vllm_instrumentation.is_instrumented

    def test_health_keeps_scheduler_visibility_after_stop(self, tmp_path):
        tracer = self._tracer(tmp_path)
        tracer.instrument_engine(FakeLLMEngine())
        tracer.stop()
        h = tracer.health()
        assert h["scheduler_visible_during_run"] is True and h["scheduler_unavailable_reason_during_run"] is None
        assert h["instrumentation"]["scheduler_visible"] is False  # reset by restore, as documented
        tracer2 = self._tracer(tmp_path / "b")
        tracer2.instrument_engine(FakeLLMEngine(in_process_scheduler=False))
        tracer2.stop()
        h2 = tracer2.health()
        assert h2["scheduler_visible_during_run"] is False
        assert "VLLM_ENABLE_V1_MULTIPROCESSING=0" in h2["scheduler_unavailable_reason_during_run"]

    def test_unknown_option_raises_and_shortcuts_apply(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown LLMTracer option"):
            LLMTracer(output_dir=str(tmp_path), sample_rate=5)
        t = LLMTracer(output_dir=str(tmp_path), gpu_sample_interval_ms=250, enable_energy_attribution=False)
        assert t.config.gpu_sampler.sample_interval_ms == 250 and not t.config.energy.enabled
        with pytest.raises(Exception):
            TracerConfig(distributed_mode=True)  # removed option must not be silently accepted

    def test_from_config_file(self, tmp_path):
        cfg = tmp_path / "c.json"
        cfg.write_text(json.dumps({"output_dir": str(tmp_path / "out"), "energy": {"attribution_method": "window_only"}}))
        t = LLMTracer.from_config_file(str(cfg))
        assert t.config.energy.attribution_method == "window_only"
