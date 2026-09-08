"""Health assessment (full record), run-directory reuse, and plans reproducing the source configuration."""

from __future__ import annotations

import copy
import json

import pytest
from click.testing import CliRunner

from llmtrace import io
from llmtrace.cli import main
from llmtrace.control_plane.decision import Target, evaluate
from llmtrace.control_plane.experiments import ExperimentPlan, plan_experiments, source_engine_kwargs
from llmtrace.control_plane.findings import SUPPORTED, Finding
from llmtrace.doctor import run_report
from llmtrace.health import assess_health
from llmtrace.manifest import RunManifest
from llmtrace.models.trace import RequestTrace
from llmtrace.runner import RunOptions, existing_run_files, prepare_run_dir, run_workload
from llmtrace.workload import ArrivalSpec, LengthSpec, RequestClass, WorkloadSpec

CLEAN = {
    "state": "stopped", "session_id": "s",
    "instrumentation": {"instrumentation_errors": 0, "active_requests": 0, "dropped_traces": 0, "dropped_batches": 0},
    "cuda_timing": {"available": True, "errors": 0, "dropped": 0, "unavailable_reason": None},
    "gpu_sampler": {"available": True, "read_errors": 0, "dropped": 0, "unavailable_reason": None},
    "vllm_stats": {"errors": 0, "dropped": 0, "unavailable_reason": None},
    "writer": {"written": {"traces": 2}, "dropped": {"traces": 0, "gpu": 0}, "write_errors": 0, "last_error": None},
    "collection_errors": 0, "last_collection_error": None,
}


def _health(**over):
    h = copy.deepcopy(CLEAN)
    for path, v in over.items():
        keys = path.split("__")
        d = h
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = v
    return h


class TestAssessHealth:
    def test_clean(self):
        a = assess_health(CLEAN)
        assert a.ok and not a.problems and not a.telemetry_problems and a.gpu_telemetry_ok and a.gpu_steps_ok and a.vllm_stats_ok

    @pytest.mark.parametrize("over, fragment", [
        ({"writer__write_errors": 3, "writer__last_error": "disk full"}, "3 writer error"),
        ({"collection_errors": 2, "last_collection_error": "boom"}, "2 collector error"),
        ({"instrumentation__instrumentation_errors": 1}, "instrumentation error"),
        ({"instrumentation__active_requests": 4}, "still active"),
        ({"instrumentation__dropped_traces": 5}, "trace(s) dropped"),
        ({"writer__dropped": {"traces": 7}}, "dropped by the writer"),
    ])
    def test_fatal_problems(self, over, fragment):
        a = assess_health(_health(**over))
        assert not a.ok and any(fragment in p for p in a.problems)

    def test_telemetry_problems_are_not_fatal(self):
        a = assess_health(_health(gpu_sampler__available=False, gpu_sampler__unavailable_reason="no NVML",
                                  cuda_timing__available=False, cuda_timing__unavailable_reason="no torch.cuda",
                                  vllm_stats__unavailable_reason="no logger_manager",
                                  writer__dropped={"traces": 0, "batches": 9}, instrumentation__dropped_batches=2))
        assert a.ok and not a.problems
        assert a.gpu_telemetry_ok is False and a.gpu_steps_ok is False and a.vllm_stats_ok is False
        assert len(a.telemetry_problems) == 5
        lossy = assess_health(_health(gpu_sampler__read_errors=3, gpu_sampler__dropped=10))
        assert lossy.ok and lossy.gpu_telemetry_ok is False

    def test_nested_only_and_missing(self):
        assert assess_health({"instrumentation_errors": 0, "active_requests": 0}).ok
        assert not assess_health({"instrumentation_errors": 2}).ok
        assert not assess_health({}).ok and not assess_health(None).ok


class TestDecideUsesFullHealth:
    def _run(self, tmp_path, name, health):
        d = tmp_path / name
        d.mkdir()
        io.write_jsonl(d / "traces_x.jsonl", [RequestTrace(request_id=f"short-{i}", start_time=0, end_time=1, prompt_length=4,
                                                            output_length=4, model_name="m", ttft_ms=10.0) for i in range(3)])
        RunManifest(engine="fake", health=health, expected_requests=3).write(str(d))
        return str(d)

    def test_writer_and_collector_errors_make_a_repeat_ineligible(self, tmp_path):
        bad = self._run(tmp_path, "bad", _health(writer__write_errors=3, collection_errors=2, last_collection_error="x"))
        good = self._run(tmp_path, "good", CLEAN)
        dec = evaluate({"bad": [bad], "good": [good]}, Target.parse("short ttft_p95 <= 20ms"))
        by = {c.name: c for c in dec.configs}
        r = by["bad"].repeats[0]
        assert r.status == "ineligible" and r.health_ok is False and any("writer" in p for p in r.health_problems)
        assert any("collector" in p for p in r.health_problems)
        assert dec.candidates == ["good"]
        assert any("bad" in n and "tracer health not clean" in n for n in dec.notes)

    def test_lossy_gpu_telemetry_keeps_latency_but_drops_energy(self, tmp_path):
        from conftest import const_power
        d = self._run(tmp_path, "lossy", _health(gpu_sampler__read_errors=4))
        io.write_jsonl(tmp_path / "lossy" / "gpu_x.jsonl", const_power(0.0, 1.0, 0.1, 100.0))
        dec = evaluate({"lossy": [d]}, Target.parse("short ttft_p95 <= 20ms"))
        r = dec.configs[0].repeats[0]
        assert r.eligible and r.meets_target and r.joules_per_output_token is None and r.device_joules is None
        assert any("read error" in t for t in r.telemetry_problems)
        assert any("energy not compared" in n for n in dec.notes)
        clean = self._run(tmp_path, "clean", CLEAN)
        io.write_jsonl(tmp_path / "clean" / "gpu_x.jsonl", const_power(0.0, 1.0, 0.1, 100.0))
        r2 = evaluate({"clean": [clean]}, Target.parse("short ttft_p95 <= 20ms")).configs[0].repeats[0]
        assert r2.device_joules is not None

    def test_doctor_reports_full_health(self, tmp_path):
        d = self._run(tmp_path, "bad", _health(writer__write_errors=1, gpu_sampler__available=False, gpu_sampler__unavailable_reason="no NVML"))
        rep = run_report(d)
        assert any(c.name == "tracer health" and c.status == "error" for c in rep.checks)
        assert any(c.name == "telemetry" and c.status == "warn" for c in rep.checks)


def _spec(n=2):
    return WorkloadSpec(name="two", classes=[RequestClass(name="a", count=n, prompt_len=LengthSpec(value=8), max_tokens=LengthSpec(value=2))])


class TestRunDirReuse:
    def test_second_run_into_same_dir_is_refused(self, tmp_path):
        out = tmp_path / "r"
        m1 = run_workload(_spec(), RunOptions(engine="fake", out_dir=str(out)))
        assert m1.status == "ok" and len(io.load_traces([str(out)])) == 2
        with pytest.raises(FileExistsError):
            run_workload(_spec(), RunOptions(engine="fake", out_dir=str(out)))
        assert len(io.load_traces([str(out)])) == 2 and RunManifest.read(str(out)).expected_requests == 2
        m2 = run_workload(_spec(), RunOptions(engine="fake", out_dir=str(out), overwrite=True))
        assert m2.status == "ok" and len(io.load_traces([str(out)])) == 2  # previous files removed, not appended
        assert len(existing_run_files(str(out))) >= 4

    def test_prepare_run_dir_keeps_foreign_files(self, tmp_path):
        out = tmp_path / "x"
        out.mkdir()
        (out / "notes.txt").write_text("keep me")
        assert existing_run_files(str(out)) == []
        prepare_run_dir(str(out))  # nothing to refuse
        (out / "traces_old.jsonl").write_text("")
        with pytest.raises(FileExistsError):
            prepare_run_dir(str(out))
        prepare_run_dir(str(out), overwrite=True)
        assert (out / "notes.txt").exists() and not (out / "traces_old.jsonl").exists()

    def test_cli_refuses_reuse_and_overwrites_on_request(self, tmp_path):
        _spec().save(str(tmp_path / "w.json"))
        r = CliRunner()
        args = ["run", "--workload", str(tmp_path / "w.json"), "--engine", "fake", "--out", str(tmp_path / "o")]
        assert r.invoke(main, args).exit_code == 0
        res = r.invoke(main, args)
        assert res.exit_code == 2 and "already holds a run" in (res.output + (res.stderr or ""))
        assert r.invoke(main, args + ["--overwrite"]).exit_code == 0
        assert len(io.load_traces([str(tmp_path / "o")])) == 2

    def test_experiment_driver_refuses_reuse(self, tmp_path):
        import subprocess
        import sys
        from pathlib import Path
        cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "experiments" / "mixed_prompts" / "run.py"),
               "--engine", "fake", "--config", "baseline", "--out", str(tmp_path / "e"), "--num-short", "2", "--num-long", "1"]
        assert subprocess.run(cmd, capture_output=True, text=True, timeout=120).returncode == 0
        second = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert second.returncode == 2 and "already holds a run" in second.stdout
        assert subprocess.run(cmd + ["--overwrite"], capture_output=True, text=True, timeout=120).returncode == 0
        assert len(io.load_traces([str(tmp_path / "e")])) == 3


class TestPlanReproducesSource:
    def test_source_engine_kwargs_from_manifest(self):
        m = RunManifest(engine="vllm", model="org/model", model_revision="abc123",
                        engine_kwargs={"max_num_batched_tokens": 512, "max_model_len": 4096},
                        scheduling_change={"long_prefill_token_threshold": 128},
                        effective_engine_config={"scheduler_config": {"max_num_batched_tokens": 512, "max_num_seqs": 256},
                                                 "parallel_config": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
                                                 "model_config": {"revision": "abc123"}})
        kw = source_engine_kwargs(m)
        assert kw == {"max_num_batched_tokens": 512, "max_model_len": 4096, "long_prefill_token_threshold": 128,
                      "tensor_parallel_size": 2, "revision": "abc123"}
        p = plan_experiments([Finding(hypothesis="queue_overload", status=SUPPORTED, summary="s")], m, [])
        assert p.source_engine == "vllm" and p.source_model == "org/model" and p.source_engine_kwargs == kw
        cfgs = p.configs()
        assert cfgs[0] == {"name": "baseline", "engine_kwargs": kw, "scheduling_change": {}}
        assert all(c["engine_kwargs"] == kw for c in cfgs)
        assert "baseline engine kwargs" in p.format() and "revision abc123" in p.format()

    def test_run_plan_baseline_keeps_source_budget(self, tmp_path):
        spec = WorkloadSpec(name="mini", seed=1, classes=[
            RequestClass(name="short", count=12, prompt_len=LengthSpec(value=32), max_tokens=LengthSpec(value=8),
                         arrival=ArrivalSpec(kind="constant", rate_per_s=200.0)),
            RequestClass(name="long", count=1, prompt_len=LengthSpec(value=1536), max_tokens=LengthSpec(value=2),
                         arrival=ArrivalSpec(kind="constant", rate_per_s=20.0, start_s=0.01))])
        spec.save(str(tmp_path / "w.json"))
        # source run with a 512-token budget passed as an engine kwarg
        src = run_workload(spec, RunOptions(engine="fake", out_dir=str(tmp_path / "src"), engine_kwargs={"max_num_batched_tokens": 512}))
        assert src.status == "ok" and src.effective_engine_config["fake_engine"]["max_num_batched_tokens"] == 512
        r = CliRunner()
        res = r.invoke(main, ["plan", str(tmp_path / "src"), "--repeats", "1", "--max-candidates", "1", "--json", str(tmp_path / "plan.json")])
        assert res.exit_code == 0, res.output
        p = ExperimentPlan.model_validate_json((tmp_path / "plan.json").read_text())
        assert p.source_engine == "fake" and p.source_engine_kwargs == {"max_num_batched_tokens": 512}
        res = r.invoke(main, ["run", "--workload", str(tmp_path / "w.json"), "--plan", str(tmp_path / "plan.json"), "--out", str(tmp_path / "exp")])
        assert res.exit_code == 0, res.output  # engine taken from the plan (fake)
        base = RunManifest.read(str(tmp_path / "exp" / "baseline" / "r0"))
        assert base.engine_kwargs == {"max_num_batched_tokens": 512}
        assert base.effective_engine_config["fake_engine"]["max_num_batched_tokens"] == 512  # not the 8192 default
        cand = RunManifest.read(str(tmp_path / "exp" / p.candidates[0].name / "r0"))
        assert cand.engine_kwargs == {"max_num_batched_tokens": 512} and cand.scheduling_change == p.candidates[0].scheduling_change
        assert cand.effective_engine_config["fake_engine"]["max_num_batched_tokens"] == 512
        # a plan from a vllm run cannot be silently run on the fake engine without a warning
        p2 = p.model_copy(update={"source_engine": "vllm", "source_model": "org/m"})
        (tmp_path / "plan2.json").write_text(p2.model_dump_json())
        res = r.invoke(main, ["run", "--workload", str(tmp_path / "w.json"), "--plan", str(tmp_path / "plan2.json"), "--engine", "fake",
                              "--out", str(tmp_path / "exp2")])
        assert res.exit_code == 0 and "made from a vllm run, running on fake" in res.output
        assert json.loads((tmp_path / "exp2" / "baseline" / "r0" / "manifest.json").read_text())["engine_kwargs"] == {"max_num_batched_tokens": 512}
