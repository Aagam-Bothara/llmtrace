"""Health assessment (full record), run-directory reuse, and plans reproducing the source configuration."""

from __future__ import annotations

import copy
import json

import pytest
from conftest import comparison_manifest
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
        # records lost in the writer queue count against the signal they belong to
        w = assess_health(_health(writer__dropped={"traces": 0, "gpu": 10, "gpu_steps": 2, "vllm_stats": 1, "batches": 3}))
        assert w.ok and w.gpu_telemetry_ok is False and w.gpu_steps_ok is False and w.vllm_stats_ok is False
        assert sum("dropped by the writer" in t for t in w.telemetry_problems) == 4

    def test_nested_only_and_missing(self):
        assert assess_health({"instrumentation_errors": 0, "active_requests": 0}).ok
        assert not assess_health({"instrumentation_errors": 2}).ok
        assert not assess_health({}).ok and not assess_health(None).ok


class TestDecideUsesFullHealth:
    @pytest.mark.parametrize("kind", ["empty", "one", "gap", "partial_allocation", "window_only"])
    def test_unavailable_energy_is_not_zero_consumption(self, tmp_path, kind):
        from conftest import const_power, mk_sample
        d = self._run(tmp_path, "energy", CLEAN)
        samples = {"empty": [], "one": [mk_sample(0.5, 100)],
                   "gap": [mk_sample(0, 100), mk_sample(2, 100)]}.get(kind, const_power(0, 1, 0.1, 100))
        if kind == "partial_allocation":
            traces = io.load_traces([d])
            traces[0].start_time, traces[0].end_time = 1, 1.1
            io.write_jsonl(tmp_path / "energy" / "traces_x.jsonl", traces)
        io.write_jsonl(tmp_path / "energy" / "gpu_x.jsonl", samples)
        dec = evaluate({"energy": [d]}, Target.parse("short ttft_p95 <= 20ms"), min_repeats=1,
                       attribution="window_only" if kind == "window_only" else "equal_share")
        r = dec.configs[0].repeats[0]
        assert r.eligible and dec.candidates == ["energy"]
        assert r.energy_withheld and r.joules_per_output_token is None and r.device_joules is None
        assert r.energy_unavailable_reason and any("energy unavailable" in p for p in r.telemetry_problems)
        if kind == "one":
            assert r.telemetry_coverage == 0

    def test_manifest_gpu_selection_is_used_by_decide_and_analyze(self, tmp_path):
        from conftest import const_power
        from llmtrace.cli import _correlate_dir
        d = self._run(tmp_path, "selected", CLEAN)
        samples = const_power(0, 1, 0.1, 100, gpu_id=0) + const_power(0, 1, 0.1, 300, gpu_id=1)
        io.write_jsonl(tmp_path / "selected" / "gpu_x.jsonl", samples)
        m = RunManifest.read(d)
        m.gpu_selection = {"mode": "explicit", "gpu_ids": [0], "devices": [{"gpu_id": 0, "uuid": "GPU-physical-0"}]}
        m.write(d)
        r = evaluate({"selected": [d]}, Target.parse("short ttft_p95 <= 20ms"), min_repeats=1).configs[0].repeats[0]
        assert r.eligible and not r.energy_withheld and r.energy_gpu_ids == [0]
        assert r.device_joules == pytest.approx(100) and r.joules_per_output_token == pytest.approx(100 / 12)
        assert _correlate_dir([d], None, "equal_share").ledger.device_joules == pytest.approx(100)
        output = tmp_path / "decision.json"
        cli = CliRunner().invoke(main, ["decide", "--target", "short ttft_p95 <= 20ms", "--config", f"selected={d}",
                                       "--min-repeats", "1", "--gpu-id", "1", "--json", str(output)])
        assert cli.exit_code == 0, cli.output
        repeat = json.loads(output.read_text())["configs"][0]["repeats"][0]
        assert repeat["energy_gpu_ids"] == [1] and repeat["device_joules"] == pytest.approx(300)

    def test_covered_zero_power_is_a_valid_zero_estimate(self, tmp_path):
        from conftest import const_power
        d = self._run(tmp_path, "zero", CLEAN)
        io.write_jsonl(tmp_path / "zero" / "gpu_x.jsonl", const_power(0, 1, 0.1, 0))
        r = evaluate({"zero": [d]}, Target.parse("short ttft_p95 <= 20ms")).configs[0].repeats[0]
        assert r.joules_per_output_token == 0 and r.telemetry_coverage == pytest.approx(1)
        assert not r.energy_withheld and r.energy_unavailable_reason is None

    def _run(self, tmp_path, name, health):
        d = tmp_path / name
        d.mkdir()
        io.write_jsonl(d / "traces_x.jsonl", [RequestTrace(request_id=f"short-{i}", start_time=0, end_time=1, prompt_length=4,
                                                            output_length=4, model_name="m", ttft_ms=10.0) for i in range(3)])
        health = {**health, "session_id": name}  # distinct tracer sessions: a repeated session is a duplicate
        comparison_manifest(3, health=health, expected_requests=3).write(str(d))
        return str(d)

    def test_writer_and_collector_errors_make_a_repeat_ineligible(self, tmp_path):
        bad = self._run(tmp_path, "bad", _health(writer__write_errors=3, collection_errors=2, last_collection_error="x"))
        good = self._run(tmp_path, "good", CLEAN)
        dec = evaluate({"bad": [bad], "good": [good]}, Target.parse("short ttft_p95 <= 20ms"), min_repeats=1)
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
        assert any("read error" in t for t in r.telemetry_problems) and r.energy_withheld
        assert any("energy not compared" in n for n in dec.notes)
        # a problem with another signal keeps the energy figures and says so
        stats_only = self._run(tmp_path, "so", _health(vllm_stats__unavailable_reason="no logger_manager"))
        io.write_jsonl(tmp_path / "so" / "gpu_x.jsonl", const_power(0.0, 1.0, 0.1, 100.0))
        d2 = evaluate({"so": [stats_only]}, Target.parse("short ttft_p95 <= 20ms"))
        r_so = d2.configs[0].repeats[0]
        assert not r_so.energy_withheld and r_so.device_joules is not None
        assert any("telemetry incomplete:" in n and "energy not compared" not in n for n in d2.notes)
        writer_lossy = self._run(tmp_path, "wl", _health(writer__dropped={"traces": 0, "gpu": 10}))
        io.write_jsonl(tmp_path / "wl" / "gpu_x.jsonl", const_power(0.0, 1.0, 0.1, 100.0))
        rw = evaluate({"wl": [writer_lossy]}, Target.parse("short ttft_p95 <= 20ms")).configs[0].repeats[0]
        assert rw.eligible and rw.joules_per_output_token is None and rw.device_joules is None
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


class TestIsolatedRun:
    def test_child_process_writes_manifest_and_parent_reads_it(self, tmp_path):
        from llmtrace.runner import run_workload_isolated
        out = tmp_path / "iso"
        m = run_workload_isolated(_spec(), RunOptions(engine="fake", out_dir=str(out)), timeout_s=300)
        assert m.status == "ok" and m.finished == 2 and len(io.load_traces([str(out)])) == 2
        with pytest.raises(FileExistsError):
            run_workload_isolated(_spec(), RunOptions(engine="fake", out_dir=str(out)))
        m2 = run_workload_isolated(_spec(), RunOptions(engine="fake", out_dir=str(out), overwrite=True), timeout_s=300)
        assert m2.status == "ok" and len(io.load_traces([str(out)])) == 2

    def test_nonzero_exit_after_manifest_is_failed_and_excluded_from_decide(self, tmp_path):
        from llmtrace.runner import run_workload_isolated
        from llmtrace.testing.fakes import isolated_entry_exit_nonzero
        good = run_workload_isolated(_spec(), RunOptions(engine="fake", out_dir=str(tmp_path / "good")), timeout_s=300)
        crashed = run_workload_isolated(_spec(), RunOptions(engine="fake", out_dir=str(tmp_path / "crashed")), timeout_s=300,
                                        entry=isolated_entry_exit_nonzero)
        assert good.status == "ok"
        assert crashed.status == "failed" and "exited with code 3" in (crashed.error or "")
        on_disk = RunManifest.read(str(tmp_path / "crashed"))
        assert on_disk.status == "failed" and on_disk.error == crashed.error
        assert len(io.load_traces([str(tmp_path / "crashed")])) == 2  # the data the child wrote is kept, but not trusted
        dec = evaluate({"good": [str(tmp_path / "good")], "crashed": [str(tmp_path / "crashed")],
                        "mixed": [str(tmp_path / "good"), str(tmp_path / "crashed")]}, Target.parse("a ttft_p95 <= 1000ms"), min_repeats=1)
        by = {c.name: c for c in dec.configs}
        assert by["crashed"].repeats[0].status == "failed" and "exited with code 3" in by["crashed"].repeats[0].error
        assert by["crashed"].all_ok is False and by["crashed"].meets_target_all_repeats is False
        assert by["mixed"].all_eligible is False and by["mixed"].meets_target_all_repeats is False
        assert dec.candidates == ["good"]
        assert any("crashed" in n and "failed" in n for n in dec.notes)

    def test_timeout_after_manifest_is_failed(self, tmp_path):
        from llmtrace.runner import run_workload_isolated
        from llmtrace.testing.fakes import isolated_entry_hang
        m = run_workload_isolated(_spec(), RunOptions(engine="fake", out_dir=str(tmp_path / "hung")), timeout_s=20,
                                  entry=isolated_entry_hang)
        assert m.status == "failed" and "timed out after 20 s" in (m.error or "")
        assert RunManifest.read(str(tmp_path / "hung")).status == "failed"
        dec = evaluate({"hung": [str(tmp_path / "hung")]}, Target.parse("a ttft_p95 <= 1000ms"))
        assert dec.configs[0].repeats[0].status == "failed" and dec.candidates == []

    def test_child_failure_is_a_failed_manifest(self, tmp_path):
        from llmtrace.runner import run_workload_isolated
        out = tmp_path / "bad"
        m = run_workload_isolated(_spec(), RunOptions(engine="fake", out_dir=str(out), engine_kwargs={"bogus": 1}), timeout_s=300)
        assert m.status == "failed" and "TypeError" in (m.error or "")
        out2 = tmp_path / "nope"
        m2 = run_workload_isolated(_spec(), RunOptions(engine="nope", out_dir=str(out2)), timeout_s=300)
        assert m2.status == "failed"


class TestProvenance:
    def test_fingerprint_is_content_based(self, tmp_path):
        from llmtrace.provenance import source_fingerprint
        a = tmp_path / "a"
        a.mkdir()
        (a / "x.py").write_text("print(1)\n")
        f1 = source_fingerprint(a)
        assert f1 == source_fingerprint(a) and f1.startswith("sha256:")
        (a / "x.py").write_text("print(2)\n")
        assert source_fingerprint(a) != f1
        (a / "x.py").write_bytes(b"print(1)\r\n")  # line endings do not change the identity
        assert source_fingerprint(a) == f1
        assert source_fingerprint() == source_fingerprint()  # the installed package

    def test_git_state_and_patch_in_a_temp_repo(self, tmp_path):
        import shutil
        import subprocess
        from llmtrace.provenance import git_state, record_provenance
        if shutil.which("git") is None:
            pytest.skip("git not available")
        repo = tmp_path / "repo"
        repo.mkdir()
        run = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True)  # noqa: E731
        run("init", "-q")
        run("config", "user.email", "t@example.com")
        run("config", "user.name", "t")
        (repo / "m.py").write_text("a = 1\n")
        run("add", "m.py")
        run("commit", "-q", "-m", "init")
        clean = git_state(repo)
        assert clean["commit"] and clean["dirty"] is False and clean["diff"] == "" and clean["untracked"] == []
        (repo / "m.py").write_text("a = 2\n")
        (repo / "new.py").write_text("b = 1\n")
        dirty = git_state(repo)
        assert dirty["dirty"] is True and "-a = 1" in dirty["diff"] and "+a = 2" in dirty["diff"] and dirty["untracked"] == ["new.py"]
        out = tmp_path / "run"
        prov = record_provenance(str(out), str(repo))
        assert prov["llmtrace_git_dirty"] is True and prov["llmtrace_source_patch"] == "source.patch"
        assert (out / "source.patch").read_text().count("+a = 2") == 1 and prov["llmtrace_untracked_files"] == ["new.py"]
        assert prov["llmtrace_git_commit_full"] == clean["commit"] and prov["llmtrace_git_commit"] == clean["commit"][:7]
        # the untracked file's contents are archived, so commit + patch + archive restore the tree
        import tarfile
        assert prov["llmtrace_untracked_archive"] == "source_untracked.tar.gz" and prov["llmtrace_snapshot_complete"] is True
        with tarfile.open(out / "source_untracked.tar.gz") as tar:
            assert tar.getnames() == ["new.py"] and tar.extractfile("new.py").read().replace(b"\r\n", b"\n") == b"b = 1\n"
        # an untracked file above the cap makes the snapshot incomplete, and says so
        from llmtrace import provenance as prov_mod
        old_cap = prov_mod.UNTRACKED_MAX_BYTES
        prov_mod.UNTRACKED_MAX_BYTES = 3
        try:
            p2 = record_provenance(str(tmp_path / "run2"), str(repo))
        finally:
            prov_mod.UNTRACKED_MAX_BYTES = old_cap
        assert p2["llmtrace_snapshot_complete"] is False and any("exceeds" in g for g in p2["llmtrace_snapshot_gaps"])
        # no git tree: identified, not restorable
        p3 = record_provenance(str(tmp_path / "run3"), str(tmp_path / "plain_dir_without_git"))
        assert p3["llmtrace_git_commit"] is None and p3["llmtrace_snapshot_complete"] is False

    def test_manifest_records_provenance(self, tmp_path):
        m = run_workload(_spec(), RunOptions(engine="fake", out_dir=str(tmp_path / "p")))
        assert m.llmtrace_source_fingerprint and m.llmtrace_source_fingerprint.startswith("sha256:")
        back = RunManifest.read(str(tmp_path / "p"))
        assert back.llmtrace_source_fingerprint == m.llmtrace_source_fingerprint
        if m.llmtrace_git_dirty:
            assert m.llmtrace_source_patch is None or (tmp_path / "p" / m.llmtrace_source_patch).exists()
            assert m.llmtrace_untracked_archive is None or (tmp_path / "p" / m.llmtrace_untracked_archive).exists()
        assert m.llmtrace_snapshot_complete is not None
        rep = run_report(str(tmp_path / "p"))
        assert any(c.name == "source" and m.llmtrace_source_fingerprint in c.detail for c in rep.checks)


class TestDuplicateRepeats:
    def _one_run(self, tmp_path, name="src"):
        spec = WorkloadSpec(name="dup", classes=[RequestClass(name="a", count=3, prompt_len=LengthSpec(value=8), max_tokens=LengthSpec(value=2))])
        run_workload(spec, RunOptions(engine="fake", out_dir=str(tmp_path / name)))
        return str(tmp_path / name)

    def test_same_directory_twice_is_one_repeat(self, tmp_path):
        d = self._one_run(tmp_path)
        alone = evaluate({"c": [d]}, Target.parse("a ttft_p95 <= 1000ms"))
        assert alone.candidates == []  # one repeat is below --min-repeats
        twice = evaluate({"c": [d, d]}, Target.parse("a ttft_p95 <= 1000ms"))
        assert twice.candidates == [] and twice.configs[0].eligible_repeats == 1
        assert twice.configs[0].repeats[1].status == "duplicate" and "same directory" in twice.configs[0].repeats[1].error
        assert any("not an independent repeat" in n for n in twice.notes)
        # a relative and an absolute spelling of the same path are the same run
        import os
        rel = os.path.relpath(d, os.getcwd())
        assert evaluate({"c": [d, rel]}, Target.parse("a ttft_p95 <= 1000ms")).configs[0].eligible_repeats == 1

    def test_copied_directory_is_a_duplicate_by_session(self, tmp_path):
        import shutil
        d = self._one_run(tmp_path)
        shutil.copytree(d, tmp_path / "copy")
        dec = evaluate({"c": [d, str(tmp_path / "copy")]}, Target.parse("a ttft_p95 <= 1000ms"))
        assert dec.candidates == [] and dec.configs[0].repeats[1].status == "duplicate"
        assert "same tracer session" in dec.configs[0].repeats[1].error
        # two genuinely separate runs are two repeats
        d2 = self._one_run(tmp_path, "src2")
        ok = evaluate({"c": [d, d2]}, Target.parse("a ttft_p95 <= 1000ms"))
        assert ok.configs[0].eligible_repeats == 2 and ok.candidates == ["c"]

    def test_same_run_under_two_configurations(self, tmp_path):
        d = self._one_run(tmp_path)
        d2 = self._one_run(tmp_path, "src2")
        dec = evaluate({"base": [d, d2], "cand": [d2, d]}, Target.parse("a ttft_p95 <= 1000ms"))
        by = {c.name: c for c in dec.configs}
        assert by["base"].eligible_repeats == 2 and by["cand"].eligible_repeats == 0
        assert all(r.status == "duplicate" for r in by["cand"].repeats) and dec.candidates == ["base"]


class TestPlanReproducesSource:
    def test_source_engine_kwargs_from_manifest(self):
        m = RunManifest(engine="vllm", model="org/model", model_revision="abc123",
                        engine_kwargs={"max_num_batched_tokens": 512, "max_model_len": 4096},
                        scheduling_change={"long_prefill_token_threshold": 128},
                        effective_engine_config={"scheduler_config": {"max_num_batched_tokens": 512, "max_num_seqs": 256},
                                                 "parallel_config": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
                                                 "model_config": {"revision": "abc123"}})
        kw, unreproduced = source_engine_kwargs(m)
        assert kw == {"max_num_batched_tokens": 512, "max_num_seqs": 256, "max_model_len": 4096, "long_prefill_token_threshold": 128,
                      "tensor_parallel_size": 2, "pipeline_parallel_size": 1, "revision": "abc123"}
        assert unreproduced == []
        p = plan_experiments([Finding(hypothesis="queue_overload", status=SUPPORTED, summary="s")], m, [])
        assert p.source_engine == "vllm" and p.source_model == "org/model" and p.source_engine_kwargs == kw
        cfgs = p.configs()
        assert cfgs[0] == {"name": "baseline", "engine_kwargs": kw, "scheduling_change": {}}
        assert all(c["engine_kwargs"] == kw for c in cfgs)
        assert "baseline engine kwargs" in p.format() and "revision abc123" in p.format()

    def test_effective_only_settings_are_reconstructed_and_unknown_ones_flagged(self):
        # nothing explicit: the budget, the sequence cap and the disabled prefix cache exist only in the effective record
        m = RunManifest(engine="vllm", model="org/model", engine_kwargs={}, scheduling_change={},
                        effective_engine_config={
                            "scheduler_config": {"max_num_batched_tokens": 512, "max_num_seqs": 32, "enable_chunked_prefill": True,
                                                 "long_prefill_token_threshold": 0, "policy": "fcfs", "max_model_len": 2048},
                            "cache_config": {"block_size": 16, "gpu_memory_utilization": 0.5, "enable_prefix_caching": False, "num_gpu_blocks": 1234},
                            "model_config": {"model": "org/model", "revision": None, "dtype": "torch.bfloat16", "max_model_len": 2048, "seed": 0,
                                             "weird_knob": "x"},
                            "parallel_config": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1, "data_parallel_size": 1}})
        kw, unreproduced = source_engine_kwargs(m)
        assert kw["max_num_batched_tokens"] == 512 and kw["max_num_seqs"] == 32 and kw["enable_prefix_caching"] is False
        assert kw["scheduling_policy"] == "fcfs" and kw["dtype"] == "bfloat16" and kw["block_size"] == 16 and kw["seed"] == 0
        assert "num_gpu_blocks" not in kw and "model" not in kw and "revision" not in kw
        assert unreproduced == ["model_config.weird_knob='x' (no engine kwarg form)"]
        p = plan_experiments([Finding(hypothesis="queue_overload", status=SUPPORTED, summary="s")], m, [])
        assert not p.reproducible and "NOT REPRODUCED: model_config.weird_knob" in p.format()
        assert any("not like-for-like" in n for n in p.notes)
        assert p.configs()[0]["engine_kwargs"]["max_num_batched_tokens"] == 512
        # explicit overrides win over the effective record
        m2 = m.model_copy(update={"scheduling_change": {"max_num_seqs": 64}})
        kw2, _ = source_engine_kwargs(m2)
        assert kw2["max_num_seqs"] == 64 and kw2["max_num_batched_tokens"] == 512
        # fake engine: every recorded knob is a constructor argument
        fake = RunManifest(engine="fake", effective_engine_config={"fake_engine": {"step_seconds": 0.0015, "max_num_batched_tokens": 512,
                                                                                   "long_prefill_token_threshold": 0, "prefill_chunk": None}})
        kwf, unf = source_engine_kwargs(fake)
        assert kwf == {"step_seconds": 0.0015, "max_num_batched_tokens": 512, "long_prefill_token_threshold": 0} and unf == []

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
        assert p.source_engine == "fake" and p.source_engine_kwargs["max_num_batched_tokens"] == 512
        assert p.source_engine_kwargs["step_seconds"] == 0.0015 and p.reproducible  # the fake's effective knobs come along
        res = r.invoke(main, ["run", "--workload", str(tmp_path / "w.json"), "--plan", str(tmp_path / "plan.json"), "--out", str(tmp_path / "exp")])
        assert res.exit_code == 0, res.output  # engine taken from the plan (fake)
        base = RunManifest.read(str(tmp_path / "exp" / "baseline" / "r0"))
        assert base.engine_kwargs["max_num_batched_tokens"] == 512
        assert base.effective_engine_config["fake_engine"]["max_num_batched_tokens"] == 512  # not the 8192 default
        cand = RunManifest.read(str(tmp_path / "exp" / p.candidates[0].name / "r0"))
        assert cand.engine_kwargs["max_num_batched_tokens"] == 512 and cand.scheduling_change == p.candidates[0].scheduling_change
        assert cand.effective_engine_config["fake_engine"]["max_num_batched_tokens"] == 512
        # a plan from a vllm run cannot be silently run on the fake engine without a warning
        p2 = p.model_copy(update={"source_engine": "vllm", "source_model": "org/m"})
        (tmp_path / "plan2.json").write_text(p2.model_dump_json())
        res = r.invoke(main, ["run", "--workload", str(tmp_path / "w.json"), "--plan", str(tmp_path / "plan2.json"), "--engine", "fake",
                              "--out", str(tmp_path / "exp2")])
        assert res.exit_code == 0 and "made from a vllm run, running on fake" in res.output
        assert json.loads((tmp_path / "exp2" / "baseline" / "r0" / "manifest.json").read_text())["engine_kwargs"]["max_num_batched_tokens"] == 512
