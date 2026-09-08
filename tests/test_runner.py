"""Workload runner on the synthetic engine: run directory contents, manifest, repeats, failures, CLI."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from llmtrace import io
from llmtrace.cli import main
from llmtrace.control_plane.decision import Target, evaluate
from llmtrace.manifest import RunManifest
from llmtrace.runner import RunOptions, run_workload
from llmtrace.workload import ArrivalSpec, LengthSpec, RequestClass, WorkloadSpec


def _spec(n_short=30, n_long=3) -> WorkloadSpec:
    return WorkloadSpec(name="mini", seed=1, classes=[
        RequestClass(name="short", count=n_short, prompt_len=LengthSpec(value=32), max_tokens=LengthSpec(value=16),
                     arrival=ArrivalSpec(kind="constant", rate_per_s=200.0)),
        RequestClass(name="long", count=n_long, prompt_len=LengthSpec(value=1536), max_tokens=LengthSpec(value=4),
                     arrival=ArrivalSpec(kind="constant", rate_per_s=20.0, start_s=0.02)),
    ])


class TestRunWorkload:
    def test_fake_run_writes_raw_data_and_manifest(self, tmp_path):
        out = tmp_path / "base"
        m = run_workload(_spec(), RunOptions(engine="fake", out_dir=str(out), config_name="baseline"))
        assert m.status == "ok" and m.synthetic and m.engine == "fake"
        assert m.finished == 33 and m.expected_requests == 33 and m.extra["problems"] == []
        assert m.workload_hash == _spec().hash() and m.seed == 1 and m.config_name == "baseline"
        assert len(m.arrivals) == 33 and m.arrival_delay_ms_max is not None
        assert (out / "workload.json").exists() and (out / "manifest.json").exists() and (out / "run_info.json").exists()
        assert WorkloadSpec.load(str(out / "workload.json")) == _spec()
        files = {p.name.split("_")[0] for p in out.glob("*.jsonl")}
        assert {"traces", "batches", "gpu", "collector"} <= files
        traces = io.load_traces([str(out)])
        assert len(traces) == 33 and all(t.status.value == "completed" for t in traces)
        assert all(t.output_length == (16 if t.request_id.startswith("short") else 4) for t in traces)
        assert all(t.scheduler_visible and t.ttft_ms is not None for t in traces)
        # no derived summaries are written into the run directory
        assert not list(out.glob("analysis*")) and not list(out.glob("findings*"))
        assert RunManifest.read(str(out)) == m

    def test_scheduling_change_is_applied_and_recorded(self, tmp_path):
        base = run_workload(_spec(), RunOptions(engine="fake", out_dir=str(tmp_path / "b")))
        capped = run_workload(_spec(), RunOptions(engine="fake", out_dir=str(tmp_path / "c"), config_name="capped",
                                                  scheduling_change={"long_prefill_token_threshold": 256}))
        assert capped.scheduling_change == {"long_prefill_token_threshold": 256}
        assert capped.effective_engine_config["fake_engine"]["long_prefill_token_threshold"] == 256
        assert base.effective_engine_config["fake_engine"]["long_prefill_token_threshold"] == 0
        assert base.workload_hash == capped.workload_hash  # identical work
        big_base = max(max(b.scheduled_tokens.values()) for b in io.load_batches([str(tmp_path / "b")]))
        big_cap = max(max(b.scheduled_tokens.values()) for b in io.load_batches([str(tmp_path / "c")]))
        assert big_base == 1536 and big_cap == 256
        # the run directories feed decide unchanged
        d = evaluate({"baseline": [str(tmp_path / "b")], "capped": [str(tmp_path / "c")]}, Target.parse("short ttft_p95 <= 50ms"))
        assert {c.name for c in d.configs} == {"baseline", "capped"} and all(c.repeats[0].status == "ok" for c in d.configs)
        assert all(c.all_eligible for c in d.configs)

    def test_failure_leaves_failed_manifest(self, tmp_path):
        out = tmp_path / "bad"
        m = run_workload(_spec(), RunOptions(engine="fake", out_dir=str(out), engine_kwargs={"no_such_kwarg": 1}))
        assert m.status == "failed" and "TypeError" in (m.error or "")
        assert RunManifest.read(str(out)).status == "failed"
        assert json.loads((out / "run_info.json").read_text())["status"] == "failed"

    def test_unknown_engine(self, tmp_path):
        m = run_workload(_spec(), RunOptions(engine="nope", out_dir=str(tmp_path / "x")))
        assert m.status == "failed" and "unknown engine" in (m.error or "")

    def test_problems_when_requests_unfinished(self, tmp_path):
        # fail_step_at makes the fake engine raise mid-run: recorded as failed, directory kept
        m = run_workload(_spec(), RunOptions(engine="fake", out_dir=str(tmp_path / "f"), engine_kwargs={"fail_step_at": 5}))
        assert m.status == "failed" and (tmp_path / "f" / "manifest.json").exists()


class TestCli:
    def test_workload_template_preview_and_run_repeat(self, tmp_path):
        r = CliRunner()
        spec_path = tmp_path / "w.json"
        res = r.invoke(main, ["workload", "template", "--output", str(spec_path)])
        assert res.exit_code == 0 and spec_path.exists()
        # shrink the template so the synthetic run is quick
        spec = WorkloadSpec.load(str(spec_path))
        spec = spec.model_copy(update={"classes": [c.model_copy(update={"count": 6}) for c in spec.classes]})
        spec.save(str(spec_path))
        res = r.invoke(main, ["workload", "preview", str(spec_path), "--json", str(tmp_path / "s.json"), "--requests", str(tmp_path / "r.jsonl")])
        assert res.exit_code == 0, res.output
        summ = json.loads((tmp_path / "s.json").read_text())
        assert summ["requests"] == 12 and len((tmp_path / "r.jsonl").read_text().splitlines()) == 12
        res = r.invoke(main, ["run", "--workload", str(spec_path), "--engine", "fake", "--out", str(tmp_path / "runs"),
                              "--repeat", "2", "--config-name", "capped", "--set", "long_prefill_token_threshold=256"])
        assert res.exit_code == 0, res.output
        for i in range(2):
            m = RunManifest.read(str(tmp_path / "runs" / f"r{i}"))
            assert m is not None and m.status == "ok" and m.config_name == "capped"
            assert m.scheduling_change == {"long_prefill_token_threshold": 256}
        assert "scheduler visible: True" in res.output

    def test_run_rejects_bad_set_and_spec(self, tmp_path):
        r = CliRunner()
        bad = tmp_path / "bad.json"
        bad.write_text('{"classes": []}')
        assert r.invoke(main, ["run", "--workload", str(bad), "--out", str(tmp_path / "o")]).exit_code == 2
        good = tmp_path / "g.json"
        _spec(2, 1).save(str(good))
        assert r.invoke(main, ["run", "--workload", str(good), "--out", str(tmp_path / "o"), "--set", "novalue"]).exit_code == 2
        assert r.invoke(main, ["workload", "preview", str(bad)]).exit_code == 2

    def test_run_reports_failure_with_exit_1(self, tmp_path):
        good = tmp_path / "g.json"
        _spec(2, 1).save(str(good))
        res = CliRunner().invoke(main, ["run", "--workload", str(good), "--out", str(tmp_path / "o"), "--engine-kwargs", '{"bogus": 1}'])
        assert res.exit_code == 1 and "FAILED" in res.output
        assert Path(tmp_path / "o" / "manifest.json").exists()
