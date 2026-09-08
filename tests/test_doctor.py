"""llmtrace doctor: environment probes (injected) and run-directory signal reports."""

from __future__ import annotations

import types

from click.testing import CliRunner

from llmtrace.cli import main
from llmtrace.doctor import Probes, environment_report, run_report
from llmtrace.runner import RunOptions, run_workload
from llmtrace.workload import ArrivalSpec, LengthSpec, RequestClass, WorkloadSpec


def _probes(vllm_version="0.11.0", mp="0", cuda=(True, "torch 2.8, 1 device"), nvml=(True, "driver 580, 1 device"),
            modules=("pyarrow", "rich")):
    def imp(name):
        if name == "vllm":
            if vllm_version is None:
                raise ImportError("no vllm")
            return types.SimpleNamespace(__version__=vllm_version)
        if name in modules:
            return types.SimpleNamespace()
        raise ImportError(name)

    return Probes(import_module=imp, env={"VLLM_ENABLE_V1_MULTIPROCESSING": mp} if mp is not None else {},
                  cuda=lambda: cuda, nvml=lambda: nvml, python_version=(3, 11, 9))


def _by_name(report):
    return {c.name: c for c in report.checks}, {s.name.split(" (")[0]: s for s in report.signals}


class TestEnvironment:
    def test_all_good_in_process(self):
        checks, signals = _by_name(environment_report(_probes()))
        assert checks["vllm"].status == "ok" and "verified target" in checks["vllm"].detail
        assert checks["engine core process"].status == "ok"
        assert all(s.available for s in signals.values())

    def test_multiprocess_default_loses_scheduler_signals(self):
        rep = environment_report(_probes(mp=None))
        checks, signals = _by_name(rep)
        assert checks["engine core process"].status == "warn" and "VLLM_ENABLE_V1_MULTIPROCESSING=0" in checks["engine core process"].consequence
        assert not signals["batch membership and chunk sizes"].available and "out of process" in signals["batch membership and chunk sizes"].reason
        assert not signals["GPU step spans"].available
        assert signals["request traces"].available and signals["vLLM per-step stats"].available
        assert signals["GPU telemetry and energy"].available
        assert rep.errors == []

    def test_wrong_vllm_version_warns_and_missing_vllm(self):
        checks, _ = _by_name(environment_report(_probes(vllm_version="0.12.0")))
        assert checks["vllm"].status == "warn" and "0.11.0" in checks["vllm"].detail
        checks, signals = _by_name(environment_report(_probes(vllm_version=None)))
        assert checks["vllm"].status == "missing" and "synthetic" in checks["vllm"].consequence
        assert not signals["request traces"].available

    def test_no_cuda_no_nvml_no_extras(self):
        checks, signals = _by_name(environment_report(_probes(cuda=(False, "no torch"), nvml=(False, "no pynvml"), modules=())))
        assert checks["torch.cuda"].status == "missing" and "gpu_steps" in checks["torch.cuda"].consequence
        assert checks["nvml"].status == "missing" and checks["pyarrow"].status == "warn" and checks["rich"].status == "warn"
        assert not signals["GPU step spans"].available and signals["GPU step spans"].reason == "torch.cuda unavailable"
        assert not signals["GPU telemetry and energy"].available
        text = environment_report(_probes(cuda=(False, "x"))).format()
        assert "signals:" in text and "GPU step spans" in text

    def test_default_probes_do_not_raise(self):
        rep = environment_report()  # real environment: no vLLM/GPU here, must still produce a report
        assert rep.kind == "environment" and any(c.name == "vllm" for c in rep.checks)


class TestRunDir:
    def _run(self, tmp_path, name="r"):
        spec = WorkloadSpec(name="mini", seed=1, classes=[
            RequestClass(name="short", count=5, prompt_len=LengthSpec(value=16), max_tokens=LengthSpec(value=4),
                         arrival=ArrivalSpec(kind="at_once"))])
        return run_workload(spec, RunOptions(engine="fake", out_dir=str(tmp_path / name)))

    def test_run_report_lists_signals(self, tmp_path):
        m = self._run(tmp_path)
        rep = run_report(str(tmp_path / "r"))
        checks, signals = _by_name(rep)
        assert checks["manifest"].status == "ok" and checks["requests finished"].status == "ok"
        assert signals["request traces"].available and signals["batch membership and chunk sizes"].available
        # the fake engine's executor is bracketed with real CUDA events when torch.cuda exists (a GPU box), else not
        cuda = bool((m.health.get("cuda_timing") or {}).get("available"))
        assert signals["GPU step spans"].available == cuda
        assert cuda or signals["GPU step spans"].reason  # without CUDA the health reason is carried over
        assert signals["vLLM per-step stats"].available  # the fake engine exposes a logger_manager
        assert rep.errors == [] and m.status == "ok"

    def test_failed_run_and_missing_dir(self, tmp_path):
        spec = WorkloadSpec(name="x", classes=[RequestClass(name="a", count=1, prompt_len=LengthSpec(value=4), max_tokens=LengthSpec(value=1))])
        run_workload(spec, RunOptions(engine="fake", out_dir=str(tmp_path / "bad"), engine_kwargs={"bogus": 1}))
        rep = run_report(str(tmp_path / "bad"))
        assert any(c.status == "error" and c.name == "manifest" for c in rep.checks)
        assert any(c.name == "traces" and c.status == "error" for c in rep.checks)
        assert run_report(str(tmp_path / "nowhere")).errors

    def test_no_manifest_dir(self, tmp_path):
        d = tmp_path / "plain"
        d.mkdir()
        rep = run_report(str(d))
        checks, _ = _by_name(rep)
        assert checks["manifest"].status == "warn" and rep.errors  # no traces either

    def test_cli(self, tmp_path):
        self._run(tmp_path)
        r = CliRunner()
        res = r.invoke(main, ["doctor", str(tmp_path / "r"), "--json", str(tmp_path / "d.json")])
        assert res.exit_code == 0, res.output
        assert "request traces" in res.output and (tmp_path / "d.json").exists()
        res = r.invoke(main, ["doctor"])
        assert res.exit_code in (0, 1) and "llmtrace doctor (environment)" in res.output
