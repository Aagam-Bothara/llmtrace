"""CLI compare/analyze behaviour, reporter percentiles and comparisons, rules engine."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest
from click.testing import CliRunner
from conftest import const_power, mk_sample

from llmtrace import io
from llmtrace.cli import EXIT_NOT_IMPLEMENTED, EXIT_OK, EXIT_REGRESSION, EXIT_USAGE, evaluate_regressions, main
from llmtrace.control_plane.reporter import compare_metric, percentile
from llmtrace.control_plane.rules_engine import RulesEngine
from llmtrace.models.config import AutopsyConfig
from llmtrace.models.trace import DiagnosisCategory, MetricComparison, RequestSpan, RequestTrace, SpanPhase


def write_run(dir_: Path, ttfts: List[Optional[float]], power: float = 100.0, duration: float = 1.0,
              tokens: int = 10, with_gpu: bool = True) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    traces = []
    for i, ttft in enumerate(ttfts):
        start = i * (duration + 0.5)
        traces.append(RequestTrace(request_id=f"r{i}", start_time=start, end_time=start + duration,
                                   prompt_length=5, output_length=tokens, model_name="synthetic",
                                   ttft_ms=ttft, tpot_ms=(ttft or 0) / 2 if ttft else None))
    io.write_jsonl(dir_ / "traces_run.jsonl", traces)
    if with_gpu:
        end = traces[-1].end_time
        io.write_jsonl(dir_ / "gpu_run.jsonl", const_power(0.0, end, 0.1, power))


class TestCompareCLI:
    def test_improvement_is_not_a_regression(self, tmp_path):
        write_run(tmp_path / "base", [100.0] * 5, power=100.0)
        write_run(tmp_path / "cur", [50.0] * 5, power=50.0)  # 50% better on both axes
        r = CliRunner().invoke(main, ["compare", "--baseline", str(tmp_path / "base"), "--current", str(tmp_path / "cur"),
                                      "--fail-on-regression", "--no-rich"])
        assert r.exit_code == EXIT_OK, r.output
        assert "PASSED" in r.output and "REGRESSION" not in r.output
        assert "-50.00%" in r.output

    def test_regression_fails_with_flag(self, tmp_path):
        write_run(tmp_path / "base", [100.0] * 5)
        write_run(tmp_path / "cur", [200.0] * 5)  # +100% TTFT
        args = ["compare", "--baseline", str(tmp_path / "base"), "--current", str(tmp_path / "cur"), "--no-rich"]
        r = CliRunner().invoke(main, args)
        assert r.exit_code == EXIT_OK and "Regressions found" in r.output
        r = CliRunner().invoke(main, args + ["--fail-on-regression"])
        assert r.exit_code == EXIT_REGRESSION and "REGRESSION p95_ttft_ms: +100.00%" in r.output

    def test_missing_metric_is_reported_and_optionally_fails(self, tmp_path):
        write_run(tmp_path / "base", [None] * 3)  # no TTFT in baseline
        write_run(tmp_path / "cur", [100.0] * 3)
        base = ["compare", "--baseline", str(tmp_path / "base"), "--current", str(tmp_path / "cur"), "--no-rich"]
        r = CliRunner().invoke(main, base + ["--fail-on-regression"])
        assert r.exit_code == EXIT_OK and "unavailable p95_ttft_ms: missing_baseline" in r.output
        r = CliRunner().invoke(main, base + ["--fail-on-missing"])
        assert r.exit_code == EXIT_REGRESSION

    def test_missing_energy_when_no_gpu_samples(self, tmp_path):
        write_run(tmp_path / "base", [100.0] * 3, with_gpu=False)
        write_run(tmp_path / "cur", [100.0] * 3, with_gpu=False)
        r = CliRunner().invoke(main, ["compare", "--baseline", str(tmp_path / "base"), "--current", str(tmp_path / "cur"),
                                      "--no-rich", "--fail-on-regression"])
        assert r.exit_code == EXIT_OK
        assert "unavailable joules_per_output_token: missing_baseline" in r.output

    def test_gpu_samples_are_matched_per_run(self, tmp_path):
        write_run(tmp_path / "base", [100.0] * 3, power=100.0)
        write_run(tmp_path / "cur", [100.0] * 3, power=300.0)  # only energy differs
        r = CliRunner().invoke(main, ["compare", "--baseline", str(tmp_path / "base"), "--current", str(tmp_path / "cur"),
                                      "--no-rich", "--fail-on-regression"])
        assert r.exit_code == EXIT_REGRESSION
        assert "REGRESSION joules_per_output_token: +200.00%" in r.output
        assert "ok         p95_ttft_ms: +0.00%" in r.output

    def test_empty_run_is_usage_error(self, tmp_path):
        write_run(tmp_path / "base", [100.0])
        (tmp_path / "cur").mkdir()
        r = CliRunner().invoke(main, ["compare", "--baseline", str(tmp_path / "base"), "--current", str(tmp_path / "cur")])
        assert r.exit_code == EXIT_USAGE

    def test_monitor_not_implemented(self):
        r = CliRunner().invoke(main, ["monitor"])
        assert r.exit_code == EXIT_NOT_IMPLEMENTED and "not implemented" in r.output

    def test_analyze_writes_json_report(self, tmp_path):
        write_run(tmp_path / "run", [100.0, 120.0])
        out = tmp_path / "report.json"
        r = CliRunner().invoke(main, ["analyze", str(tmp_path / "run"), "--no-rich", "--output", str(out)])
        assert r.exit_code == 0, r.output
        assert "Device energy in run window" in r.output
        assert out.exists() and '"num_requests": 2' in out.read_text()

    def test_init_config_roundtrip(self, tmp_path):
        out = tmp_path / "cfg.json"
        r = CliRunner().invoke(main, ["init-config", "--output", str(out)])
        assert r.exit_code == 0
        from llmtrace.models.config import TracerConfig
        TracerConfig.model_validate(io.read_json(out))


class TestEvaluateRegressions:
    def test_sign_convention(self):
        cmps = {
            "p95_ttft_ms": MetricComparison(metric="p95_ttft_ms", baseline=100, current=50, pct_change=-50.0),
            "joules_per_output_token": MetricComparison(metric="j", baseline=1, current=1.2, pct_change=20.0),
        }
        reg, ok, missing = evaluate_regressions(cmps, {"p95_ttft_ms": 5.0, "joules_per_output_token": 10.0})
        assert reg == ["joules_per_output_token: +20.00% (threshold +10.0%)"]
        assert ok == ["p95_ttft_ms: -50.00% (threshold +5.0%)"]
        assert missing == []

    def test_at_threshold_is_not_regression(self):
        cmps = {"p95_ttft_ms": MetricComparison(metric="p95_ttft_ms", baseline=100, current=105, pct_change=5.0)}
        reg, ok, _ = evaluate_regressions(cmps, {"p95_ttft_ms": 5.0})
        assert not reg and ok


class TestReporterHelpers:
    def test_percentile_nearest_rank(self):
        assert percentile([], 95) is None
        assert percentile([7.0], 99) == 7.0
        vals = list(range(1, 101))
        assert percentile(vals, 50) == 50 and percentile(vals, 95) == 95 and percentile(vals, 99) == 99
        assert percentile([1.0, 2.0], 50) == 1.0 and percentile([1.0, 2.0], 51) == 2.0

    def test_compare_metric_edge_cases(self):
        assert compare_metric("m", None, 1.0).status == "missing_baseline"
        assert compare_metric("m", 1.0, None).status == "missing_current"
        z = compare_metric("m", 0.0, 1.0)
        assert z.status == "zero_baseline" and z.pct_change is None
        zz = compare_metric("m", 0.0, 0.0)
        assert zz.status == "ok" and zz.pct_change == 0.0
        ok = compare_metric("m", 100.0, 90.0)
        assert ok.status == "ok" and ok.pct_change == pytest.approx(-10.0)

    def test_zero_ttft_is_a_value_not_missing(self):
        from llmtrace.control_plane.reporter import Reporter
        t = RequestTrace(request_id="r", start_time=0, end_time=1, prompt_length=1, output_length=1, model_name="m", ttft_ms=0.0)
        a = Reporter().generate_analysis([t])
        assert a.num_with_ttft == 1 and a.p95_ttft_ms == 0.0


class TestRulesEngine:
    def _trace(self, **kw) -> RequestTrace:
        base = dict(request_id="r", start_time=0.0, end_time=1.0, prompt_length=10, output_length=5, model_name="m")
        base.update(kw)
        return RequestTrace(**base)

    def test_queueing_overload_needs_real_queue_span(self):
        engine = RulesEngine(AutopsyConfig(queue_overload_threshold_ms=50.0))
        q = RequestSpan(phase=SpanPhase.QUEUE, start_time=0.0, end_time=0.1, duration_ms=100.0)
        d = engine.diagnose_request(self._trace(spans=[q]))
        assert d is not None and d.category == DiagnosisCategory.QUEUEING_OVERLOAD and 0 < d.score <= 1
        ttft = RequestSpan(phase=SpanPhase.TIME_TO_FIRST_TOKEN, start_time=0.0, end_time=0.1, duration_ms=100.0)
        assert engine.diagnose_request(self._trace(spans=[ttft])) is None

    def test_throttling(self):
        engine = RulesEngine(AutopsyConfig(throttle_severity_threshold=2))
        samples = [mk_sample(0.1 * i, throttled=True) for i in range(5)]
        d = engine.diagnose_request(self._trace(gpu_samples=samples))
        assert d is not None and d.category == DiagnosisCategory.GPU_THROTTLING and d.evidence

    def test_memory_pressure_uses_capacity_not_bandwidth(self):
        engine = RulesEngine(AutopsyConfig(memory_pressure_threshold_pct=90.0))
        low_capacity = [mk_sample(0.1 * i, mem_used=8000.0) for i in range(3)]  # 50% used
        assert engine.diagnose_request(self._trace(gpu_samples=low_capacity)) is None
        high_capacity = [mk_sample(0.1 * i, mem_used=15500.0) for i in range(3)]  # ~97% used
        d = engine.diagnose_request(self._trace(gpu_samples=high_capacity))
        assert d is not None and d.category == DiagnosisCategory.MEMORY_PRESSURE

    def test_rules_skip_missing_fields(self):
        engine = RulesEngine(AutopsyConfig())
        samples = [mk_sample(0.1 * i, util=None, mem_used=None) for i in range(3)]
        assert engine.diagnose_request(self._trace(gpu_samples=samples)) is None
