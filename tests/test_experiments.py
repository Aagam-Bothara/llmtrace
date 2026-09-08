"""Findings context, goodput and bootstrap in decide, experiment planner, and the plan -> run -> decide loop on the fake engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from llmtrace import io
from llmtrace.cli import main
from llmtrace.control_plane.decision import Slo, Target, bootstrap_interval, evaluate, format_decision, goodput
from llmtrace.control_plane.experiments import ExperimentPlan, effective_knobs, plan_experiments
from llmtrace.control_plane.findings import INSUFFICIENT, SUPPORTED, Finding, check_long_prompt_interference, check_queue_overload, evaluate_all
from llmtrace.manifest import RunManifest
from llmtrace.models.trace import BatchMetadata, RequestTrace
from llmtrace.runner import RunOptions, run_workload
from llmtrace.workload import ArrivalSpec, LengthSpec, RequestClass, WorkloadSpec


def _batch(i, ids, tokens, dur, running=None):
    return BatchMetadata(batch_id=f"b{i}", step_index=i, timestamp=float(i), monotonic=float(i), step_end_monotonic=float(i) + dur,
                         num_requests=len(ids), num_prefill=0, num_decode=len(ids), total_scheduled_tokens=sum(tokens.values()),
                         request_ids=list(ids), scheduled_tokens=tokens, num_running=running)


def _trace(rid, ttft=5.0, tpot=None, tokens=4):
    return RequestTrace(request_id=rid, start_time=0, end_time=0.5, prompt_length=8, output_length=tokens, model_name="m", ttft_ms=ttft, tpot_ms=tpot)


class TestFindingContext:
    def test_every_finding_carries_context_and_insufficient_status(self):
        fs = evaluate_all([_trace("short-0")], [], [], [], gpu_steps=[])
        assert {f.hypothesis for f in fs} == {"queue_overload", "long_prompt_interference", "kv_cache_pressure", "host_overhead", "tracer_observer_effect"}
        for f in fs:
            assert f.status == INSUFFICIENT and not f.evaluable
            assert f.assumptions and f.competing_explanations and f.confidence_limits and f.missing_evidence
        assert "not a" in " ".join(fs[1].confidence_limits) or "cause" in " ".join(fs[1].confidence_limits)

    def test_supported_finding_keeps_context_and_json_roundtrip(self):
        batches = [_batch(0, ["short-0"], {"short-0": 1}, 0.002), _batch(1, ["short-0", "long-0"], {"short-0": 1, "long-0": 1500}, 0.010)]
        traces = [_trace("short-0", ttft=12.0), _trace("long-0", ttft=10.0)]
        traces[0].batch_ids = ["b0", "b1"]
        traces[1].batch_ids = ["b1"]
        f = check_long_prompt_interference(traces, batches)
        assert f.status == SUPPORTED and any("queue" in c for c in f.competing_explanations)
        back = Finding.model_validate_json(f.model_dump_json())
        assert back.assumptions == f.assumptions and back.confidence_limits == f.confidence_limits

    def test_queue_overload_from_vllm_queued_time_without_scheduler(self):
        from llmtrace.data_plane.vllm_stats import VLLMFinishedRequestStats, VLLMIterationRecord
        stats = [VLLMIterationRecord(timestamp=1, step_seq=1, num_waiting_reqs=3,
                                     finished_requests=[VLLMFinishedRequestStats(queued_time_s=0.25)])]
        f = check_queue_overload([_trace("a-0")], [], stats, threshold_ms=100.0)
        assert f.status == SUPPORTED and f.affected_count == 0
        assert any("no request ids" in lim for lim in f.confidence_limits)


class TestGoodputAndBootstrap:
    def test_slo_parse_and_goodput(self):
        s = Slo.parse("short: ttft <= 10ms, tpot<=2")
        assert s.request_class == "short" and s.bounds == {"ttft": 10.0, "tpot": 2.0}
        with pytest.raises(ValueError):
            Slo.parse("short ttft <= 10")
        with pytest.raises(ValueError):
            Slo.parse("short: ttft_p95 <= 10")
        traces = [_trace("short-0", ttft=5, tpot=1), _trace("short-1", ttft=15, tpot=1), _trace("short-2", ttft=5, tpot=None),
                  _trace("long-0", ttft=500, tpot=1)]
        gp, n, cov = goodput(traces, [s])
        assert n == 3 and gp == pytest.approx(1 / 3) and cov == pytest.approx(2 / 3)  # missing tpot counts as not good
        gp_all, n_all, _ = goodput(traces, [Slo.parse("*: ttft <= 100ms")])
        assert n_all == 4 and gp_all == pytest.approx(3 / 4)
        assert goodput(traces, []) == (None, 0, None)

    def test_bootstrap_is_seeded_and_bracketing(self):
        vals = [10.0] * 90 + [50.0] * 10
        ci = bootstrap_interval(vals, "p95", resamples=300, seed=1)
        assert ci == bootstrap_interval(vals, "p95", resamples=300, seed=1)
        assert ci[0] <= 50.0 <= ci[1] or ci[0] <= 10.0  # contains the plausible p95 values
        assert bootstrap_interval([1.0], "p50") is None

    def _run(self, tmp_path, name, ttfts, tpots=None):
        d = tmp_path / name
        d.mkdir()
        tpots = tpots or [1.0] * len(ttfts)
        io.write_jsonl(d / "traces_x.jsonl", [_trace(f"short-{i}", ttft=v, tpot=p) for i, (v, p) in enumerate(zip(ttfts, tpots))])
        return str(d)

    def test_evaluate_reports_goodput_ci_and_marginal(self, tmp_path):
        tight = [self._run(tmp_path, "t0", [10.0] * 19 + [30.0]), self._run(tmp_path, "t1", [10.0] * 19 + [31.0])]
        loose = [self._run(tmp_path, "l0", [10.0] * 10 + [40.0] * 10), self._run(tmp_path, "l1", [10.0] * 10 + [40.0] * 10)]
        dec = evaluate({"tight": tight, "loose": loose}, Target.parse("short ttft_p95 <= 35ms"),
                       slos=[Slo.parse("short: ttft <= 20ms, tpot <= 5ms")], bootstrap_resamples=200)
        by = {c.name: c for c in dec.configs}
        assert by["tight"].goodput_median == pytest.approx(0.95) and by["loose"].goodput_median == pytest.approx(0.5)
        assert by["tight"].target_ci95_ms is not None and by["tight"].pooled_requests == 40
        assert by["loose"].meets_target_all_repeats is False
        assert dec.candidates == ["tight"] and "goodput" in dec.recommendation
        text = format_decision(dec)
        assert "95% CI" in text and "goodput" in text and "slo: short: ttft <= 20 ms, tpot <= 5 ms" in text
        # marginal: every repeat's p95 is 30/31 ms (< 35) but the pooled bootstrap upper bound can reach 31 -> not marginal here;
        # build a case where the upper bound exceeds the target
        edge = [self._run(tmp_path, "e0", [10.0] * 18 + [34.0, 36.0]), self._run(tmp_path, "e1", [10.0] * 18 + [34.0, 36.0])]
        dec2 = evaluate({"edge": edge}, Target.parse("short ttft_p95 <= 35ms"), bootstrap_resamples=300)
        c = dec2.configs[0]
        # per-repeat p95 (nearest rank of 20 values) = 34 -> meets; pooled bootstrap p95 reaches 36 -> marginal
        assert c.meets_target_all_repeats is True and c.target_ci95_ms[1] >= 36.0 and dec2.marginal == ["edge"]
        assert any("marginal" in n for n in dec2.notes) and "marginal" in dec2.recommendation

    def test_cli_decide_with_slo(self, tmp_path):
        a = self._run(tmp_path, "a0", [10, 20])
        r = CliRunner().invoke(main, ["decide", "--target", "short ttft_p95 <= 25ms", "--config", f"A={a}",
                                      "--slo", "short: ttft <= 15ms", "--json", str(tmp_path / "d.json")])
        assert r.exit_code == 0, r.output
        d = json.loads((tmp_path / "d.json").read_text())
        assert d["slos"] == ["short: ttft <= 15 ms"] and d["configs"][0]["goodput_median"] == 0.5
        assert CliRunner().invoke(main, ["decide", "--target", "short ttft_p95 <= 25ms", "--config", f"A={a}", "--slo", "bad"]).exit_code == 2


def _manifest(**knobs):
    return RunManifest(engine="vllm", workload_hash="sha256:abc",
                       effective_engine_config={"scheduler_config": {k: v for k, v in knobs.items() if k in ("max_num_batched_tokens", "max_num_seqs", "long_prefill_token_threshold")},
                                                "cache_config": {k: v for k, v in knobs.items() if k in ("gpu_memory_utilization", "enable_prefix_caching")}})


def _finding(h, status=SUPPORTED, **params):
    return Finding(hypothesis=h, status=status, summary="s", parameters=params)


class TestPlanner:
    def test_interference_plan_is_bounded_by_observed_chunk_and_current_cap(self):
        batches = [_batch(0, ["long-0"], {"long-0": 1536}, 0.01)]
        m = _manifest(max_num_batched_tokens=8192, max_num_seqs=256, long_prefill_token_threshold=0)
        p = plan_experiments([_finding("long_prompt_interference", chunk_threshold=128)], m, batches, max_candidates=4)
        names = [c.name for c in p.candidates]
        assert names == ["cap1024", "cap512"]  # halving an 8192 budget to 4096 would not cap a 1536 chunk: not proposed
        p_small = plan_experiments([_finding("long_prompt_interference", chunk_threshold=128)],
                                   _manifest(max_num_batched_tokens=2048, max_num_seqs=256, long_prefill_token_threshold=0), batches)
        assert [c.name for c in p_small.candidates] == ["cap1024", "cap512", "budget1024"]
        assert all(c.scheduling_change and c.source_finding == "long_prompt_interference" and c.expected_cost for c in p.candidates)
        # the baseline reproduces the source's effective settings even though none were explicit kwargs
        assert p.configs()[0] == {"name": "baseline", "scheduling_change": {},
                                  "engine_kwargs": {"max_num_batched_tokens": 8192, "max_num_seqs": 256, "long_prefill_token_threshold": 0}}
        assert p.reproducible
        # already capped at 256: only smaller values, none below the chunk threshold of 256 -> skipped with a reason
        m2 = _manifest(max_num_batched_tokens=8192, max_num_seqs=256, long_prefill_token_threshold=256)
        p2 = plan_experiments([_finding("long_prompt_interference", chunk_threshold=256)], m2, batches)
        assert not [c for c in p2.candidates if "long_prefill_token_threshold" in c.scheduling_change]
        assert any("already 256" in s for s in p2.skipped)

    def test_queue_kv_host_rules_and_cap(self):
        m = _manifest(max_num_batched_tokens=2048, max_num_seqs=64, gpu_memory_utilization=0.5, enable_prefix_caching=False)
        batches = [_batch(0, ["a-0"], {"a-0": 1}, 0.01, running=64)]
        fs = [_finding("queue_overload"), _finding("kv_cache_pressure"), _finding("host_overhead"), _finding("long_prompt_interference", status="not_supported")]
        p = plan_experiments(fs, m, batches, max_candidates=10)
        changes = {c.name: c.scheduling_change for c in p.candidates}
        assert changes["seqs128"] == {"max_num_seqs": 128}  # queue: running hit max_num_seqs (deduped with host_overhead's identical change)
        assert changes["budget4096"] == {"max_num_batched_tokens": 4096}
        assert changes["mem60"] == {"gpu_memory_utilization": 0.6} and changes["seqs32"] == {"max_num_seqs": 32}
        assert changes["prefix_cache"] == {"enable_prefix_caching": True}
        assert sum(1 for c in p.candidates if c.scheduling_change == {"max_num_seqs": 128}) == 1
        assert any("long_prompt_interference: not_supported" in s for s in p.skipped)
        capped = plan_experiments(fs, m, batches, max_candidates=2)
        assert len(capped.candidates) == 2 and any("beyond --max-candidates" in s for s in capped.skipped)

    def test_candidates_ranked_by_affected_requests_and_finding_filter(self):
        m = _manifest(max_num_batched_tokens=8192, max_num_seqs=256, gpu_memory_utilization=0.06)
        batches = [_batch(0, ["kv-0"], {"kv-0": 256}, 0.01)]
        interference = Finding(hypothesis="long_prompt_interference", status=SUPPORTED, summary="s", affected_count=38,
                               parameters={"chunk_threshold": 128})
        kv = Finding(hypothesis="kv_cache_pressure", status=SUPPORTED, summary="s", affected_count=48)
        p = plan_experiments([interference, kv], m, batches, max_candidates=2)
        assert [c.name for c in p.candidates] == ["mem16", "seqs128"]  # the finding touching more requests comes first
        assert any("cap" in s and "beyond --max-candidates" in s for s in p.skipped)
        p2 = plan_experiments([interference, kv], m, batches, max_candidates=4)
        assert [c.source_finding for c in p2.candidates] == ["kv_cache_pressure"] * 2 + ["long_prompt_interference"] * 1
        only = plan_experiments([interference, kv], m, batches, max_candidates=4, only_findings=["long_prompt_interference"])
        assert {c.source_finding for c in only.candidates} == {"long_prompt_interference"}
        assert any("restricted to" in n for n in only.notes)
        none = plan_experiments([interference, kv], m, batches, only_findings=["no_such"])
        assert none.candidates == [] and any("no such finding" in s for s in none.skipped)

    def test_no_supported_findings_or_manifest(self):
        p = plan_experiments([_finding("queue_overload", status=INSUFFICIENT)], None, [])
        assert p.candidates == [] and any("no manifest" in n for n in p.notes) and "no candidates" in p.format()
        assert effective_knobs(None)["max_num_seqs"] is None
        fake = RunManifest(engine="fake", effective_engine_config={"fake_engine": {"max_num_batched_tokens": 8192, "long_prefill_token_threshold": 0}})
        assert effective_knobs(fake)["max_num_batched_tokens"] == 8192

    def test_plan_json_roundtrip(self, tmp_path):
        m = _manifest(max_num_batched_tokens=8192, max_num_seqs=256)
        p = plan_experiments([_finding("long_prompt_interference", chunk_threshold=128)], m, [_batch(0, ["l-0"], {"l-0": 1536}, 0.01)])
        path = tmp_path / "plan.json"
        path.write_text(p.model_dump_json())
        back = ExperimentPlan.model_validate_json(path.read_text())
        assert back == p and len(back.configs()) == len(p.candidates) + 1


class TestPlanRunDecideLoop:
    def test_fake_engine_loop(self, tmp_path):
        spec = WorkloadSpec(name="mini", seed=1, classes=[
            RequestClass(name="short", count=24, prompt_len=LengthSpec(value=32), max_tokens=LengthSpec(value=16),
                         arrival=ArrivalSpec(kind="constant", rate_per_s=200.0)),
            RequestClass(name="long", count=2, prompt_len=LengthSpec(value=1536), max_tokens=LengthSpec(value=4),
                         arrival=ArrivalSpec(kind="constant", rate_per_s=20.0, start_s=0.02))])
        spec.save(str(tmp_path / "w.json"))
        base = run_workload(spec, RunOptions(engine="fake", out_dir=str(tmp_path / "base")))
        assert base.status == "ok"
        r = CliRunner()
        res = r.invoke(main, ["plan", str(tmp_path / "base"), "--repeats", "2", "--max-candidates", "2", "--json", str(tmp_path / "plan.json")])
        assert res.exit_code == 0, res.output
        p = ExperimentPlan.model_validate_json((tmp_path / "plan.json").read_text())
        assert p.candidates and p.candidates[0].scheduling_change == {"long_prefill_token_threshold": 1024}
        assert p.workload_hash == spec.hash()
        res = r.invoke(main, ["run", "--workload", str(tmp_path / "w.json"), "--plan", str(tmp_path / "plan.json"), "--engine", "fake",
                              "--out", str(tmp_path / "exp")])
        assert res.exit_code == 0, res.output
        for cfg in p.configs():
            for i in range(2):
                m = RunManifest.read(str(tmp_path / "exp" / cfg["name"] / f"r{i}"))
                assert m is not None and m.status == "ok" and m.scheduling_change == cfg["scheduling_change"]
        assert "llmtrace decide" in res.output
        names = [c["name"] for c in p.configs()]
        cfgs = [f"--config={n}=" + ",".join(str(tmp_path / "exp" / n / f"r{i}") for i in range(2)) for n in names]
        res = r.invoke(main, ["decide", "--target", "short ttft_p95 <= 6ms", "--slo", "short: ttft <= 6ms"] + cfgs)
        assert res.exit_code == 0, res.output
        assert "goodput" in res.output and "baseline" in res.output
        # the workload hash mismatch is warned about, not fatal
        other = spec.model_copy(update={"seed": 2})
        other.save(str(tmp_path / "w2.json"))
        res = r.invoke(main, ["run", "--workload", str(tmp_path / "w2.json"), "--plan", str(tmp_path / "plan.json"), "--engine", "fake",
                              "--out", str(tmp_path / "exp2")])
        assert res.exit_code == 0 and "warning: plan was made from workload" in res.output
        assert Path(tmp_path / "exp2" / "baseline" / "r0" / "manifest.json").exists()
