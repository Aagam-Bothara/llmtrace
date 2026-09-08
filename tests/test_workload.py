"""Workload specs: determinism, distributions, arrival processes, validation, equivalence with the experiment workload."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmtrace.manifest import workload_hash
from llmtrace.workload import ArrivalSpec, LengthSpec, RequestClass, WorkloadSpec, class_of, make_prompt, template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "mixed_prompts"))
from workload import WorkloadConfig, build_workload, to_workload_spec  # noqa: E402


def _spec(**kw) -> WorkloadSpec:
    base = dict(name="t", seed=3, classes=[
        RequestClass(name="short", count=20, prompt_len=LengthSpec(kind="uniform", low=16, high=64),
                     max_tokens=LengthSpec(kind="choice", choices=[8, 32, 128], weights=[1, 2, 1]),
                     arrival=ArrivalSpec(kind="poisson", rate_per_s=50.0)),
        RequestClass(name="long", count=5, prompt_len=LengthSpec(kind="lognormal", median=1024, sigma=0.3, low=512, high=2000),
                     max_tokens=LengthSpec(kind="fixed", value=8),
                     arrival=ArrivalSpec(kind="gamma", rate_per_s=5.0, burstiness=0.5, start_s=0.1)),
    ])
    base.update(kw)
    return WorkloadSpec(**base)


class TestGeneration:
    def test_deterministic_and_seed_sensitive(self):
        a, b = _spec().generate(), _spec().generate()
        assert a == b
        assert workload_hash(a) == workload_hash(b) and _spec().hash() == _spec().hash()
        c = _spec(seed=4).generate()
        assert c != a and workload_hash(c) != workload_hash(a) and _spec(seed=4).hash() != _spec().hash()
        # fixed lengths + constant arrivals: the request list is seed-independent, the prompt token ids are not
        fixed = dict(classes=[RequestClass(name="a", count=3, prompt_len=LengthSpec(value=8), max_tokens=LengthSpec(value=2),
                                           arrival=ArrivalSpec(kind="constant", rate_per_s=10.0))])
        assert WorkloadSpec(seed=1, **fixed).generate() == WorkloadSpec(seed=2, **fixed).generate()
        assert WorkloadSpec(seed=1, **fixed).hash() != WorkloadSpec(seed=2, **fixed).hash()

    def test_ids_classes_ordering_and_lengths(self):
        specs = _spec().generate()
        assert len(specs) == 25
        assert [s.arrival_s for s in specs] == sorted(s.arrival_s for s in specs)
        assert {class_of(s.request_id) for s in specs} == {"short", "long"}
        for s in specs:
            assert s.kind == class_of(s.request_id)
            if s.kind == "short":
                assert 16 <= s.prompt_len <= 64 and s.max_tokens in (8, 32, 128)
            else:
                assert 512 <= s.prompt_len <= 2000 and s.max_tokens == 8
        longs = [s for s in specs if s.kind == "long"]
        assert longs[0].arrival_s == pytest.approx(0.1)  # start_s honoured; later ones are random gaps
        assert all(x.arrival_s > 0.1 for x in longs[1:])

    def test_prompt_token_ids_exact_length_and_range(self):
        spec = _spec(vocab_size=1000, min_token_id=200)
        for s in spec.generate():
            ids = make_prompt(s, spec.vocab_size, spec.seed, spec.min_token_id)["prompt_token_ids"]
            assert len(ids) == s.prompt_len and all(200 <= i < 1000 for i in ids)

    def test_summary_and_roundtrip(self, tmp_path):
        spec = _spec()
        p = spec.save(str(tmp_path / "w.json"))
        loaded = WorkloadSpec.load(str(p))
        assert loaded == spec and loaded.generate() == spec.generate()
        summ = spec.summary()
        assert summ["requests"] == 25 and set(summ["classes"]) == {"short", "long"}
        assert summ["classes"]["short"]["count"] == 20 and summ["workload_hash"] == spec.hash()
        assert "uniform[16, 64]" in summ["classes"]["short"]["prompt_len"]
        assert json.loads(p.read_text())["classes"][0]["name"] == "short"


class TestArrivals:
    def test_constant_at_once_burst(self):
        rng = random.Random(0)
        assert ArrivalSpec(kind="constant", rate_per_s=4.0).offsets(3, rng) == pytest.approx([0.0, 0.25, 0.5])
        assert ArrivalSpec(kind="at_once", start_s=2.0).offsets(3, rng) == [2.0, 2.0, 2.0]
        assert ArrivalSpec(kind="burst", burst_size=2, burst_every_s=1.0, start_s=0.5).offsets(5, rng) == [0.5, 0.5, 1.5, 1.5, 2.5]
        assert ArrivalSpec(kind="constant", rate_per_s=4.0).offsets(0, rng) == []

    def test_poisson_and_gamma_mean_gap(self):
        n = 20000
        pois = ArrivalSpec(kind="poisson", rate_per_s=100.0).offsets(n, random.Random(1))
        gam = ArrivalSpec(kind="gamma", rate_per_s=100.0, burstiness=0.25).offsets(n, random.Random(1))
        for offs in (pois, gam):
            assert offs == sorted(offs)
            assert offs[-1] / (n - 1) == pytest.approx(0.01, rel=0.05)  # mean gap = 1/rate
        # burstiness < 1 means a more variable gap than Poisson
        gaps_p = [b - a for a, b in zip(pois, pois[1:])]
        gaps_g = [b - a for a, b in zip(gam, gam[1:])]
        var = lambda xs: sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / len(xs)  # noqa: E731
        assert var(gaps_g) > 2 * var(gaps_p)


class TestValidation:
    @pytest.mark.parametrize("bad", [
        dict(kind="fixed"), dict(kind="uniform", low=10, high=5), dict(kind="choice", choices=[]),
        dict(kind="choice", choices=[1, 2], weights=[1]), dict(kind="lognormal", median=10), dict(kind="fixed", value=0),
    ])
    def test_length_spec_rejects(self, bad):
        with pytest.raises(ValidationError):
            LengthSpec(**bad)

    @pytest.mark.parametrize("bad", [dict(kind="poisson"), dict(kind="burst", burst_size=2), dict(kind="constant", rate_per_s=0)])
    def test_arrival_spec_rejects(self, bad):
        with pytest.raises(ValidationError):
            ArrivalSpec(**bad)

    def test_spec_rejects_bad_names_duplicates_unknown_fields(self):
        cls = dict(count=1, prompt_len=LengthSpec(value=8), max_tokens=LengthSpec(value=8))
        with pytest.raises(ValidationError):
            RequestClass(name="short-a", **cls)  # dash would break class_of
        with pytest.raises(ValidationError):
            WorkloadSpec(classes=[RequestClass(name="a", **cls), RequestClass(name="a", **cls)])
        with pytest.raises(ValidationError):
            WorkloadSpec(classes=[RequestClass(name="a", **cls)], typo=1)
        with pytest.raises(ValidationError):
            WorkloadSpec(classes=[RequestClass(name="a", **cls)], vocab_size=300, min_token_id=300)
        with pytest.raises(ValidationError):
            WorkloadSpec(classes=[])


class TestExperimentEquivalence:
    # The experiment pads long ids to two digits ("long-00") and the generic spec to four ("long-0000");
    # everything else (class, arrival, lengths, order) must be identical, so the runs are work-identical.
    @staticmethod
    def _key(s):
        return (s.kind, int(s.request_id.split("-")[1]), s.arrival_s, s.prompt_len, s.max_tokens)

    def test_template_matches_experiment_defaults(self):
        assert [self._key(s) for s in template().generate()] == [self._key(s) for s in build_workload(WorkloadConfig())]

    def test_to_workload_spec_equals_build_workload(self):
        cfg = WorkloadConfig(num_short=30, short_rate_per_s=25.0, num_long=4, long_every_s=0.25, first_long_at_s=0.3, seed=7)
        assert [self._key(s) for s in to_workload_spec(cfg).generate()] == [self._key(s) for s in build_workload(cfg)]
        assert workload_hash(build_workload(cfg)) != workload_hash(build_workload(WorkloadConfig(seed=8)))
