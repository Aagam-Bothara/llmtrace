"""Energy integration, allocation and conservation."""

from __future__ import annotations

import pytest
from conftest import const_power, mk_sample

from llmtrace.control_plane.correlator import Correlator, CumulativePower
from llmtrace.models.config import EnergyConfig
from llmtrace.models.trace import BatchMetadata, RequestSpan, RequestTrace, SpanPhase


def trace(rid: str, start: float, end: float, out_tokens: int = 10, prompt: int = 10, spans=None) -> RequestTrace:
    return RequestTrace(
        request_id=rid,
        start_time=start,
        end_time=end,
        prompt_length=prompt,
        output_length=out_tokens,
        model_name="m",
        spans=spans or [],
    )


def corr(**kw) -> Correlator:
    return Correlator(EnergyConfig(**kw))


class TestIntegration:
    def test_unrelated_gpu_is_never_implicitly_allocated(self):
        g0 = const_power(0, 1, 0.1, 100, gpu_id=0)
        g1 = const_power(0, 1, 0.1, 300, gpu_id=1)
        ambiguous = corr().correlate([trace("r", 0, 1)], g0 + g1)
        assert ambiguous.ledger.device_joules is None
        assert "ambiguous" in ambiguous.traces[0].energy.unavailable_reason
        selected = corr(gpu_ids=[0]).correlate([trace("r", 0, 1)], g0 + g1)
        assert selected.ledger.device_joules == pytest.approx(100)
        assert selected.traces[0].energy.attributed_joules == pytest.approx(100)
        assert {s.gpu_id for s in selected.traces[0].gpu_samples} == {0}
        parallel = corr(gpu_ids=[0, 1]).correlate([trace("r", 0, 1)], g0 + g1)
        assert parallel.ledger.device_joules == pytest.approx(400)

    def test_missing_selected_gpu_is_not_hidden_by_another_gpu(self):
        g0 = const_power(0, 1, 0.1, 100, gpu_id=0)
        for g1 in ([], [mk_sample(0.5, None, gpu_id=1)], [mk_sample(0.5, 300, gpu_id=1)]):
            res = corr(gpu_ids=[0, 1]).correlate([trace("r", 0, 1)], g0 + g1)
            assert res.ledger.device_joules is None and not res.traces[0].energy.is_allocated

    def test_single_gpu_constant_power_known_total(self):
        samples = const_power(0.0, 1.0, 0.1, 200.0)
        res = corr().correlate([trace("r", 0.0, 1.0)], samples)
        assert res.ledger.device_joules == pytest.approx(200.0)
        assert res.ledger.per_gpu_joules == {0: pytest.approx(200.0)}
        e = res.traces[0].energy
        assert e.window_device_joules == pytest.approx(200.0)
        assert e.attributed_joules == pytest.approx(200.0)
        assert e.joules_per_output_token == pytest.approx(20.0)
        assert e.is_allocated and e.is_estimate
        assert e.coverage.coverage_fraction == pytest.approx(1.0) and e.coverage.num_samples == 11

    def test_linear_ramp_trapezoid(self):
        samples = [mk_sample(0.0, 0.0), mk_sample(2.0, 100.0)]
        assert corr(max_sample_gap_s=5.0).integrate_power(samples) == pytest.approx(100.0)
        assert corr().integrate_power(samples) == 0.0  # 2s apart exceeds the default 1s gap: not integrated
        c = CumulativePower([(0.0, 0.0), (2.0, 100.0)], max_gap_s=5.0)
        assert c.energy(0.0, 1.0) == pytest.approx(25.0)  # interpolated partial trapezoid
        assert c.energy(1.0, 2.0) == pytest.approx(75.0)

    def test_multi_gpu_with_offset_timestamps(self):
        g0 = const_power(0.0, 1.0, 0.1, 100.0, gpu_id=0)
        g1 = const_power(0.03, 1.03, 0.1, 50.0, gpu_id=1)  # not aligned with GPU 0
        res = corr(gpu_ids=[0, 1]).correlate([trace("r", 0.0, 1.0)], g0 + g1)
        # GPU1 covers [0.03, 1.0] of the window = 0.97s * 50W
        assert res.ledger.per_gpu_joules[0] == pytest.approx(100.0)
        assert res.ledger.per_gpu_joules[1] == pytest.approx(48.5)
        assert res.ledger.device_joules == pytest.approx(148.5)
        assert res.traces[0].energy.attributed_joules == pytest.approx(148.5)

    def test_duplicate_timestamps_keep_last(self):
        samples = const_power(0.0, 1.0, 0.5, 100.0) + [mk_sample(0.5, 900.0)]
        c = CumulativePower([(s.timestamp, s.power_draw_watts) for s in samples], 5.0)
        assert c.num_points == 3
        assert c.energy(0.0, 1.0) == pytest.approx(0.5 * (100 + 900) * 0.5 + 0.5 * (900 + 100) * 0.5)

    def test_gap_is_not_integrated(self):
        samples = const_power(0.0, 1.0, 0.1, 100.0) + const_power(5.0, 6.0, 0.1, 100.0)
        res = corr(max_sample_gap_s=1.0).correlate([trace("r", 0.0, 6.0)], samples)
        assert res.ledger.device_joules == pytest.approx(200.0)  # only the two covered seconds
        cov = res.ledger.coverage
        assert cov.covered_s == pytest.approx(2.0) and cov.coverage_fraction == pytest.approx(2 / 6)
        assert cov.max_gap_s == pytest.approx(4.0)
        # Coverage below the threshold: no per-request figure, but energy stays conserved.
        e = res.traces[0].energy
        assert e.attributed_joules is None and "insufficient telemetry" in e.unavailable_reason
        assert res.ledger.unattributable_joules == pytest.approx(200.0)
        assert res.ledger.conservation_error_joules < 1e-9

    def test_samples_without_power_are_ignored_and_counted(self):
        samples = const_power(0.0, 1.0, 0.1, 100.0) + [mk_sample(0.55, None)]
        res = corr().correlate([trace("r", 0.0, 1.0)], samples)
        assert res.ledger.samples_without_power == 1
        assert res.ledger.device_joules == pytest.approx(100.0)


class TestUnavailable:
    def test_no_samples(self):
        res = corr().correlate([trace("r", 0.0, 1.0)], [])
        assert res.ledger.device_joules is None
        e = res.traces[0].energy
        assert e is not None and e.attributed_joules is None and e.window_device_joules is None
        assert not e.is_allocated and "no GPU power telemetry" in e.unavailable_reason
        assert res.ledger.num_requests_without_telemetry == 1

    def test_single_sample_is_not_zero_energy(self):
        res = corr().correlate([trace("r", 0.0, 1.0)], [mk_sample(0.5, 200.0)])
        assert res.ledger.device_joules is None  # nothing bracketed
        e = res.traces[0].energy
        assert e.attributed_joules is None and e.window_device_joules is None
        assert "1 power samples" in e.unavailable_reason

    def test_sparse_telemetry_outside_request(self):
        samples = const_power(0.0, 1.0, 0.1, 100.0)
        res = corr().correlate([trace("late", 5.0, 6.0)], samples)
        e = res.traces[0].energy
        assert e.attributed_joules is None and e.coverage.num_samples == 0

    def test_disabled(self):
        res = corr(enabled=False).correlate([trace("r", 0.0, 1.0)], const_power(0.0, 1.0, 0.1, 100.0))
        assert res.traces[0].energy is None and "disabled" in res.ledger.notes[0]


class TestAllocation:
    def test_overlapping_requests_conserve_energy(self):
        samples = const_power(0.0, 3.0, 0.1, 100.0)  # 300 J over the run window
        traces = [trace("a", 0.0, 2.0), trace("b", 1.0, 3.0)]
        res = corr().correlate(traces, samples)
        by = {t.request_id: t.energy for t in res.traces}
        # a alone in [0,1] = 100 J, shared [1,2] = 50 J each, b alone in [2,3] = 100 J
        assert by["a"].attributed_joules == pytest.approx(150.0)
        assert by["b"].attributed_joules == pytest.approx(150.0)
        assert by["a"].window_device_joules == pytest.approx(200.0)  # shared window energy, not consumption
        assert res.ledger.attributed_joules == pytest.approx(300.0)
        assert res.ledger.idle_joules == 0.0
        assert res.ledger.conservation_error_joules < 1e-9
        assert res.ledger.membership_source == "request_window"

    def test_idle_energy_is_unallocated(self):
        samples = const_power(0.0, 4.0, 0.1, 100.0)
        traces = [trace("a", 0.0, 1.0), trace("b", 3.0, 4.0)]
        res = corr().correlate(traces, samples)
        assert res.ledger.device_joules == pytest.approx(400.0)
        assert res.ledger.attributed_joules == pytest.approx(200.0)
        assert res.ledger.idle_joules == pytest.approx(200.0)
        assert res.ledger.conservation_error_joules < 1e-9

    def test_proportional_tokens(self):
        samples = const_power(0.0, 1.0, 0.1, 100.0)
        traces = [trace("a", 0.0, 1.0, out_tokens=30, prompt=0), trace("b", 0.0, 1.0, out_tokens=10, prompt=0)]
        res = corr(attribution_method="proportional_tokens").correlate(traces, samples)
        by = {t.request_id: t.energy.attributed_joules for t in res.traces}
        assert by["a"] == pytest.approx(75.0) and by["b"] == pytest.approx(25.0)
        assert res.ledger.conservation_error_joules < 1e-9

    def test_window_only_does_not_allocate(self):
        samples = const_power(0.0, 2.0, 0.1, 100.0)
        traces = [trace("a", 0.0, 2.0), trace("b", 0.0, 2.0)]
        res = corr(attribution_method="window_only").correlate(traces, samples)
        for t in res.traces:
            assert t.energy.window_device_joules == pytest.approx(200.0)
            assert t.energy.attributed_joules is None and not t.energy.is_allocated
            assert "window_only" in t.energy.unavailable_reason
        assert res.ledger.attributed_joules == 0.0
        assert res.ledger.unattributable_joules == pytest.approx(200.0)
        assert res.ledger.conservation_error_joules < 1e-9

    def test_phase_breakdown_follows_spans(self):
        spans = [
            RequestSpan(phase=SpanPhase.QUEUE, start_time=0.0, end_time=0.5, duration_ms=500),
            RequestSpan(phase=SpanPhase.PREFILL, start_time=0.5, end_time=1.0, duration_ms=500),
            RequestSpan(phase=SpanPhase.DECODE, start_time=1.0, end_time=2.0, duration_ms=1000),
        ]
        samples = const_power(0.0, 2.0, 0.1, 100.0)
        res = corr().correlate([trace("a", 0.0, 2.0, spans=spans)], samples)
        e = res.traces[0].energy
        assert e.queue_joules == pytest.approx(50.0)
        assert e.prefill_joules == pytest.approx(50.0)
        assert e.decode_joules == pytest.approx(100.0)
        assert e.time_to_first_token_joules is None

    def test_phase_energy_integrates_varying_power(self):
        # Power ramps 0 -> 200 W over [0, 2]: first second holds 50 J, second second 150 J.
        samples = [mk_sample(0.1 * i, 100.0 * i * 0.1 * 1.0 * 1.0 * 1.0, 0) for i in range(21)]
        spans = [
            RequestSpan(phase=SpanPhase.QUEUE, start_time=0.0, end_time=1.0, duration_ms=1000),
            RequestSpan(phase=SpanPhase.DECODE, start_time=1.0, end_time=2.0, duration_ms=1000),
        ]
        res = corr().correlate([trace("a", 0.0, 2.0, spans=spans)], samples)
        e = res.traces[0].energy
        assert e.attributed_joules == pytest.approx(200.0)
        assert e.queue_joules == pytest.approx(50.0)
        assert e.decode_joules == pytest.approx(150.0)
        assert res.ledger.conservation_error_joules < 1e-9

    def test_phase_energy_with_overlapping_requests_and_varying_power(self):
        samples = [mk_sample(0.1 * i, 10.0 * i, 0) for i in range(21)]  # 0 -> 200 W over [0, 2]
        a = trace("a", 0.0, 2.0, spans=[RequestSpan(phase=SpanPhase.DECODE, start_time=0.5, end_time=2.0, duration_ms=1500)])
        b = trace("b", 1.0, 2.0, spans=[RequestSpan(phase=SpanPhase.DECODE, start_time=1.0, end_time=1.5, duration_ms=500)])
        res = corr().correlate([a, b], samples)
        by = {t.request_id: t.energy for t in res.traces}
        # device energy on [0,2] = 200 J; [0,1] = 50 J (a alone); [1,2] = 150 J shared -> 75 each
        assert by["a"].attributed_joules == pytest.approx(125.0)
        assert by["b"].attributed_joules == pytest.approx(75.0)
        # a's decode span [0.5,2]: alone on [0.5,1] = 50-12.5 = 37.5 J, plus half of [1,2] = 75 -> 112.5
        assert by["a"].decode_joules == pytest.approx(112.5)
        # b's decode span [1,1.5]: half of integral on [1,1.5] = 0.5 * (100+150)/2*0.5 = 31.25
        assert by["b"].decode_joules == pytest.approx(31.25)
        assert res.ledger.conservation_error_joules < 1e-9

    def test_cost(self):
        res = corr(energy_price_usd_per_kwh=0.36).correlate([trace("r", 0.0, 1.0)], const_power(0.0, 1.0, 0.1, 3600.0))
        assert res.traces[0].energy.attributed_joules == pytest.approx(3600.0)
        assert res.traces[0].energy.cost_usd == pytest.approx(0.36 / 1000)


class TestClockAndBatches:
    def _mono_trace(self, rid, start, end, domain="d"):
        t = trace(rid, 1e9 + start, 1e9 + end)
        t.start_monotonic, t.end_monotonic, t.clock_domain = start, end, domain
        return t

    def test_monotonic_clock_chosen_when_domains_match(self):
        samples = [mk_sample(1e9 + 0.1 * i, 100.0, mono=0.1 * i, domain="d") for i in range(11)]
        # Wall clock of samples deliberately shifted: monotonic must be used.
        for s in samples:
            s.timestamp += 500.0
        res = corr().correlate([self._mono_trace("r", 0.0, 1.0)], samples)
        assert res.ledger.clock == "monotonic"
        assert res.ledger.device_joules == pytest.approx(100.0)
        assert res.traces[0].energy.attributed_joules == pytest.approx(100.0)

    def test_wall_clock_fallback_when_domains_differ(self):
        samples = [mk_sample(1e9 + 0.1 * i, 100.0, mono=0.1 * i, domain="other") for i in range(11)]
        res = corr().correlate([self._mono_trace("r", 0.0, 1.0)], samples)
        assert res.ledger.clock == "wall"
        assert res.ledger.device_joules == pytest.approx(100.0)

    def test_batch_membership_beats_request_window(self):
        # Two requests alive for [0,2] but only executed in disjoint steps.
        samples = [mk_sample(1e9 + 0.1 * i, 100.0, mono=0.1 * i, domain="d") for i in range(21)]
        traces = [self._mono_trace("a", 0.0, 2.0), self._mono_trace("b", 0.0, 2.0)]
        batches = [
            BatchMetadata(batch_id="b1", step_index=1, timestamp=1e9, monotonic=0.0, step_start_monotonic=0.0,
                          step_end_monotonic=1.0, num_requests=1, num_prefill=1, num_decode=0,
                          total_scheduled_tokens=10, request_ids=["a"]),
            BatchMetadata(batch_id="b2", step_index=2, timestamp=1e9 + 1, monotonic=1.0, step_start_monotonic=1.0,
                          step_end_monotonic=2.0, num_requests=1, num_prefill=1, num_decode=0,
                          total_scheduled_tokens=10, request_ids=["b"]),
        ]
        res = corr().correlate(traces, samples, batches)
        assert res.ledger.membership_source == "batch_metadata"
        by = {t.request_id: t.energy.attributed_joules for t in res.traces}
        assert by["a"] == pytest.approx(100.0) and by["b"] == pytest.approx(100.0)
        assert res.ledger.conservation_error_joules < 1e-9

    def test_batches_without_step_end_fall_back(self):
        samples = [mk_sample(1e9 + 0.1 * i, 100.0, mono=0.1 * i, domain="d") for i in range(11)]
        batches = [BatchMetadata(batch_id="b1", step_index=1, timestamp=1e9, monotonic=0.0, num_requests=1,
                                 num_prefill=1, num_decode=0, total_scheduled_tokens=1, request_ids=["a"])]
        res = corr().correlate([self._mono_trace("a", 0.0, 1.0)], samples, batches)
        assert res.ledger.membership_source == "request_window"
        assert any("without a step end time" in n for n in res.ledger.notes)
