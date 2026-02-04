"""Basic tests for llmtrace components."""

import pytest
import asyncio
import time
from unittest.mock import Mock, MagicMock
from llmtrace.models.trace import (
    RequestTrace,
    RequestSpan,
    SpanPhase,
    GPUSample,
    ThrottleReason,
)
from llmtrace.models.config import EnergyConfig, AutopsyConfig
from llmtrace.control_plane.correlator import Correlator
from llmtrace.control_plane.rules_engine import RulesEngine


class TestCorrelator:
    """Test energy correlator."""

    @pytest.mark.asyncio
    async def test_basic_correlation(self):
        """Test basic trace-GPU sample correlation."""
        # Create mock trace
        trace = RequestTrace(
            request_id="test-1",
            start_time=1000.0,
            end_time=1001.0,
            prompt_length=10,
            output_length=5,
            model_name="test-model",
        )

        # Create mock GPU samples
        samples = [
            GPUSample(
                timestamp=1000.0 + i * 0.1,
                gpu_id=0,
                device_name="Test GPU",
                gpu_utilization_pct=80.0,
                memory_utilization_pct=70.0,
                memory_used_mb=8000.0,
                memory_total_mb=16000.0,
                power_draw_watts=200.0,
                power_limit_watts=300.0,
                temperature_c=70.0,
                sm_clock_mhz=1500,
                memory_clock_mhz=5000,
                throttle_reasons=[ThrottleReason.NONE],
            )
            for i in range(10)
        ]

        # Correlate
        correlator = Correlator(EnergyConfig())
        correlated = await correlator.correlate_traces([trace], samples)

        assert len(correlated) == 1
        assert len(correlated[0].gpu_samples) == 10
        assert correlated[0].energy is not None
        assert correlated[0].energy.total_joules > 0

    @pytest.mark.asyncio
    async def test_energy_integration(self):
        """Test power integration for energy calculation."""
        correlator = Correlator(EnergyConfig())

        # Two samples 1 second apart, 200W each
        samples = [
            GPUSample(
                timestamp=0.0,
                gpu_id=0,
                device_name="Test",
                gpu_utilization_pct=100.0,
                memory_utilization_pct=50.0,
                memory_used_mb=8000.0,
                memory_total_mb=16000.0,
                power_draw_watts=200.0,
                power_limit_watts=300.0,
                temperature_c=70.0,
                sm_clock_mhz=1500,
                memory_clock_mhz=5000,
                throttle_reasons=[ThrottleReason.NONE],
            ),
            GPUSample(
                timestamp=1.0,
                gpu_id=0,
                device_name="Test",
                gpu_utilization_pct=100.0,
                memory_utilization_pct=50.0,
                memory_used_mb=8000.0,
                memory_total_mb=16000.0,
                power_draw_watts=200.0,
                power_limit_watts=300.0,
                temperature_c=70.0,
                sm_clock_mhz=1500,
                memory_clock_mhz=5000,
                throttle_reasons=[ThrottleReason.NONE],
            ),
        ]

        energy = correlator._integrate_power(samples)

        # Trapezoidal integration: 0.5 * (200 + 200) * 1 = 200 J
        assert energy == pytest.approx(200.0, rel=0.01)


class TestRulesEngine:
    """Test diagnosis rules engine."""

    @pytest.mark.asyncio
    async def test_queueing_overload_detection(self):
        """Test queueing overload rule."""
        engine = RulesEngine(AutopsyConfig(queue_overload_threshold_ms=50.0))

        # Create trace with long queue time
        trace = RequestTrace(
            request_id="test-1",
            start_time=0.0,
            end_time=0.2,
            prompt_length=10,
            output_length=5,
            model_name="test-model",
            spans=[
                RequestSpan(
                    phase=SpanPhase.QUEUE,
                    start_time=0.0,
                    end_time=0.1,
                    duration_ms=100.0,  # Long queue!
                )
            ],
        )

        diagnosis = await engine.diagnose_request(trace)

        assert diagnosis is not None
        assert diagnosis.category.value == "queueing_overload"
        assert diagnosis.confidence > 0.5

    @pytest.mark.asyncio
    async def test_throttling_detection(self):
        """Test GPU throttling rule."""
        engine = RulesEngine(AutopsyConfig(throttle_severity_threshold=2))

        # Create trace with throttled GPU samples
        trace = RequestTrace(
            request_id="test-1",
            start_time=0.0,
            end_time=1.0,
            prompt_length=10,
            output_length=5,
            model_name="test-model",
            gpu_samples=[
                GPUSample(
                    timestamp=0.0 + i * 0.1,
                    gpu_id=0,
                    device_name="Test GPU",
                    gpu_utilization_pct=80.0,
                    memory_utilization_pct=70.0,
                    memory_used_mb=8000.0,
                    memory_total_mb=16000.0,
                    power_draw_watts=200.0,
                    power_limit_watts=300.0,
                    temperature_c=85.0,
                    sm_clock_mhz=1200,  # Reduced clock
                    memory_clock_mhz=5000,
                    throttle_reasons=[ThrottleReason.HW_THERMAL],  # Throttled!
                )
                for i in range(5)
            ],
        )

        diagnosis = await engine.diagnose_request(trace)

        assert diagnosis is not None
        assert diagnosis.category.value == "gpu_throttling"
        assert len(diagnosis.evidence) > 0


class TestModels:
    """Test data models."""

    def test_request_trace_properties(self):
        """Test RequestTrace computed properties."""
        trace = RequestTrace(
            request_id="test-1",
            start_time=0.0,
            end_time=1.0,
            prompt_length=10,
            output_length=20,
            model_name="test-model",
            spans=[
                RequestSpan(
                    phase=SpanPhase.QUEUE,
                    start_time=0.0,
                    end_time=0.2,
                    duration_ms=200.0,
                ),
                RequestSpan(
                    phase=SpanPhase.PREFILL,
                    start_time=0.2,
                    end_time=0.3,
                    duration_ms=100.0,
                ),
                RequestSpan(
                    phase=SpanPhase.DECODE,
                    start_time=0.3,
                    end_time=1.0,
                    duration_ms=700.0,
                ),
            ],
        )

        assert trace.total_duration_ms == 1000.0
        assert trace.queue_duration_ms == 200.0
        assert trace.prefill_duration_ms == 100.0
        assert trace.decode_duration_ms == 700.0

    def test_gpu_sample_throttle_detection(self):
        """Test GPU sample throttle detection."""
        # Not throttled
        sample1 = GPUSample(
            timestamp=0.0,
            gpu_id=0,
            device_name="Test",
            gpu_utilization_pct=100.0,
            memory_utilization_pct=50.0,
            memory_used_mb=8000.0,
            memory_total_mb=16000.0,
            power_draw_watts=200.0,
            power_limit_watts=300.0,
            temperature_c=70.0,
            sm_clock_mhz=1500,
            memory_clock_mhz=5000,
            throttle_reasons=[ThrottleReason.NONE],
        )

        assert not sample1.is_throttled

        # Throttled
        sample2 = GPUSample(
            timestamp=0.0,
            gpu_id=0,
            device_name="Test",
            gpu_utilization_pct=100.0,
            memory_utilization_pct=50.0,
            memory_used_mb=8000.0,
            memory_total_mb=16000.0,
            power_draw_watts=200.0,
            power_limit_watts=300.0,
            temperature_c=85.0,
            sm_clock_mhz=1200,
            memory_clock_mhz=5000,
            throttle_reasons=[ThrottleReason.HW_THERMAL],
        )

        assert sample2.is_throttled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
