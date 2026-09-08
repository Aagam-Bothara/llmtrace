"""Shared fixtures: deterministic clocks, fake engines and sample builders."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import FakeClock, FakeLLMEngine  # noqa: E402

from llmtrace.data_plane.vllm_instrumentation import VLLMInstrumentation  # noqa: E402
from llmtrace.models.trace import GPUSample, ThrottleReason  # noqa: E402


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def engine(clock: FakeClock) -> FakeLLMEngine:
    return FakeLLMEngine(clock=clock)


@pytest.fixture
def instr(clock: FakeClock) -> VLLMInstrumentation:
    return VLLMInstrumentation(clock_domain="test", monotonic=clock.monotonic, wall=clock.time)


def mk_sample(
    t: float,
    power: Optional[float] = 200.0,
    gpu_id: int = 0,
    mono: Optional[float] = None,
    domain: Optional[str] = None,
    throttled: bool = False,
    util: Optional[float] = 90.0,
    mem_used: Optional[float] = 8000.0,
) -> GPUSample:
    return GPUSample(
        timestamp=t,
        monotonic=mono,
        clock_domain=domain,
        gpu_id=gpu_id,
        device_name=f"GPU{gpu_id}",
        gpu_utilization_pct=util,
        memory_utilization_pct=40.0,
        memory_used_mb=mem_used,
        memory_total_mb=16000.0,
        power_draw_watts=power,
        power_limit_watts=300.0,
        temperature_c=60.0,
        sm_clock_mhz=1500,
        memory_clock_mhz=5000,
        throttle_reasons=[ThrottleReason.HW_THERMAL if throttled else ThrottleReason.NONE],
    )


def const_power(t0: float, t1: float, step: float, power: float, gpu_id: int = 0, **kw) -> List[GPUSample]:
    out = []
    n = int(round((t1 - t0) / step))
    for i in range(n + 1):
        out.append(mk_sample(t0 + i * step, power, gpu_id, **kw))
    return out
