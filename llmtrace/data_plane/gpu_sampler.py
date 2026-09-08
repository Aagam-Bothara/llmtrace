"""GPU telemetry sampler.

Runs in a background *thread* so it keeps sampling while the synchronous vLLM
``LLM.generate()`` call blocks the calling thread (and any asyncio loop).

The NVML dependency (``nvidia-ml-py``) is imported lazily inside
``NVMLBackend`` so this module imports on machines without a GPU. Tests inject
a fake backend implementing the same three methods.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Protocol

from llmtrace.models.config import GPUSamplerConfig
from llmtrace.models.trace import GPUSample, ThrottleReason

logger = logging.getLogger(__name__)


class SamplerBackend(Protocol):
    def open(self, gpu_ids: Optional[List[int]]) -> List[Dict[str, Any]]:
        """Initialise and return [{"gpu_id": int, "name": str}, ...] for monitored devices."""

    def read(self, gpu_id: int) -> Dict[str, Any]:
        """Return raw fields for one device; missing/failed fields are absent or None."""

    def close(self) -> None: ...


class NVMLBackend:
    """Reads telemetry through ``pynvml`` (package ``nvidia-ml-py``)."""

    def __init__(self) -> None:
        self._nvml: Any = None
        self._handles: Dict[int, Any] = {}

    def open(self, gpu_ids: Optional[List[int]]) -> List[Dict[str, Any]]:
        try:
            import pynvml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pynvml not installed (pip install 'llmtrace[nvml]')") from exc
        pynvml.nvmlInit()
        self._nvml = pynvml
        count = pynvml.nvmlDeviceGetCount()
        wanted = list(range(count)) if not gpu_ids else gpu_ids
        devices = []
        for gpu_id in wanted:
            if gpu_id < 0 or gpu_id >= count:
                logger.warning("GPU %d not present (device count %d); skipping", gpu_id, count)
                continue
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            self._handles[gpu_id] = handle
            devices.append({"gpu_id": gpu_id, "name": str(name)})
        return devices

    def _try(self, fn, *args):  # type: ignore[no-untyped-def]
        try:
            return fn(*args)
        except self._nvml.NVMLError:
            return None

    def read(self, gpu_id: int) -> Dict[str, Any]:
        nv = self._nvml
        h = self._handles[gpu_id]
        raw: Dict[str, Any] = {}
        util = self._try(nv.nvmlDeviceGetUtilizationRates, h)
        if util is not None:
            raw["gpu_utilization_pct"] = float(util.gpu)
            raw["memory_utilization_pct"] = float(util.memory)
        mem = self._try(nv.nvmlDeviceGetMemoryInfo, h)
        if mem is not None:
            raw["memory_used_mb"] = mem.used / (1024 * 1024)
            raw["memory_total_mb"] = mem.total / (1024 * 1024)
        power_mw = self._try(nv.nvmlDeviceGetPowerUsage, h)
        if power_mw is not None:
            raw["power_draw_watts"] = power_mw / 1000.0
        limit_mw = self._try(nv.nvmlDeviceGetPowerManagementLimit, h)
        if limit_mw is not None:
            raw["power_limit_watts"] = limit_mw / 1000.0
        temp = self._try(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU)
        if temp is not None:
            raw["temperature_c"] = float(temp)
        sm = self._try(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_SM)
        if sm is not None:
            raw["sm_clock_mhz"] = int(sm)
        mc = self._try(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_MEM)
        if mc is not None:
            raw["memory_clock_mhz"] = int(mc)
        bits = self._try(nv.nvmlDeviceGetCurrentClocksThrottleReasons, h)
        raw["throttle_bits"] = bits
        if bits is not None:
            raw["throttle_map"] = {
                nv.nvmlClocksThrottleReasonGpuIdle: ThrottleReason.GPU_IDLE,
                nv.nvmlClocksThrottleReasonApplicationsClocksSetting: ThrottleReason.APPLICATIONS_CLOCKS,
                nv.nvmlClocksThrottleReasonSwPowerCap: ThrottleReason.SW_POWER_CAP,
                nv.nvmlClocksThrottleReasonHwSlowdown: ThrottleReason.HW_SLOWDOWN,
                nv.nvmlClocksThrottleReasonSyncBoost: ThrottleReason.SYNC_BOOST,
                nv.nvmlClocksThrottleReasonSwThermalSlowdown: ThrottleReason.SW_THERMAL,
                nv.nvmlClocksThrottleReasonHwThermalSlowdown: ThrottleReason.HW_THERMAL,
                nv.nvmlClocksThrottleReasonHwPowerBrakeSlowdown: ThrottleReason.HW_POWER_BRAKE,
                nv.nvmlClocksThrottleReasonDisplayClockSetting: ThrottleReason.DISPLAY_CLOCK,
            }
        return raw

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception as exc:  # pragma: no cover
                logger.warning("nvmlShutdown failed: %s", exc)
            self._nvml = None
            self._handles.clear()


def throttle_reasons_from_raw(raw: Dict[str, Any]) -> List[ThrottleReason]:
    bits = raw.get("throttle_bits")
    if bits is None:
        return [ThrottleReason.UNKNOWN]
    reasons = [r for bit, r in raw.get("throttle_map", {}).items() if bits & bit]
    return reasons or [ThrottleReason.NONE]


class GPUSampler:
    """Continuously samples GPU telemetry into a bounded in-memory buffer."""

    def __init__(
        self,
        config: GPUSamplerConfig,
        backend: Optional[SamplerBackend] = None,
        clock_domain: Optional[str] = None,
    ):
        self.config = config
        self._backend: SamplerBackend = backend or NVMLBackend()
        self.clock_domain = clock_domain or uuid.uuid4().hex[:12]

        self._devices: List[Dict[str, Any]] = []
        self._buffer: Deque[GPUSample] = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.available = False
        self.unavailable_reason: Optional[str] = "not started"
        self._samples_taken = 0
        self._dropped = 0
        self._read_errors = 0
        self._last_error: Optional[str] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            logger.warning("GPU sampler already running")
            return
        try:
            self._devices = self._backend.open(self.config.gpu_ids)
            if not self._devices:
                raise RuntimeError("no GPUs selected/found")
        except Exception as exc:
            self.available = False
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"
            try:
                self._backend.close()
            except Exception:
                pass
            if self.config.require_gpu:
                raise RuntimeError(f"GPU telemetry required but unavailable: {exc}") from exc
            logger.warning("GPU telemetry unavailable; continuing without it (%s)", exc)
            return

        self.available = True
        self.unavailable_reason = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="llmtrace-gpu-sampler", daemon=True)
        self._thread.start()
        logger.info(
            "GPU sampler started: %s at %dms", [d["name"] for d in self._devices], self.config.sample_interval_ms
        )

    def stop(self) -> None:
        """Stop the sampling thread and release NVML. Idempotent; buffered samples are kept."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.config.sample_interval_ms / 1000.0 * 5))
            if self._thread.is_alive():  # pragma: no cover
                logger.error("GPU sampler thread did not stop in time")
            self._thread = None
        if self.available:
            self._backend.close()
        logger.info("GPU sampler stopped (%d samples taken, %d dropped)", self._samples_taken, self._dropped)

    def sample_once(self) -> List[GPUSample]:
        """Take one sample from every device. Public for tests and synchronous use."""
        samples: List[GPUSample] = []
        for dev in self._devices:
            try:
                raw = self._backend.read(dev["gpu_id"])
            except Exception as exc:
                self._read_errors += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.error("Error sampling GPU %s: %s", dev["gpu_id"], exc)
                continue
            samples.append(
                GPUSample(
                    timestamp=time.time(),
                    monotonic=time.monotonic(),
                    clock_domain=self.clock_domain,
                    gpu_id=dev["gpu_id"],
                    device_name=dev["name"],
                    gpu_utilization_pct=raw.get("gpu_utilization_pct"),
                    memory_utilization_pct=raw.get("memory_utilization_pct"),
                    memory_used_mb=raw.get("memory_used_mb"),
                    memory_total_mb=raw.get("memory_total_mb"),
                    power_draw_watts=raw.get("power_draw_watts"),
                    power_limit_watts=raw.get("power_limit_watts"),
                    temperature_c=raw.get("temperature_c"),
                    sm_clock_mhz=raw.get("sm_clock_mhz"),
                    memory_clock_mhz=raw.get("memory_clock_mhz"),
                    throttle_reasons=throttle_reasons_from_raw(raw),
                )
            )
        return samples

    def _loop(self) -> None:
        interval = self.config.sample_interval_ms / 1000.0
        next_at = time.monotonic()
        while not self._stop.is_set():
            samples = self.sample_once()
            with self._lock:
                for s in samples:
                    if len(self._buffer) >= self.config.max_buffered_samples:
                        self._buffer.popleft()
                        self._dropped += 1
                    self._buffer.append(s)
                self._samples_taken += len(samples)
            next_at += interval
            wait = next_at - time.monotonic()
            if wait <= 0:
                next_at = time.monotonic()  # fell behind; do not burst to catch up
                continue
            self._stop.wait(wait)

    def drain(self) -> List[GPUSample]:
        """Atomically remove and return all buffered samples (no read/clear race)."""
        with self._lock:
            out = list(self._buffer)
            self._buffer.clear()
        return out

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            buffered = len(self._buffer)
        return {
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "devices": list(self._devices),
            "samples_taken": self._samples_taken,
            "buffered": buffered,
            "dropped": self._dropped,
            "read_errors": self._read_errors,
            "last_error": self._last_error,
        }
