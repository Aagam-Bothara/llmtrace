"""GPU telemetry sampler using NVML."""

import asyncio
import logging
import time
from typing import Dict, List, Optional

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from llmtrace.models.config import GPUSamplerConfig
from llmtrace.models.trace import GPUSample, ThrottleReason

logger = logging.getLogger(__name__)


class GPUSampler:
    """
    Continuously samples GPU telemetry using NVML.

    Runs in a background thread/task and stores timestamped samples.
    Designed for low overhead (<1% CPU typically).
    """

    def __init__(self, config: GPUSamplerConfig):
        if not NVML_AVAILABLE:
            raise RuntimeError(
                "pynvml not available. Install with: pip install nvidia-ml-py"
            )

        self.config = config
        self.running = False
        self.samples: List[GPUSample] = []
        self._sample_lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._handles: List[pynvml.c_nvmlDevice_t] = []
        self._gpu_info: List[Dict] = []

    def start(self) -> None:
        """Initialize NVML and start sampling."""
        logger.info("Initializing NVML for GPU sampling")
        pynvml.nvmlInit()

        # Get GPU handles
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_ids = self.config.gpu_ids if self.config.gpu_ids else list(range(device_count))

        for gpu_id in gpu_ids:
            if gpu_id >= device_count:
                logger.warning(f"GPU {gpu_id} not found, skipping")
                continue

            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            self._handles.append(handle)
            self._gpu_info.append({"gpu_id": gpu_id, "name": name})
            logger.info(f"Monitoring GPU {gpu_id}: {name}")

        # Start sampling loop
        self.running = True
        self._task = asyncio.create_task(self._sampling_loop())
        logger.info(
            f"GPU sampler started with {len(self._handles)} GPUs at "
            f"{self.config.sample_interval_ms}ms interval"
        )

    async def stop(self) -> None:
        """Stop sampling and cleanup."""
        logger.info("Stopping GPU sampler")
        self.running = False

        if self._task:
            await self._task

        pynvml.nvmlShutdown()
        logger.info(f"GPU sampler stopped. Collected {len(self.samples)} samples.")

    async def _sampling_loop(self) -> None:
        """Main sampling loop."""
        interval_s = self.config.sample_interval_ms / 1000.0

        while self.running:
            start = time.perf_counter()

            # Sample all GPUs
            samples = []
            for handle, info in zip(self._handles, self._gpu_info):
                try:
                    sample = self._sample_gpu(handle, info)
                    samples.append(sample)
                except Exception as e:
                    logger.error(f"Error sampling GPU {info['gpu_id']}: {e}")

            # Store samples
            async with self._sample_lock:
                self.samples.extend(samples)

            # Sleep for remainder of interval
            elapsed = time.perf_counter() - start
            sleep_time = max(0, interval_s - elapsed)
            await asyncio.sleep(sleep_time)

    def _sample_gpu(self, handle: pynvml.c_nvmlDevice_t, info: Dict) -> GPUSample:
        """Sample a single GPU."""
        timestamp = time.time()

        # Utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = float(utilization.gpu)
        mem_util = float(utilization.memory)

        # Memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used_mb = mem_info.used / (1024 * 1024)
        mem_total_mb = mem_info.total / (1024 * 1024)

        # Power
        try:
            power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
        except pynvml.NVMLError:
            power_draw = 0.0

        try:
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        except pynvml.NVMLError:
            power_limit = 0.0

        # Temperature
        try:
            temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        except pynvml.NVMLError:
            temp = 0.0

        # Clocks
        try:
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        except pynvml.NVMLError:
            sm_clock = 0

        try:
            mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except pynvml.NVMLError:
            mem_clock = 0

        # Throttling reasons
        throttle_reasons = self._get_throttle_reasons(handle)

        # Optional: Tensor utilization (requires newer GPUs/drivers)
        tensor_util = None
        if self.config.collect_tensor_utilization:
            try:
                # This is GPU-specific and may not be available
                # Placeholder for now - would need GPU-specific implementation
                tensor_util = None
            except Exception:
                pass

        return GPUSample(
            timestamp=timestamp,
            gpu_id=info["gpu_id"],
            device_name=info["name"],
            gpu_utilization_pct=gpu_util,
            memory_utilization_pct=mem_util,
            memory_used_mb=mem_used_mb,
            memory_total_mb=mem_total_mb,
            power_draw_watts=power_draw,
            power_limit_watts=power_limit,
            temperature_c=temp,
            sm_clock_mhz=sm_clock,
            memory_clock_mhz=mem_clock,
            throttle_reasons=throttle_reasons,
            tensor_utilization_pct=tensor_util,
        )

    def _get_throttle_reasons(self, handle: pynvml.c_nvmlDevice_t) -> List[ThrottleReason]:
        """Get current throttling reasons."""
        try:
            reasons_bits = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        except pynvml.NVMLError:
            return [ThrottleReason.NONE]

        reasons = []

        # Map NVML throttle bits to our enum
        throttle_map = {
            pynvml.nvmlClocksThrottleReasonGpuIdle: ThrottleReason.GPU_IDLE,
            pynvml.nvmlClocksThrottleReasonApplicationsClocksSetting: ThrottleReason.APPLICATIONS_CLOCKS,
            pynvml.nvmlClocksThrottleReasonSwPowerCap: ThrottleReason.SW_POWER_CAP,
            pynvml.nvmlClocksThrottleReasonHwSlowdown: ThrottleReason.HW_SLOWDOWN,
            pynvml.nvmlClocksThrottleReasonSyncBoost: ThrottleReason.SYNC_BOOST,
            pynvml.nvmlClocksThrottleReasonSwThermalSlowdown: ThrottleReason.SW_THERMAL,
            pynvml.nvmlClocksThrottleReasonHwThermalSlowdown: ThrottleReason.HW_THERMAL,
            pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown: ThrottleReason.HW_POWER_BRAKE,
            pynvml.nvmlClocksThrottleReasonDisplayClockSetting: ThrottleReason.DISPLAY_CLOCK,
        }

        for bit, reason in throttle_map.items():
            if reasons_bits & bit:
                reasons.append(reason)

        return reasons if reasons else [ThrottleReason.NONE]

    async def get_samples(
        self, start_time: Optional[float] = None, end_time: Optional[float] = None
    ) -> List[GPUSample]:
        """
        Get samples within a time window.

        Args:
            start_time: Unix timestamp (inclusive)
            end_time: Unix timestamp (inclusive)

        Returns:
            List of samples in the time window
        """
        async with self._sample_lock:
            if start_time is None and end_time is None:
                return list(self.samples)

            filtered = []
            for sample in self.samples:
                if start_time is not None and sample.timestamp < start_time:
                    continue
                if end_time is not None and sample.timestamp > end_time:
                    continue
                filtered.append(sample)

            return filtered

    async def clear_samples(self) -> None:
        """Clear stored samples (useful to manage memory)."""
        async with self._sample_lock:
            count = len(self.samples)
            self.samples.clear()
            logger.debug(f"Cleared {count} GPU samples")
