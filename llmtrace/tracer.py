"""Main LLMTracer orchestrator.

Synchronous lifecycle, safe to use around the blocking ``vllm.LLM.generate()``
call: GPU sampling, collection and writing all run in background threads, so
they progress while the calling thread is inside vLLM.

    tracer = LLMTracer(output_dir="./traces")
    tracer.instrument_engine(llm.llm_engine)   # starts collection
    llm.generate(prompts, sampling_params)
    tracer.stop()                              # restores engine, drains, flushes
    analysis = tracer.analyze()
"""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmtrace import io
from llmtrace.control_plane.correlator import Correlator
from llmtrace.control_plane.reporter import Reporter
from llmtrace.control_plane.rules_engine import RulesEngine
from llmtrace.data_plane.gpu_sampler import GPUSampler, SamplerBackend
from llmtrace.data_plane.trace_writer import TraceWriter
from llmtrace.data_plane.vllm_instrumentation import VLLMInstrumentation
from llmtrace.models.config import TracerConfig
from llmtrace.models.trace import TraceAnalysis

logger = logging.getLogger(__name__)

_CONVENIENCE_KWARGS = {
    # LLMTracer(**kwargs) shortcuts -> nested config paths
    "gpu_sample_interval_ms": ("gpu_sampler", "sample_interval_ms"),
    "enable_energy_attribution": ("energy", "enabled"),
    "attribution_method": ("energy", "attribution_method"),
    "energy_price_usd_per_kwh": ("energy", "energy_price_usd_per_kwh"),
}


class LLMTracer:
    def __init__(
        self,
        config: Optional[TracerConfig] = None,
        *,
        gpu_backend: Optional[SamplerBackend] = None,
        **kwargs: Any,
    ):
        """Create a tracer.

        ``kwargs`` may be top-level ``TracerConfig`` fields or the shortcuts in
        ``_CONVENIENCE_KWARGS``. Unknown options raise ``ValueError`` instead of
        being ignored.
        """
        base = (config or TracerConfig()).model_dump()
        for key, value in kwargs.items():
            if key in _CONVENIENCE_KWARGS:
                section, field = _CONVENIENCE_KWARGS[key]
                base[section][field] = value
            elif key in TracerConfig.model_fields:
                base[key] = value
            else:
                raise ValueError(
                    f"Unknown LLMTracer option '{key}'. Valid: "
                    f"{sorted(list(TracerConfig.model_fields) + list(_CONVENIENCE_KWARGS))}"
                )
        self.config = TracerConfig.model_validate(base)

        self.session_id = uuid.uuid4().hex[:12]
        self.gpu_sampler = GPUSampler(self.config.gpu_sampler, backend=gpu_backend, clock_domain=self.session_id)
        self.vllm_instrumentation = VLLMInstrumentation(
            enable_batch_metadata=self.config.enable_batch_metadata,
            max_buffered=self.config.max_buffered_events,
            strict=self.config.strict_instrumentation,
            clock_domain=self.session_id,
        )
        self.trace_writer = TraceWriter(
            output_dir=self.config.output_dir,
            output_format=self.config.output_format,
            background=self.config.background_writes,
            max_queue=self.config.max_write_queue,
        )
        self.correlator = Correlator(self.config.energy)
        self.rules_engine = RulesEngine(self.config.autopsy)
        self.reporter = Reporter(self.config.reporter)

        self._state = "new"  # new | running | stopped
        self._stop_event = threading.Event()
        self._collector: Optional[threading.Thread] = None
        self._collection_errors = 0
        self._last_collection_error: Optional[str] = None
        self._incomplete_written = 0

    # -------------------------------------------------------------- lifecycle

    def instrument_engine(self, engine: Any) -> None:
        """Instrument a vLLM LLMEngine and start collection."""
        self.vllm_instrumentation.instrument_engine(engine)
        self.start()

    def start(self) -> None:
        if self._state == "running":
            logger.warning("Tracer already running")
            return
        if self._state == "stopped":
            raise RuntimeError("LLMTracer cannot be restarted; create a new instance")
        self.trace_writer.start()
        self.gpu_sampler.start()  # raises only if require_gpu=True and NVML is unavailable
        self._stop_event.clear()
        self._collector = threading.Thread(target=self._collect_loop, name="llmtrace-collector", daemon=True)
        self._collector.start()
        self._state = "running"
        logger.info("llmtrace started (session %s, output %s)", self.session_id, self.config.output_dir)

    def stop(self) -> None:
        """Stop collection, restore the engine, drain buffers and flush files. Idempotent."""
        if self._state != "running":
            if self._state == "new":
                logger.warning("Tracer not running")
            return
        self._state = "stopped"
        self._stop_event.set()
        if self._collector is not None:
            self._collector.join()
            self._collector = None

        # Restore the engine first so no new events arrive, then drain everything.
        leftovers = self.vllm_instrumentation.uninstrument_engine()
        self.gpu_sampler.stop()
        self._collect_once()
        if leftovers:
            if self.config.write_incomplete_requests:
                self.trace_writer.write_traces(leftovers)
                self._incomplete_written = len(leftovers)
            logger.warning("%d requests were still active at stop (written as incomplete=%s)",
                           len(leftovers), self.config.write_incomplete_requests)
        self.trace_writer.stop()
        health = self.health()
        if health["instrumentation"]["instrumentation_errors"] or health["writer"]["write_errors"] \
                or any(health["writer"]["dropped"].values()) or self._collection_errors:
            logger.error("llmtrace stopped with problems: %s", health)
        else:
            logger.info("llmtrace stopped cleanly: %s", health)

    def __enter__(self) -> "LLMTracer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ------------------------------------------------------------- collection

    def _collect_loop(self) -> None:
        while not self._stop_event.wait(self.config.collection_interval_s):
            self._collect_once()

    def _collect_once(self) -> None:
        try:
            traces = self.vllm_instrumentation.drain_completed_traces()
            batches = self.vllm_instrumentation.drain_batch_metadata()
            samples = self.gpu_sampler.drain()
            if traces:
                self.trace_writer.write_traces(traces)
            if batches:
                self.trace_writer.write_batch_metadata(batches)
            if samples:
                self.trace_writer.write_gpu_samples(samples)
        except Exception as exc:
            self._collection_errors += 1
            self._last_collection_error = f"{type(exc).__name__}: {exc}"
            logger.error("Collection error: %s", exc, exc_info=True)

    def health(self) -> Dict[str, Any]:
        return {
            "state": self._state,
            "session_id": self.session_id,
            "instrumentation": self.vllm_instrumentation.health(),
            "gpu_sampler": self.gpu_sampler.stats(),
            "writer": self.trace_writer.stats(),
            "collection_errors": self._collection_errors,
            "last_collection_error": self._last_collection_error,
            "incomplete_requests_written": self._incomplete_written,
        }

    # --------------------------------------------------------------- analysis

    def analyze(self, baseline_dir: Optional[str] = None) -> TraceAnalysis:
        """Analyze this session's files in ``output_dir`` (optionally against a baseline dir)."""
        out_dir = Path(self.config.output_dir)
        files = self.trace_writer.get_output_files()
        traces = io.load_traces(files.get("traces", []))
        if not traces:
            logger.warning("No traces found for session %s in %s", self.trace_writer.session_id, out_dir)
            return self.reporter._empty_analysis()
        samples = io.load_gpu_samples(files.get("gpu", []))
        batches = io.load_batches(files.get("batches", []))
        result = self.correlator.correlate(traces, samples, batches)
        for t in result.traces:
            t.diagnosis = self.rules_engine.diagnose_request(t)

        baseline_traces = None
        baseline_ledger = None
        if baseline_dir:
            baseline_traces = io.load_traces([baseline_dir])
            if baseline_traces:
                b = self.correlator.correlate(
                    baseline_traces, io.load_gpu_samples([baseline_dir]), io.load_batches([baseline_dir])
                )
                baseline_traces, baseline_ledger = b.traces, b.ledger
        return self.reporter.generate_analysis(result.traces, baseline_traces, result.ledger, baseline_ledger)

    def print_analysis(self, analysis: TraceAnalysis) -> None:
        self.reporter.print_analysis(analysis)

    def get_output_files(self) -> Dict[str, List[str]]:
        return self.trace_writer.get_output_files()

    @classmethod
    def from_config_file(cls, config_path: str, **kwargs: Any) -> "LLMTracer":
        path = Path(config_path)
        if path.suffix != ".json":
            raise ValueError(f"Unsupported config format: {path.suffix} (use .json)")
        return cls(TracerConfig.model_validate(io.read_json(path)), **kwargs)
