"""Main LLMTracer orchestrator class."""

import asyncio
import logging
from typing import Any, List, Optional
from pathlib import Path

from llmtrace.models.config import TracerConfig
from llmtrace.models.trace import RequestTrace, TraceAnalysis
from llmtrace.data_plane.gpu_sampler import GPUSampler
from llmtrace.data_plane.vllm_instrumentation import VLLMInstrumentation
from llmtrace.data_plane.trace_writer import TraceWriter
from llmtrace.control_plane.correlator import Correlator
from llmtrace.control_plane.rules_engine import RulesEngine
from llmtrace.control_plane.reporter import Reporter

logger = logging.getLogger(__name__)


class LLMTracer:
    """
    Main tracer orchestrator for llmtrace.

    Coordinates data plane (collection) and control plane (analysis) components.

    Usage:
        tracer = LLMTracer(config)
        tracer.instrument_engine(llm_engine)
        # ... run inference ...
        tracer.stop()
        analysis = tracer.analyze()
    """

    def __init__(self, config: Optional[TracerConfig] = None, **kwargs):
        """
        Initialize LLMTracer.

        Args:
            config: TracerConfig object, or None to use defaults
            **kwargs: Config overrides (e.g., output_dir="./traces")
        """
        # Build config
        if config is None:
            config = TracerConfig(**kwargs)
        elif kwargs:
            # Override config fields
            config = config.model_copy(update=kwargs)

        self.config = config

        # Initialize components
        self.gpu_sampler = GPUSampler(config.gpu_sampler)
        self.vllm_instrumentation = VLLMInstrumentation(
            enable_batch_metadata=config.enable_batch_metadata,
            enable_kv_cache=config.enable_kv_cache_tracking,
        )
        self.trace_writer = TraceWriter(
            output_dir=config.output_dir,
            output_format=config.output_format,
            async_write=config.async_write,
            buffer_size=config.buffer_size,
        )
        self.correlator = Correlator(config.energy)
        self.rules_engine = RulesEngine(config.autopsy)
        self.reporter = Reporter(
            config=TracerConfig.model_validate({"cli_rich_output": True, "export_formats": ["jsonl"]})
            if not hasattr(config, 'reporter') else config.reporter
        )

        self._running = False
        self._collection_task: Optional[asyncio.Task] = None

        logger.info("LLMTracer initialized")

    def instrument_engine(self, engine: Any) -> None:
        """
        Instrument a vLLM LLMEngine for tracing.

        Args:
            engine: vLLM LLMEngine instance
        """
        logger.info("Instrumenting vLLM engine")
        self.vllm_instrumentation.instrument_engine(engine)

        # Start data collection
        self.start()

    def start(self) -> None:
        """Start tracing (GPU sampling and data collection)."""
        if self._running:
            logger.warning("Tracer already running")
            return

        logger.info("Starting llmtrace data collection")

        # Start GPU sampler
        self.gpu_sampler.start()

        # Start trace writer
        self.trace_writer.start()

        # Start periodic collection task
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())

        logger.info("llmtrace started successfully")

    async def stop(self) -> None:
        """Stop tracing and flush all data."""
        if not self._running:
            logger.warning("Tracer not running")
            return

        logger.info("Stopping llmtrace")

        self._running = False

        # Stop collection task
        if self._collection_task:
            await self._collection_task

        # Stop components
        await self.gpu_sampler.stop()
        await self.trace_writer.stop()

        # Uninstrument engine
        self.vllm_instrumentation.uninstrument_engine()

        logger.info("llmtrace stopped")

    async def _collection_loop(self) -> None:
        """
        Periodic collection loop.

        Collects completed traces, GPU samples, and batch metadata,
        then writes them to disk.
        """
        while self._running:
            await asyncio.sleep(1.0)  # Collect every second

            try:
                # Collect completed traces
                traces = await self.vllm_instrumentation.get_completed_traces(clear=True)

                # Collect batch metadata
                batches = await self.vllm_instrumentation.get_batch_metadata(clear=True)

                # Collect GPU samples
                # Note: We keep samples until correlation, then can clear
                # For now, we'll write them incrementally
                gpu_samples = await self.gpu_sampler.get_samples()

                # Write to disk
                if traces:
                    await self.trace_writer.write_traces(traces)
                    logger.debug(f"Collected and wrote {len(traces)} traces")

                if batches:
                    await self.trace_writer.write_batch_metadata(batches)
                    logger.debug(f"Collected and wrote {len(batches)} batch metadata entries")

                if gpu_samples:
                    await self.trace_writer.write_gpu_samples(gpu_samples)
                    # Clear old samples to manage memory
                    await self.gpu_sampler.clear_samples()
                    logger.debug(f"Collected and wrote {len(gpu_samples)} GPU samples")

            except Exception as e:
                logger.error(f"Error in collection loop: {e}", exc_info=True)

    async def analyze(
        self,
        baseline_dir: Optional[str] = None,
    ) -> TraceAnalysis:
        """
        Analyze collected traces and generate report.

        Args:
            baseline_dir: Optional path to baseline traces for regression detection

        Returns:
            TraceAnalysis object
        """
        logger.info("Analyzing collected traces")

        # Load traces from output files
        traces = await self._load_traces()

        if not traces:
            logger.warning("No traces found to analyze")
            return self.reporter._empty_analysis()

        # Load GPU samples
        gpu_samples = await self._load_gpu_samples()

        # Correlate traces with GPU samples
        correlated_traces = await self.correlator.correlate_traces(traces, gpu_samples)

        # Run diagnosis on each trace
        for trace in correlated_traces:
            diagnosis = await self.rules_engine.diagnose_request(trace)
            trace.diagnosis = diagnosis

        # Run batch-level diagnosis if we have batch metadata
        # (This would require loading batch metadata and grouping traces by batch)

        # Load baseline if provided
        baseline_traces = None
        if baseline_dir:
            baseline_traces = await self._load_traces(baseline_dir)

        # Generate analysis
        analysis = self.reporter.generate_analysis(correlated_traces, baseline_traces)

        logger.info("Analysis complete")
        return analysis

    def print_analysis(self, analysis: TraceAnalysis) -> None:
        """Print analysis to console."""
        self.reporter.print_analysis(analysis)

    async def _load_traces(self, directory: Optional[str] = None) -> List[RequestTrace]:
        """Load traces from output directory."""
        dir_path = Path(directory) if directory else Path(self.config.output_dir)

        if not dir_path.exists():
            return []

        traces = []

        # Load from JSONL files
        if self.config.output_format == "jsonl":
            import json

            for trace_file in dir_path.glob("traces_*.jsonl"):
                with open(trace_file, "r") as f:
                    for line in f:
                        if line.strip():
                            trace_dict = json.loads(line)
                            trace = RequestTrace.model_validate(trace_dict)
                            traces.append(trace)

        # Load from Parquet files
        elif self.config.output_format == "parquet":
            import pyarrow.parquet as pq

            for trace_file in dir_path.glob("traces_*.parquet"):
                table = pq.read_table(trace_file)
                for row in table.to_pylist():
                    trace = RequestTrace.model_validate(row)
                    traces.append(trace)

        logger.info(f"Loaded {len(traces)} traces from {dir_path}")
        return traces

    async def _load_gpu_samples(self, directory: Optional[str] = None):
        """Load GPU samples from output directory."""
        from llmtrace.models.trace import GPUSample

        dir_path = Path(directory) if directory else Path(self.config.output_dir)

        if not dir_path.exists():
            return []

        samples = []

        # Load from JSONL files
        if self.config.output_format == "jsonl":
            import json

            for sample_file in dir_path.glob("gpu_*.jsonl"):
                with open(sample_file, "r") as f:
                    for line in f:
                        if line.strip():
                            sample_dict = json.loads(line)
                            sample = GPUSample.model_validate(sample_dict)
                            samples.append(sample)

        # Load from Parquet files
        elif self.config.output_format == "parquet":
            import pyarrow.parquet as pq

            for sample_file in dir_path.glob("gpu_*.parquet"):
                table = pq.read_table(sample_file)
                for row in table.to_pylist():
                    sample = GPUSample.model_validate(row)
                    samples.append(sample)

        logger.info(f"Loaded {len(samples)} GPU samples from {dir_path}")
        return samples

    def get_output_files(self) -> dict:
        """Get paths to output files."""
        return self.trace_writer.get_output_files()

    @classmethod
    def from_config_file(cls, config_path: str) -> "LLMTracer":
        """
        Create LLMTracer from a config file.

        Args:
            config_path: Path to JSON or YAML config file

        Returns:
            LLMTracer instance
        """
        import json
        from pathlib import Path

        config_file = Path(config_path)

        if config_file.suffix == ".json":
            with open(config_file) as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_file.suffix}")

        config = TracerConfig.model_validate(config_dict)
        return cls(config)
