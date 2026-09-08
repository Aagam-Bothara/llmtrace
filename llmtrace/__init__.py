"""llmtrace - flight recorder, attribution, and autopsy for vLLM inference.

Importing this package never imports vLLM or NVML; those are loaded lazily by
the components that need them.
"""

__version__ = "0.2.0"

from llmtrace.models.config import TracerConfig  # noqa: E402
from llmtrace.models.trace import GPUSample, RequestTrace, TraceAnalysis  # noqa: E402
from llmtrace.tracer import LLMTracer  # noqa: E402

__all__ = ["LLMTracer", "TracerConfig", "RequestTrace", "GPUSample", "TraceAnalysis", "__version__"]
