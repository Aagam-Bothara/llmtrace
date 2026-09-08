"""Workload: a steady stream of short requests with long prompts injected on a schedule.

Deterministic given ``seed``. Prompts are token-id lists (``TokensPrompt``), so
prompt lengths are exact on any tokenizer.

Hypothesis under test (vLLM 0.11.0 V1 scheduler): with chunked prefill, a long
prompt's prefill chunk is co-scheduled with the decode steps of concurrent short
requests. Steps carrying a large prefill chunk take longer, so short requests
that share those steps see inflated per-token latency (TPOT) and short requests
arriving during them see inflated TTFT. Capping the per-step prefill tokens of
long prompts (``long_prefill_token_threshold``) should bound step time and
shrink the short-request tail, at the cost of the long request's own TTFT.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from llmtrace.workload import ArrivalSpec, LengthSpec, RequestClass, RequestSpec, WorkloadSpec, make_prompt  # noqa: F401

# RequestSpec and make_prompt now live in llmtrace.workload (re-exported here unchanged).


@dataclass(frozen=True)
class WorkloadConfig:
    num_short: int = 120
    short_rate_per_s: float = 40.0  # short arrivals per second (deterministic spacing)
    short_prompt_len: int = 32
    short_max_tokens: int = 128  # long enough that several short requests are in flight when a long prompt lands
    num_long: int = 12
    long_prompt_len: int = 1536  # opt-125m max_model_len is 2048
    long_max_tokens: int = 8
    first_long_at_s: float = 0.4
    long_every_s: float = 0.2  # dense enough that most short requests overlap at least one long prompt
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_workload(cfg: WorkloadConfig = WorkloadConfig()) -> List[RequestSpec]:
    specs: List[RequestSpec] = []
    spacing = 1.0 / cfg.short_rate_per_s
    for i in range(cfg.num_short):
        specs.append(RequestSpec(f"short-{i:04d}", "short", i * spacing, cfg.short_prompt_len, cfg.short_max_tokens))
    for j in range(cfg.num_long):
        specs.append(RequestSpec(f"long-{j:02d}", "long", cfg.first_long_at_s + j * cfg.long_every_s,
                                 cfg.long_prompt_len, cfg.long_max_tokens))
    specs.sort(key=lambda s: (s.arrival_s, s.request_id))
    return specs


def to_workload_spec(cfg: WorkloadConfig = WorkloadConfig()) -> WorkloadSpec:
    """The same workload as ``build_workload(cfg)`` expressed as a generic ``WorkloadSpec`` (for ``llmtrace run``).
    Identical classes, arrivals and lengths (asserted by the tests); only the zero padding of long-request ids
    differs (``long-00`` here, ``long-0000`` from the spec)."""
    return WorkloadSpec(
        name="mixed_prompts", seed=cfg.seed,
        classes=[
            RequestClass(name="short", count=cfg.num_short, prompt_len=LengthSpec(kind="fixed", value=cfg.short_prompt_len),
                         max_tokens=LengthSpec(kind="fixed", value=cfg.short_max_tokens),
                         arrival=ArrivalSpec(kind="constant", rate_per_s=cfg.short_rate_per_s)),
            RequestClass(name="long", count=cfg.num_long, prompt_len=LengthSpec(kind="fixed", value=cfg.long_prompt_len),
                         max_tokens=LengthSpec(kind="fixed", value=cfg.long_max_tokens),
                         arrival=ArrivalSpec(kind="constant", rate_per_s=1.0 / cfg.long_every_s, start_s=cfg.first_long_at_s)),
        ])


def kind_of(request_id: str) -> str:
    return "long" if request_id.startswith("long-") else "short" if request_id.startswith("short-") else "other"
