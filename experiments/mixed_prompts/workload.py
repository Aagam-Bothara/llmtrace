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

import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    kind: str  # "short" | "long"
    arrival_s: float  # seconds after the run starts
    prompt_len: int
    max_tokens: int


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


def make_prompt(spec: RequestSpec, vocab_size: int = 50000, seed: int = 0) -> Dict[str, List[int]]:
    """Deterministic pseudo-random token ids (avoids the lowest ids, which are usually special tokens)."""
    rng = random.Random(f"{seed}:{spec.request_id}")
    return {"prompt_token_ids": [rng.randrange(100, vocab_size) for _ in range(spec.prompt_len)]}


def kind_of(request_id: str) -> str:
    return "long" if request_id.startswith("long-") else "short" if request_id.startswith("short-") else "other"
