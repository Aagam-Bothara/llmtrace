"""Configuration-driven, deterministic workloads.

A :class:`WorkloadSpec` describes request classes (prompt-length and
output-length distributions, an arrival process, a count) and a seed, and
generates a fixed list of :class:`RequestSpec` (request id, class, arrival
offset, exact prompt length in tokens, ``max_tokens``). Generation is a pure
function of the spec: the same JSON file always yields the same requests, and
``workload_hash`` in a run manifest identifies them.

Prompts are token-id lists (``TokensPrompt``), so prompt lengths are exact on
any tokenizer and no tokenizer is needed to build a workload. Request ids are
``<class>-<index>``; the request class is recovered from the id by
:func:`class_of` (used by ``llmtrace decide`` and the experiment analysis).

Arrival processes (offsets from the class's ``start_s``):

* ``at_once``: every request at ``start_s`` (closed-loop burst);
* ``constant``: evenly spaced at ``rate_per_s``;
* ``poisson``: exponential gaps with mean ``1/rate_per_s``;
* ``gamma``: gamma-distributed gaps with shape ``burstiness`` and mean
  ``1/rate_per_s`` (``burstiness=1`` is Poisson; below 1 is burstier, above 1
  more regular; the parameterization ``vllm bench serve --burstiness`` uses);
* ``burst``: groups of ``burst_size`` requests every ``burst_every_s``.

The load generator that replays these offsets records scheduled versus actual
arrival per request in the manifest, so generator lateness is visible.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    kind: str  # request class name
    arrival_s: float  # seconds after the run starts
    prompt_len: int
    max_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LengthSpec(_Strict):
    """A distribution over token counts. ``fixed`` needs ``value``; ``uniform`` needs
    ``low``/``high`` (inclusive); ``choice`` needs ``choices`` (optional ``weights``);
    ``lognormal`` needs ``median`` and ``sigma`` (log-space) and is clipped to
    ``low``/``high`` when given. Every kind yields integers >= 1."""

    kind: Literal["fixed", "uniform", "choice", "lognormal"] = "fixed"
    value: Optional[int] = Field(default=None, ge=1)
    low: Optional[int] = Field(default=None, ge=1)
    high: Optional[int] = Field(default=None, ge=1)
    choices: Optional[List[int]] = None
    weights: Optional[List[float]] = None
    median: Optional[float] = Field(default=None, gt=0)
    sigma: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> "LengthSpec":
        if self.kind == "fixed" and self.value is None:
            raise ValueError("fixed length needs 'value'")
        if self.kind == "uniform":
            if self.low is None or self.high is None or self.low > self.high:
                raise ValueError("uniform length needs low <= high")
        if self.kind == "choice":
            if not self.choices or any(c < 1 for c in self.choices):
                raise ValueError("choice length needs non-empty 'choices' of integers >= 1")
            if self.weights is not None and (len(self.weights) != len(self.choices) or any(w < 0 for w in self.weights)
                                             or sum(self.weights) <= 0):
                raise ValueError("weights must match choices and be non-negative with a positive sum")
        if self.kind == "lognormal":
            if self.median is None or self.sigma is None:
                raise ValueError("lognormal length needs 'median' and 'sigma'")
            if self.low is not None and self.high is not None and self.low > self.high:
                raise ValueError("lognormal clip needs low <= high")
        return self

    def sample(self, rng: random.Random) -> int:
        if self.kind == "fixed":
            return int(self.value)  # type: ignore[arg-type]
        if self.kind == "uniform":
            return rng.randint(int(self.low), int(self.high))  # type: ignore[arg-type]
        if self.kind == "choice":
            return int(rng.choices(self.choices, weights=self.weights, k=1)[0])  # type: ignore[arg-type]
        v = int(round(rng.lognormvariate(_ln(float(self.median)), float(self.sigma))))  # type: ignore[arg-type]
        v = max(1, v)
        if self.low is not None:
            v = max(v, self.low)
        if self.high is not None:
            v = min(v, self.high)
        return v

    def describe(self) -> str:
        if self.kind == "fixed":
            return f"{self.value}"
        if self.kind == "uniform":
            return f"uniform[{self.low}, {self.high}]"
        if self.kind == "choice":
            return f"choice{self.choices}"
        clip = f" clipped to [{self.low}, {self.high}]" if (self.low is not None or self.high is not None) else ""
        return f"lognormal(median={self.median:g}, sigma={self.sigma:g}){clip}"


def _ln(x: float) -> float:
    import math

    return math.log(x)


class ArrivalSpec(_Strict):
    kind: Literal["at_once", "constant", "poisson", "gamma", "burst"] = "at_once"
    rate_per_s: Optional[float] = Field(default=None, gt=0, description="constant/poisson/gamma: mean arrivals per second")
    start_s: float = Field(default=0.0, ge=0, description="offset of the class's first arrival from run start")
    burstiness: float = Field(default=1.0, gt=0, description="gamma shape; 1 = Poisson")
    burst_size: Optional[int] = Field(default=None, ge=1)
    burst_every_s: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check(self) -> "ArrivalSpec":
        if self.kind in ("constant", "poisson", "gamma") and self.rate_per_s is None:
            raise ValueError(f"{self.kind} arrivals need 'rate_per_s'")
        if self.kind == "burst" and (self.burst_size is None or self.burst_every_s is None):
            raise ValueError("burst arrivals need 'burst_size' and 'burst_every_s'")
        return self

    def offsets(self, n: int, rng: random.Random) -> List[float]:
        """``n`` arrival offsets in seconds from run start, non-decreasing."""
        if n <= 0:
            return []
        if self.kind == "at_once":
            return [self.start_s] * n
        if self.kind == "constant":
            gap = 1.0 / float(self.rate_per_s)  # type: ignore[arg-type]
            return [self.start_s + i * gap for i in range(n)]
        if self.kind == "burst":
            size, every = int(self.burst_size), float(self.burst_every_s)  # type: ignore[arg-type]
            return [self.start_s + (i // size) * every for i in range(n)]
        mean_gap = 1.0 / float(self.rate_per_s)  # type: ignore[arg-type]
        out, t = [], self.start_s
        for i in range(n):
            if i > 0:
                if self.kind == "poisson":
                    t += rng.expovariate(1.0 / mean_gap)
                else:  # gamma with shape=burstiness, scale=mean_gap/burstiness (mean = mean_gap)
                    t += rng.gammavariate(self.burstiness, mean_gap / self.burstiness)
            out.append(t)
        return out

    def describe(self) -> str:
        if self.kind == "at_once":
            return f"all at {self.start_s:g}s"
        if self.kind == "burst":
            return f"bursts of {self.burst_size} every {self.burst_every_s:g}s from {self.start_s:g}s"
        extra = f", burstiness={self.burstiness:g}" if self.kind == "gamma" else ""
        return f"{self.kind} at {self.rate_per_s:g}/s from {self.start_s:g}s{extra}"


class RequestClass(_Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9_]+$", description="class name; request ids are '<name>-<index>' (no dashes)")
    count: int = Field(ge=1)
    prompt_len: LengthSpec
    max_tokens: LengthSpec
    arrival: ArrivalSpec = Field(default_factory=ArrivalSpec)


class WorkloadSpec(_Strict):
    name: str = "workload"
    description: Optional[str] = None
    seed: int = 0
    vocab_size: int = Field(default=50000, ge=200, description="prompt token ids are drawn from [min_token_id, vocab_size)")
    min_token_id: int = Field(default=100, ge=0, description="lowest token id used (skips the usual special tokens)")
    classes: List[RequestClass] = Field(min_length=1)

    @field_validator("classes")
    @classmethod
    def _unique(cls, v: List[RequestClass]) -> List[RequestClass]:
        names = [c.name for c in v]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate class names: {names}")
        return v

    @model_validator(mode="after")
    def _vocab(self) -> "WorkloadSpec":
        if self.min_token_id >= self.vocab_size:
            raise ValueError("min_token_id must be below vocab_size")
        return self

    # ---------------------------------------------------------------- generation

    def generate(self) -> List[RequestSpec]:
        """The request list: deterministic in (seed, classes), sorted by arrival then id."""
        specs: List[RequestSpec] = []
        for c in self.classes:
            r_arr = random.Random(f"{self.seed}:{c.name}:arrival")
            r_pl = random.Random(f"{self.seed}:{c.name}:prompt_len")
            r_mt = random.Random(f"{self.seed}:{c.name}:max_tokens")
            offsets = c.arrival.offsets(c.count, r_arr)
            width = max(4, len(str(c.count - 1)))
            for i in range(c.count):
                specs.append(RequestSpec(f"{c.name}-{i:0{width}d}", c.name, float(offsets[i]),
                                         c.prompt_len.sample(r_pl), c.max_tokens.sample(r_mt)))
        specs.sort(key=lambda s: (s.arrival_s, s.request_id))
        return specs

    def hash(self) -> str:
        """Identity of the replayed work: the request list plus the seed and vocabulary settings that fix the prompt token ids."""
        from llmtrace.manifest import workload_hash

        return workload_hash(self.generate() + [{"seed": self.seed, "vocab_size": self.vocab_size, "min_token_id": self.min_token_id}])

    def summary(self, specs: Optional[List[RequestSpec]] = None) -> Dict[str, Any]:
        specs = specs if specs is not None else self.generate()
        by_class: Dict[str, List[RequestSpec]] = {}
        for s in specs:
            by_class.setdefault(s.kind, []).append(s)
        classes = {}
        for c in self.classes:
            ss = by_class.get(c.name, [])
            classes[c.name] = {
                "count": len(ss),
                "prompt_len": c.prompt_len.describe(),
                "max_tokens": c.max_tokens.describe(),
                "arrival": c.arrival.describe(),
                "prompt_tokens_total": sum(s.prompt_len for s in ss),
                "max_output_tokens_total": sum(s.max_tokens for s in ss),
                "first_arrival_s": min((s.arrival_s for s in ss), default=None),
                "last_arrival_s": max((s.arrival_s for s in ss), default=None),
            }
        return {
            "name": self.name, "seed": self.seed, "requests": len(specs), "workload_hash": self.hash(),
            "span_s": (max(s.arrival_s for s in specs) if specs else 0.0),
            "classes": classes,
        }

    # ----------------------------------------------------------------------- io

    @classmethod
    def load(cls, path: str) -> "WorkloadSpec":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(exclude_none=True), indent=2) + "\n", encoding="utf-8")
        return p


def make_prompt(spec: RequestSpec, vocab_size: int = 50000, seed: int = 0, min_token_id: int = 100) -> Dict[str, List[int]]:
    """Deterministic pseudo-random token ids for ``spec`` (a vLLM ``TokensPrompt`` dict).

    Avoids the lowest ids, which are usually special tokens. Identical to what the
    mixed-prompt experiment has used on the GPU runs (same RNG seeding).
    """
    rng = random.Random(f"{seed}:{spec.request_id}")
    return {"prompt_token_ids": [rng.randrange(min_token_id, vocab_size) for _ in range(spec.prompt_len)]}


def class_of(request_id: str) -> str:
    """Request class from an id of the form ``<class>-<index>``; ids without a dash are their own class."""
    return request_id.split("-")[0]


def template() -> WorkloadSpec:
    """A steady stream of short requests with long prompts injected on a schedule (the
    mixed-prompt experiment's shape, at its GPU-run defaults)."""
    return WorkloadSpec(
        name="mixed_prompts",
        description="Short requests at a constant rate with long prompts injected every 0.2 s; "
                    "exercises chunked-prefill interference on the short-request tail.",
        seed=0,
        classes=[
            RequestClass(name="short", count=120, prompt_len=LengthSpec(kind="fixed", value=32),
                         max_tokens=LengthSpec(kind="fixed", value=128),
                         arrival=ArrivalSpec(kind="constant", rate_per_s=40.0)),
            RequestClass(name="long", count=12, prompt_len=LengthSpec(kind="fixed", value=1536),
                         max_tokens=LengthSpec(kind="fixed", value=8),
                         arrival=ArrivalSpec(kind="constant", rate_per_s=5.0, start_s=0.4)),
        ],
    )


__all__ = ["RequestSpec", "LengthSpec", "ArrivalSpec", "RequestClass", "WorkloadSpec", "make_prompt", "class_of", "template"]
