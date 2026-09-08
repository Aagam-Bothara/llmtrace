"""Deterministic fakes mirroring the vLLM 0.11.0 V1 ``LLMEngine`` surface that llmtrace uses.

These mimic the *shapes* verified from the vLLM 0.11.0 source (see
``llmtrace/data_plane/vllm_instrumentation.py``); they do not prove real vLLM
compatibility. Timing is driven by an injectable fake clock so tests are exact.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class RequestOutputKind(enum.Enum):
    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2


@dataclass
class SamplingParams:
    max_tokens: int = 16
    n: int = 1
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE


@dataclass
class PoolingParams:
    task: str = "embed"


@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: List[int]
    cumulative_logprob: Optional[float] = None
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None
    stop_reason: Any = None


@dataclass
class RequestOutput:
    request_id: str
    prompt: Optional[str]
    prompt_token_ids: Optional[List[int]]
    prompt_logprobs: Any
    outputs: List[CompletionOutput]
    finished: bool
    metrics: Any = None  # V1 never populates this
    num_cached_tokens: Optional[int] = None


@dataclass
class PoolingOutput:
    data: Any


@dataclass
class PoolingRequestOutput:
    request_id: str
    outputs: PoolingOutput
    prompt_token_ids: List[int]
    finished: bool


@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: Optional[List[int]]
    num_computed_tokens: int


@dataclass
class CachedRequestData:
    req_ids: List[str] = field(default_factory=list)
    num_computed_tokens: List[int] = field(default_factory=list)


@dataclass
class SchedulerOutput:
    scheduled_new_reqs: List[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: Dict[str, int]
    total_num_scheduled_tokens: int
    finished_req_ids: set


class FakeClock:
    """Monotonic + wall clocks that only advance when told to."""

    def __init__(self, start_monotonic: float = 1000.0, start_wall: float = 1_700_000_000.0):
        self.mono = start_monotonic
        self.wall = start_wall

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return self.wall

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.wall += seconds


@dataclass
class _Req:
    """Mirrors vllm.v1.request.Request: num_computed_tokens counts every token whose KV
    has been scheduled for computation (prompt and generated), like the real scheduler."""

    request_id: str
    prompt_token_ids: List[int]
    params: Any
    num_prompt_tokens: int
    num_computed_tokens: int = 0
    generated: List[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: Optional[str] = None
    emitted: int = 0  # tokens already sent in DELTA mode
    arrival_step: int = 0
    scheduled_once: bool = False

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.generated)


class FakeKVCacheManager:
    def __init__(self) -> None:
        self.usage = 0.0


class FakeScheduler:
    """Shapes: schedule() -> SchedulerOutput; .requests dict; .running/.waiting; .kv_cache_manager.usage."""

    def __init__(self, engine: "FakeLLMEngine") -> None:
        self.engine = engine
        self.requests: Dict[str, _Req] = engine._requests
        self.running: List[_Req] = []
        self.waiting: List[_Req] = []
        self.kv_cache_manager = FakeKVCacheManager()

    def schedule(self) -> SchedulerOutput:
        out = self.engine._do_schedule()
        self.running = [r for r in self.requests.values() if not r.finished]
        self.waiting = []
        self.kv_cache_manager.usage = min(1.0, 0.1 * len(self.running))
        return out


class FakeInprocClient:
    def __init__(self, engine: "FakeLLMEngine") -> None:
        self.engine_core = _FakeEngineCore(engine)


class FakeModelExecutor:
    """Shape of EngineCore.model_executor.execute_model(scheduler_output) (vLLM 0.11.0 UniProcExecutor)."""

    def __init__(self, engine: "FakeLLMEngine") -> None:
        self.engine = engine
        self.calls = 0

    def execute_model(self, scheduler_output: Any) -> Any:
        self.calls += 1
        self.engine._execute(scheduler_output)
        return object()  # ModelRunnerOutput stand-in


class _FakeEngineCore:
    def __init__(self, engine: "FakeLLMEngine") -> None:
        self.scheduler = FakeScheduler(engine)
        self.model_executor = FakeModelExecutor(engine)


class FakeSyncMPClient:
    """Stands in for SyncMPClient: no scheduler reachable."""


class FakeModelConfig:
    model = "fake/model-v1"


class FakeLLMEngine:
    """Synchronous V1-style engine.

    Each ``step()`` advances every unfinished request: prefill consumes up to
    ``prefill_chunk`` prompt tokens per step; once prefill is complete the step
    produces ``tokens_per_step`` output tokens (all in one step, e.g. speculative
    decoding, when ``tokens_per_step > 1``). Outputs are emitted per
    ``SamplingParams.output_kind``.
    """

    def __init__(
        self,
        clock: Optional[FakeClock] = None,
        in_process_scheduler: bool = True,
        prefill_chunk: int = 10_000,
        tokens_per_step: int = 1,
        step_seconds: float = 0.01,
        fail_step_at: Optional[int] = None,
        step_sleep_s: float = 0.0,
        step_seconds_per_token: float = 0.0,
        max_num_batched_tokens: Optional[int] = None,
        long_prefill_token_threshold: int = 0,
    ) -> None:
        self.clock = clock or FakeClock()
        self.prefill_chunk = prefill_chunk
        self.tokens_per_step = tokens_per_step
        self.step_seconds = step_seconds  # base fake-clock time per step
        self.step_seconds_per_token = step_seconds_per_token  # extra fake-clock time per scheduled token
        self.step_sleep_s = step_sleep_s  # real time.sleep inside step() (for real-clock demos)
        # Scheduler knobs mirroring vLLM 0.11.0 SchedulerConfig semantics (token budget per step;
        # per-step cap on prefill tokens for prompts longer than the threshold; 0 = off).
        self.max_num_batched_tokens = max_num_batched_tokens
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.last_scheduled_tokens = 0
        self.fail_step_at = fail_step_at
        self.model_config = FakeModelConfig()
        self._requests: Dict[str, _Req] = {}
        self._steps = 0
        self.engine_core: Any = FakeInprocClient(self) if in_process_scheduler else FakeSyncMPClient()
        self.calls: List[str] = []
        self.vllm_config = object()
        self.logger_manager: Any = FakeStatLoggerManager()  # like LLMEngine with log_stats=True

    # --- vLLM 0.11.0 LLMEngine surface ---------------------------------------

    def add_request(self, request_id: str, prompt: Any, params: Any, arrival_time: Optional[float] = None,
                    lora_request: Any = None, tokenization_kwargs: Any = None, trace_headers: Any = None,
                    priority: int = 0) -> None:
        self.calls.append(f"add_request:{request_id}")
        if isinstance(prompt, dict) and "prompt_token_ids" in prompt:
            ids = list(prompt["prompt_token_ids"])
        elif isinstance(prompt, str):
            ids = [hash(w) % 1000 for w in prompt.split()]  # the "tokenizer"
        else:
            raise TypeError("unsupported prompt")
        self._requests[request_id] = _Req(request_id, ids, params, len(ids), arrival_step=self._steps)

    def abort_request(self, request_ids: List[str]) -> None:
        self.calls.append(f"abort:{','.join(request_ids)}")
        for rid in request_ids:
            req = self._requests.get(rid)
            if req is not None and not req.finished:
                req.finished = True
                req.finish_reason = "abort"

    def has_unfinished_requests(self) -> bool:
        return any(not r.finished for r in self._requests.values())

    def step(self) -> List[Any]:
        self._steps += 1
        self.calls.append(f"step:{self._steps}")
        if self.fail_step_at is not None and self._steps == self.fail_step_at:
            raise RuntimeError("engine failure injected")
        sched = self._scheduler()
        if sched is not None:
            out = sched.schedule()
            self.engine_core.engine_core.model_executor.execute_model(out)  # in-process: executor reachable, like EngineCore.step()
        else:
            out = self._do_schedule()  # same bookkeeping, just not reachable from outside
            self._execute(out)
        self.clock.advance(self.step_seconds + self.step_seconds_per_token * self.last_scheduled_tokens)
        # LLMEngine.step() records stats after processing outputs; emulate the important fields.
        if self.logger_manager is not None:
            running = [r for r in self._requests.values() if not r.finished]
            self.logger_manager.record(
                FakeSchedulerStats(num_running_reqs=len(running), kv_cache_usage=min(1.0, 0.1 * len(running))),
                FakeIterationStats(num_generation_tokens=len(running), num_prompt_tokens=0),
            )
        if self.step_sleep_s:
            import time

            time.sleep(self.step_sleep_s)
        outputs: List[Any] = []
        for req in self._order():
            if req.finished:
                continue
            if req.num_computed_tokens < req.num_prompt_tokens:
                continue  # chunked prefill still running: no output this step
            if not req.scheduled_once or req.num_computed_tokens < req.num_tokens:
                continue  # not scheduled in this step (token budget exhausted)
            if isinstance(req.params, PoolingParams):
                req.finished = True
                outputs.append(PoolingRequestOutput(req.request_id, PoolingOutput([0.0]), list(req.prompt_token_ids), True))
                continue
            max_tokens = req.params.max_tokens
            n_new = min(self.tokens_per_step, max_tokens - len(req.generated))
            req.generated.extend(range(len(req.generated), len(req.generated) + n_new))
            if len(req.generated) >= max_tokens:
                req.finished = True
                req.finish_reason = "length"
            out = self._make_output(req)
            if out is not None:
                outputs.append(out)
        return outputs

    # --- helpers --------------------------------------------------------------

    def _execute(self, scheduler_output: Any) -> None:
        """The 'GPU work' of a step: nothing to do in the fake; the clock advance happens in step()."""

    def _do_schedule(self) -> SchedulerOutput:
        """Real vLLM 0.11.0 order: build SchedulerOutput, then _update_after_schedule()
        advances num_computed_tokens by the scheduled tokens *before* schedule() returns."""
        new, cached, tokens = [], CachedRequestData(), {}
        budget = self.max_num_batched_tokens if self.max_num_batched_tokens is not None else 10**9
        # Real order: RUNNING requests first (already scheduled once), then WAITING ones.
        ordered = [r for r in self._order() if r.scheduled_once] + [r for r in self._order() if not r.scheduled_once]
        for req in ordered:
            if req.finished or budget <= 0:
                continue
            remaining = req.num_tokens - req.num_computed_tokens  # decode: exactly the new token(s)
            n = min(remaining, self.prefill_chunk) if req.num_computed_tokens < req.num_prompt_tokens else remaining
            if 0 < self.long_prefill_token_threshold < n:
                n = self.long_prefill_token_threshold
            n = min(n, budget)
            budget -= n
            if not req.scheduled_once:
                new.append(NewRequestData(req.request_id, list(req.prompt_token_ids), req.num_computed_tokens))
                req.scheduled_once = True
            else:
                cached.req_ids.append(req.request_id)
                cached.num_computed_tokens.append(req.num_computed_tokens)
            tokens[req.request_id] = n
        out = SchedulerOutput(new, cached, tokens, sum(tokens.values()), set())
        self.last_scheduled_tokens = out.total_num_scheduled_tokens
        for rid, n in tokens.items():  # _update_after_schedule()
            self._requests[rid].num_computed_tokens += n
        return out

    def _scheduler(self) -> Optional[FakeScheduler]:
        core = getattr(self.engine_core, "engine_core", None)
        return getattr(core, "scheduler", None)

    def _order(self) -> List[_Req]:
        return list(self._requests.values())

    def _make_output(self, req: _Req) -> Optional[RequestOutput]:
        kind = req.params.output_kind
        if kind is RequestOutputKind.FINAL_ONLY and not req.finished:
            return None
        if kind is RequestOutputKind.DELTA:
            token_ids = req.generated[req.emitted:]
            req.emitted = len(req.generated)
        else:
            token_ids = list(req.generated)
        comp = CompletionOutput(0, "", token_ids, finish_reason=req.finish_reason if req.finished else None)
        return RequestOutput(req.request_id, None, list(req.prompt_token_ids), None, [comp], req.finished)


def run_to_completion(engine: FakeLLMEngine, max_steps: int = 10_000) -> List[Any]:
    """Mirror LLM._run_engine: step until no unfinished requests, collect finished outputs."""
    finished = []
    steps = 0
    while engine.has_unfinished_requests():
        steps += 1
        if steps > max_steps:
            raise RuntimeError("fake engine did not finish")
        for out in engine.step():
            if out.finished:
                finished.append(out)
    return finished


@dataclass
class FakePrefixCacheStats:
    reset: bool = False
    requests: int = 0
    queries: int = 0
    hits: int = 0


@dataclass
class FakeSchedulerStats:
    """Shape of vllm.v1.metrics.stats.SchedulerStats (0.11.0)."""

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    kv_cache_usage: float = 0.0
    prefix_cache_stats: FakePrefixCacheStats = field(default_factory=FakePrefixCacheStats)
    num_corrupted_reqs: int = 0


@dataclass
class FakeFinishedRequestStats:
    finish_reason: str = "stop"
    e2e_latency: float = 0.0
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    max_tokens_param: Optional[int] = None
    queued_time: float = 0.0
    prefill_time: float = 0.0
    inference_time: float = 0.0
    decode_time: float = 0.0
    mean_time_per_output_token: float = 0.0


@dataclass
class FakeIterationStats:
    """Shape of vllm.v1.metrics.stats.IterationStats (0.11.0)."""

    num_generation_tokens: int = 0
    num_prompt_tokens: int = 0
    num_preempted_reqs: int = 0
    finished_requests: List[FakeFinishedRequestStats] = field(default_factory=list)
    time_to_first_tokens_iter: List[float] = field(default_factory=list)
    inter_token_latencies_iter: List[float] = field(default_factory=list)


class FakeStatLoggerManager:
    """Shape of vllm.v1.metrics.loggers.StatLoggerManager: per_engine_logger_dict + record() loop."""

    def __init__(self, factories=(), vllm_config=None) -> None:
        self.per_engine_logger_dict: Dict[int, list] = {0: [f(vllm_config, 0) for f in factories]}

    def record(self, scheduler_stats, iteration_stats, engine_idx: int = 0) -> None:
        for lg in self.per_engine_logger_dict[engine_idx]:
            lg.record(scheduler_stats, iteration_stats, engine_idx)


class FakeAsyncLLM:
    """Shape of vllm.v1.engine.async_llm.AsyncLLM (0.11.0): generate() is an async generator that
    aborts on CancelledError/GeneratorExit; abort() is a coroutine; logger_manager exists."""

    def __init__(self, clock: Optional[FakeClock] = None, step_seconds: float = 0.01, fail_at_token: Optional[int] = None) -> None:
        self.clock = clock or FakeClock()
        self.step_seconds = step_seconds
        self.fail_at_token = fail_at_token
        self.model_config = FakeModelConfig()
        self.vllm_config = object()
        self.logger_manager: Any = FakeStatLoggerManager()
        self.aborted: List[str] = []
        self.calls: List[str] = []

    async def generate(self, prompt: Any, sampling_params: Any, request_id: str, lora_request: Any = None,
                       trace_headers: Any = None, priority: int = 0, data_parallel_rank: Any = None):
        import asyncio

        self.calls.append(f"generate:{request_id}")
        ids = list(prompt["prompt_token_ids"]) if isinstance(prompt, dict) else [hash(w) % 1000 for w in str(prompt).split()]
        generated: List[int] = []
        try:
            while len(generated) < sampling_params.max_tokens:
                await asyncio.sleep(0)
                self.clock.advance(self.step_seconds)
                generated.append(len(generated))
                if self.fail_at_token is not None and len(generated) == self.fail_at_token:
                    raise RuntimeError("engine generate error injected")
                finished = len(generated) >= sampling_params.max_tokens
                kind = sampling_params.output_kind
                toks = generated[-1:] if kind is RequestOutputKind.DELTA else list(generated)
                if kind is RequestOutputKind.FINAL_ONLY and not finished:
                    continue
                yield RequestOutput(request_id, None, ids, None,
                                    [CompletionOutput(0, "", toks, finish_reason="length" if finished else None)], finished)
        except (asyncio.CancelledError, GeneratorExit):
            await self.abort(request_id)
            raise

    async def abort(self, request_id: Any) -> None:
        self.calls.append(f"abort:{request_id}")
        self.aborted.append(str(request_id))


class FakeNVMLBackend:
    """SamplerBackend fake: constant or scripted power per GPU."""

    def __init__(self, gpus: Dict[int, float], fail_open: bool = False, read_error_on: Optional[int] = None):
        self.gpus = gpus
        self.fail_open = fail_open
        self.read_error_on = read_error_on
        self.opened = False
        self.closed = False
        self.reads = 0

    def open(self, gpu_ids: Optional[List[int]]) -> List[Dict[str, Any]]:
        if self.fail_open:
            raise RuntimeError("NVML init failed (fake)")
        self.opened = True
        wanted = gpu_ids if gpu_ids else sorted(self.gpus)
        return [{"gpu_id": g, "name": f"Fake GPU {g}"} for g in wanted if g in self.gpus]

    def read(self, gpu_id: int) -> Dict[str, Any]:
        self.reads += 1
        if self.read_error_on == gpu_id:
            raise RuntimeError("read failed (fake)")
        return {
            "gpu_utilization_pct": 90.0,
            "memory_utilization_pct": 40.0,
            "memory_used_mb": 8000.0,
            "memory_total_mb": 16000.0,
            "power_draw_watts": self.gpus[gpu_id],
            "power_limit_watts": 300.0,
            "temperature_c": 60.0,
            "sm_clock_mhz": 1500,
            "memory_clock_mhz": 5000,
            "throttle_bits": 0,
            "throttle_map": {},
        }

    def close(self) -> None:
        self.closed = True
