"""Helpers for driving a vLLM 0.11.0 ``LLMEngine`` directly when timing is needed.

``vllm.LLM.generate()`` forces ``SamplingParams.output_kind = FINAL_ONLY``
(``LLM._validate_and_add_requests`` in vLLM 0.11.0), so every request emits a
single output at completion and the first-token time is not observable. Under
``generate()`` llmtrace records completion, token counts and energy, but
``ttft_ms``/``tpot_ms`` are ``None`` with ``output_kind=FINAL_ONLY`` as the reason.

To measure TTFT/TPOT, drive the synchronous engine the way ``LLM._run_engine``
does but with cumulative outputs. This module does exactly that and imports
nothing from vLLM (duck-typed), so it also works with the test fakes.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any, Iterable, List, Optional, Sequence

_counter = itertools.count()


def with_cumulative_outputs(params: Any) -> Any:
    """Return a copy of ``params`` whose ``output_kind`` is CUMULATIVE (if it has one)."""
    clone = params.clone() if hasattr(params, "clone") else copy.copy(params)
    kind = getattr(clone, "output_kind", None)
    if kind is not None:
        cumulative = getattr(type(kind), "CUMULATIVE", None)
        if cumulative is not None:
            clone.output_kind = cumulative
    return clone


def run_engine_with_timing(
    engine: Any,
    prompts: Sequence[Any],
    params: Any,
    request_ids: Optional[Iterable[str]] = None,
) -> List[Any]:
    """Add ``prompts`` to a synchronous ``LLMEngine`` and step it to completion.

    Uses cumulative outputs so llmtrace can observe first-token and per-token
    timing. Returns the finished ``RequestOutput`` objects **in input order**
    (requests may complete in any order). ``engine`` is ``llm.llm_engine`` for
    a ``vllm.LLM`` instance.
    """
    ids = list(request_ids) if request_ids is not None else [f"llmtrace-{next(_counter)}" for _ in prompts]
    if len(ids) != len(prompts):
        raise ValueError("request_ids must match prompts")
    p = with_cumulative_outputs(params)
    for rid, prompt in zip(ids, prompts):
        engine.add_request(rid, prompt, p)
    finished: dict = {}
    while engine.has_unfinished_requests():
        for out in engine.step():
            if getattr(out, "finished", False):
                finished[str(out.request_id)] = out
    missing = [rid for rid in ids if rid not in finished]
    if missing:
        raise RuntimeError(f"engine finished without outputs for {missing}")
    return [finished[rid] for rid in ids]
