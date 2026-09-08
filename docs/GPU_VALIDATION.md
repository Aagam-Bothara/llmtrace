# GPU Validation

The first GPU run happened on 2026-09-08 (results below). The checklist that
follows is the procedure for repeating it; record results with exact commands
and outputs, never from memory.

## Environment

```bash
# Linux, NVIDIA driver with NVML, CUDA-capable GPU, Python 3.10-3.12
python -m venv .venv && source .venv/bin/activate
pip install -e ".[vllm,dev]"          # vllm==0.11.0, nvidia-ml-py, pytest
python -c "import vllm; print(vllm.__version__)"   # must print 0.11.0
nvidia-smi
python -m pytest                       # CPU suite must pass on the GPU box too
```

Record: GPU model, driver version, `nvidia-smi` output, vLLM version,
Python version, `VLLM_ENABLE_V1_MULTIPROCESSING` value.

## Run A: default engine (multiprocess engine core)

```bash
unset VLLM_ENABLE_V1_MULTIPROCESSING
python examples/vllm_smoke_test.py --model facebook/opt-125m --out ./traces_A
```

Expected: scheduler not visible (reason names `VLLM_ENABLE_V1_MULTIPROCESSING=0`),
no `batches_*` file, spans are `time_to_first_token` + `decode`.

## Run B: in-process engine core

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
python examples/vllm_smoke_test.py --model facebook/opt-125m --out ./traces_B
```

Expected: `scheduler_visible=true`, batches recorded, `queue` and `prefill`
spans, every trace has `batch_ids`, ledger `membership_source=batch_metadata`.

## Checks (both runs)

Telemetry availability
- [ ] `health()["gpu_sampler"]["available"]` is true and `unavailable_reason` is null
- [ ] `samples_taken > 0` and `gpu_*.jsonl` exists; `dropped == 0`, `read_errors == 0`
- [ ] samples have non-null `power_draw_watts`; note any null fields per GPU
- [ ] sample interval observed in the file is close to the configured 50 ms (report median and max gap)

Request completion
- [ ] number of traces == number of prompts; all `status == completed`
- [ ] `health()["instrumentation"]["active_requests"] == 0` after `stop()`
- [ ] `instrumentation_errors == 0`, `last_error == null`
- [ ] `engine.__dict__` has no `step` / `add_request` / `abort_request` after `stop()`
- [ ] a second `llm.generate()` after `stop()` works normally (engine restored)
- [ ] abort path: add a long request via the raw engine, call `engine.abort_request([...])` while traced, confirm `status == aborted`

Timing checks (the smoke test runs two phases: A = `LLM.generate()`, which forces FINAL_ONLY outputs; B = raw engine loop with CUMULATIVE outputs via `run_engine_with_timing`)
- [ ] Phase A: `output_kind == final_only`, `ttft_ms`/`tpot_ms` are null with a FINAL_ONLY reason; token counts still match
- [ ] Phase B: generated text identical to phase A (temperature 0)
- [ ] `output_length` equals the engine's token count per request; `prompt_length` equals `len(prompt_token_ids)` and `prompt_length_source == engine_prompt_token_ids`
- [ ] Phase B: `ttft_ms` > 0 for every completed request with output; `tpot_ms` > 0 where `output_length >= 2`
- [ ] Run B only: each batch's `num_prefill`/`num_decode` matches expectations (first batch all prefill; with chunked prefill on, long prompts stay prefill for several steps)
- [ ] `total_duration_ms` of each request <= `generate()` wall time
- [ ] p50 TPOT is plausible for the model on that GPU (compare with vLLM's own logged stats if `--disable-log-stats` is off)
- [ ] Run B only: queue + prefill == TTFT for each request (to floating point)
- [ ] `tokens_at_first_observation` is 1 without speculative decoding

Energy checks
- [ ] ledger `conservation_error_joules < 1e-6`
- [ ] `device_joules` roughly equals mean power × run window from `nvidia-smi --query-gpu=power.draw --format=csv -lms 100` sampled in parallel (order of magnitude; write down both numbers)
- [ ] every request has either an allocation or an `unavailable_reason`; count each
- [ ] idle energy is non-zero if there were gaps between requests, zero otherwise

Tracing enabled vs disabled
- [ ] run `--no-trace --repeat 5` and `--repeat 5`; record per-run `generate()` wall times for both
- [ ] report median traced/untraced ratio; do not claim an overhead figure before this exists
- [ ] outputs (generated text) are identical between traced and untraced runs at temperature 0

Failure surfacing
- [ ] with `strict_instrumentation=True`, inject a fault (e.g. monkeypatch `_on_step_completed`) and confirm the exception surfaces after the engine call
- [ ] with `require_gpu=True` and NVML blocked, confirm `instrument_engine()` raises **and** the engine is restored (`"step" not in engine.__dict__`), no `llmtrace-*` threads remain

## Results: first GPU run (2026-09-08)

Environment: RunPod Secure Cloud, 1x NVIDIA RTX A5000 (24 GB), driver
580.159.04, CUDA 13.0 runtime, Python 3.11.11, vLLM 0.11.0, transformers 4.57.6
(after the pin below), llmtrace working tree at this commit. Model
`facebook/opt-125m`, temperature 0. Raw logs and trace files are in the run
artifacts (`gpu_smoke_results/`, `long_results/`); numbers below are copied
from them.

Executed via `scripts/gpu_smoke_run.sh`, then a longer workload
(64 prompts x 256 tokens) with `scripts/gpu_overhead.py` and an independent
`nvidia-smi --query-gpu=power.draw -lms 50` log.

| Step | Result |
|------|--------|
| CPU suite on the GPU box | 106 passed |
| Smoke, multiprocess engine core (default) | ALL CHECKS PASSED: `SyncMPClient`, scheduler reported unreachable with the documented reason, no batches, membership `request_window` |
| Smoke, in-process engine core (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) | ALL CHECKS PASSED: `InprocClient`, scheduler found at `engine.engine_core.engine_core.scheduler`, 32 batches for 8 x 32-token requests, every trace linked to batches, membership `batch_metadata`, queue + prefill == TTFT |
| Phase A (`LLM.generate()`) | `output_kind=final_only`, TTFT/TPOT unavailable with the FINAL_ONLY reason, token counts equal to the engine's, text identical to untraced |
| Phase B (raw engine loop, cumulative) | TTFT and TPOT measured for all requests, `tokens_at_first_observation == 1`, text identical to `generate()` |
| Restoration | `step`/`add_request` back to originals after every `stop()`; no leaked requests; no instrumentation errors; no dropped writes |
| NVML | all fields populated (power, limit, util, memory, clocks, temperature, throttle `none`); sampler interval median 48.6 ms at a 50 ms setting |
| Real batch classification | first step `num_prefill=8, num_decode=0`; all later steps decode-only; `kv_cache_usage_fraction` populated |
| Energy ledger | conservation error <= 2e-13 J on every run |

Energy cross-check (64 x 256 tokens, in-process; llmtrace 50 ms sampler vs
independent `nvidia-smi` 50 ms log, both integrated over the same wall-clock
request window):

| Run | window | llmtrace device J | nvidia-smi J |
|-----|--------|-------------------|--------------|
| generate 0 | 0.802 s | 149.29 | 143.49 |
| generate 1 | 0.805 s | 162.67 | 152.68 |
| generate 2 | 0.824 s | 164.97 | 165.38 |
| engine loop 0 | 0.877 s | 177.79 | 167.96 |
| engine loop 1 | 0.876 s | 177.15 | 177.30 |
| engine loop 2 | 0.881 s | 176.55 | 177.05 |

Agreement is within 6% with 16 to 18 samples per window on each side, i.e.
within one sample interval of edge effect. Three of 64 requests per run had
too little coverage for a per-request figure and are reported as such.

Overhead (`scripts/gpu_overhead.py`, 5 interleaved repeats, medians of
`generate()`-equivalent wall time; opt-125m steps are ~3 ms, so this is a
worst-case relative figure for a tiny model, not a general one):

| Configuration | untraced | traced | ratio | per engine step |
|---------------|----------|--------|-------|-----------------|
| `LLM.generate()` (FINAL_ONLY) | 0.7686 s | 0.8012 s | 1.042 | +0.13 ms |
| engine loop (CUMULATIVE) | 0.7966 s | 0.8708 s | 1.093 | +0.29 ms |

Not yet exercised on hardware: larger models, chunked prefill across steps,
preemption, speculative decoding, `n > 1`, multi-GPU, abort under load,
`require_gpu` failure path on real NVML.

## Findings from GPU runs

* 2026-09-08, RunPod RTX A5000 (driver 580.159.04), Python 3.11, vLLM 0.11.0:
  `pip install vllm==0.11.0` resolved `transformers` to 5.16.1, and every
  `LLM(...)` construction failed with
  `AttributeError: GPT2Tokenizer has no attribute all_special_tokens_extended`
  (in `vllm/transformers_utils/tokenizer.py`). Not an llmtrace bug; the `vllm`
  extra now pins `transformers>=4.56,<5`.

## Things likely to need adjustment after the first run

* `_attach_scheduler` path if `InprocClient` attribute names differ at runtime.
* `NVMLBackend` throttle-reason constants on older `nvidia-ml-py`.
* `min_coverage_fraction` for very short requests at 50-100 ms sampling.
* The `prefill` span including the first decode step (step granularity).

## Out of scope for this phase

Renting hardware, deploying servers, AMD/ROCm, DCGM, dashboards, additional
inference frameworks, ML-based diagnosis, distributed multi-node runs.
