# First GPU Validation Checklist

Nothing in llmtrace has run on a GPU yet. This is the plan for the first run.
Record results in this file (or an issue) with the exact commands and outputs;
do not summarise from memory.

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

## Things likely to need adjustment after the first run

* `_attach_scheduler` path if `InprocClient` attribute names differ at runtime.
* `NVMLBackend` throttle-reason constants on older `nvidia-ml-py`.
* `min_coverage_fraction` for very short requests at 50-100 ms sampling.
* The `prefill` span including the first decode step (step granularity).

## Out of scope for this phase

Renting hardware, deploying servers, AMD/ROCm, DCGM, dashboards, additional
inference frameworks, ML-based diagnosis, distributed multi-node runs.
