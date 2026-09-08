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
580.159.04 (driver supports CUDA 13.0; PyTorch 2.8.0+cu128, i.e. CUDA 12.8 runtime), Python 3.11.11, vLLM 0.11.0, transformers 4.57.6
(after the pin below). Model `facebook/opt-125m`, temperature 0.

Tested code: the working tree was uploaded to the pod uncommitted. Its
`llmtrace/`, `examples/` and `scripts/gpu_smoke_run.sh` are byte-identical to
commit `98c0cd7`; `tests/` differs from `98c0cd7` only by the pyarrow skip fix
described under "CPU suite" below, made after the run. `pyproject.toml` gained
the transformers pin after the run started. The overhead script was run as
`/root/overhead.py`, a copy of which is in the evidence directory.

All raw artifacts (logs, request/batch/GPU trace files, the nvidia-smi log,
overhead JSON, cross-check scripts) are committed under
`docs/gpu_runs/2026-09-08-rtx-a5000/`. Every number below is computed from
those files; the cross-check table can be recomputed with
`python docs/gpu_runs/2026-09-08-rtx-a5000/long/crosscheck_bracketed.py docs/gpu_runs/2026-09-08-rtx-a5000/long`.

Executed via `scripts/gpu_smoke_run.sh`, then a longer workload
(64 prompts x 256 tokens) with `scripts/gpu_overhead.py` and an independent
`nvidia-smi --query-gpu=power.draw -lms 50` log.

| Step | Result |
|------|--------|
| CPU suite on the GPU box | 83 passed, `tests/test_collection.py` skipped as a whole (`smoke/cpu_tests.log`). Cause: a class-level `pytest.importorskip("pyarrow")` skipped the entire module on machines without pyarrow, so the sampler/writer/tracer tests did not run on the pod. Fixed after the run (per-test `find_spec` skip); the full suite is 106 tests, of which 105 run without pyarrow. The GPU box has not been re-run since. |
| Smoke, multiprocess engine core (default) | ALL CHECKS PASSED: `SyncMPClient`, scheduler reported unreachable with the documented reason, no batches, membership `request_window` |
| Smoke, in-process engine core (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) | ALL CHECKS PASSED: `InprocClient`, scheduler found at `engine.engine_core.engine_core.scheduler`, 32 batches for 8 x 32-token requests, every trace linked to batches, membership `batch_metadata`, queue + prefill == TTFT |
| Phase A (`LLM.generate()`) | `output_kind=final_only`, TTFT/TPOT unavailable with the FINAL_ONLY reason, token counts equal to the engine's, text identical to untraced |
| Phase B (raw engine loop, cumulative) | TTFT and TPOT measured for all requests, `tokens_at_first_observation == 1`, text identical to `generate()` |
| Restoration | `step`/`add_request` back to originals after every `stop()`; no leaked requests; no instrumentation errors; no dropped writes |
| NVML | all fields populated (power, limit, util, memory, clocks, temperature, throttle `none`); sampler interval median 48.6 ms at a 50 ms setting |
| Real batch classification | first step `num_prefill=8, num_decode=0`; all later steps decode-only; `kv_cache_usage_fraction` populated |
| Energy ledger | conservation error <= 2e-13 J on every run |

Energy consistency check (64 x 256 tokens, in-process). This compares
llmtrace's NVML sampling (50 ms) against a *separately collected* telemetry
stream, `nvidia-smi --query-gpu=power.draw -lms 50`, which reads the same NVML
power sensor. It checks llmtrace's sampling and integration, not the accuracy
of the sensor; it is not an independent energy measurement. Both streams are
integrated with the same trapezoid (`CumulativePower`, edge interpolation)
over identical bracketed boundaries: from the latest of (request window start,
first sample of either stream) to the earliest of (request window end, last
sample of either stream). Script: `long/crosscheck_bracketed.py`.

| Run | shared window | llmtrace J | nvidia-smi J | diff |
|-----|---------------|------------|--------------|------|
| generate 0 | 0.788 s | 149.29 | 149.51 | -0.15% |
| generate 1 | 0.800 s | 162.67 | 162.62 | +0.03% |
| generate 2 | 0.800 s | 164.97 | 165.00 | -0.02% |
| engine loop 0 | 0.850 s | 177.79 | 178.00 | -0.12% |
| engine loop 1 | 0.851 s | 177.15 | 177.17 | -0.01% |
| engine loop 2 | 0.850 s | 176.55 | 176.60 | -0.03% |

An earlier version of this table (`long/crosscheck.py`, kept for the record)
integrated the nvidia-smi stream only between samples strictly inside the
request window while llmtrace interpolated to the window edges, and reported
up to 6% differences; that gap was the unequal boundaries, not sampling.
Three of 64 requests per run fell below the per-request coverage threshold and
are reported as unavailable rather than given a figure.

Overhead, small-model benchmark (`scripts/gpu_overhead.py` as run, 5
interleaved repeats, medians). The model is `facebook/opt-125m`, whose engine
steps take about 3 ms here; how the ratio changes with larger models was not
measured. Engine steps were not recorded by the overhead script as run; the
count comes from the batch records of the identical workload in
`long/traces/run*_engine` (256 scheduled steps in every run, because 57 of 64
requests reached `max_tokens=256`; 7 stopped earlier at EOS, minimum 24
tokens). The untraced runs are assumed to take the same 256 steps (same
prompts, temperature 0, identical outputs verified by the smoke test). The
script now records `steps_observed` directly.

| Configuration | untraced | traced | ratio | per step (256 steps) |
|---------------|----------|--------|-------|----------------------|
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
