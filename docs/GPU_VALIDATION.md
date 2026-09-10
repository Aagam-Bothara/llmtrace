# GPU validation

This document records the GPU experiments and how to repeat the checks.
Use [status](STATUS.md) for a short overview or jump to the
[repeated A100 runs](#validation-set-from-a-clean-commit-with-independent-repeats-2026-09-09-a100-80gb-qwen25-7b-session-4)
for the queue and KV-cache results.

Scripts, manifests and reports are kept in each session directory. Download
raw traces and verify their checksums using the
[evidence guide](gpu_runs/README.md).

The results below describe the code and environment used in each session.
They are historical measurements, not a new validation of later changes.
In particular, the latest energy-selection and writer-shutdown fixes were
CPU-tested without rerunning GPU inference. Current analysis may withhold
older results that lack required evidence or an explicit GPU selection.

## Repeat the checks

Save the exact commands, environment and output for each new session.
The smoke commands below suit a host with one NVML-visible GPU. On a
multi-GPU host, configure `gpu_sampler.gpu_ids` with the engine's physical
NVML indices, or use the generic runner with repeatable `--gpu-id` flags.

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

### Telemetry availability

- [ ] `health()["gpu_sampler"]["available"]` is true and `unavailable_reason` is null
- [ ] `samples_taken > 0` and `gpu_*.jsonl` exists; `dropped == 0`, `read_errors == 0`
- [ ] samples have non-null `power_draw_watts`; note any null fields per GPU
- [ ] sample interval observed in the file is close to the configured 50 ms (report median and max gap)

### Request completion

- [ ] number of traces == number of prompts; all `status == completed`
- [ ] `health()["instrumentation"]["active_requests"] == 0` after `stop()`
- [ ] `instrumentation_errors == 0`, `last_error == null`
- [ ] `engine.__dict__` has no `step` / `add_request` / `abort_request` after `stop()`
- [ ] a second `llm.generate()` after `stop()` works normally (engine restored)
- [ ] abort path: add a long request via the raw engine, call `engine.abort_request([...])` while traced, confirm `status == aborted`

### Timing

Phase A uses final-only `LLM.generate()`; phase B uses cumulative outputs through `run_engine_with_timing`.

- [ ] Phase A: `output_kind == final_only`, `ttft_ms`/`tpot_ms` are null with a FINAL_ONLY reason; token counts still match
- [ ] Phase B: generated text identical to phase A (temperature 0)
- [ ] `output_length` equals the engine's token count per request; `prompt_length` equals `len(prompt_token_ids)` and `prompt_length_source == engine_prompt_token_ids`
- [ ] Phase B: `ttft_ms` > 0 for every completed request with output; `tpot_ms` > 0 where `output_length >= 2`
- [ ] Run B only: each batch's `num_prefill`/`num_decode` matches expectations (first batch all prefill; with chunked prefill on, long prompts stay prefill for several steps)
- [ ] `total_duration_ms` of each request <= `generate()` wall time
- [ ] p50 TPOT is plausible for the model on that GPU (compare with vLLM's own logged stats if `--disable-log-stats` is off)
- [ ] Run B only: queue + prefill == TTFT for each request (to floating point)
- [ ] `tokens_at_first_observation` is 1 without speculative decoding

### GPU step spans (CUDA events; in-process run only)

- [x] `health()["executor_visible_during_run"]` is true and `gpu_steps_*.jsonl` has one record per step with `gpu_span_ms` (RTX 4000 Ada run)
- [x] `gpu_span_ms <= host_step_ms` for every step, and `cuda_timing.dropped == 0`, `errors == 0`
- [x] median host share per step recorded: 9 to 13% on opt-125m (the CUDA span still includes launch gaps); the `host_overhead` finding reports not supported (the expectation of a large host share was wrong)
- [x] with `enable_nvtx=True` under `nsys profile`, llmtrace step ranges appear next to the kernels, and every range matched a `gpu_steps` record; `gpu_span_ms >= ` Nsight busy time on every step (RTX A5000 run, `scripts/nsys_step_compare.py`)
- [x] traced-vs-untraced wall time with `gpu_step_timing` on vs off (event recording cost): 0.11 ms per step, +3.0% (RTX A5000 session 2)
- [x] `llmtrace run --workload w.json --engine vllm` (the generic runner) reproduces `experiments/mixed_prompts/run.py` on the same GPU: same effective config, work-identical hash, verdicts agree (RTX A5000 session 2)
- [x] `llmtrace plan` -> `llmtrace run --plan` -> `llmtrace decide` on real vLLM, one process per engine (RTX A5000 session 2)
- [x] vLLM per-step stats through the post-hoc attach on the sync engine (`disable_log_stats=False`; RTX A5000 session 2)

### AsyncLLM (`examples/vllm_async_smoke_test.py`)

- [x] concurrent `generate()` streams traced with TTFT, token counts equal to the consumer's, status completed (RTX 4000 Ada run)
- [x] a stream the client stops reading is recorded `aborted` with `abort_cause=client_cancelled` and a partial token count, and `AsyncLLM.abort` was called
- [x] `generate`/`abort` restored after `stop()`; scheduler and executor reported unavailable with the AsyncLLM reason; `vllm_stats_*.jsonl` present

### vLLM stats (stat_loggers hook)

- [x] `health()["vllm_stats"]["unavailable_reason"]` is null and `vllm_stats_*.jsonl` exists with one record per engine step (AsyncLLM run; sync-engine attachment was checked later in session 2)
- [ ] `kv_cache_usage`, `num_running_reqs`, `num_waiting_reqs` populated; `num_preempted_reqs` is 0 in the smoke run
- [ ] vLLM's own TTFT samples (`time_to_first_tokens_s`) agree with llmtrace `ttft_ms` within a step for the raw-engine phase
- [ ] the logger is gone from `engine.logger_manager.per_engine_logger_dict[0]` after `stop()`

### Energy checks

- [ ] Record every participating physical GPU index and UUID; exclude unrelated devices.
- [ ] Confirm enough integrated coverage and an allocation or reason for every request.
- [ ] ledger `conservation_error_joules < 1e-6`
- [ ] `device_joules` roughly equals mean power × run window from `nvidia-smi --query-gpu=power.draw --format=csv -lms 100` sampled in parallel (order of magnitude; write down both numbers)
- [ ] every request has either an allocation or an `unavailable_reason`; count each
- [ ] idle energy is non-zero if there were gaps between requests, zero otherwise

### Tracing enabled vs disabled

- [ ] run `--no-trace --repeat 5` and `--repeat 5`; record per-run `generate()` wall times for both
- [ ] report median traced/untraced ratio; do not claim an overhead figure before this exists
- [ ] outputs (generated text) are identical between traced and untraced runs at temperature 0

### Failure surfacing

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
| CPU suite on the GPU box | 83 passed, `tests/test_collection.py` skipped as a whole (`smoke/cpu_tests.log`). Cause: a class-level `pytest.importorskip("pyarrow")` skipped the entire module on machines without pyarrow, so the sampler/writer/tracer tests did not run on the pod. Fixed after the run (per-test `find_spec` skip); at that point the suite was 106 tests. The CPU suite has not been re-run on a GPU box since (it needs nothing from the GPU). |
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

## AsyncLLM (2026-09-08, RTX 4000 Ada, opt-125m)

Evidence: `docs/gpu_runs/2026-09-08-rtx-4000-ada-asyncllm/`. `examples/vllm_async_smoke_test.py`:
ALL CHECKS PASSED. Six concurrent `generate()` streams traced with completion
status, engine prompt token counts, output counts equal to what the consumer
received, and TTFT (~190 ms for the first concurrent batch on this card);
a stream the client stopped reading after 4 tokens was recorded as aborted
with `abort_cause=client_cancelled` and `AsyncLLM.abort` observed; `generate`
and `abort` restored after `stop()`; scheduler and executor reported
unavailable with the AsyncLLM reason; 34 vLLM per-step stat records (KV usage,
running/waiting) arrived through the `stat_loggers` hook over the
multiprocess engine core.

## 7B model on A100 (2026-09-08, 2x A100-SXM4-80GB, Qwen2.5-7B)

Evidence: `docs/gpu_runs/2026-09-08-a100-qwen2.5-7b/`. Same workload shape as
the small-model experiment at a lower arrival rate (10 short/s, 128 output
tokens each, 12 long 1536-token prompts every 0.4 s), `ignore_eos` so work is
identical across configurations (confirmed by `decide`: same work = yes).

| Check | Result |
|-------|--------|
| In-process smoke, TP=1 | ALL CHECKS PASSED; GPU span p50 10.85 ms of an 11.04 ms step; host share 2% |
| TP=1 experiment (3 repeats) | improved 3/3: short TTFT p95 103.4 -> 29.0 ms (-72%), worst short stall -71 to -72%, long TTFT p50 104 -> 170 ms (+64%); mean short TTFT change decomposes entirely into prefill |
| TP=1 GPU spans | long-chunk steps 102.6 ms vs 11.1 ms for the rest (baseline), 27.6 ms under the cap; host overhead p50 0.21 ms, host share 2%: the interference is GPU prefill compute |
| TP=2 experiment (2 repeats, NVLink) | improved 2/2: short TTFT p95 60.5 -> 18.1 ms (-70%), worst stall -66 to -72%, long TTFT +77 to +79%; GPU spans correctly refused (`MultiprocExecutor` runs workers out of process) |
| Decision, target short TTFT p95 <= 50 ms | candidates: `tp1_capped`, `tp2_capped`; neither baseline meets it. Energy per output token 0.34 J (TP=1) vs 0.49 J (TP=2) at ~1160 vs ~1200 tok/s |
| Findings, TP=1 baseline | `long_prompt_interference` supported (49 affected requests, 103 vs 11 ms steps); queue overload, KV pressure (max 1% usage), host overhead (3%) not supported; tracer self-effect not supported after the fix below |
| Load generator | arrival delay max 8 to 24 ms (recorded in manifests); short TTFT from intended arrival p95 108 ms baseline vs 45 ms capped at TP=1 |

Caveats stated in the outputs: throughput over an open-loop run reflects the
arrival schedule unless the system saturates (the ~3% tok/s difference between
TP=1 and TP=2 is not a capacity comparison); GPU spans are upper bounds on busy
time; energy is an allocation with coverage 1.00 here.

Tooling issues found by this run and fixed afterwards (the recorded analyses
were recomputed locally with the fixed code, and the on-pod versions are kept
in the logs): the `gpu_*` loader also matched `gpu_steps_*` files and crashed
`decide`; manifests counted only the workload while the traced settling phase
added 4 requests, so every repeat was ineligible until `--exclude-class settle`
existed and the driver counted them; the tracer self-check flagged 100 ms
prefill steps merely for overlapping a 0.2 ms drain; a parallel
`pip install hf_transfer` upgraded huggingface_hub past what transformers
4.57 accepts.

## CUDA-event step spans (2026-09-08, RTX 4000 Ada)

Evidence: `docs/gpu_runs/2026-09-08-rtx-4000-ada-cuda-spans/`. First run of
the GPU span timing on hardware; also the third GPU (second architecture) for
the smoke test and the mixed-prompt experiment.

| Check | Result |
|-------|--------|
| In-process smoke (8 x 32 tokens) | ALL CHECKS PASSED; 32 spans for 32 batches; `0 < gpu_span <= host_step` on every step; timer clean (0 dropped, 0 errors, 0 pending); every event resolved by the end of its own step (vLLM syncs on the sampled-token copy inside `execute_model`) |
| Multiprocess smoke | ALL CHECKS PASSED; executor reported unreachable (`SyncMPClient`), no `gpu_steps` file, as designed |
| Decode steps (smoke, 8 requests) | GPU span p50 1.60 ms of a 1.78 to 1.80 ms host step; host overhead p50 0.20 ms; first (prefill, 112 tokens) step 2.90 ms span of 3.25 ms |
| Experiment, ~1490 steps per run | GPU span p50 1.71 ms; host overhead p50 0.16 ms; median host share 9% (finding `host_overhead`: not supported, steps are GPU-bound) |
| Long-chunk steps (1536-token prefill) | GPU span 7.92 ms vs 1.71 ms for other steps in baseline; 2.74 ms under `long_prefill_token_threshold=256`. The interference cost is GPU prefill compute co-scheduled with decodes, not host work |
| Experiment verdict on this GPU | improved 3/3: short TTFT p95 -62.2/-62.7/-62.4%, ITL max -54.8/-45.5/-54.0%, long TTFT +113.6 to +116.1%; mean TTFT change decomposes entirely into prefill (queue 0) |
| TTFT from intended arrival | short p95 10.9 ms baseline vs 5.1 ms capped (engine TTFT p95 8.5 vs 3.2 ms); load-generator delay max 3 to 4 ms, now recorded in manifests |
| Tracer self-check | 33 collector drains per run, longest 0.09 ms; no step overlapping a drain was unusually long |

Not measured here: GPU busy time (see the Nsight cross-check below for how
far the span is from it) and the event-recording overhead itself.

## Validation set from a clean commit with independent repeats (2026-09-09, A100 80GB, Qwen2.5-7B, session 4)

Evidence: `docs/gpu_runs/2026-09-09-a100-qwen2.5-7b-clean-repeats/`. Source:
commit `b3973ed` uploaded as a clean `git archive`; every manifest carries the
fingerprint `sha256:bd5774b11fa83b41598d`, equal to that commit's fingerprint
computed locally (the pod had no `.git`, so the manifests say so rather than
claiming a commit). CPU suite on the pod: 251 passed, 1 skipped. Each engine
ran in its own process; `decide --min-repeats 3`; every repeat is a separate
run directory with its own tracer session (duplicates would be rejected).

| Check | Result |
|-------|--------|
| Queue overload, 4 independent repeats per configuration (bursts of 32 x 64/64 tokens every second, `max_num_seqs=8`) | `queue_overload` supported on the source run (88 of 96 waited over 100 ms). `decide`, target burst TTFT p95 <= 3000 ms, SLO ttft <= 3000 ms and tpot <= 40 ms: baseline 6435 ms, run-to-run range [6419..6456] (spread 38 ms), goodput 50%; `seqs16` 2180 ms [2164..2200] (spread 36 ms), goodput 100%, the only candidate; `budget16384` 6484 ms [6465..6498], goodput 50%, no effect. The request bootstrap intervals are narrower than or comparable to the run-to-run ranges, which is the point of reporting both. Steps 384 vs 768; J/token 0.23 vs 0.43 |
| KV-cache pressure on the 7B model, 3 independent repeats (64 x 256/1536 tokens at once, `gpu_memory_utilization=0.25`) | `kv_cache_pressure` supported on the source run: usage 100% in 733 of 2148 steps, 22 preemptions from vLLM's stats; `long_prompt_interference` (7 prefill steps of 526 ms vs 16.5 ms) and `queue_overload` (32 requests waited: the 16k prompt tokens exceed one step's budget) also supported. Plan (ranked by affected requests): `mem35` (0.25 to 0.35) and `seqs128` |
| Its `decide`, target kv e2e p95 <= 60000 ms (loose by design) | baseline 38.0 s [37.9..38.1] (spread 0.15 s); `mem35` 29.9 s [29.9..30.1] (-21%; 1538 steps vs 2148; J/token 0.085 vs 0.111; `findings` on it: usage max 51%, no preemptions, `kv_cache_pressure` not supported); `seqs128` 36.6 s [36.3..36.7] (18 preemptions, usage still 100% in 637 steps). All three meet the loose target; recommendation `mem35` on goodput then throughput |
| Provenance in the manifests | `llmtrace_source_fingerprint` set on every run, `llmtrace_git_commit` null, `llmtrace_snapshot_complete` false with the gap "no git tree" (a fingerprint identifies code but cannot restore it); the README ties the fingerprint to commit `b3973ed` |
| Memory after 23 engines | 0 MiB |

Still not measured: anything above 7B; `host_overhead` and
`tracer_observer_effect` as positive findings; more than four repeats.

## Two induced bottlenecks and the 7B overhead matrix (2026-09-08, RTX A5000 and A100 80GB, session 3)

Evidence: `docs/gpu_runs/2026-09-08-rtx-a5000-bottlenecks/` (opt-125m) and
`docs/gpu_runs/2026-09-08-a100-qwen2.5-7b-overhead-queue/` (Qwen2.5-7B).
Each bottleneck was induced with a workload spec and one engine setting, then
taken through `findings`, `plan --max-candidates 2 --repeats 2`, `run --plan`
(one spawned process per engine) and `decide`. Two repeats per configuration
is the minimum `decide` accepts, not a good estimate of run-to-run variation;
the min..max ranges below rest on two runs each, and the bootstrap intervals
are within-run statements over requests that share engine steps. Provenance:
these sessions ran the session-2 and session-3 fixes uploaded before they
were committed (the pods had no `.git`); the READMEs name the base commit,
and the fixes are in the commit that adds this evidence. Runs made from now
on carry a source fingerprint and, when dirty, `source.patch`.

| Check | Result |
|-------|--------|
| Queue overload, opt-125m: bursts of 32 requests (64 prompt, 64 output tokens) every second, `max_num_seqs=8` | `queue_overload` supported: 72 of 96 requests waited over 100 ms (in-process queue spans); the other four hypotheses not supported. Plan: `seqs16` (running count reached the cap) and `budget16384` |
| Its `decide`, target burst TTFT p95 <= 300 ms, SLO ttft <= 300 ms and tpot <= 10 ms | baseline 379 ms [362..396], CI [363..396], goodput 75%; `seqs16` 166 ms [138..194], CI [160..194], goodput 100%, the only candidate; `budget16384` 346 ms [342..351], goodput 75%: doubling the token budget changed nothing because the sequence cap, not the budget, was the limit (a negative control the planner proposed alongside). `seqs16` halved the step count (384 vs 768) and the energy per token (0.0107 vs 0.0240 J) |
| Same queue loop on Qwen2.5-7B (A100 80GB) | 88 requests waited; `seqs16` cut TTFT p95 from 6.38 s [6.377..6.378] to 2.14 s [2.131..2.146] with J/token 0.23 vs 0.43; `budget16384` again changed nothing (6.36 s). No candidate met the 300 ms target, which was set for the small model; `decide` said so |
| KV-cache pressure, opt-125m: 48 requests of 256 prompt and 1024 output tokens at once, `gpu_memory_utilization=0.06` | `kv_cache_pressure` supported: usage 100% in 1547 of 2473 steps with 56 preemptions from vLLM's per-step stats (the sync-engine stats hook at work); `long_prompt_interference` also supported (38 requests, 18 long-chunk steps, 6.9 vs 2.8 ms) |
| First plan attempt | With candidates in rule order and the cap at two, the interference candidates `cap1024`/`cap512` were proposed and the KV ones skipped as beyond the cap. `decide` on them: e2e p95 7.48 s baseline vs 7.40 and 7.43 s, no candidate, both replays still at 100% usage with 55 to 56 preemptions. A correct negative result that exposed a planner defect: candidates are now ranked by the affected-request count of their finding (48 KV vs 38 interference here), and `plan --finding` restricts them |
| KV rerun after the fix | Plan: `mem16` (`gpu_memory_utilization` 0.06 to 0.16) and `seqs128`. `decide`, target kv e2e p95 <= 4000 ms: baseline 7.43 s [7.41..7.46], `mem16` 5.39 s [5.29..5.49] (-28%; 1025 steps vs 2473; J/token 0.0206 vs 0.0276; `findings` on it: KV usage max 70%, no preemptions, `kv_cache_pressure` not supported), `seqs128` 6.55 s [6.46..6.63] (28 preemptions, half). No candidate met the target (set before the numbers were known); goodput under e2e <= 4 s was 33% for the baseline and 0% for both candidates because without preemption every request finishes at about the same, later time, while preemption lets a third finish early: the p95 and the SLO share can move in opposite directions |
| Overhead matrix on Qwen2.5-7B (64 x 256 tokens, 3 repeats) | untraced `generate()` 3.360 s, untraced engine loop 3.378 s, traced `generate()` 3.400 s (+1.2%, 0.16 ms per step), traced engine loop 3.436 s (+1.7%, 0.23 ms per step), without GPU step timing 3.415 s: CUDA-event recording 0.08 ms per step (+0.6%). The relative cost falls with model size as expected (opt-125m: +7.7% / +14.4%, 0.11 ms per step for events) |
| Memory after the runs | 1 MiB (A5000), 0 MiB (A100): per-process engines released everything |

Not measured: KV pressure on the 7B model (only queue overload was repeated
there); the `host_overhead` and `tracer_observer_effect` findings as positive
results; anything above 7B.

## Generic runner, plan loop, stats hook, overhead matrix (2026-09-08, RTX A5000, session 2)

Evidence: `docs/gpu_runs/2026-09-08-rtx-a5000-runner-plan/` (its README lists
the stages and the two defects found and fixed during the session).
Workload: the `llmtrace workload template` (120 short requests at 40/s of 32
prompt tokens and 128 output tokens, 12 long prompts of 1536 tokens every
0.2 s), opt-125m, in-process core, `ignore_eos`.

| Check | Result |
|-------|--------|
| `llmtrace doctor` on the box | vLLM 0.11.0 verified target, in-process core, CUDA and NVML found; every signal available except parquet output |
| CPU suite on the box | 238 passed, 1 skipped (parquet), 1 failed: a doctor test assumed no CUDA (the fake engine's executor is bracketed with real CUDA events on a GPU box); fixed |
| Generic runner vs experiment driver, 256-token cap | Both verdicts `improved`. Runner: short TTFT p95 7.93 to 2.85 ms (-64.0%) and 8.28 to 2.89 ms (-65.1%), short ITL max -51.9% and -61.3%, long TTFT p50 +112% and +103%. Driver: 7.90 to 3.09 ms (-60.9%), ITL max -47.3%, long TTFT +115% |
| `decide` with runner + driver runs as three repeats each, target short TTFT p95 <= 5 ms, SLO ttft <= 5 ms and tpot <= 3 ms | baseline 7.9 ms [7.9..8.3], 95% CI [7.9..8.1], goodput 90%; capped 2.9 ms [2.9..3.1], CI [2.9..3.1], goodput 100%; energy 0.0349 vs 0.0355 J/token at 99% coverage; work identical across all six runs; candidate: capped |
| Findings on the runner baseline | long_prompt_interference supported (97 requests in 12 long-chunk steps, 7.6 vs 1.5 ms); host_overhead not supported (15% host share); queue, KV pressure, observer effect not supported |
| `run --plan`, first attempt | Baseline r0 ran; every later engine failed to start: "Free memory on device 11.75/23.55 GiB is less than desired 0.5" because the previous engine's memory stayed allocated in the process. Fixed: one spawned process per real engine (`run_workload_isolated`) plus best-effort teardown; GPU memory after six engines: 1 MiB |
| vLLM stats on the sync engine, first attempt | No `logger_manager`: `vllm.LLM` sets `disable_log_stats=True` unless told otherwise (read from `vllm/entrypoints/llm.py`). Fixed: the runner passes `disable_log_stats=False` by default; then 1714 per-step records with KV usage and 136 finished-request stats with `queued_time` |
| Plan from the baseline run | `long_prefill_token_threshold` 1024 and 512 (largest observed chunk 1536); queue, KV and host findings not supported so no other candidates; every source setting reproduced (no `NOT REPRODUCED`) |
| Planned runs (2 repeats each), `decide` on the same target and SLO | baseline 8.3 ms [8.1..8.4], CI [8.0..8.6], goodput 89%; cap1024 6.1 ms [6.0..6.2], CI [5.9..6.7], goodput 89%; cap512 4.8 ms [4.8..4.9], CI [4.7..5.0], goodput 98%, the only candidate. Per-run analysis against the planned baseline: cap1024 short TTFT p95 -24.2%, ITL max -21.4%, long TTFT +28.2%; cap512 -40.9%, -43.7%, +62.1%; with the earlier 256 cap at -64%, -52%, +112% the effect and its cost grow monotonically as the cap shrinks |
| Overhead matrix, 64 x 256 tokens, 3 interleaved repeats | untraced `generate()` 0.818 s, untraced engine loop 0.843 s, traced `generate()` 0.881 s (+7.7%, 0.25 ms per step), traced engine loop 0.964 s (+14.4%, 0.47 ms per step), traced engine loop without GPU step timing 0.936 s: CUDA-event recording 0.11 ms per step (+3.0%). Higher than the first session's +4% / +9%, which had no step timing; the remaining difference is between sessions on different boxes and is not separated here |

Not measured: any of this on a model larger than opt-125m; the planner's
other rules (queue, KV pressure, host overhead) on real vLLM, since those
findings were not supported on this workload.

## Nsight Systems cross-check of the step spans (2026-09-08, RTX A5000)

Evidence: `docs/gpu_runs/2026-09-08-rtx-a5000-nsys/`. The mixed-prompt
experiment (reduced workload: 40 short requests at 40/s, 6 long prompts,
opt-125m, in-process core) was run for `baseline` and `capped` under
`nsys profile -t cuda,nvtx` with `--enable-nvtx`, exported to SQLite and
joined per step by `scripts/nsys_step_compare.py`: for each `llmtrace step N`
NVTX range, the union of kernel executions and CUDA-graph executions inside
the range (Nsight's GPU busy time) against llmtrace's CUDA-event `gpu_span_ms`
for the same step index.

| Check | baseline | capped (`long_prefill_token_threshold=256`) |
|-------|----------|------|
| NVTX step ranges / matched to `gpu_steps` records | 626 / 626 | 654 / 654 |
| Steps with the span below Nsight's busy time | 0 | 0 |
| NVTX range p50 vs `host_step_ms` p50 | 1.786 vs 1.788 ms | 1.725 vs 1.726 ms |
| Decode-only steps: busy p50 / span p50 / ratio p50 (p10 to p90) | 0.855 / 1.50 ms / 0.58 (0.48 to 0.59) | 0.853 / 1.46 ms / 0.58 (0.53 to 0.62) |
| Long-chunk steps: busy p50 / span p50 / ratio | 6.28 / 7.48 ms / 0.84 (1536-token chunks, 130 kernels, eager) | 1.68 / 2.56 ms / 0.66 (256-token chunks, 33 kernels + 13 graph launches) |
| GPU work per decode step | 9 kernels + 1 CUDA-graph execution | same |
| Sum over the traced phase: busy / span / host | 572 / 1042 / 1198 ms | 589 / 1052 / 1207 ms |

Reading: the span is what it was defined to be, an upper bound that includes
launch gaps on the stream. On this 125M model a decode step is launch-bound:
the GPU is idle for about 0.64 ms of a 1.5 ms span between nine small eager
kernels (sampler, input preparation) and one graph launch. On the eager
1536-token prefill steps the bound is tight (84% busy). So on opt-125m
`host_overhead_ms` (host minus span, 9 to 13%) understates the host-side
share of the step; the span-minus-busy gap (about 40% here) is the launch
overhead that only a profiler can see. The bound was not checked on the 7B
model; expect it to be tighter there (longer kernels, same gaps).

Caveat found while doing this: vLLM runs decode steps as CUDA graphs, and with
Nsight's default `--cuda-graph-trace=graph` those appear in
`CUPTI_ACTIVITY_KIND_GRAPH_TRACE`, not in the kernel table. A first version
of the comparison that read only kernels reported busy/span of 0.10 on decode
steps; the script now unions both tables (and handles
`--cuda-graph-trace=node` profiles, where the kernels appear individually).
Profiling overhead under `nsys` was not separated from the run; the numbers
above are not a benchmark of llmtrace or vLLM.

## Diagnosis experiment (2026-09-08, RTX A4500)

See `experiments/mixed_prompts/README.md` (results section) and
`docs/gpu_runs/2026-09-08-rtx-a4500-mixed-prompts/`. Summary: baseline traces
attribute the short-request TTFT tail to steps carrying a 1536-token prefill
chunk (8 ms vs 1.6 ms steps); `long_prefill_token_threshold=256` reduced
short TTFT p95 by 63% and the worst short-request stall by 54 to 62% in 3/3
repeats, while doubling long-request TTFT.

## Findings from GPU runs

* 2026-09-08, RTX A4500, mixed-prompt experiment run 1: with the default 1.0 s
  collector interval, the engine step that coincided with each drain (~500
  batch records serialized by the writer thread under the GIL) took ~10 ms
  longer than the token model predicts, in every run and both configs
  (`docs/gpu_runs/2026-09-08-rtx-a4500-mixed-prompts/run1`). With a 0.1 s
  interval (run 2) no such stalls appeared. `collection_interval_s` now
  defaults to 0.1. Serialization cost is still paid on the engine's process;
  it is only spread out, not removed.
* Same runs: the first traced step after a single-prompt warm-up took ~23 ms
  (first mixed batch shape), and the first traced step of each run ~8 ms; the
  experiment driver now warms up with a mixed workload and runs a traced
  settling phase before measuring.

* 2026-09-08, RunPod RTX A5000 (driver 580.159.04), Python 3.11, vLLM 0.11.0:
  `pip install vllm==0.11.0` resolved `transformers` to 5.16.1, and every
  `LLM(...)` construction failed with
  `AttributeError: GPT2Tokenizer has no attribute all_special_tokens_extended`
  (in `vllm/transformers_utils/tokenizer.py`). Not an llmtrace bug; the `vllm`
  extra now pins `transformers>=4.56,<5`.

## Checks to revisit when changing the environment

* `_attach_scheduler` path if `InprocClient` attribute names differ at runtime.
* `NVMLBackend` throttle-reason constants on older `nvidia-ml-py`.
* `min_coverage_fraction` for very short requests at 50-100 ms sampling.
* The `prefill` span including the first decode step (step granularity).

## Outside this validation scope

Renting hardware, deploying servers, AMD/ROCm, DCGM, dashboards, additional
inference frameworks, ML-based diagnosis, distributed multi-node runs.
