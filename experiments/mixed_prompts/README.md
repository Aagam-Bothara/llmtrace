# Experiment: short requests mixed with long prompts

Status: run on one GPU (results below). Everything before the results section
is the design; the synthetic dry run at the end is not evidence.

## Question

When a steady stream of short requests is interrupted by long-prompt requests,
do llmtrace's traces explain the short-request slowdown, and does one
scheduling change improve it?

## Hypothesis (vLLM 0.11.0 V1 scheduler, verified from source)

Chunked prefill is on. Each step has a token budget (`max_num_batched_tokens`);
running requests are scheduled first, then waiting ones. A long prompt is
prefilled in chunks up to the remaining budget, so a decode step can carry a
prefill chunk of a thousand-plus tokens. Such steps take much longer than
decode-only steps. Short requests decoding in those steps see inflated TPOT;
short requests arriving during them see inflated TTFT.

`long_prefill_token_threshold=N` caps the prefill tokens a prompt longer than N
receives per step (`Scheduler.schedule()`: `num_new_tokens = min(..., threshold)`).
Prediction: step time is bounded, the short-request TPOT/TTFT tail shrinks, and
the long request's own TTFT grows (more steps to finish its prefill).

## Workload (`workload.py`)

Deterministic: 120 short requests (32-token prompt, 128 output tokens) at 40/s,
plus 12 long requests (1536-token prompt, 8 output tokens) every 0.2 s from
0.4 s. Prompts are token-id lists so lengths are exact. Adjustable via `run.py`
flags (`--num-long`, `--long-every`, `--short-tokens`, ...); the workload config
is saved in each run's `run_info.json`.

## Configurations (`run.py`)

| name | change vs vLLM defaults |
|------|-------------------------|
| `baseline` | none (`enable_chunked_prefill=True`, `max_model_len=2048`) |
| `capped` | `long_prefill_token_threshold=256` |

The effective scheduler config (`max_num_batched_tokens`, `max_num_seqs`,
threshold, policy) is printed and saved so the actual defaults on the machine
are recorded, not assumed.

## Diagnosis (`analyze.py`)

From llmtrace's files only (`traces_*`, `batches_*`), no extra instrumentation:

1. TTFT/TPOT percentiles per request class; TTFT from the *intended* arrival
   (engine TTFT plus the load generator's recorded delay); `step_ms`, the
   durations of the engine steps a request was scheduled in (a compute proxy);
   and `itl_ms`, the real inter-token latency: intervals between the request's
   successive step ends from its first-token step onward, including steps it
   was not scheduled in. Average TPOT hides one slow interval; ITL max does not.
2. Per-step duration and scheduled tokens from batch metadata; steps carrying a
   prefill chunk above `--chunk-threshold` (default 128 tokens) are flagged.
3. Short-request interference: share of each short request's step time spent in
   flagged steps, and TPOT/TTFT of affected vs unaffected short requests.
4. Step-time model: least-squares `duration = a + b * scheduled_tokens`.
5. `--compare`: side-by-side change with an explicit verdict (threshold 20%)
   on two stall metrics, short-request **TTFT p95** and **ITL max**: `improved`
   only if both improve, `worse` if either regresses, else `no_meaningful_change`;
   `unavailable` (with the missing metrics named) if either is missing in
   either run. ITL p99, step max, TPOT, TTFT from intended arrival and the
   long-request TTFT cost are reported alongside. Without batch metadata in
   both runs the stall part falls back to TPOT p95.

Why not ITL p99: the mechanism produces rare stalls (in the synthetic dry run
the long-chunk steps are under 1% of steps), and capping spreads each stall
over several milder steps, so an all-token p99 rises under the cap while the
worst stall falls. p99 is still reported; it is the throughput-side cost.

Items 2 to 4 are co-occurrence evidence from traces; the controlled comparison
in item 5 is the causal test.

## Run

CPU, synthetic (fake engine with an invented step-cost model; demonstrates the
pipeline, proves nothing about vLLM):

```bash
python experiments/mixed_prompts/run.py --engine fake --config baseline --out ./exp/fake_baseline
python experiments/mixed_prompts/run.py --engine fake --config capped   --out ./exp/fake_capped
python experiments/mixed_prompts/analyze.py ./exp/fake_baseline --compare ./exp/fake_capped
```

GPU (Linux, NVIDIA, `pip install -e ".[vllm]"`), three repeats per config:

```bash
bash experiments/mixed_prompts/run_gpu.sh facebook/opt-125m ./exp_gpu
```

`VLLM_ENABLE_V1_MULTIPROCESSING=0` is required: the diagnosis needs batch
metadata, which only exists with the in-process scheduler.

## Manifests and work-identical replay

Every run directory gets `manifest.json` (workload hash, seed, model revision,
vLLM and llmtrace versions, git commit, effective engine config, GPU, tracer
config, per-request scheduled vs actual arrival with delay p50/max, status).
A configuration that fails to start (e.g. out of memory) leaves a manifest
with `status: failed` and the error, and `llmtrace decide` lists it as failed.
Real runs use `ignore_eos=True` so every request generates exactly
`max_tokens`; otherwise batch composition moves where EOS lands and the work
differs across configurations (`--no-ignore-eos` to disable).
`--enable-nvtx` adds an NVTX range per engine step; run under
`nsys profile -t cuda,nvtx`, export with `nsys export --type sqlite`, and
`scripts/nsys_step_compare.py <sqlite> <run_dir>` compares Nsight's per-step
GPU busy time with llmtrace's CUDA-event spans (see `docs/GPU_VALIDATION.md`).

Then: `llmtrace findings <run_dir>` and
`llmtrace decide --target "short ttft_p95 <= 5ms" --config baseline=... --config capped=...`.

## What built-in metrics would have shown

vLLM's own metrics (Prometheus / `LoggingStatLogger`, and the same numbers
llmtrace records through the `stat_loggers` hook) give per-step aggregates:
TTFT and inter-token latency histograms, queued/prefill/decode time per
finished request without request ids, running and waiting counts, KV usage,
preemptions. From those alone one can see that short-request TTFT p95 was
8.4 ms and that nothing was queued for long. They cannot say *which* steps were
slow, *what else* was in them, or *which* requests paid for it: there is no
step-level record of scheduled tokens per request and no link from a request
to the steps it shared. The trace-level evidence that identified the mechanism
here was exactly that join (`batches_*.jsonl` request ids and scheduled
tokens per step, joined to `traces_*.jsonl` batch ids), plus the TTFT
decomposition into queue and prefill spans that `compare` prints. The
`findings` command reports the same hypothesis as `insufficient_evidence`, with the
missing evidence named, when only the aggregate stats are available.

## Visualize

```bash
llmtrace visualize ./exp_gpu/baseline_0 --compare ./exp_gpu/capped_0 --html-out report.html --trace-out baseline.perfetto.json
```

The HTML report shows the request Gantt (long prompts as wide bars, short
requests stacking up behind the red long-chunk steps), the step-duration
timeline with long-chunk steps in red, and the step-time-vs-tokens plot the
diagnosis fits. The Perfetto trace lets you click a slow short request and see
exactly which step it waited on and what else was in that step.

## What would count as a result

* The traces explain the slowdown if flagged steps are markedly longer than
  unflagged ones, the step-time model has a clear per-token slope, and affected
  short requests have a worse TPOT/TTFT tail than unaffected ones, in the
  baseline run.
* The scheduling change helps if `capped` reduces both short-request TTFT p95
  and ITL max by at least 20% versus `baseline` in all repeats, with ITL p99,
  TPOT and the long-request TTFT cost reported alongside. Anything less is
  reported as no meaningful change.


## Results: GPU run 2026-09-08 (RTX A4500, vLLM 0.11.0, opt-125m)

Evidence: `docs/gpu_runs/2026-09-08-rtx-a4500-mixed-prompts/` (four run sets;
the first three exposed measurement artifacts that were fixed in the driver and
in llmtrace itself, see that directory's README). Final set: `run4_full_warmup`,
three interleaved repeats per configuration, untraced full-workload warm-up, a
traced settling phase, collector interval 0.1 s. Effective scheduler config
recorded by the driver: `max_num_batched_tokens=8192`, `max_num_seqs=256`,
chunked prefill on, FCFS; `long_prefill_token_threshold` 0 vs 256.

**Do the traces explain the slowdown?** Yes, in every baseline run:

* 12 of ~1700 steps carried a 1536-token prefill chunk; they took 8.0 to 8.5 ms
  against 1.6 ms for the rest (step-time fit 1.6 ms + 4.1 to 4.5 us per
  scheduled token, r2 0.82 to 0.89).
* 97 of 120 short requests shared at least one such step. Short requests that
  did had TTFT p95 8.4 ms vs 2.5 ms for those that did not, and TPOT p95 2.0 vs
  1.8 ms.
* No step exceeded the token model by more than 2x the median (no unexplained
  stalls) in the final set.

**Where does the extra step time go?** (RTX 4000 Ada run, CUDA-event spans,
`docs/gpu_runs/2026-09-08-rtx-4000-ada-cuda-spans/exp`): in the baseline the
12 long-chunk steps have a GPU span of 7.92 ms against 1.71 ms for the other
~1480 steps, with host overhead unchanged (p50 0.16 ms, median host share 9%).
Under the cap the long-chunk span is 2.74 ms. So the interference is GPU
prefill compute co-scheduled with the short requests' decode, not host work;
the mean short TTFT change decomposes entirely into the prefill component.
Verdicts on this second GPU: improved 3/3 (short TTFT p95 -62%, ITL max -45
to -55%, long TTFT +114 to +116%).

## Results: Qwen2.5-7B on A100 (2026-09-08)

Evidence: `docs/gpu_runs/2026-09-08-a100-qwen2.5-7b/`. Workload at 10 short
requests/s (128 output tokens, `ignore_eos`), 12 long 1536-token prompts every
0.4 s; three repeats at TP=1 (in-process, GPU spans available) and two at TP=2
(in-process, spans refused for the out-of-process executor).

| | TP=1 baseline | TP=1 capped (256) | TP=2 baseline | TP=2 capped |
|---|---|---|---|---|
| short TTFT p95 (median of repeats) | 103.4 ms | 29.0 ms (-72%) | 60.5 ms | 18.1 ms (-70%) |
| worst short stall (ITL max) | 105 ms | 30 ms (-71%) | 68 ms | 19 ms (-72%) |
| long TTFT p50 (cost) | 104 ms | 170 ms (+64%) | 61 ms | 108 ms (+78%) |
| GPU span, long-chunk steps vs others | 102.6 vs 11.1 ms | 27.6 vs 11.1 ms | n/a (TP=2) | n/a |
| host share of a step | 2% | 2% | n/a | n/a |
| output tokens/s (open-loop, see caveat) | 1159 | 1160 | 1198 | 1198 |
| energy per output token | 0.337 J | 0.341 J | 0.490 J | 0.486 J |
| meets short TTFT p95 <= 50 ms in every repeat | no | yes | no | yes |

Reading: on a 7B model the mechanism is the same and larger in absolute terms:
a 1536-token prefill chunk turns an 11 ms step into a 103 ms step, and every
short request in flight or arriving during it waits for it. Capping the chunk
at 256 tokens bounds that to 28 ms, cuts the short-request p95 TTFT and the
worst stall by about 70%, and costs the long request 64 to 78% more time to
first token. TP=2 halves the stall (its steps are faster) but at this arrival
rate delivers almost the same tokens per second while spending 45% more energy
per token; with an open-loop workload the throughput column reflects the
arrival schedule, not capacity, and `decide` says so.

**Does the scheduling change help on opt-125m?** Verdict `improved` in 3 of 3 repeats (RTX A4500):

| metric (short requests unless noted) | baseline | capped (256) | change |
|---|---|---|---|
| TTFT p95 | 8.33 to 8.38 ms | 3.03 to 3.08 ms | -63% (all repeats) |
| ITL max (worst stall) | 8.05 to 8.54 ms | 3.08 to 3.93 ms | -54% to -62% |
| ITL p99 | 2.5 to 2.6 ms | 2.9 to 3.0 ms | +12% to +16% (more, milder chunk steps) |
| TPOT p95 | 2.00 ms | 1.96 ms | -2% |
| long-request TTFT p50 (cost) | 8.8 ms | 18.5 to 18.6 ms | +110% to +113% |
| long-request TPOT p50 | 2.04 ms | 2.11 ms | +3% |

Reading: capping long-prompt prefill at 256 tokens per step bounds the step
time that short requests and new arrivals wait on, at the price of doubling
the long request's own time to first token. Whether that trade is worth it is
a product decision; the traces make it visible and quantified.

Scope: one tiny model on one GPU with ~1.6 ms decode steps. Larger models have
longer steps and different prefill/decode cost ratios; the mechanism is the
same but the magnitudes are not transferable.

**Measurement lessons recorded from the first three sets** (all visible in
the committed traces): llmtrace's own collector caused ~10 ms stalls once per
second at the old 1.0 s default (now 0.1 s); first-time batch shapes cost 8 to
23 ms and must be warmed up with the actual workload; the per-step
"unexplained stall" report in `analyze.py` was added because these artifacts
initially masqueraded as scheduler effects.

## Synthetic dry run (fake engine, invented cost model; not evidence)

With the fake's cost model (1.5 ms + 8 us per scheduled token): baseline
long-chunk steps take ~14 ms against ~1.6 ms decode steps, the step-time fit
recovers the invented slope exactly, 98 of 120 short requests share a step with
a long chunk, and short TTFT p95 is 14.1 ms vs 1.8 ms for unaffected ones.
`capped` bounds steps to ~3.6 ms: short TTFT p95 14.1 -> 3.9 ms and ITL max
14.1 -> 3.9 ms (both about -73%), ITL p99 1.8 -> 3.6 ms (more, milder affected
steps), long TTFT +55%. This shows the pipeline and verdict logic work on an
invented model; the real question is answered by the GPU run.
