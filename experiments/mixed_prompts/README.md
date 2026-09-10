# Experiment: short requests mixed with long prompts

Can long prompts slow short requests that share their engine steps? This
experiment records that interaction, caps long-prompt processing, and measures
the benefit to short requests and the cost to long ones.

It was run on opt-125m and Qwen2.5-7B. Results are below; the fake-engine run
only demonstrates the workflow. See the [GPU record](../../docs/GPU_VALIDATION.md)
for session details and [raw evidence](../../docs/gpu_runs/README.md) for downloads.

## The idea

With chunked prefill enabled, a step can contain both short-request decode
work and a large chunk of a long prompt. A larger step can delay short
requests already running or arriving during it.

`long_prefill_token_threshold=256` limits a long prompt to 256 prefill tokens
per step. The prediction is shorter stalls for short requests, at the cost
of more steps before a long request produces its first token.

## Workload and settings

The default workload uses exact token-ID prompts and a fixed seed:

| Class | Count | Prompt tokens | Output tokens | Arrival schedule |
|-------|-------|---------------|---------------|------------------|
| Short | 120 | 32 | 128 | 40 requests/s |
| Long | 12 | 1,536 | 8 | Every 0.2 s, starting at 0.4 s |

`baseline` enables chunked prefill with `max_model_len=2048`. `capped` adds
`long_prefill_token_threshold=256`. The driver saves the effective scheduler
settings so the machine's defaults are recorded. Flags such as `--num-long`
and `--short-tokens` change the workload; the 7B runs below used a slower
arrival schedule.

## Run the example

CPU demo:

```bash
python experiments/mixed_prompts/run.py --engine fake --config baseline --out ./exp/fake_baseline
python experiments/mixed_prompts/run.py --engine fake --config capped --out ./exp/fake_capped
python experiments/mixed_prompts/analyze.py ./exp/fake_baseline --compare ./exp/fake_capped
```

On Linux with an NVIDIA GPU and `pip install -e ".[vllm]"`:

```bash
bash experiments/mixed_prompts/run_gpu.sh facebook/opt-125m ./exp_gpu
```

This script runs three repeats per configuration. It needs
`VLLM_ENABLE_V1_MULTIPROCESSING=0` for batch metadata. For explicit GPU
selection on a multi-GPU host, use the generic runner's `--gpu-id` option
or configure `gpu_sampler.gpu_ids` in the experiment driver.

Real runs use `ignore_eos=True` to keep output counts fixed. The driver saves
the workload, model, seed, settings, source information, intended and actual
arrivals, and health in `manifest.json`. A failed start leaves a failed
manifest with its error.

## Read the comparison

`analyze.py` joins requests to scheduler steps and reports:

- TTFT and TPOT by request class, plus TTFT from the intended arrival time.
- Inter-token latency (ITL), including steps when a request was not scheduled.
- Step duration and token counts, with large prefill chunks flagged.
- Latency for affected versus unaffected short requests.
- A fitted relationship between scheduled tokens and step duration.

The comparison reports `improved` only when both short-request TTFT p95 and
worst ITL improve by at least 20%. If either regresses by that threshold,
it reports `worse`; otherwise `no_meaningful_change`. Missing metrics produce
`unavailable`. Without batch metadata in both runs, the stall check uses
TPOT p95 instead of worst ITL.

Read ITL p99 and long-request TTFT alongside the verdict. A cap can replace a
few large stalls with more small ones: the worst stall improves while p99
gets worse. Shared steps are supporting evidence; the controlled change tests
the proposed explanation.

## Visualize the evidence

```bash
llmtrace visualize ./exp_gpu/baseline_0 --compare ./exp_gpu/capped_0 \
    --html-out report.html --trace-out baseline.perfetto.json
```

The HTML report shows request timelines, long-chunk steps and step duration
versus token count. Open the Perfetto file to inspect a request and the steps
it shared. vLLM's aggregate latency and queue metrics show the symptom;
request IDs and scheduled tokens link it to specific shared steps.

`llmtrace findings <run_dir>` reports the hypothesis and missing evidence.
Use `decide` with actual repeat paths to check a target. Newer comparison
checks may withhold recommendations or energy for older records that lack
required metadata or GPU selection.

## Comparing CUDA spans with Nsight

`--enable-nvtx` adds an NVTX range per step. Record with
`nsys profile -t cuda,nvtx`, export with `nsys export --type sqlite`, then use
`scripts/nsys_step_compare.py <sqlite> <run_dir>`. CUDA-event spans include
launch gaps; the Nsight comparison checks them against measured busy intervals.

## Results: GPU run 2026-09-08 (RTX A4500, vLLM 0.11.0, opt-125m)

Evidence: `docs/gpu_runs/2026-09-08-rtx-a4500-mixed-prompts/` (four run sets;
the first three exposed measurement artifacts that were fixed in the driver and
in llmtrace itself, see that directory's README). Final set: `run4_full_warmup`,
three interleaved repeats per configuration, untraced full-workload warm-up, a
traced settling phase, collector interval 0.1 s. Effective scheduler config
recorded by the driver: `max_num_batched_tokens=8192`, `max_num_seqs=256`,
chunked prefill on, FCFS; `long_prefill_token_threshold` 0 vs 256.

**Baseline observations:**

* 12 of ~1700 steps carried a 1536-token prefill chunk; they took 8.0 to 8.5 ms
  against 1.6 ms for the rest (step-time fit 1.6 ms + 4.1 to 4.5 us per
  scheduled token, r2 0.82 to 0.89).
* 97 of 120 short requests shared at least one such step. Short requests that
  did had TTFT p95 8.4 ms vs 2.5 ms for those that did not, and TPOT p95 2.0 vs
  1.8 ms.
* No step exceeded the token model by more than 2x the median (no unexplained
  stalls) in the final set.

**Step timing on a second GPU.** (RTX 4000 Ada run, CUDA-event spans,
`docs/gpu_runs/2026-09-08-rtx-4000-ada-cuda-spans/exp`): in the baseline the
12 long-chunk steps have a GPU span of 7.92 ms against 1.71 ms for the other
~1480 steps, with host overhead unchanged (p50 0.16 ms, median host share 9%).
Under the cap the long-chunk span is 2.74 ms. The extra elapsed time falls inside the CUDA-event span of steps that
combine prefill with short-request decode. The span includes launch gaps,
so it does not isolate kernel execution time. The mean short TTFT change
falls in the recorded prefill component.
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

**Effect on opt-125m:** verdict `improved` in all three repeats (RTX A4500):

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

Scope of the opt-125m table: one model on one GPU with ~1.6 ms decode steps. Larger models have
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
