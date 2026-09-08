# Experiment: short requests mixed with long prompts

Status: prepared and exercised on the CPU fake engine only. **No GPU run yet.**
Nothing here shows that the diagnosis is right on real hardware.

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

1. TTFT/TPOT percentiles per request class, plus inter-token latency (ITL):
   the durations of every engine step a request was scheduled in, from batch
   metadata. Average TPOT hides one slow step among 128; ITL p99 does not.
2. Per-step duration and scheduled tokens from batch metadata; steps carrying a
   prefill chunk above `--chunk-threshold` (default 128 tokens) are flagged.
3. Short-request interference: share of each short request's step time spent in
   flagged steps, and TPOT/TTFT of affected vs unaffected short requests.
4. Step-time model: least-squares `duration = a + b * scheduled_tokens`.
5. `--compare`: side-by-side change with an explicit verdict (threshold 20%)
   on two stall metrics, short-request **TTFT p95** and **ITL max**: `improved`
   only if both improve, `worse` if either regresses, else `no_meaningful_change`.
   ITL p99, TPOT and the long-request TTFT cost are reported alongside. Without
   batch metadata the ITL part falls back to TPOT p95.

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

## What would count as a result

* The traces explain the slowdown if flagged steps are markedly longer than
  unflagged ones, the step-time model has a clear per-token slope, and affected
  short requests have a worse TPOT/TTFT tail than unaffected ones, in the
  baseline run.
* The scheduling change helps if `capped` reduces both short-request TTFT p95
  and ITL max by at least 20% versus `baseline` in all repeats, with ITL p99,
  TPOT and the long-request TTFT cost reported alongside. Anything less is
  reported as no meaningful change.

## Synthetic dry run (fake engine, invented cost model; not evidence)

With the fake's cost model (1.5 ms + 8 us per scheduled token): baseline
long-chunk steps take ~14 ms against ~1.6 ms decode steps, the step-time fit
recovers the invented slope exactly, 98 of 120 short requests share a step with
a long chunk, and short TTFT p95 is 14.1 ms vs 1.8 ms for unaffected ones.
`capped` bounds steps to ~3.6 ms: short TTFT p95 14.1 -> 3.9 ms and ITL max
14.1 -> 3.9 ms (both about -73%), ITL p99 1.8 -> 3.6 ms (more, milder affected
steps), long TTFT +55%. This shows the pipeline and verdict logic work on an
invented model; the real question is answered by the GPU run.
