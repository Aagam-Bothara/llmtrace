# GPU run 2026-09-08, RunPod A100 80GB PCIe: Qwen2.5-7B overhead matrix and queue-overload loop

Environment (`environment.txt`): NVIDIA A100 80GB PCIe, driver 595.91.07,
Python 3.11.11, torch 2.8.0+cu128, vLLM 0.11.0, transformers 4.57.6, llmtrace
0.2.0 (commit 9d13003 plus the session-2 and session-3 fixes uploaded
uncommitted: per-process engines, `disable_log_stats=False`, failed-exit
manifests). `VLLM_ENABLE_V1_MULTIPROCESSING=0`, model `Qwen/Qwen2.5-7B`
(bf16, single GPU, `gpu_memory_utilization=0.5`, `max_model_len=2048`).

`session_7b.sh` is the exact driver (`summary.txt` its log):

* `overhead.json` / `overhead.log`: `scripts/gpu_overhead.py --num-prompts 64
  --max-tokens 256 --repeat 3 --gpu-step-timing both`, five interleaved
  configurations (untraced `generate()`, untraced engine loop, traced
  `generate()`, traced engine loop with GPU step timing, traced engine loop
  without it).
* `bn/`: the queue-overload loop run by `session_bottlenecks.sh` with
  `WHICH=queue` (`bn/summary.txt`): `w_queue.json` (96 requests of 64 prompt
  and 64 output tokens in bursts of 32 every second) on a source engine with
  `--set max_num_seqs=8` (`bn/queue/source`), `findings_source.*`, `plan.*`
  (candidates `seqs16` and `budget16384`), `planned/<config>/r<i>` (one
  spawned process per engine), `decision.*` (target burst TTFT p95 <= 300 ms,
  SLO ttft <= 300 ms and tpot <= 10 ms), `findings_<candidate>.*`.
  `bn_driver.log` is that script's stdout.

Numbers are in `docs/GPU_VALIDATION.md`. Model download log omitted.

Provenance: this session ran on a tarball upload without `.git`, with fixes that were uploaded before being committed, so its manifests carry no commit or fingerprint. The exact code is commit 9d13003 plus the fixes committed together with this evidence (the commit that added this directory). Runs made after that commit record `llmtrace_source_fingerprint`, the full commit, and `source.patch` when dirty.

Raw trace files (`*.jsonl`) for this session are not in git: they are the `2026-09-08-a100-qwen2.5-7b-overhead-queue-raw-traces.tar.gz` asset of the GitHub release [`evidence-2026-09`](https://github.com/Aagam-Bothara/llmtrace/releases/tag/evidence-2026-09); extract it at the repository root to restore them under this directory. Manifests, summaries and every derived output are here.
