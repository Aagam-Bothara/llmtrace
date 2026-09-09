# GPU run 2026-09-08, RunPod RTX A5000 (session 3): two induced bottlenecks through findings, plan, run and decide

Environment (`environment.txt`): NVIDIA RTX A5000 24 GB, driver 580.159.04,
Python 3.11.11, torch 2.8.0+cu128, vLLM 0.11.0, transformers 4.57.6, llmtrace
0.2.0 (commit 9d13003 plus the session-2 and session-3 fixes uploaded
uncommitted). `VLLM_ENABLE_V1_MULTIPROCESSING=0`, model `facebook/opt-125m`.

`session_bottlenecks.sh` is the exact driver (`summary.txt` its log). For
each bottleneck it runs a source configuration under `llmtrace run`, then
`doctor`, `findings --verbose`, `plan --repeats 2 --max-candidates 2`,
`run --plan` (one spawned process per engine) and `decide`, and finally
`findings` on each candidate's first repeat.

* `queue/`: `w_queue.json` (96 requests of 64 prompt and 64 output tokens in
  bursts of 32 every second) on `--set max_num_seqs=8` (`source/`);
  candidates `seqs16` and `budget16384`; target burst TTFT p95 <= 300 ms,
  SLO ttft <= 300 ms and tpot <= 10 ms.
* `kv/`: `w_kv.json` (48 requests of 256 prompt and 1024 output tokens at
  once) on `--set gpu_memory_utilization=0.06` (a KV cache far too small for
  the workload). First attempt: the planner ranked candidates in rule order,
  so with the cap at two the `long_prompt_interference` candidates
  (`cap1024`, `cap512`) crowded out the KV ones, and `decide` correctly found
  no change on the KV-bound tail. Only the text/JSON outputs of that attempt
  are kept (`plan.*`, `decision.*`, `findings_*.*`); its run directories
  (about 40 MB) were dropped because `kv_rerun/` repeats the same source
  configuration.
* `kv_rerun/`: after the fix (candidates ranked by the affected-request count
  of their source finding; `plan --finding` added): same source run,
  candidates `mem16` (`gpu_memory_utilization` 0.06 to 0.16) and `seqs128`;
  target kv e2e p95 <= 4000 ms, SLO e2e <= 4000 ms. `kv_rerun/summary.txt`
  is that rerun's log.

Every run directory holds `traces_*`, `batches_*`, `gpu_*`, `gpu_steps_*`,
`vllm_stats_*`, `collector_*`, `workload.json`, `manifest.json`,
`run_info.json`. Numbers are in `docs/GPU_VALIDATION.md`.

Provenance: this session ran on a tarball upload without `.git`, with fixes that were uploaded before being committed, so its manifests carry no commit or fingerprint. The exact code is commit 9d13003 plus the fixes committed together with this evidence (the commit that added this directory). Runs made after that commit record `llmtrace_source_fingerprint`, the full commit, and `source.patch` when dirty.

Raw trace files (`*.jsonl`) for this session are not in git: they are the `2026-09-08-rtx-a5000-bottlenecks-raw-traces.tar.gz` asset of the GitHub release [`evidence-2026-09`](https://github.com/Aagam-Bothara/llmtrace/releases/tag/evidence-2026-09); extract it at the repository root to restore them under this directory. Manifests, summaries and every derived output are here.
