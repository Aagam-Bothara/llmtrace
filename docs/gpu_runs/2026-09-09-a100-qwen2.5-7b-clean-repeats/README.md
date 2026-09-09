# GPU run 2026-09-09, RunPod A100 80GB PCIe (session 4): validation set from a clean commit with independent repeats

Source: commit `b3973edff61c9da3ff32065baaccda24f1cbbe9c`, uploaded as
`git archive HEAD` (clean tree, no uncommitted changes). The pod had no
`.git`, so each manifest's `llmtrace_git_commit` is null and
`llmtrace_snapshot_complete` is false with the gap "no git tree"; the
identity is the source fingerprint `sha256:bd5774b11fa83b41598d`, recorded in
every manifest, printed on the pod (`environment.txt`) and equal to the
fingerprint of that commit computed locally (`SOURCE_COMMIT`). The CPU suite
on the pod: 251 passed, 1 skipped (`pytest.log`).

Environment (`environment.txt`): NVIDIA A100 80GB PCIe, driver 570.172.08,
Python 3.11.11, torch 2.8.0+cu128, vLLM 0.11.0, transformers 4.57.6,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, model `Qwen/Qwen2.5-7B` (bf16, one GPU,
`max_model_len=2048`). `session_clean.sh` is the exact driver, `summary.txt`
its log. Every engine ran in its own spawned process; `decide` was run with
`--min-repeats 3`.

* `queue/`: `w_queue.json` (96 requests of 64 prompt and 64 output tokens in
  bursts of 32 every second) on `--set max_num_seqs=8` (`source/`); plan
  candidates `seqs16` and `budget16384`; `planned/<config>/r0..r3` (four
  independent repeats per configuration); `decision.*` (target burst TTFT p95
  <= 3000 ms, SLO ttft <= 3000 ms and tpot <= 40 ms); `findings_*`.
* `kv/`: `w_kv7b.json` (64 requests of 256 prompt and 1536 output tokens at
  once) on `--set gpu_memory_utilization=0.25` (`source/`; 20 GB for weights
  plus cache, which the workload's 115k tokens overflow); plan candidates
  `mem35` (0.25 to 0.35) and `seqs128`; `planned/<config>/r0..r2` (three
  independent repeats); `decision.*` (target kv e2e p95 <= 60000 ms, SLO e2e
  <= 60000 ms, deliberately loose so the comparison rather than a pass/fail
  is the result); `findings_*`.

Numbers are in `docs/GPU_VALIDATION.md`.

Raw trace files (`*.jsonl`) for this session are not in git: they are the `2026-09-09-a100-qwen2.5-7b-clean-repeats-raw-traces.tar.gz` asset of the GitHub release [`evidence-2026-09`](https://github.com/Aagam-Bothara/llmtrace/releases/tag/evidence-2026-09); extract it at the repository root to restore them under this directory. Manifests, summaries and every derived output are here.
