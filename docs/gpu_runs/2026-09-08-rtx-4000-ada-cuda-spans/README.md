# GPU run 2026-09-08, RunPod RTX 4000 Ada: CUDA-event step spans

Environment: NVIDIA RTX 4000 Ada Generation (20 GB), driver 550.127.05, Python
3.11.11, vLLM 0.11.0, torch 2.8.0+cu128, transformers 4.57.6, `facebook/opt-125m`.
Code: the working tree committed right after this run (CUDA-event timing commit).

* `smoke_inproc/`: `examples/vllm_smoke_test.py` with `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  (ALL CHECKS PASSED, including 32 GPU spans for 32 batches, spans <= host step, timer clean).
* `smoke_multiproc/`: default engine core (ALL CHECKS PASSED; executor correctly reported
  unreachable: `SyncMPClient`).
* `exp/`: mixed-prompt experiment, 3 interleaved repeats per config, manifests with arrival
  delays, `gpu_steps_*.jsonl` per run. Recompute with
  `python experiments/mixed_prompts/analyze.py exp/baseline_0 --compare exp/capped_0`
  and `llmtrace findings exp/baseline_0`.

Raw trace files (`*.jsonl`) for this session are not in git: they are the `2026-09-08-rtx-4000-ada-cuda-spans-raw-traces.tar.gz` asset of the GitHub release [`evidence-2026-09`](https://github.com/Aagam-Bothara/llmtrace/releases/tag/evidence-2026-09); extract it at the repository root to restore them under this directory. Manifests, summaries and every derived output are here.
