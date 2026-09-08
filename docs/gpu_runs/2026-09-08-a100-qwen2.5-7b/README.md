# GPU run 2026-09-08, RunPod 2x A100-SXM4-80GB (NVLink): Qwen2.5-7B

Environment: 2x NVIDIA A100-SXM4-80GB, driver 580.159.04, Python 3.11.11,
vLLM 0.11.0, torch 2.8.0+cu128, transformers 4.57.6, huggingface_hub 0.36.2,
model `Qwen/Qwen2.5-7B` (bf16, snapshot d149729398750b98c0af14eb82c78cfe92750796).
Script: `experiments/mixed_prompts/run_7b.sh` (workload: 120 short requests at
10/s with 32-token prompts and 128 output tokens, `ignore_eos`; 12 long
1536-token prompts every 0.4 s; `max_model_len=2048`,
`gpu_memory_utilization=0.5`; untraced full-workload warm-up, traced settling
phase, collector interval 0.1 s).

* `smoke_tp1/`, `smoke_tp1.log`: in-process smoke test on GPU 0 with CUDA-event
  spans (ALL CHECKS PASSED).
* `tp1_*`: TP=1, in-process engine core (scheduler and executor visible):
  batch metadata and `gpu_steps_*.jsonl` per run. 3 repeats per config.
* `tp2_*`: TP=2 (`tensor_parallel_size=2`), in-process engine core: batch
  metadata visible; GPU spans refused because the executor is
  `MultiprocExecutor` (workers in other processes). 2 repeats per config.
* `analysis_*.txt/json`: `experiments/mixed_prompts/analyze.py --compare`.
* `findings_*.txt`: `llmtrace findings` (recomputed locally after the
  tracer-self-check fix; the on-pod version flagged the 12 long-chunk steps
  merely for overlapping a 0.2 ms drain).
* `decision_50ms.json`, `decision_150ms.json`: `llmtrace decide --exclude-class settle`
  (recomputed locally: the on-pod `decide` crashed because the `gpu_*` loader
  also matched `gpu_steps_*` files, fixed afterwards; the settle exclusion is
  needed because these manifests counted the workload only, also fixed).
* `install.log`, `dl.log`: note that installing `hf_transfer` for the download
  pulled huggingface_hub 1.30, which transformers 4.57 rejects; pinned back to
  0.36 before the runs (first attempt's failed manifests were discarded).
