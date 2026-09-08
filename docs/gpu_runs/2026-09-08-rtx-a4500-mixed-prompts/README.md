# Mixed-prompt experiment, GPU runs, 2026-09-08, RunPod RTX A4500

Raw evidence for `experiments/mixed_prompts/README.md` (results section) and
`docs/GPU_VALIDATION.md`. Environment: RTX A4500 (20 GB), driver 550.127.05,
Python 3.11.11, vLLM 0.11.0, torch 2.8.0+cu128, transformers 4.57.6,
`facebook/opt-125m`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`.

Each `runN_*/` holds `summary.txt`, per-run driver logs, `analysis_*.txt/json`
(baseline vs capped, repeats 0..2), and full trace files
(`traces_*`, `batches_*`, `gpu_*`) for every run directory, so every number can
be recomputed with `python experiments/mixed_prompts/analyze.py <baseline_dir> --compare <capped_dir>`.

| set | driver differences | what it showed |
|-----|--------------------|----------------|
| `run1_collect1.0s` | tracer collector interval 1.0 s (then the default); single-prompt warm-up | ~10 ms stalls at exactly 1.0 s and 2.0 s in every run (collector drains ~500 batch records under the GIL); ~23 ms first-mixed-batch warm-up stall at 0.10 s |
| `run2_collect0.1s` | collector 0.1 s; 8-short + 1-long warm-up | collector stalls gone; first traced step of each run ~8-9 ms (lone 32-token prefill) dominated ITL max |
| `run3_settle` | + traced settling phase (`settle-*` requests) | settling step fine, but the first measured request's lone 32-token prefill (~8 ms) and a 36-token mixed step (~17-20 ms, 2 of 3 capped runs) remained |
| `run4_full_warmup` | untraced warm-up replays the whole workload; then settling | no step exceeds the token model; final result |

The code that produced each set is the repository at the corresponding
commit; the driver changes between sets are exactly the ones described above
(see git history of `experiments/mixed_prompts/run.py`).
