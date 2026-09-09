# GPU run 2026-09-08, RunPod RTX A5000: Nsight Systems cross-check of CUDA-event step spans

Environment (`environment.txt`): NVIDIA RTX A5000, driver 580.159.04, Python
3.11.11, torch 2.8.0+cu128, vLLM 0.11.0, transformers 4.57.6, Nsight Systems
2026.1.3. `VLLM_ENABLE_V1_MULTIPROCESSING=0` (in-process engine core, so
llmtrace's `execute_model` bracket and NVTX ranges are active).

`nsys_run.sh` is the exact script: for `baseline` and `capped`
(`long_prefill_token_threshold=256`) it ran

    nsys profile -t cuda,nvtx -o <cfg> python experiments/mixed_prompts/run.py \
        --engine vllm --config <cfg> --model facebook/opt-125m --enable-nvtx \
        --num-short 40 --short-rate 40 --num-long 6 --long-every 0.2 --collection-interval 0.1
    nsys export --type sqlite -o <cfg>.sqlite <cfg>.nsys-rep
    python scripts/nsys_step_compare.py <cfg>.sqlite <cfg>_run --json <cfg>_compare.json

(a reduced workload to keep the profiles small; the warm-up phase is untraced
and carries no NVTX ranges).

Files: `<cfg>_run/` (llmtrace output: `gpu_steps_*`, `batches_*`, `traces_*`,
`gpu_*`, `collector_*`, `manifest.json`, `run_info.json`),
`<cfg>_compare.json` (per-step join of the NVTX range, Nsight busy time,
kernel and graph-launch counts, llmtrace's `gpu_span_ms` / `host_step_ms` and
the scheduled chunk sizes), `<cfg>_compare.txt` (summary as printed),
`*_profile.log`, `*_export.log`, `summary.txt` (exit codes).

The Nsight profiles themselves (`<cfg>.nsys-rep`, 4.5 and 3.2 MB) and their
SQLite exports are not committed: Nsight Systems records the profiled
process's environment variables, and a RunPod pod's environment includes a
RunPod API key (GitHub's secret scanning rejected the push that contained
them). The per-step join in `<cfg>_compare.json` (NVTX range, Nsight busy
time, kernel and graph-launch counts, llmtrace span and host time, chunk
sizes) is the derived data every number in `docs/GPU_VALIDATION.md` comes
from. To reproduce the raw profile, run `nsys_run.sh` on a GPU host; if you
keep profiles, strip or avoid secrets in the profiled environment first.

The comparison in `*_compare.*` is from the corrected script: the first
version counted only `CUPTI_ACTIVITY_KIND_KERNEL`, which misses the CUDA-graph
executions vLLM uses for decode steps (Nsight records those in
`CUPTI_ACTIVITY_KIND_GRAPH_TRACE` with the default `--cuda-graph-trace=graph`),
and reported a busy/span ratio of about 0.10 on decode steps. See
`docs/GPU_VALIDATION.md` for the numbers.

Raw trace files (`*.jsonl`) for this session are not in git: they are the `2026-09-08-rtx-a5000-nsys-raw-traces.tar.gz` asset of the GitHub release [`evidence-2026-09`](https://github.com/Aagam-Bothara/llmtrace/releases/tag/evidence-2026-09); extract it at the repository root to restore them under this directory. Manifests, summaries and every derived output are here.
