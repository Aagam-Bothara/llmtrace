# GPU run 2026-09-08, RunPod RTX A5000 (session 2): generic runner, plan loop, stats hook, overhead matrix

Environment (`environment.txt`): NVIDIA RTX A5000 24 GB, driver 580.159.04,
Python 3.11.11, torch 2.8.0+cu128, vLLM 0.11.0, transformers 4.57.6, llmtrace
0.2.0 at commit 9d13003 plus the fixes made during the session (below).
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, model `facebook/opt-125m`.

Three stages, driven by the committed scripts:

* `gpu_session.sh` (stage 1, `summary.txt`): `llmtrace doctor` (`doctor_env.*`),
  the CPU suite on the GPU box (`pytest.log`: 238 passed, 1 failed, 1 skipped;
  the failure was `tests/test_doctor.py::TestRunDir::test_run_report_lists_signals`
  assuming no CUDA, fixed the same day), the workload template (`w.json`,
  `w_preview.json`), and the overhead matrix (`overhead.json`, `overhead.log`).
  The engine runs of this stage did not start: the script redirected their
  logs into directories it had not created (a bug in the session script, not
  in llmtrace).
* `gpu_session2.sh` (stage 2, `summary2.txt`): `runner/` = `llmtrace run
  --engine vllm` baseline x2 and capped x2 (`--set long_prefill_token_threshold=256`);
  `driver/` = `experiments/mixed_prompts/run.py` baseline and capped once;
  `doctor_run.*`, `findings_runner_baseline_0.*`, `analysis_*.{json,txt}`
  (the experiment's analysis on both drivers' runs), `decision_5ms.*`
  (three repeats per configuration: two runner runs plus the driver run).
  Two defects surfaced here: `run --plan` failed from its second engine on
  (GPU memory of the previous engine not released inside one process), and
  the sync engine had no `logger_manager` (`vllm.LLM` sets
  `disable_log_stats=True` unless told otherwise). Both fixed before stage 3;
  `planned.log`/`decision_planned.*` were overwritten by stage 3.
* `gpu_session3.sh` (stage 3, `summary3.txt`): `runner/baseline_2` with the
  new `disable_log_stats=False` default (`doctor_run_2.txt`; 1714 vLLM stats
  records, 136 finished-request stats with `queued_time`); `plan.*` from that
  run; `planned/` = `llmtrace run --plan` executing baseline, cap1024 and
  cap512 twice each, one spawned process per engine; `decision_planned.*`;
  `analysis_planned_*.{json,txt}` (each planned config's r0 against the
  planned baseline r0). GPU memory after the six engines: 1 MiB.

Every run directory holds `traces_*`, `batches_*`, `gpu_*`, `gpu_steps_*`,
`collector_*`, `workload.json`, `manifest.json`, `run_info.json`
(`vllm_stats_*` only from `runner/baseline_2` and `planned/` onward). The
`.log` files next to them are the drivers' stdout/stderr. Numbers are in
`docs/GPU_VALIDATION.md`.

Provenance: this session ran on a tarball upload without `.git`, with fixes that were uploaded before being committed, so its manifests carry no commit or fingerprint. The exact code is commit 9d13003 plus the fixes committed together with this evidence (the commit that added this directory). Runs made after that commit record `llmtrace_source_fingerprint`, the full commit, and `source.patch` when dirty.

Raw trace files (`*.jsonl`) for this session are not in git: they are the `2026-09-08-rtx-a5000-runner-plan-raw-traces.tar.gz` asset of the GitHub release [`evidence-2026-09`](https://github.com/Aagam-Bothara/llmtrace/releases/tag/evidence-2026-09); extract it at the repository root to restore them under this directory. Manifests, summaries and every derived output are here.
