# GPU run 2026-09-08, RunPod RTX A5000

Raw evidence for the results table in `docs/GPU_VALIDATION.md`.

* `smoke/`: `scripts/gpu_smoke_run.sh` output (`facebook/opt-125m`, 8 prompts x 32 tokens):
  CPU suite, multiprocess and in-process smoke runs, traced/untraced repeats.
* `long/`: 64 prompts x 256 tokens. `untraced.log`/`traced.log` from the smoke
  test, `overhead.json`/`overhead.log` from the interleaved overhead script
  (`scripts/gpu_overhead.py`), `nvidia_smi_power.csv` (independent 50 ms power
  log) and `crosscheck.py`, the script that compared llmtrace's device energy
  against it. `install.log` shows the transformers 5.x failure and the pin.

Trace files (`traces_*.jsonl`, `batches_*.jsonl`, `gpu_*.jsonl`) are
committed under `smoke/traces_*/` and `long/traces/run*/` so the energy and
timing numbers can be recomputed from the recorded data.

Cross-check scripts in `long/`: `crosscheck.py` is the version run on the pod
(flawed: unequal integration boundaries, kept for the record);
`crosscheck_bracketed.py` integrates both streams over identical bracketed
boundaries and produced the table in `docs/GPU_VALIDATION.md`.

Code tested: `llmtrace/`, `examples/`, `scripts/gpu_smoke_run.sh` as in commit
`98c0cd7`; `tests/` as in `98c0cd7` except the later pyarrow skip fix;
`long/overhead.py` is the exact overhead script that ran (`/root/overhead.py`).
`smoke/cpu_tests.log` shows 83 tests because `tests/test_collection.py` was
skipped as a module on the pod (no pyarrow there); see `docs/GPU_VALIDATION.md`.
