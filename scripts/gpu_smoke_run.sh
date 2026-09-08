#!/usr/bin/env bash
# One-shot GPU smoke run for llmtrace (e.g. on a RunPod pod). Linux + NVIDIA GPU only.
#
#   bash scripts/gpu_smoke_run.sh [MODEL] [OUT_DIR]
#
# Produces OUT_DIR/{A_multiproc,B_inproc,untraced}.log, the trace directories,
# the CPU test log, and an environment snapshot, then tars everything into
# OUT_DIR.tar.gz for download. Every step's exit code is recorded in summary.txt.
set -u
MODEL="${1:-facebook/opt-125m}"
OUT="${2:-$PWD/gpu_smoke_results}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"

run() {  # run <name> <command...>
  local name="$1"; shift
  echo "=== $name: $*" | tee -a "$SUMMARY"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$name exit=$rc" | tee -a "$SUMMARY"
  tail -n 40 "$OUT/$name.log"
  return $rc
}

{
  echo "date: $(date -u +%FT%TZ)"
  nvidia-smi
  python -c "import sys; print('python', sys.version)"
  python -c "import vllm; print('vllm', vllm.__version__)" 2>&1
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1
  pip freeze | grep -iE "^(vllm|torch|nvidia-ml-py|llmtrace)"
} > "$OUT/environment.txt" 2>&1
cat "$OUT/environment.txt"

run cpu_tests python -m pytest -rs -p no:cacheprovider  # pyproject addopts already has -q; -rs lists skips

# Phase A/B with the default multiprocess engine core (scheduler NOT visible).
unset VLLM_ENABLE_V1_MULTIPROCESSING
run A_multiproc python examples/vllm_smoke_test.py --model "$MODEL" --out "$OUT/traces_multiproc"

# Phase A/B with the in-process engine core (scheduler visible, batch metadata expected).
export VLLM_ENABLE_V1_MULTIPROCESSING=0
run B_inproc python examples/vllm_smoke_test.py --model "$MODEL" --out "$OUT/traces_inproc"

# Traced vs untraced wall time, 5 repeats each (report medians; do not quote a single number).
run untraced python examples/vllm_smoke_test.py --model "$MODEL" --no-trace --repeat 5
run traced_repeat python examples/vllm_smoke_test.py --model "$MODEL" --out "$OUT/traces_repeat" --repeat 5
unset VLLM_ENABLE_V1_MULTIPROCESSING

# Optional cross-check for the energy ledger: independent power log via nvidia-smi.
nvidia-smi --query-gpu=timestamp,index,power.draw --format=csv -lms 100 > "$OUT/nvidia_smi_power.csv" 2>&1 &
SMI=$!
export VLLM_ENABLE_V1_MULTIPROCESSING=0
run C_power_crosscheck python examples/vllm_smoke_test.py --model "$MODEL" --out "$OUT/traces_crosscheck"
kill $SMI 2>/dev/null

echo "--- summary ---"; cat "$SUMMARY"
tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "results: $OUT.tar.gz"
