#!/usr/bin/env bash
# Controlled GPU run of the mixed-prompt experiment: baseline vs capped, REPEAT times each, interleaved.
#   bash experiments/mixed_prompts/run_gpu.sh [MODEL] [OUT_DIR] [REPEAT]
set -u
MODEL="${1:-facebook/opt-125m}"
OUT="${2:-$PWD/exp_gpu}"
REPEAT="${3:-3}"
COLLECT="${4:-0.1}"   # tracer collection interval (s); see run.py --collection-interval
mkdir -p "$OUT"
export VLLM_ENABLE_V1_MULTIPROCESSING=0   # in-process scheduler: batch metadata is required
SUMMARY="$OUT/summary.txt"; : > "$SUMMARY"
{
  date -u +%FT%TZ; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
} > "$OUT/environment.txt" 2>&1
for i in $(seq 0 $((REPEAT - 1))); do
  for cfg in baseline capped; do
    d="$OUT/${cfg}_$i"
    echo "=== $cfg run $i" | tee -a "$SUMMARY"
    python experiments/mixed_prompts/run.py --engine vllm --config "$cfg" --model "$MODEL" --out "$d" \
        --collection-interval "$COLLECT" > "$d.log" 2>&1
    echo "$cfg run $i exit=$?" | tee -a "$SUMMARY"
    grep -E "effective scheduler config|steps|wall_s|PROBLEM" "$d.log" | tee -a "$SUMMARY"
  done
done
echo "=== analysis" | tee -a "$SUMMARY"
for i in $(seq 0 $((REPEAT - 1))); do
  python experiments/mixed_prompts/analyze.py "$OUT/baseline_$i" --compare "$OUT/capped_$i" --json "$OUT/analysis_$i.json" \
    > "$OUT/analysis_$i.txt" 2>&1
  grep -E "^verdict|short_tpot_ms_p95|short_ttft_ms_p95|long_ttft_ms_p50" "$OUT/analysis_$i.txt" | tee -a "$SUMMARY"
done
tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "results: $OUT.tar.gz"
