#!/usr/bin/env bash
# 7B-class campaign (Linux, NVIDIA, >= 40 GB per GPU): CUDA spans at TP=1, then the mixed-prompt
# experiment at TP=1 (in-process, spans + batch metadata) and TP=2 (in-process; spans are refused
# for the out-of-process executor and the run must say so).
#   bash experiments/mixed_prompts/run_7b.sh [MODEL] [OUT_DIR]
set -u
MODEL="${1:-Qwen/Qwen2.5-7B}"
OUT="${2:-$PWD/exp_7b}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"; : > "$SUMMARY"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HOME="${HF_HOME:-/workspace/hf}"
# Slower steps than opt-125m: 10 short/s (~30 in flight at 128 tokens), 12 long prompts every 0.4 s.
WL="--num-short 120 --short-rate 10 --num-long 12 --long-every 0.4 --collection-interval 0.1"
{
  date -u +%FT%TZ; nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
  python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
} > "$OUT/environment.txt" 2>&1

echo "=== smoke TP=1 in-process" | tee -a "$SUMMARY"
CUDA_VISIBLE_DEVICES=0 python examples/vllm_smoke_test.py --model "$MODEL" --out "$OUT/smoke_tp1" --num-prompts 8 --max-tokens 32 \
  > "$OUT/smoke_tp1.log" 2>&1
echo "smoke_tp1 exit=$?" | tee -a "$SUMMARY"
grep -E "RESULT|gpu span p50|executor visible" "$OUT/smoke_tp1.log" | tail -4 | tee -a "$SUMMARY"

for i in 0 1 2; do
  for cfg in baseline capped; do
    d="$OUT/tp1_${cfg}_$i"
    echo "=== TP=1 $cfg run $i" | tee -a "$SUMMARY"
    CUDA_VISIBLE_DEVICES=0 python experiments/mixed_prompts/run.py --engine vllm --config "$cfg" --model "$MODEL" --out "$d" $WL \
      > "$d.log" 2>&1
    echo "tp1 $cfg run $i exit=$?" | tee -a "$SUMMARY"
    grep -E "effective scheduler config|\"steps\"|\"wall_s\"|PROBLEM|FAILED" "$d.log" | tee -a "$SUMMARY"
  done
done

for i in 0 1; do
  for cfg in baseline capped; do
    d="$OUT/tp2_${cfg}_$i"
    echo "=== TP=2 $cfg run $i" | tee -a "$SUMMARY"
    python experiments/mixed_prompts/run.py --engine vllm --config "$cfg" --model "$MODEL" --out "$d" $WL \
      --engine-kwargs '{"tensor_parallel_size": 2}' > "$d.log" 2>&1
    echo "tp2 $cfg run $i exit=$?" | tee -a "$SUMMARY"
    grep -E "effective scheduler config|\"steps\"|\"wall_s\"|PROBLEM|FAILED" "$d.log" | tee -a "$SUMMARY"
  done
done

echo "=== analysis" | tee -a "$SUMMARY"
for i in 0 1 2; do
  python experiments/mixed_prompts/analyze.py "$OUT/tp1_baseline_$i" --compare "$OUT/tp1_capped_$i" --json "$OUT/analysis_tp1_$i.json" \
    > "$OUT/analysis_tp1_$i.txt" 2>&1
  grep -E "^verdict|GPU span|short TTFT" "$OUT/analysis_tp1_$i.txt" | tee -a "$SUMMARY"
done
for i in 0 1; do
  python experiments/mixed_prompts/analyze.py "$OUT/tp2_baseline_$i" --compare "$OUT/tp2_capped_$i" --json "$OUT/analysis_tp2_$i.json" \
    > "$OUT/analysis_tp2_$i.txt" 2>&1
  grep -E "^verdict|GPU span|short TTFT" "$OUT/analysis_tp2_$i.txt" | tee -a "$SUMMARY"
done
python -m llmtrace.cli findings "$OUT/tp1_baseline_0" > "$OUT/findings_tp1_baseline_0.txt" 2>&1
python -m llmtrace.cli findings "$OUT/tp2_baseline_0" > "$OUT/findings_tp2_baseline_0.txt" 2>&1
python -m llmtrace.cli decide --target "short ttft_p95 <= 150ms" \
  --config "tp1_baseline=$OUT/tp1_baseline_0,$OUT/tp1_baseline_1,$OUT/tp1_baseline_2" \
  --config "tp1_capped=$OUT/tp1_capped_0,$OUT/tp1_capped_1,$OUT/tp1_capped_2" \
  --config "tp2_baseline=$OUT/tp2_baseline_0,$OUT/tp2_baseline_1" \
  --config "tp2_capped=$OUT/tp2_capped_0,$OUT/tp2_capped_1" \
  --json "$OUT/decision.json" > "$OUT/decision.txt" 2>&1
tail -n 8 "$OUT/decision.txt" | tee -a "$SUMMARY"
tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "results: $OUT.tar.gz" | tee -a "$SUMMARY"
echo ALL_DONE >> "$SUMMARY"
