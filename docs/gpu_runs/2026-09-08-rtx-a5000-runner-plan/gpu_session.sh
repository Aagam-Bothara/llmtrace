#!/usr/bin/env bash
# GPU session 2: generic runner vs experiment driver, plan -> run --plan -> decide on real vLLM, overhead matrix, doctor.
set -u
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
OUT=/workspace/gpu
REPO=/workspace/llmtrace
MODEL=facebook/opt-125m
mkdir -p "$OUT"
S="$OUT/summary.txt"; : > "$S"
log() { echo "$*" | tee -a "$S"; }
cd "$REPO"

{ date -u +%FT%TZ; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  python -c "import sys,torch,vllm,transformers,llmtrace;print(sys.version.split()[0],'torch',torch.__version__,'vllm',vllm.__version__,'transformers',transformers.__version__,'llmtrace',llmtrace.__version__)"
  git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo "git n/a"; } > "$OUT/environment.txt" 2>&1
cat "$OUT/environment.txt" | tee -a "$S"

log "=== doctor (environment)"
llmtrace doctor --json "$OUT/doctor_env.json" > "$OUT/doctor_env.txt" 2>&1; log "doctor exit=$?"

log "=== CPU suite on the GPU box"
python -m pytest -p no:cacheprovider > "$OUT/pytest.log" 2>&1; log "pytest exit=$? $(tail -1 "$OUT/pytest.log")"

log "=== workload template"
llmtrace workload template --output "$OUT/w.json" | tee -a "$S"
llmtrace workload preview "$OUT/w.json" --json "$OUT/w_preview.json" > /dev/null

# A. generic runner: baseline x2, capped x2 (interleaved) on real vLLM
log "=== A. llmtrace run --engine vllm"
for i in 0 1; do
  for cfg in baseline capped; do
    d="$OUT/runner/${cfg}_$i"
    if [ "$cfg" = capped ]; then extra=(--config-name capped --set long_prefill_token_threshold=256); else extra=(--config-name baseline); fi
    llmtrace run --workload "$OUT/w.json" --engine vllm --model "$MODEL" --out "$d" "${extra[@]}" > "$d.log" 2>&1
    log "runner $cfg $i exit=$? :: $(grep -E '^\[' "$d.log" | tail -1 | cut -c1-200)"
    grep -E "scheduler visible|PROBLEM|telemetry:" "$d.log" | tee -a "$S"
  done
done

# B. experiment driver on the same workload defaults, one each, for a like-for-like comparison of the two drivers
log "=== B. experiments/mixed_prompts/run.py"
for cfg in baseline capped; do
  d="$OUT/driver/${cfg}_0"
  python experiments/mixed_prompts/run.py --engine vllm --config "$cfg" --model "$MODEL" --out "$d" > "$d.log" 2>&1
  log "driver $cfg exit=$? :: $(grep -E 'effective scheduler config' "$d.log" | cut -c1-200)"
done

log "=== doctor (run dir)"
llmtrace doctor "$OUT/runner/baseline_0" --json "$OUT/doctor_run.json" > "$OUT/doctor_run.txt" 2>&1; log "doctor run exit=$?"
head -20 "$OUT/doctor_run.txt" | tee -a "$S"

log "=== findings (runner baseline_0)"
llmtrace findings "$OUT/runner/baseline_0" --verbose --json "$OUT/findings_runner_baseline_0.json" > "$OUT/findings_runner_baseline_0.txt" 2>&1
grep -E "^\[" "$OUT/findings_runner_baseline_0.txt" | tee -a "$S"

log "=== analyze.py: runner vs driver runs (same analysis code on both)"
python experiments/mixed_prompts/analyze.py "$OUT/runner/baseline_0" --compare "$OUT/runner/capped_0" --json "$OUT/analysis_runner_0.json" > "$OUT/analysis_runner_0.txt" 2>&1
python experiments/mixed_prompts/analyze.py "$OUT/runner/baseline_1" --compare "$OUT/runner/capped_1" --json "$OUT/analysis_runner_1.json" > "$OUT/analysis_runner_1.txt" 2>&1
python experiments/mixed_prompts/analyze.py "$OUT/driver/baseline_0" --compare "$OUT/driver/capped_0" --json "$OUT/analysis_driver_0.json" > "$OUT/analysis_driver_0.txt" 2>&1
for f in analysis_runner_0 analysis_runner_1 analysis_driver_0; do
  log "--- $f"; grep -E "^verdict|short_ttft_ms_p95|short_itl_ms_max|long_ttft_ms_p50|long_chunk" "$OUT/$f.txt" | tee -a "$S"
done

log "=== decide: runner repeats + driver repeat as a third repeat of each config"
llmtrace decide --target "short ttft_p95 <= 5ms" --slo "short: ttft <= 5ms, tpot <= 3ms" --exclude-class settle --exclude-class other \
  --config baseline="$OUT/runner/baseline_0,$OUT/runner/baseline_1,$OUT/driver/baseline_0" \
  --config capped="$OUT/runner/capped_0,$OUT/runner/capped_1,$OUT/driver/capped_0" \
  --json "$OUT/decision_5ms.json" > "$OUT/decision_5ms.txt" 2>&1; log "decide exit=$?"
cat "$OUT/decision_5ms.txt" | tee -a "$S"

# C. plan -> run --plan -> decide on real vLLM
log "=== C. plan from runner baseline_0"
llmtrace plan "$OUT/runner/baseline_0" --repeats 2 --max-candidates 2 --json "$OUT/plan.json" > "$OUT/plan.txt" 2>&1; log "plan exit=$?"
cat "$OUT/plan.txt" | tee -a "$S"
llmtrace run --workload "$OUT/w.json" --plan "$OUT/plan.json" --out "$OUT/planned" > "$OUT/planned.log" 2>&1; log "run --plan exit=$?"
grep -E "^\[|PROBLEM|warning|compare with" "$OUT/planned.log" | cut -c1-220 | tee -a "$S"
CFGS=$(python - <<'EOF'
import json; p = json.load(open("/workspace/gpu/plan.json")); names = ["baseline"] + [c["name"] for c in p["candidates"]]
print(" ".join(f"--config {n}=/workspace/gpu/planned/{n}/r0,/workspace/gpu/planned/{n}/r1" for n in names))
EOF
)
llmtrace decide --target "short ttft_p95 <= 5ms" --slo "short: ttft <= 5ms, tpot <= 3ms" --exclude-class settle $CFGS \
  --json "$OUT/decision_planned.json" > "$OUT/decision_planned.txt" 2>&1; log "decide planned exit=$?"
cat "$OUT/decision_planned.txt" | tee -a "$S"

# D. overhead matrix (from /root so the checkout does not shadow the package)
log "=== D. overhead matrix (gpu_step_timing on/off)"
( cd /root && python "$REPO/scripts/gpu_overhead.py" --model "$MODEL" --repeat 3 --gpu-step-timing both \
    --out "$OUT/overhead_traces" --json "$OUT/overhead.json" ) > "$OUT/overhead.log" 2>&1; log "overhead exit=$?"
grep -E "median|traced/untraced|gpu_step_timing on/off|OVERHEAD_DONE" "$OUT/overhead.log" | tee -a "$S"

rm -rf "$OUT/overhead_traces"
log "ALL_DONE"
