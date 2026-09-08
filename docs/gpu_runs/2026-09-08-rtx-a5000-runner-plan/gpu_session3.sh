#!/usr/bin/env bash
# GPU session 2, stage 3: after the fixes (disable_log_stats=False, one process per engine, teardown).
set -u
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
OUT=/workspace/gpu
REPO=/workspace/llmtrace
MODEL=facebook/opt-125m
S="$OUT/summary3.txt"; : > "$S"
log() { echo "$*" | tee -a "$S"; }
cd "$REPO"
rm -rf "$OUT/planned" "$OUT/planned.log"

log "=== A2. runner baseline_2 with vLLM stats logging enabled (disable_log_stats=False default)"
d="$OUT/runner/baseline_2"
llmtrace run --workload "$OUT/w.json" --engine vllm --model "$MODEL" --out "$d" --config-name baseline > "$d.log" 2>&1
log "runner baseline_2 exit=$?"
grep -E "^\[/|scheduler visible|PROBLEM|telemetry:|FAILED" "$d.log" | cut -c1-220 | tee -a "$S"
llmtrace doctor "$d" > "$OUT/doctor_run_2.txt" 2>&1; grep -E "vLLM per-step stats|telemetry|tracer health" "$OUT/doctor_run_2.txt" | tee -a "$S"
python - <<'EOF' | tee -a /workspace/gpu/summary3.txt
from llmtrace import io
s = io.load_vllm_stats(["/workspace/gpu/runner/baseline_2"])
print(f"vllm_stats records: {len(s)}; with kv usage: {sum(1 for r in s if r.kv_cache_usage is not None)}; "
      f"finished-request stats with queued_time: {sum(1 for r in s for f in r.finished_requests if f.queued_time_s is not None)}")
EOF

log "=== C2. plan -> run --plan (one process per engine) -> decide"
llmtrace plan "$OUT/runner/baseline_2" --repeats 2 --max-candidates 2 --json "$OUT/plan.json" > "$OUT/plan.txt" 2>&1; log "plan exit=$?"
grep -E "^\[|^source|NOT REPRODUCED|^skipped" "$OUT/plan.txt" | cut -c1-200 | tee -a "$S"
llmtrace run --workload "$OUT/w.json" --plan "$OUT/plan.json" --out "$OUT/planned" > "$OUT/planned.log" 2>&1; log "run --plan exit=$?"
grep -E "^\[/|PROBLEM|warning|FAILED|telemetry:" "$OUT/planned.log" | cut -c1-220 | tee -a "$S"
CFGS=$(python - <<'EOF'
import json; p = json.load(open("/workspace/gpu/plan.json")); names = ["baseline"] + [c["name"] for c in p["candidates"]]
print(" ".join(f"--config {n}=/workspace/gpu/planned/{n}/r0,/workspace/gpu/planned/{n}/r1" for n in names))
EOF
)
llmtrace decide --target "short ttft_p95 <= 5ms" --slo "short: ttft <= 5ms, tpot <= 3ms" --exclude-class settle $CFGS \
  --json "$OUT/decision_planned.json" > "$OUT/decision_planned.txt" 2>&1; log "decide planned exit=$?"
cat "$OUT/decision_planned.txt" | tee -a "$S"
for n in baseline cap1024 cap512; do
  python experiments/mixed_prompts/analyze.py "$OUT/planned/baseline/r0" --compare "$OUT/planned/$n/r0" --json "$OUT/analysis_planned_$n.json" > "$OUT/analysis_planned_$n.txt" 2>&1
  log "--- planned $n vs baseline r0"; grep -E "^verdict|short_ttft_ms_p95|short_itl_ms_max|long_ttft_ms_p50" "$OUT/analysis_planned_$n.txt" | cut -c1-200 | tee -a "$S"
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$S"
log "ALL_DONE3"
