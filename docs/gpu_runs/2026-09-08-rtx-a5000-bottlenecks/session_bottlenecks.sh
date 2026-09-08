#!/usr/bin/env bash
# Session 3a (opt-125m): two induced bottlenecks, each through findings -> plan -> run --plan -> decide.
#   bash session_bottlenecks.sh [MODEL] [OUT]
set -u
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
MODEL="${1:-facebook/opt-125m}"
OUT="${2:-/workspace/bn}"
WHICH="${3:-all}"   # all | queue | kv
REPO=/workspace/llmtrace
mkdir -p "$OUT"
S="$OUT/summary.txt"; : > "$S"
log() { echo "$*" | tee -a "$S"; }
cd "$REPO"
{ date -u +%FT%TZ; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  python -c "import sys,torch,vllm,transformers,llmtrace;print(sys.version.split()[0],'torch',torch.__version__,'vllm',vllm.__version__,'transformers',transformers.__version__,'llmtrace',llmtrace.__version__)"
} > "$OUT/environment.txt" 2>&1; cat "$OUT/environment.txt" | tee -a "$S"

loop() {  # name, workload, source config name, target, slo, source --set args...
  local name="$1" wl="$2" cfg="$3" target="$4" slo="$5"; shift 5
  local d="$OUT/$name"; mkdir -p "$d"
  log "=== $name: source run ($cfg: $*)"
  llmtrace run --workload "$wl" --engine vllm --model "$MODEL" --out "$d/source" --config-name "$cfg" "$@" > "$d/source.log" 2>&1
  log "source exit=$? :: $(grep -E '^\[/' "$d/source.log" | tail -1 | cut -c1-200)"
  grep -E "PROBLEM|telemetry:|FAILED" "$d/source.log" | tee -a "$S"
  llmtrace doctor "$d/source" > "$d/doctor_source.txt" 2>&1
  llmtrace findings "$d/source" --verbose --json "$d/findings_source.json" > "$d/findings_source.txt" 2>&1
  grep -E "^\[" "$d/findings_source.txt" | cut -c1-230 | tee -a "$S"
  llmtrace plan "$d/source" --repeats 2 --max-candidates 2 --json "$d/plan.json" > "$d/plan.txt" 2>&1; log "plan exit=$?"
  grep -E "^\[|^source|^baseline engine|NOT REPRODUCED|^skipped|^note" "$d/plan.txt" | cut -c1-230 | tee -a "$S"
  llmtrace run --workload "$wl" --plan "$d/plan.json" --out "$d/planned" > "$d/planned.log" 2>&1; log "run --plan exit=$?"
  grep -E "^\[/|PROBLEM|warning|FAILED" "$d/planned.log" | cut -c1-200 | tee -a "$S"
  local cfgs
  cfgs=$(python - "$d/plan.json" "$d/planned" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1])); base = sys.argv[2]
names = ["baseline"] + [c["name"] for c in p["candidates"]]
print(" ".join(f"--config {n}={base}/{n}/r0,{base}/{n}/r1" for n in names))
EOF
)
  llmtrace decide --target "$target" --slo "$slo" --exclude-class settle $cfgs --json "$d/decision.json" > "$d/decision.txt" 2>&1; log "decide exit=$?"
  cat "$d/decision.txt" | tee -a "$S"
  for n in $(python -c "import json;p=json.load(open('$d/plan.json'));print(' '.join(c['name'] for c in p['candidates']))"); do
    llmtrace findings "$d/planned/$n/r0" --json "$d/findings_$n.json" > "$d/findings_$n.txt" 2>&1
    log "--- findings on $n r0"; grep -E "^\[" "$d/findings_$n.txt" | cut -c1-200 | tee -a "$S"
  done
}

# 1. queue overload: 32-request bursts on an engine capped at 8 sequences
[ "$WHICH" = all ] || [ "$WHICH" = queue ] && loop queue "$OUT/w_queue.json" seqs8 "burst ttft_p95 <= 300ms" "burst: ttft <= 300ms, tpot <= 10ms" --set max_num_seqs=8

# 2. KV-cache pressure: 48 long-output requests with a tiny KV cache
[ "$WHICH" = all ] || [ "$WHICH" = kv ] && loop kv "$OUT/w_kv.json" mem06 "kv e2e_p95 <= 4000ms" "kv: e2e <= 4000ms" --set gpu_memory_utilization=0.06

nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$S"
log "ALL_DONE_$WHICH"
