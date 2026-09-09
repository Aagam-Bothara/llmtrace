#!/usr/bin/env bash
# Session 4 (Qwen2.5-7B, A100 80GB): reproducible validation set from a clean commit with independent repeats.
#   queue overload (4 repeats per configuration) and KV-cache pressure (3 repeats), each: source -> findings -> plan -> run --plan -> decide.
set -u
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
MODEL=Qwen/Qwen2.5-7B
OUT=/workspace/s4
REPO=/workspace/llmtrace
mkdir -p "$OUT"
S="$OUT/summary.txt"; : > "$S"
log() { echo "$*" | tee -a "$S"; }
cd "$REPO"
{ date -u +%FT%TZ; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  python -c "import sys,torch,vllm,transformers,llmtrace;print(sys.version.split()[0],'torch',torch.__version__,'vllm',vllm.__version__,'transformers',transformers.__version__,'llmtrace',llmtrace.__version__)"
  cat /workspace/SOURCE_COMMIT 2>/dev/null
  python -c "from llmtrace.provenance import source_fingerprint; print('fingerprint', source_fingerprint())"
} > "$OUT/environment.txt" 2>&1; cat "$OUT/environment.txt" | tee -a "$S"
llmtrace doctor --json "$OUT/doctor_env.json" > "$OUT/doctor_env.txt" 2>&1
python -m pytest -p no:cacheprovider > "$OUT/pytest.log" 2>&1; log "pytest exit=$? $(tail -1 "$OUT/pytest.log")"

log "=== download model"
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', allow_patterns=['*.json','*.safetensors','*.txt','*.model'])" > "$OUT/dl.log" 2>&1; log "download exit=$?"

loop() {  # name, workload, source config name, repeats, target, slo, source --set args...
  local name="$1" wl="$2" cfg="$3" reps="$4" target="$5" slo="$6"; shift 6
  local d="$OUT/$name"; mkdir -p "$d"
  log "=== $name: source run ($cfg: $*), $reps repeats per configuration"
  llmtrace run --workload "$wl" --engine vllm --model "$MODEL" --out "$d/source" --config-name "$cfg" "$@" > "$d/source.log" 2>&1
  log "source exit=$? :: $(grep -E '^\[/' "$d/source.log" | tail -1 | cut -c1-200)"
  grep -E "PROBLEM|telemetry:|FAILED" "$d/source.log" | tee -a "$S"
  llmtrace doctor "$d/source" > "$d/doctor_source.txt" 2>&1
  llmtrace findings "$d/source" --verbose --json "$d/findings_source.json" > "$d/findings_source.txt" 2>&1
  grep -E "^\[" "$d/findings_source.txt" | cut -c1-230 | tee -a "$S"
  llmtrace plan "$d/source" --repeats "$reps" --max-candidates 2 --json "$d/plan.json" > "$d/plan.txt" 2>&1; log "plan exit=$?"
  grep -E "^\[|^source|NOT REPRODUCED|^skipped" "$d/plan.txt" | cut -c1-230 | tee -a "$S"
  llmtrace run --workload "$wl" --plan "$d/plan.json" --out "$d/planned" > "$d/planned.log" 2>&1; log "run --plan exit=$?"
  grep -E "^\[/|PROBLEM|warning|FAILED" "$d/planned.log" | cut -c1-200 | tee -a "$S"
  local cfgs
  cfgs=$(python - "$d/plan.json" "$d/planned" "$reps" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1])); base = sys.argv[2]; reps = int(sys.argv[3])
names = ["baseline"] + [c["name"] for c in p["candidates"]]
print(" ".join(f"--config {n}=" + ",".join(f"{base}/{n}/r{i}" for i in range(reps)) for n in names))
EOF
)
  llmtrace decide --target "$target" --slo "$slo" --exclude-class settle --min-repeats 3 $cfgs --json "$d/decision.json" > "$d/decision.txt" 2>&1; log "decide exit=$?"
  cat "$d/decision.txt" | tee -a "$S"
  for n in $(python -c "import json;p=json.load(open('$d/plan.json'));print(' '.join(c['name'] for c in p['candidates']))"); do
    llmtrace findings "$d/planned/$n/r0" --json "$d/findings_$n.json" > "$d/findings_$n.txt" 2>&1
    log "--- findings on $n r0"; grep -E "^\[" "$d/findings_$n.txt" | cut -c1-200 | tee -a "$S"
  done
}

loop queue /workspace/w_queue.json seqs8 4 "burst ttft_p95 <= 3000ms" "burst: ttft <= 3000ms, tpot <= 40ms" --set max_num_seqs=8
loop kv /workspace/w_kv7b.json mem25 3 "kv e2e_p95 <= 60000ms" "kv: e2e <= 60000ms" --set gpu_memory_utilization=0.25

nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$S"
log "ALL_DONE"
