#!/usr/bin/env bash
# Session 3b (Qwen2.5-7B): overhead matrix with GPU step timing on/off, then the queue-overload loop on the 7B model.
set -u
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
MODEL=Qwen/Qwen2.5-7B
OUT=/workspace/b7
REPO=/workspace/llmtrace
mkdir -p "$OUT"
S="$OUT/summary.txt"; : > "$S"
log() { echo "$*" | tee -a "$S"; }
cd "$REPO"
{ date -u +%FT%TZ; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  python -c "import sys,torch,vllm,transformers,llmtrace;print(sys.version.split()[0],'torch',torch.__version__,'vllm',vllm.__version__,'transformers',transformers.__version__,'llmtrace',llmtrace.__version__)"
} > "$OUT/environment.txt" 2>&1; cat "$OUT/environment.txt" | tee -a "$S"

log "=== download model"
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', allow_patterns=['*.json','*.safetensors','*.txt','*.model'])" > "$OUT/dl.log" 2>&1; log "download exit=$?"

log "=== overhead matrix on $MODEL (64 x 256 tokens, 3 repeats, gpu_step_timing both)"
( cd /root && python "$REPO/scripts/gpu_overhead.py" --model "$MODEL" --num-prompts 64 --max-tokens 256 --repeat 3 --gpu-step-timing both \
    --out "$OUT/overhead_traces" --json "$OUT/overhead.json" ) > "$OUT/overhead.log" 2>&1; log "overhead exit=$?"
grep -E "median|traced/untraced|gpu_step_timing on/off|OVERHEAD_DONE|Error|error" "$OUT/overhead.log" | tail -12 | tee -a "$S"
rm -rf "$OUT/overhead_traces"

log "=== queue overload loop on $MODEL"
cp /workspace/w_queue.json /workspace/w_kv.json "$OUT/bn/" 2>/dev/null || { mkdir -p "$OUT/bn"; cp /workspace/w_queue.json /workspace/w_kv.json "$OUT/bn/"; }
bash /workspace/session_bottlenecks.sh "$MODEL" "$OUT/bn" queue > "$OUT/bn_driver.log" 2>&1; log "queue loop exit=$?"
cat "$OUT/bn/summary.txt" | tee -a "$S"
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$S"
log "ALL_DONE"
