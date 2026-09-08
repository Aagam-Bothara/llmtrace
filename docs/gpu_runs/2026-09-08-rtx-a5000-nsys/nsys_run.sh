set -u
cd /workspace/llmtrace
export VLLM_ENABLE_V1_MULTIPROCESSING=0
mkdir -p /workspace/nsys
WL="--num-short 40 --short-rate 40 --num-long 6 --long-every 0.2 --collection-interval 0.1 --enable-nvtx"
for cfg in baseline capped; do
  echo "=== profiling $cfg" >> /workspace/nsys/summary.txt
  nsys profile -t cuda,nvtx -o /workspace/nsys/$cfg --force-overwrite true \
    python experiments/mixed_prompts/run.py --engine vllm --config $cfg --model facebook/opt-125m --out /workspace/nsys/${cfg}_run $WL \
    > /workspace/nsys/${cfg}_profile.log 2>&1
  echo "$cfg profile exit=$?" >> /workspace/nsys/summary.txt
  nsys export --type sqlite -o /workspace/nsys/$cfg.sqlite --force-overwrite true /workspace/nsys/$cfg.nsys-rep > /workspace/nsys/${cfg}_export.log 2>&1
  echo "$cfg export exit=$?" >> /workspace/nsys/summary.txt
  python scripts/nsys_step_compare.py /workspace/nsys/$cfg.sqlite /workspace/nsys/${cfg}_run --json /workspace/nsys/${cfg}_compare.json > /workspace/nsys/${cfg}_compare.txt 2>&1
  echo "$cfg compare exit=$?" >> /workspace/nsys/summary.txt
done
echo ALL_DONE >> /workspace/nsys/summary.txt
