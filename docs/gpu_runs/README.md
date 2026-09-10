# GPU evidence

Each session has a directory containing its driver scripts, environment,
manifests, summaries and reports. The measurements in
[GPU validation](../GPU_VALIDATION.md) come from these records.

## Download the raw traces

Raw JSONL files are available in the
[evidence-2026-09 release](https://github.com/Aagam-Bothara/llmtrace/releases/tag/evidence-2026-09):
one `<session>-raw-traces.tar.gz` archive per session, plus `SHA256SUMS`.
The ten archives contain about 290 MB of uncompressed data.

From the repository root:

```bash
gh release download evidence-2026-09 --repo Aagam-Bothara/llmtrace \
    --pattern '*-raw-traces.tar.gz' --pattern SHA256SUMS
sha256sum -c SHA256SUMS
tar xzf 2026-09-09-a100-qwen2.5-7b-clean-repeats-raw-traces.tar.gz
llmtrace findings docs/gpu_runs/2026-09-09-a100-qwen2.5-7b-clean-repeats/queue/source
```

You can also download the assets from the release page. Extracting an archive
restores `docs/gpu_runs/<session>/**/*.jsonl`.

## Verification

On September 9, 2026, all ten archives and `SHA256SUMS` were downloaded
without authentication and matched the [committed checksums](SHA256SUMS).
All 588 archived JSONL files also matched commit `6840592` byte-for-byte.
The raw files remain in Git history up to that commit.

The archives contain request traces, batch records, power samples, CUDA
spans, vLLM stats and collector timings. Historical reports reflect the code
used in each session. Current checks may require explicit GPU selection or
withhold recommendations for older records with missing evidence.

## Sessions

| Session | What it established |
|---------|---------------------|
| `2026-09-08-rtx-a5000` | first smoke tests, NVML fields, energy integral vs an `nvidia-smi` stream, overhead on opt-125m |
| `2026-09-08-rtx-a4500-mixed-prompts` | the long-prompt interference experiment, four run sets including the collector-interval and warm-up artifacts |
| `2026-09-08-rtx-4000-ada-cuda-spans` | CUDA-event GPU span per step |
| `2026-09-08-rtx-4000-ada-asyncllm` | `AsyncLLM` request-level tracing and the stats hook |
| `2026-09-08-a100-qwen2.5-7b` | the interference experiment on Qwen2.5-7B at TP=1 and TP=2 |
| `2026-09-08-rtx-a5000-nsys` | Nsight Systems cross-check of the step spans (profiles not committed: they embed the pod environment) |
| `2026-09-08-rtx-a5000-runner-plan` | generic runner vs experiment driver, first plan loop, sync-engine stats, overhead with step timing |
| `2026-09-08-rtx-a5000-bottlenecks` | queue overload and KV pressure induced on opt-125m; the planner ranking fix |
| `2026-09-08-a100-qwen2.5-7b-overhead-queue` | overhead matrix and queue overload on Qwen2.5-7B |
| `2026-09-09-a100-qwen2.5-7b-clean-repeats` | queue overload (4 repeats) and KV pressure (3 repeats) on Qwen2.5-7B from a fingerprinted clean commit |
