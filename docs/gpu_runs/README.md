# GPU evidence

One directory per GPU session, each with its own README, the exact driver
script, `environment.txt`, every manifest (`manifest.json`, `run_info.json`,
`workload.json`), and every derived output (`analysis_*`, `findings_*`,
`plan.*`, `decision.*`, `doctor_*`, summaries and logs). The numbers quoted in
`docs/GPU_VALIDATION.md` come from those derived files.

The raw per-request, per-step and telemetry records (`traces_*`, `batches_*`,
`gpu_*`, `gpu_steps_*`, `vllm_stats_*`, `collector_*` JSONL files, about 290 MB
across the sessions) are not in git. They are attached to the GitHub release
[`evidence-2026-09`](https://github.com/Aagam-Bothara/llmtrace/releases/tag/evidence-2026-09) as one archive per session, named
`<session>-raw-traces.tar.gz`. To put a session's raw data back where the
manifests expect it, from the repository root:

```bash
gh release download evidence-2026-09 --repo Aagam-Bothara/llmtrace \
    --pattern '*-raw-traces.tar.gz' --pattern SHA256SUMS
sha256sum -c SHA256SUMS
tar xzf 2026-09-09-a100-qwen2.5-7b-clean-repeats-raw-traces.tar.gz   # restores docs/gpu_runs/<session>/**/*.jsonl
llmtrace findings docs/gpu_runs/2026-09-09-a100-qwen2.5-7b-clean-repeats/queue/source
```

Every archive was produced from the committed tree with `git ls-files
'<session>/**/*.jsonl' | tar czf ... -T -`, so the restored files are the
ones the derived outputs were computed from. Commits before this split
(up to `6840592`) still contain the raw files in history.

Published and verified on September 9, 2026: all ten archives and `SHA256SUMS`
were downloaded anonymously from the public release and checked against the
[committed SHA-256 checksums](SHA256SUMS). All 588 archived JSONL files were
also checked byte-for-byte against commit `6840592`. You can download assets
directly from the release page if you do not use the GitHub CLI.

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
