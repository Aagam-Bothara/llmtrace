# GPU run 2026-09-08, RunPod RTX 4000 Ada: AsyncLLM smoke test

Environment: NVIDIA RTX 4000 Ada Generation, driver 550.127.05, Python 3.11,
vLLM 0.11.0, torch 2.8.0+cu128, transformers 4.57.6, `facebook/opt-125m`.
Script: `examples/vllm_async_smoke_test.py` (6 concurrent `AsyncLLM.generate()`
streams of 32 tokens with `ignore_eos`, plus one stream the client stops
reading after 4 tokens; `AsyncLLM.from_engine_args(..., stat_loggers=[tracer.stat_logger_factory()])`
and `tracer.instrument_async_engine(engine)`).

`async_smoke.log`: ALL CHECKS PASSED. `traces/`: the recorded `traces_*`,
`gpu_*`, `vllm_stats_*`, `collector_*` files (no `batches_*`: the engine core
is out of process). `install.log`, `environment.txt`.
