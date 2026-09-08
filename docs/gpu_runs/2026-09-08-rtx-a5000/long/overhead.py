import os, sys, time, statistics, json
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING","0")
from vllm import LLM, SamplingParams
from llmtrace import LLMTracer, TracerConfig
from llmtrace.vllm_helpers import run_engine_with_timing
N=64; MAX=256; REP=5
llm=LLM(model="facebook/opt-125m", gpu_memory_utilization=0.5); eng=llm.llm_engine
prompts=[f"Prompt number {i}: tell me something about the number {i}." for i in range(N)]
sp=SamplingParams(temperature=0.0, max_tokens=MAX)
llm.generate(prompts[:4], sp)
res={"untraced_generate":[], "untraced_engine_loop":[], "traced_generate":[], "traced_engine_loop":[]}
def traced(fn, name, i):
    tr=LLMTracer(TracerConfig(output_dir=f"/workspace/overhead_traces/{name}_{i}", gpu_sampler={"sample_interval_ms":50}))
    tr.instrument_engine(eng); t0=time.perf_counter(); fn(); w=time.perf_counter()-t0; tr.stop()
    h=tr.health(); assert h["instrumentation"]["instrumentation_errors"]==0 and h["instrumentation"]["active_requests"]==0
    return w
for i in range(REP):
    t0=time.perf_counter(); llm.generate(prompts, sp); res["untraced_generate"].append(time.perf_counter()-t0)
    t0=time.perf_counter(); run_engine_with_timing(eng, prompts, sp); res["untraced_engine_loop"].append(time.perf_counter()-t0)
    res["traced_generate"].append(traced(lambda: llm.generate(prompts, sp), "gen", i))
    res["traced_engine_loop"].append(traced(lambda: run_engine_with_timing(eng, prompts, sp), "loop", i))
for k,v in res.items(): print(f"{k:22} median {statistics.median(v):.4f}s  all {[round(x,4) for x in v]}")
json.dump(res, open("/workspace/overhead.json","w"))
print("OVERHEAD_DONE")
