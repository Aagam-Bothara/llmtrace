"""Basic llmtrace usage with vLLM 0.11.0 (requires Linux + NVIDIA GPU; not yet validated on hardware).

    pip install -e ".[vllm]"
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python examples/basic_usage.py   # =0 exposes scheduler batch metadata
"""

from vllm import LLM, SamplingParams

from llmtrace import LLMTracer


def main() -> None:
    tracer = LLMTracer(output_dir="./traces", gpu_sample_interval_ms=100)

    llm = LLM(model="facebook/opt-125m")
    tracer.instrument_engine(llm.llm_engine)  # patches add_request/step/abort_request, starts collection

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "To be or not to be,",
    ]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50))
    for output in outputs:
        print(f"{output.prompt!r} -> {output.outputs[0].text!r}")

    tracer.stop()  # restores the engine, drains buffers, flushes files
    print("health:", tracer.health())

    analysis = tracer.analyze()
    tracer.print_analysis(analysis)
    print("files:", tracer.get_output_files())


if __name__ == "__main__":
    main()
