"""Basic usage example for llmtrace."""

import asyncio
from vllm import LLM, SamplingParams
from llmtrace import LLMTracer


async def main():
    # Initialize llmtrace
    tracer = LLMTracer(
        output_dir="./traces",
        gpu_sample_interval_ms=100,
        enable_energy_attribution=True,
    )

    # Initialize vLLM
    llm = LLM(model="facebook/opt-125m")  # Small model for demo

    # Instrument the engine
    tracer.instrument_engine(llm.llm_engine)

    # Run inference
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "To be or not to be,",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

    print("Running inference with llmtrace...")
    outputs = llm.generate(prompts, sampling_params)

    # Print outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Stop tracer
    await tracer.stop()

    # Analyze traces
    print("\nAnalyzing traces...")
    analysis = await tracer.analyze()

    # Print analysis
    tracer.print_analysis(analysis)

    # Get output files
    output_files = tracer.get_output_files()
    print(f"\nTrace files written to:")
    for data_type, filepath in output_files.items():
        print(f"  {data_type}: {filepath}")


if __name__ == "__main__":
    asyncio.run(main())
