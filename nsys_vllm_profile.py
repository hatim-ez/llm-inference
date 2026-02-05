from vllm import LLM, SamplingParams

llm = LLM(
    model="/home/ubuntu/models/llama-3.2-3b",
    max_model_len=4096,            # reduce KV cache requirement
    gpu_memory_utilization=0.90,   # optional: allow more GPU usage
)
params = SamplingParams(max_tokens=100)

llm.generate(["warmup"], params)
llm.generate(["Hey what a wonderful day to build inference servers"], params)
