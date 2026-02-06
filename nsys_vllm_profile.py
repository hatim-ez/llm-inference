import multiprocessing as mp
import torch
from vllm import LLM, SamplingParams

def main():
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0))

    llm = LLM(
        model="/home/ubuntu/models/llama-3.2-3b",
        max_model_len=4096,
        disable_log_stats=True,
    )
    params = SamplingParams(max_tokens=100)

    llm.generate(["warmup"], params)
    llm.generate(["Hey what a wonderful day to build inference servers"], params)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
