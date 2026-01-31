#!/usr/bin/env python3
"""Benchmark LLM inference performance."""

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkResult:
    """Benchmark result for a single configuration."""

    config_name: str
    batch_size: int
    num_prompts: int
    total_time: float
    throughput: float  # requests/second
    avg_latency: float
    p50_latency: float
    p90_latency: float
    p95_latency: float
    p99_latency: float
    tokens_generated: int
    tokens_per_second: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None


def get_gpu_stats() -> Optional[dict]:
    """Get current GPU statistics."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return {
                "memory_used_mb": float(parts[0]),
                "memory_total_mb": float(parts[1]),
                "gpu_utilization": float(parts[2]),
            }
    except Exception:
        pass
    return None


def run_benchmark(
    model_path: str,
    batch_sizes: list[int],
    num_prompts: int,
    max_tokens: int,
    warmup_runs: int = 3,
    gpu_memory_utilization: float = 0.90,
) -> list[BenchmarkResult]:
    """Run inference benchmarks.

    Args:
        model_path: Path to model weights.
        batch_sizes: List of batch sizes to test.
        num_prompts: Number of prompts per batch.
        max_tokens: Maximum tokens to generate.
        warmup_runs: Number of warmup runs.
        gpu_memory_utilization: GPU memory utilization.

    Returns:
        List of benchmark results.
    """
    from vllm import LLM, SamplingParams

    print("Loading model...")
    start_load = time.time()

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")

    # Test prompts of varying lengths
    test_prompts = [
        "Explain machine learning",
        "What is deep learning and how does it work?",
        "Describe the process of training a neural network from scratch",
        "Write a detailed explanation of transformer architecture",
    ]

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
    )

    results = []

    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Testing batch size: {batch_size}")
        print(f"{'='*60}")

        # Prepare prompts
        prompts = [test_prompts[i % len(test_prompts)] for i in range(batch_size)]

        # Warmup
        print(f"Warming up ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            _ = llm.generate(prompts[:1], sampling_params)

        # Benchmark
        print(f"Running benchmark ({num_prompts} prompts)...")
        latencies = []
        total_tokens = 0

        for i in range(num_prompts):
            start = time.time()
            outputs = llm.generate(prompts, sampling_params)
            latency = time.time() - start
            latencies.append(latency)

            # Count tokens
            for output in outputs:
                for completion in output.outputs:
                    total_tokens += len(completion.token_ids)

            print(f"  Run {i+1}/{num_prompts}: {latency:.3f}s", end="\r")

        print()

        # Get GPU stats
        gpu_stats = get_gpu_stats()

        # Calculate statistics
        total_time = sum(latencies)
        avg_latency = statistics.mean(latencies)
        throughput = (num_prompts * batch_size) / total_time

        sorted_latencies = sorted(latencies)

        def percentile(data, p):
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (data[c] - data[f]) * (k - f)

        result = BenchmarkResult(
            config_name=f"batch_{batch_size}",
            batch_size=batch_size,
            num_prompts=num_prompts,
            total_time=total_time,
            throughput=throughput,
            avg_latency=avg_latency,
            p50_latency=percentile(sorted_latencies, 50),
            p90_latency=percentile(sorted_latencies, 90),
            p95_latency=percentile(sorted_latencies, 95),
            p99_latency=percentile(sorted_latencies, 99),
            tokens_generated=total_tokens,
            tokens_per_second=total_tokens / total_time,
            gpu_memory_mb=gpu_stats["memory_used_mb"] if gpu_stats else None,
            gpu_utilization=gpu_stats["gpu_utilization"] if gpu_stats else None,
        )

        results.append(result)

        # Print results
        print(f"\nResults for batch size {batch_size}:")
        print(f"  Throughput: {result.throughput:.2f} requests/second")
        print(f"  Tokens/sec: {result.tokens_per_second:.2f}")
        print(f"  Latency (avg): {result.avg_latency:.3f}s")
        print(f"  Latency (P50): {result.p50_latency:.3f}s")
        print(f"  Latency (P95): {result.p95_latency:.3f}s")
        print(f"  Latency (P99): {result.p99_latency:.3f}s")
        if gpu_stats:
            print(f"  GPU Memory: {result.gpu_memory_mb:.0f}MB")
            print(f"  GPU Util: {result.gpu_utilization:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM inference")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/ubuntu/models/llama-3.2-11b-vision",
        help="Path to model weights",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16,32",
        help="Comma-separated list of batch sizes",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts per batch size",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization (0.0-1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print("=" * 60)
    print("LLM Inference Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Prompts per batch: {args.num_prompts}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")

    results = run_benchmark(
        model_path=args.model,
        batch_sizes=batch_sizes,
        num_prompts=args.num_prompts,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"{'Batch Size':<12} {'Throughput':<15} {'Tokens/s':<12} "
        f"{'P50 (s)':<10} {'P95 (s)':<10} {'P99 (s)':<10}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r.batch_size:<12} {r.throughput:<15.2f} {r.tokens_per_second:<12.2f} "
            f"{r.p50_latency:<10.3f} {r.p95_latency:<10.3f} {r.p99_latency:<10.3f}"
        )

    # Find optimal configurations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    max_throughput = max(results, key=lambda x: x.throughput)
    min_latency = min(results, key=lambda x: x.p95_latency)

    print(f"For maximum throughput: batch size {max_throughput.batch_size}")
    print(f"  Throughput: {max_throughput.throughput:.2f} req/s")
    print(f"  P95 latency: {max_throughput.p95_latency:.3f}s")

    print(f"\nFor minimum latency: batch size {min_latency.batch_size}")
    print(f"  Throughput: {min_latency.throughput:.2f} req/s")
    print(f"  P95 latency: {min_latency.p95_latency:.3f}s")

    # Save results
    if args.output:
        output_data = {
            "config": {
                "model": args.model,
                "batch_sizes": batch_sizes,
                "num_prompts": args.num_prompts,
                "max_tokens": args.max_tokens,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            },
            "results": [
                {
                    "config_name": r.config_name,
                    "batch_size": r.batch_size,
                    "throughput": r.throughput,
                    "tokens_per_second": r.tokens_per_second,
                    "avg_latency": r.avg_latency,
                    "p50_latency": r.p50_latency,
                    "p90_latency": r.p90_latency,
                    "p95_latency": r.p95_latency,
                    "p99_latency": r.p99_latency,
                    "tokens_generated": r.tokens_generated,
                    "gpu_memory_mb": r.gpu_memory_mb,
                    "gpu_utilization": r.gpu_utilization,
                }
                for r in results
            ],
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
