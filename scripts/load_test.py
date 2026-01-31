#!/usr/bin/env python3
"""Load testing script for LLM inference API."""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp


@dataclass
class RequestResult:
    """Result of a single request."""

    request_id: int
    start_time: float
    end_time: float
    duration: float
    status_code: int
    success: bool
    tokens: int = 0
    error: Optional[str] = None


@dataclass
class LoadTestResults:
    """Aggregated load test results."""

    config: dict
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    throughput: float
    total_tokens: int
    tokens_per_second: float
    latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def percentile(self, p: float) -> float:
        """Calculate latency percentile."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        k = (len(sorted_latencies) - 1) * p / 100
        f = int(k)
        c = min(f + 1, len(sorted_latencies) - 1)
        return sorted_latencies[f] + (sorted_latencies[c] - sorted_latencies[f]) * (k - f)


class LoadTester:
    """Load tester for LLM inference API."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        """Initialize load tester.

        Args:
            base_url: API base URL.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def send_request(
        self,
        session: aiohttp.ClientSession,
        request_id: int,
        prompt: str,
        max_tokens: int,
    ) -> RequestResult:
        """Send a single chat completion request.

        Args:
            session: aiohttp session.
            request_id: Request identifier.
            prompt: Prompt text.
            max_tokens: Maximum tokens to generate.

        Returns:
            RequestResult with timing and status.
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }

        start_time = time.time()

        try:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                end_time = time.time()
                result_json = await response.json()

                success = response.status == 200
                tokens = 0
                error = None

                if success and "choices" in result_json:
                    content = result_json["choices"][0]["message"]["content"]
                    tokens = len(content.split())
                elif not success:
                    error = result_json.get("error", {}).get("message", "Unknown error")

                return RequestResult(
                    request_id=request_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    status_code=response.status,
                    success=success,
                    tokens=tokens,
                    error=error,
                )

        except asyncio.TimeoutError:
            end_time = time.time()
            return RequestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                status_code=0,
                success=False,
                error="Timeout",
            )
        except Exception as e:
            end_time = time.time()
            return RequestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                status_code=0,
                success=False,
                error=str(e),
            )

    async def run_test(
        self,
        num_requests: int,
        concurrency: int,
        prompts: list[str],
        max_tokens: int,
        ramp_up_time: float = 0,
    ) -> LoadTestResults:
        """Run load test.

        Args:
            num_requests: Total number of requests.
            concurrency: Number of concurrent requests.
            prompts: List of prompts to use.
            max_tokens: Maximum tokens per request.
            ramp_up_time: Time to ramp up to full concurrency.

        Returns:
            LoadTestResults with aggregated metrics.
        """
        print(f"\nStarting load test:")
        print(f"  Total requests: {num_requests}")
        print(f"  Concurrency: {concurrency}")
        print(f"  Max tokens: {max_tokens}")
        print(f"  Ramp up time: {ramp_up_time}s")

        results: list[RequestResult] = []
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_request(req_id: int, prompt: str) -> RequestResult:
            async with semaphore:
                return await self.send_request(session, req_id, prompt, max_tokens)

        connector = aiohttp.TCPConnector(limit=concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            test_start = time.time()

            # Create tasks with optional ramp-up
            tasks = []
            delay_per_request = ramp_up_time / num_requests if ramp_up_time > 0 else 0

            for i in range(num_requests):
                prompt = prompts[i % len(prompts)]
                task = asyncio.create_task(limited_request(i, prompt))
                tasks.append(task)

                if delay_per_request > 0 and i < num_requests - 1:
                    await asyncio.sleep(delay_per_request)

            # Wait for all requests
            print("\nRunning requests...")
            completed = 0
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                completed += 1
                if completed % 10 == 0:
                    print(f"  Completed: {completed}/{num_requests}", end="\r")

            test_end = time.time()

        print(f"\n  Completed: {completed}/{num_requests}")

        # Calculate results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        total_duration = test_end - test_start
        total_tokens = sum(r.tokens for r in successful)

        load_results = LoadTestResults(
            config={
                "num_requests": num_requests,
                "concurrency": concurrency,
                "max_tokens": max_tokens,
                "ramp_up_time": ramp_up_time,
            },
            total_requests=num_requests,
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_duration=total_duration,
            throughput=len(successful) / total_duration if total_duration > 0 else 0,
            total_tokens=total_tokens,
            tokens_per_second=total_tokens / total_duration if total_duration > 0 else 0,
            latencies=[r.duration for r in successful],
            errors=[r.error for r in failed if r.error],
        )

        return load_results


def print_results(results: LoadTestResults) -> None:
    """Print load test results."""
    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)

    print(f"\nRequests:")
    print(f"  Total: {results.total_requests}")
    print(f"  Successful: {results.successful_requests}")
    print(f"  Failed: {results.failed_requests}")
    print(f"  Success rate: {results.successful_requests / results.total_requests * 100:.1f}%")

    print(f"\nPerformance:")
    print(f"  Total duration: {results.total_duration:.2f}s")
    print(f"  Throughput: {results.throughput:.2f} requests/second")
    print(f"  Tokens/second: {results.tokens_per_second:.2f}")

    if results.latencies:
        print(f"\nLatency:")
        print(f"  Min: {min(results.latencies):.3f}s")
        print(f"  Max: {max(results.latencies):.3f}s")
        print(f"  Mean: {statistics.mean(results.latencies):.3f}s")
        print(f"  Median: {statistics.median(results.latencies):.3f}s")
        print(f"  P50: {results.percentile(50):.3f}s")
        print(f"  P90: {results.percentile(90):.3f}s")
        print(f"  P95: {results.percentile(95):.3f}s")
        print(f"  P99: {results.percentile(99):.3f}s")
        print(f"  Std Dev: {statistics.stdev(results.latencies) if len(results.latencies) > 1 else 0:.3f}s")

    if results.errors:
        print(f"\nErrors (showing first 5):")
        for error in results.errors[:5]:
            print(f"  - {error}")


def main():
    parser = argparse.ArgumentParser(description="Load test LLM inference API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API base URL",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Total number of requests",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens per request",
    )
    parser.add_argument(
        "--ramp-up",
        type=float,
        default=0,
        help="Ramp up time in seconds",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    # Test prompts
    prompts = [
        "What is machine learning?",
        "Explain deep learning",
        "How do neural networks work?",
        "What is artificial intelligence?",
        "Describe natural language processing",
    ]

    # Run load test
    tester = LoadTester(args.url, timeout=args.timeout)
    results = asyncio.run(
        tester.run_test(
            num_requests=args.requests,
            concurrency=args.concurrency,
            prompts=prompts,
            max_tokens=args.max_tokens,
            ramp_up_time=args.ramp_up,
        )
    )

    # Print results
    print_results(results)

    # Save results
    if args.output:
        output_data = {
            "config": results.config,
            "summary": {
                "total_requests": results.total_requests,
                "successful_requests": results.successful_requests,
                "failed_requests": results.failed_requests,
                "total_duration": results.total_duration,
                "throughput": results.throughput,
                "tokens_per_second": results.tokens_per_second,
            },
            "latency": {
                "min": min(results.latencies) if results.latencies else None,
                "max": max(results.latencies) if results.latencies else None,
                "mean": statistics.mean(results.latencies) if results.latencies else None,
                "p50": results.percentile(50),
                "p90": results.percentile(90),
                "p95": results.percentile(95),
                "p99": results.percentile(99),
            },
            "errors": results.errors[:20],  # First 20 errors
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
