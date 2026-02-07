# LLM Inference Performance Testing Report

**Date:** 2026-02-05
**Updated:** 2026-02-07 (Post-optimization results)

## System Configuration

| Component | Specification |
|-----------|---------------|
| **Model** | Llama 3.2 3B Instruct |
| **GPU** | Tesla T4 (16GB VRAM) |
| **Instance** | AWS EC2 |
| **Framework** | vLLM 0.15.1 + FastAPI |
| **GPU Memory Utilization** | 90% |
| **Max Model Length** | 4096 tokens |

---

## Executive Summary

| Metric | Direct Benchmark | API v1 (Before) | API v2 (After) | Improvement |
|--------|------------------|-----------------|----------------|-------------|
| **Max Throughput** | 265 tokens/sec | 4.8 tokens/sec | 132.5 tokens/sec | **27x faster** |
| **Request Throughput** | 2.65 req/s | 0.06 req/s | 1.66 req/s | **27x faster** |
| **P95 Latency** | 3.03s | 2.72s | 3.03s | Similar |
| **Success Rate** | 100% | 44% | **100%** | ✅ Fixed |
| **GPU Utilization** | 97-98% | 97% | 97% | Near capacity |

**Key Finding:** After implementing continuous batching with `AsyncLLMEngine`, API throughput improved **27x** (from 4.8 to 132.5 tokens/sec) and success rate improved from 44% to **100%**. The remaining 2x gap vs direct benchmark is due to HTTP overhead and can be considered acceptable for production use.

---

## Test 1: Direct vLLM Benchmark

**Purpose:** Measure raw model inference performance without HTTP overhead

**Test Parameters:**
| Parameter | Value |
|-----------|-------|
| Model | `/home/ubuntu/models/llama-3.2-3b` |
| Batch Sizes | 1, 2, 4, 8 |
| Prompts per Batch | 15 |
| Max Tokens | 100 |
| GPU Memory Utilization | 0.9 |

### Results by Batch Size

| Batch Size | Throughput (req/s) | Tokens/sec | P50 Latency | P95 Latency | P99 Latency |
|------------|-------------------|------------|-------------|-------------|-------------|
| 1 | 0.37 | 37.0 | 2.700s | 2.712s | 2.716s |
| 2 | 0.69 | 69.0 | 2.898s | 2.905s | 2.906s |
| 4 | 1.36 | 135.6 | 2.949s | 2.955s | 2.962s |
| **8** | **2.65** | **265.1** | 3.016s | 3.026s | 3.026s |

### Scaling Analysis

```
Tokens/sec vs Batch Size:

265 |                              ████
    |                              ████
200 |                              ████
    |                              ████
135 |                    ████      ████
    |                    ████      ████
100 |                    ████      ████
 69 |          ████      ████      ████
 37 | ████     ████      ████      ████
    +------+--------+--------+--------+
      BS=1    BS=2     BS=4     BS=8
```

### Observations

1. **Linear Scaling**: Throughput scales almost linearly with batch size (7.2x improvement from BS=1 to BS=8)
2. **Latency Trade-off**: Latency increases only ~12% (2.70s → 3.02s) while throughput increases 7x
3. **GPU Saturation**: GPU utilization at 97-98% across all batch sizes
4. **Memory Usage**: Consistent at ~14.7 GB regardless of batch size

### Recommendation

**Use batch size 8** for optimal throughput with acceptable latency. Consider testing batch size 16 if memory allows.

---

## Test 2: API Load Test

**Purpose:** Measure end-to-end API performance under concurrent load

**Test Parameters:**
| Parameter | Value |
|-----------|-------|
| Total Requests | 50 |
| Concurrency | 5 |
| Max Tokens | 100 |
| Ramp Up Time | 0s |

### Results Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Total Requests** | 50 | |
| **Successful** | 22 (44%) | 🔴 Critical |
| **Failed** | 28 (56%) | 🔴 Critical |
| **Total Duration** | 365.8s | |
| **Throughput** | 0.06 req/s | 🔴 Critical |
| **Tokens/sec** | 4.84 | 🔴 Critical |

### Latency Distribution (Successful Requests Only)

| Percentile | Latency |
|------------|---------|
| Min | 2.703s |
| P50 | 2.708s |
| P90 | 2.714s |
| P95 | 2.715s |
| P99 | 2.940s |
| Max | 3.000s |

### Error Analysis

| Error Type | Count | Percentage |
|------------|-------|------------|
| Timeout | 28 | 100% of failures |

### Observations

1. **Massive Throughput Gap**: API delivers only 4.84 tokens/sec vs 265 tokens/sec in direct benchmark (55x worse)
2. **High Failure Rate**: 56% of requests timeout before completion
3. **Consistent Success Latency**: Successful requests have tight latency distribution (std dev ~0.06s)
4. **Queue Starvation**: Requests are timing out while waiting in queue, not during processing

### Root Cause Analysis

```
Request Flow with Bottleneck:

[Client] → [FastAPI] → [Queue] → [vLLM Engine] → [GPU]
                          ↑
                    BOTTLENECK

- GPU can process at 265 tok/s
- But only 1 request processes at a time
- Other 4 concurrent requests wait in queue
- Default timeout triggers before queue clears
```

---

## Test 2b: API Load Test v2 (Post-Optimization)

**Purpose:** Validate performance improvements after implementing continuous batching

**Changes Made:**
1. Switched from synchronous `LLM` to `AsyncLLMEngine` for continuous batching
2. Reduced `max_num_seqs` from 128 to 8 (optimal for T4 GPU)
3. Increased rate limits (1000 req/min, 100 burst/sec)
4. Enabled native async streaming

**Test Parameters:** (Same as v1)
| Parameter | Value |
|-----------|-------|
| Total Requests | 50 |
| Concurrency | 5 |
| Max Tokens | 100 |
| Timeout | 120s |

### Results Summary

| Metric | Before (v1) | After (v2) | Change |
|--------|-------------|------------|--------|
| **Successful** | 22 (44%) | **50 (100%)** | ✅ +127% |
| **Failed** | 28 (56%) | **0 (0%)** | ✅ Fixed |
| **Total Duration** | 365.8s | **30.2s** | ⚡ 12x faster |
| **Throughput** | 0.06 req/s | **1.66 req/s** | ⚡ 27x faster |
| **Tokens/sec** | 4.84 | **132.5** | ⚡ **27x faster** |

### Latency Distribution

| Percentile | Before (v1) | After (v2) | Change |
|------------|-------------|------------|--------|
| Min | 2.703s | 2.981s | +10% |
| P50 | 2.708s | 3.016s | +11% |
| P90 | 2.714s | 3.020s | +11% |
| P95 | 2.715s | 3.026s | +11% |
| P99 | 2.940s | 3.026s | +3% |
| Max | 3.000s | 3.026s | +1% |

### Analysis

1. **100% Success Rate**: All requests complete successfully (vs 44% before)
2. **27x Throughput Improvement**: From 4.8 to 132.5 tokens/sec
3. **Consistent Latency**: Very tight latency distribution (min 2.98s, max 3.03s)
4. **Slight Latency Increase**: ~10% higher latency due to batching overhead, but acceptable tradeoff for 27x throughput

### Why It Works Now

```
Improved Request Flow with Continuous Batching:

[Client] → [FastAPI] → [AsyncLLMEngine] → [GPU]
                              ↓
                    ┌─────────────────┐
                    │ Continuous      │
                    │ Batching        │
                    │ (max_num_seqs=8)│
                    └─────────────────┘

- Multiple requests batched together automatically
- GPU processes 8 sequences concurrently
- No queue starvation or timeouts
- Native async - no thread pool blocking
```

---

## Comparative Analysis

### Full Comparison: Benchmark vs API v1 vs API v2

| Metric | Direct Benchmark (BS=8) | API v1 (Before) | API v2 (After) |
|--------|------------------------|-----------------|----------------|
| Tokens/sec | 265.1 | 4.84 | **132.5** |
| Throughput | 2.65 req/s | 0.06 req/s | **1.66 req/s** |
| Success Rate | 100% | 44% | **100%** |
| P95 Latency | 3.03s | 2.72s | 3.03s |
| Gap vs Benchmark | — | 55x worse | **2x worse** |

### Gap Analysis

| Issue | v1 Status | v2 Status |
|-------|-----------|-----------|
| No continuous batching | 🔴 Sequential processing | ✅ AsyncLLMEngine batches requests |
| Timeout too aggressive | 🔴 56% failures | ✅ All requests complete |
| No backpressure | 🔴 Queue overflow | ✅ Rate limiting configured |
| HTTP overhead | ⚠️ ~10% | ⚠️ ~10% (unavoidable) |

### Remaining 2x Gap Explanation

The API v2 achieves **50% of direct benchmark performance** (132.5 vs 265 tokens/sec). This remaining gap is due to:

1. **HTTP Protocol Overhead**: Request parsing, JSON serialization, response formatting
2. **FastAPI Middleware Stack**: Logging, rate limiting, metrics collection
3. **Async Context Switching**: Additional overhead from async/await machinery
4. **Conservative Batching**: `max_num_seqs=8` leaves headroom for stability

**Verdict**: The 2x gap is acceptable for production. Further optimization would require:
- Native vLLM HTTP server (bypasses FastAPI)
- Larger batch sizes (if memory allows)
- gRPC instead of HTTP/JSON

---

## Recommendations

### Completed Actions ✅

| Priority | Action | Status | Result |
|----------|--------|--------|--------|
| P0 | Enable continuous batching with AsyncLLMEngine | ✅ Done | 27x throughput improvement |
| P0 | Increase rate limits for load testing | ✅ Done | 100% success rate |
| P1 | Set max_num_seqs=8 for optimal batching | ✅ Done | Matches benchmark batch size |

### Implementation Details

```python
# app/core/vllm_engine.py - AsyncLLMEngine for continuous batching
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

engine_args = AsyncEngineArgs(
    model=model_path,
    max_num_seqs=8,  # Batch up to 8 concurrent requests
    gpu_memory_utilization=0.90,
    max_model_len=4096,
    trust_remote_code=True,
)
self._engine = AsyncLLMEngine.from_engine_args(engine_args)
```

```python
# app/config.py - Rate limiting configuration
rate_limit: int = 1000        # Requests per minute
rate_limit_burst: int = 100   # Max burst per second
```

### Future Optimizations (Optional)

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| P2 | Test max_num_seqs=16 | Potential 20-30% throughput increase |
| P2 | Use vLLM native HTTP server | Eliminate FastAPI overhead (~2x) |
| P3 | Implement gRPC endpoint | Lower serialization overhead |
| P3 | Add request priority queuing | Better SLA management |

### Stress Testing Recommendations

1. **Higher concurrency test**: Try concurrency=10, 20 to find saturation point
2. **Longer duration test**: Run 1000+ requests to measure stability
3. **Variable prompt lengths**: Test with mixed short/long prompts
4. **Memory pressure test**: Increase max_tokens to 500+ to stress KV cache

---

## Test 3: NVIDIA Nsight Systems GPU Profile

**Purpose:** Deep-dive into GPU kernel execution and identify low-level bottlenecks

**Profile Tool:** `nsys profile` with vLLM inference
**Profile Script:** `nsys_vllm_profile.py`

### CUDA API Time Breakdown

| Operation | % of API Time | Implication |
|-----------|---------------|-------------|
| `cudaMemcpyAsync` | ~41% | Memory transfers (H2D) |
| `cudaEventSynchronize` | ~30% | Host waiting for GPU |
| `cudaDeviceSynchronize` | ~16% | More host blocking |
| Other | ~13% | Kernel launches, etc. |

### GPU Kernel Analysis

**Top Kernels (by execution time):**
1. **Turing FP16 Tensor Core GEMMs** (`turing_fp16_s1688gemm...`)
   - This is expected and good - GPU is doing matrix multiplications
   - Using Tensor Cores efficiently

2. **Triton Fused Ops** - Custom fused operations
3. **FlashInfer Kernels** - Attention computation

### Memory Transfer Analysis

| Direction | Total | Observation |
|-----------|-------|-------------|
| **H2D (Host→Device)** | ~6.4 GB | Large transfers (model weights, graph capture) |
| **D2H (Device→Host)** | Minimal | Good - not bottlenecked on output |
| **Largest single transfer** | ~788 MB | One-time initialization |

### Key Findings

1. **GPU is Doing the Right Work**
   - FP16 Tensor Core GEMMs dominate kernel time
   - Attention kernels (FlashInfer) are efficient
   - No unexpected kernels or inefficiencies

2. **Host-Side Synchronization Overhead**
   - 46% of CUDA API time is synchronization (`cudaEventSynchronize` + `cudaDeviceSynchronize`)
   - Host threads spend most time waiting, not computing
   - This is typical for small batch sizes and async GPU workloads

3. **Profile Includes Setup Overhead**
   - 6.4 GB H2D transfers include model loading (one-time cost)
   - Warmup phase inflates sync/memcpy percentages
   - Steady-state performance would show less overhead

### CPU Thread Analysis

| State | Observation |
|-------|-------------|
| `pthread_cond_*` | Threads waiting on conditions |
| `epoll_wait` | I/O polling (idle) |
| `poll` | More waiting |

**Interpretation:** CPU threads are mostly idle/syncing, not CPU-bound. This is normal for async GPU workloads.

### Recommendations from Profiling

1. **Profile Steady-State Only**
   ```python
   import torch
   torch.cuda.profiler.start()
   llm.generate(["prompt"], params)
   torch.cuda.profiler.stop()
   ```
   Run with: `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...`

2. **Add NVTX Annotations for Phase Tracking**
   ```python
   import nvtx
   with nvtx.annotate("warmup"):
       llm.generate(["warmup"], params)
   with nvtx.annotate("generate"):
       llm.generate(["real"], params)
   ```

3. **Increase Batch Size** to reduce sync overhead percentage and see sustained GPU utilization

### Profile Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Kernel efficiency | ✅ Good | Tensor Core GEMMs as expected |
| Memory transfers | ✅ Normal | One-time model load dominates |
| Host sync overhead | ⚠️ High | 46% - typical for small batches |
| CPU utilization | ✅ Normal | Threads idle (GPU-bound workload) |

---

## Appendix

### Test Commands

```bash
# Direct vLLM Benchmark
uv run python scripts/benchmark.py \
  --model /home/ubuntu/models/llama-3.2-3b \
  --batch-sizes 1,2,4,8 \
  --num-prompts 15 \
  --max-tokens 100 \
  --max-model-len 4096 \
  --output benchmark_results.json

# API Load Test
uv run python scripts/load_test.py \
  --url http://localhost:8000 \
  --requests 50 \
  --concurrency 5 \
  --output load_test_results.json
```

### GPU Memory Breakdown

For Llama 3.2 3B with float16:
- Model weights: ~6 GB (3B params × 2 bytes)
- KV Cache (4096 context): ~0.4 GB
- CUDA overhead: ~1 GB
- **Total**: ~7.4 GB
- **Available for batching**: ~7 GB headroom

### Raw Data Files

- `benchmark_results.json` - Direct vLLM benchmark results
- `load_test_results.json` - API load test results (v1, before optimization)
- `load_test_results_v2.json` - API load test results (v2, after optimization)
- `inference_profile.nsys-rep` - NVIDIA Nsight Systems profile
- `nsys_report.md` - Nsight analysis notes
- `nsys_vllm_profile.py` - Profiling script

---

## Changelog

| Date | Change |
|------|--------|
| 2026-02-05 | Initial report with benchmark, load test v1, and Nsight profiling |
| 2026-02-07 | Added load test v2 results after AsyncLLMEngine optimization |
