# LLM Inference Performance Testing Report

**Date:** 2026-02-05

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

| Metric | Direct Benchmark | API Load Test | Status |
|--------|------------------|---------------|--------|
| **Max Throughput** | 265 tokens/sec | 4.8 tokens/sec | API bottleneck |
| **Best Batch Size** | 8 | N/A | |
| **P95 Latency** | 3.03s | 2.72s | |
| **Success Rate** | 100% | 44% | 🔴 Critical |
| **GPU Utilization** | 97-98% | 97% | Near capacity |

**Key Finding:** The direct vLLM benchmark shows excellent performance (265 tokens/sec at batch size 8), but the API load test reveals severe issues with only 44% success rate due to timeouts. The bottleneck is in the API/queuing layer, not the GPU.

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

## Comparative Analysis

### Direct Benchmark vs API Load Test

| Metric | Direct Benchmark (BS=8) | API Load Test | Ratio |
|--------|------------------------|---------------|-------|
| Tokens/sec | 265.1 | 4.84 | 55x gap |
| Throughput | 2.65 req/s | 0.06 req/s | 44x gap |
| Success Rate | 100% | 44% | |
| P95 Latency | 3.03s | 2.72s | Similar |

### Why the Gap?

1. **No Batching in API**: API processes requests sequentially, not in batches
2. **Timeout Too Aggressive**: Default timeout doesn't account for queue wait time
3. **No Backpressure**: Server accepts all requests without admission control
4. **HTTP Overhead**: Additional latency from request parsing, validation, response serialization

---

## Recommendations

### Immediate Actions (Critical)

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| P0 | Increase API timeout to 30-60s | Reduce timeout failures |
| P0 | Implement request queuing with backpressure | Prevent queue overflow |
| P1 | Enable continuous batching in vLLM | 5-10x throughput improvement |

### Configuration Changes

```python
# vllm_engine.py - Enable batching
llm = LLM(
    model=model_path,
    max_num_seqs=8,  # Allow batching up to 8 requests
    gpu_memory_utilization=0.85,  # Leave headroom
    max_model_len=4096,
)
```

```python
# load_test.py - Increase timeout
timeout = 60  # seconds instead of default
```

### Future Tests

1. **Test with concurrency=1**: Establish true baseline without queuing
2. **Test with longer timeout**: Capture actual latency for queued requests
3. **Test batch sizes 16, 32**: Find memory limit on T4
4. **Test with continuous batching enabled**: Measure real-world improvement

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
- `load_test_results.json` - API load test results
