# LLM Inference System

A production-grade multimodal LLM inference system built with vLLM and FastAPI, designed for learning GPU optimization and ML operations.

## Overview

This project implements a complete inference system for the Llama 3.2 11B Vision model, including:

- **FastAPI Application**: OpenAI-compatible REST API
- **vLLM Engine**: High-performance model serving with PagedAttention
- **Monitoring Stack**: Prometheus, Grafana, and GPU metrics
- **Infrastructure as Code**: Terraform for AWS deployment
- **Benchmarking Tools**: Load testing and performance profiling

Edit: The goal was to use Llama 3.2 11B Vision model but instead I'm using LLama-3.2-3b as I understand that the vLLM v1 engine does not yet support multimodal models that rely on cross-attention

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Internet                           │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
      ┌───────────────────────┐
      │  Nginx (Reverse Proxy)│
      └───────────┬───────────┘
                  │
                  ▼
      ┌────────────────────────────────┐
      │  FastAPI Application           │
      │  - Request Validation          │
      │  - Auth & Rate Limiting        │
      │  - Metrics Collection          │
      └────────────┬───────────────────┘
                  │
                  ▼
      ┌────────────────────────────────┐
      │  vLLM Engine                   │
      │  - Model Loading              │
      │  - Batch Processing           │
      │  - KV Cache Management        │
      └────────────┬───────────────────┘
                  │
                  ▼
      ┌────────────────────────────────┐
      │  NVIDIA A10G GPU (24GB)       │
      └────────────────────────────────┘
```

## Quick Start

### Local Development

```bash
# Clone the repository
git clone <repo-url>
cd llm-inference

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Copy environment file
cp config/.env.example config/.env
# Edit .env with your settings

# Run the API (without model for testing)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or activate the virtual environment manually
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### AWS Deployment

```bash
# Configure AWS credentials
aws configure

# Deploy infrastructure
cd infrastructure/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings

terraform init
terraform plan
terraform apply

# Get instance IP
INSTANCE_IP=$(terraform output -raw public_ip)

# SSH into instance
ssh -i your-key.pem ubuntu@$INSTANCE_IP
```

### EC2 Instance Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@$INSTANCE_IP

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Source the uv environment
source ~/.local/bin/env

# Create and activate a virtual environment
uv venv ~/venv
source ~/venv/bin/activate

# Install huggingface_hub
uv pip install huggingface_hub[cli]
```

### Download Model

```bash
# Authenticate with Hugging Face (get token from https://huggingface.co/settings/tokens)
hf auth login

# Create model directory
mkdir -p /home/ubuntu/models/llama-3.2-11b-vision

# Download the model
hf download meta-llama/Llama-3.2-11B-Vision-Instruct \
  --local-dir /home/ubuntu/models/llama-3.2-11b-vision

# Start the API
sudo systemctl start llm-api
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Streaming Response

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain AI"}],
    "max_tokens": 200,
    "stream": true
  }'
```

## Project Structure

```
llm-inference/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── config.py                  # Settings
│   ├── models.py                  # Pydantic models
│   ├── core/
│   │   ├── vllm_engine.py         # vLLM wrapper
│   │   ├── image_processor.py     # Multimodal support
│   │   └── cache.py               # Caching utilities
│   ├── api/
│   │   ├── chat.py                # Chat endpoints
│   │   └── health.py              # Health checks
│   ├── middleware/
│   │   ├── auth.py                # Authentication
│   │   ├── rate_limit.py          # Rate limiting
│   │   └── logging.py             # Request logging
│   └── utils/
│       ├── metrics.py             # Prometheus metrics
│       └── gpu_monitor.py         # GPU monitoring
├── config/
│   ├── .env.example
│   └── production.yaml
├── monitoring/
│   ├── prometheus.yml
│   ├── docker-compose.yml
│   └── grafana/
├── scripts/
│   ├── download_model.py
│   ├── benchmark.py
│   ├── load_test.py
│   └── monitor_gpu.py
├── infrastructure/
│   └── terraform/
│       ├── main.tf
│       ├── variables.tf
│       └── user_data.sh
├── tests/
├── requirements.txt
└── README.md
```

## Monitoring

### Start Monitoring Stack

```bash
cd monitoring
docker-compose up -d
```

### Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### GPU Monitoring

```bash
# Interactive GPU monitor
uv run python scripts/monitor_gpu.py

# Or use nvtop
nvtop
```

## Benchmarking

### Performance Benchmark

```bash
uv run python scripts/benchmark.py \
  --model /path/to/model \
  --batch-sizes 1,4,8,16 \
  --num-prompts 10 \
  --output results.json
```

### Load Testing

```bash
uv run python scripts/load_test.py \
  --url http://localhost:8000 \
  --requests 100 \
  --concurrency 10 \
  --output load_test.json
```

## Development with uv

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync

# Install with dev dependencies
uv sync --dev

# Add a new dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>

# Run tests
uv run pytest

# Run linting
uv run ruff check .

# Run type checking
uv run mypy app/
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to model weights | `/home/ubuntu/models/llama-3.2-11b-vision` |
| `GPU_MEMORY_UTILIZATION` | GPU memory fraction | `0.90` |
| `MAX_MODEL_LENGTH` | Maximum sequence length | `4096` |
| `MAX_NUM_SEQS` | Max concurrent sequences | `128` |
| `RATE_LIMIT` | Requests per minute | `100` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Performance Targets

- **Throughput**: 50-100 requests/second (text-only)
- **Latency**: <3s P95 (text), <5s P95 (with images)
- **GPU Utilization**: 70-90% under load
- **Error Rate**: <1%

## Learning Resources

This project is designed for learning. Key areas to explore:

1. **GPU Architecture**: Understanding A10G specs, memory hierarchy
2. **vLLM Internals**: PagedAttention, continuous batching
3. **Performance Profiling**: nvidia-smi, Nsight Systems
4. **Optimization**: Memory utilization, batch sizing, quantization
5. **Production Operations**: Monitoring, alerting, runbooks

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce GPU memory utilization
export GPU_MEMORY_UTILIZATION=0.85

# Or reduce batch size
export MAX_NUM_SEQS=64
```

### High Latency

```bash
# Check GPU utilization
nvidia-smi

# Profile with Nsight
nsys profile python -m uvicorn app.main:app
```

### Model Not Loading

```bash
# Check model path
ls -la /home/ubuntu/models/llama-3.2-11b-vision

# Check GPU availability
nvidia-smi

# Check logs
sudo journalctl -u llm-api -f
```

## Future Improvements

This project provides a solid foundation for LLM inference. Here are ideas for further enhancements:

### Architecture Improvements

#### 1. Decouple API Layer from GPU Inference

Separate the FastAPI server from the vLLM engine into independent services:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Load Balancer  │     │  Load Balancer  │     │                 │
└────────┬────────┘     └────────┬────────┘     │                 │
         │                       │              │                 │
    ┌────┴────┐             ┌────┴────┐        │                 │
    ▼         ▼             ▼         ▼        │                 │
┌───────┐ ┌───────┐    ┌───────┐ ┌───────┐    │   Message       │
│FastAPI│ │FastAPI│    │ vLLM  │ │ vLLM  │◄───│   Queue         │
│  #1   │ │  #2   │───▶│ GPU#1 │ │ GPU#2 │    │  (Redis/Kafka)  │
└───────┘ └───────┘    └───────┘ └───────┘    │                 │
                                               └─────────────────┘
```

**Benefits:**
- Scale API and GPU layers independently
- Restart API without reloading model (model stays warm)
- Add/remove GPUs without API downtime
- Better resource utilization

**Implementation options:**
- Redis Streams for request queuing
- gRPC for low-latency communication
- Kubernetes with separate deployments

#### 2. Request Priority Queuing

Implement priority lanes for different request types:

```python
# High priority: real-time chat
# Medium priority: batch processing
# Low priority: background tasks
```

#### 3. Model Sharding Across GPUs

For larger models, implement tensor parallelism:

```bash
# Example: 70B model across 4 GPUs
TENSOR_PARALLEL_SIZE=4 uvicorn app.main:app
```

### Performance Optimizations

#### 4. Speculative Decoding

Use a smaller "draft" model to speed up generation:
- Draft model proposes tokens
- Main model verifies in parallel
- Can provide 2-3x speedup

#### 5. Quantization

Reduce memory and increase throughput:

```python
# AWQ quantization (4-bit)
engine_args = AsyncEngineArgs(
    model=model_path,
    quantization="awq",
)
```

#### 6. Prefix Caching

Cache common prompt prefixes (system prompts) across requests:

```python
engine_args = AsyncEngineArgs(
    model=model_path,
    enable_prefix_caching=True,
)
```

### Operational Improvements

#### 7. Graceful Degradation

- Circuit breaker for GPU failures
- Fallback to smaller model under load
- Request shedding with backpressure

#### 8. A/B Testing Infrastructure

Route traffic between model versions:

```python
@app.post("/v1/chat/completions")
async def chat(request: Request):
    model = select_model(user_id, experiment="new_model_test")
    return await generate(model, request)
```

#### 9. Cost Optimization

- Spot instance support with checkpointing
- Auto-scaling based on queue depth
- Scheduled scale-down during off-peak

### Observability Enhancements

#### 10. Distributed Tracing

Add OpenTelemetry for end-to-end request tracing:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@app.post("/v1/chat/completions")
async def chat(request: Request):
    with tracer.start_as_current_span("chat_completion"):
        # ... generation logic
```

#### 11. Advanced Metrics

- Token-level latency breakdown (prefill vs decode)
- KV cache hit rates
- Batch utilization histograms
- Per-user usage tracking

### Security Hardening

#### 12. Input Validation & Guardrails

- Prompt injection detection
- Output filtering for PII/harmful content
- Token budget enforcement per user

#### 13. Authentication & Multi-tenancy

- JWT-based authentication
- Per-tenant rate limiting and quotas
- Usage metering for billing

### Developer Experience

#### 14. Local Development Mode

- Mock vLLM engine for testing without GPU
- Recorded responses for deterministic tests
- Docker Compose for full local stack

#### 15. CLI Tools

```bash
# Interactive chat
llm-inference chat --model llama-3.2-3b

# Batch processing
llm-inference batch --input prompts.jsonl --output results.jsonl

# Model management
llm-inference models list
llm-inference models download meta-llama/Llama-3.2-3B
```

### Contributing

Contributions are welcome! Pick an item from the list above or propose your own improvement.

## License

MIT License
