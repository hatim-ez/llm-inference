# LLM Inference System

A production-grade multimodal LLM inference system built with vLLM and FastAPI, designed for learning GPU optimization and ML operations.

## Overview

This project implements a complete inference system for the Llama 3.2 11B Vision model, including:

- **FastAPI Application**: OpenAI-compatible REST API
- **vLLM Engine**: High-performance model serving with PagedAttention
- **Monitoring Stack**: Prometheus, Grafana, and GPU metrics
- **Infrastructure as Code**: Terraform for AWS deployment
- **Benchmarking Tools**: Load testing and performance profiling

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

## License

MIT License
