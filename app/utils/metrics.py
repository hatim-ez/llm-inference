"""Prometheus metrics for monitoring."""

from prometheus_client import Counter, Gauge, Histogram, Info

# Application info
APP_INFO = Info(
    "llm_inference_app",
    "LLM Inference application information",
)

# Request metrics
REQUEST_COUNT = Counter(
    "llm_inference_requests_total",
    "Total number of inference requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "llm_inference_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# Token metrics
TOKENS_GENERATED = Counter(
    "llm_inference_tokens_generated_total",
    "Total number of tokens generated",
)

TOKENS_PROMPT = Counter(
    "llm_inference_prompt_tokens_total",
    "Total number of prompt tokens processed",
)

GENERATION_TIME = Histogram(
    "llm_inference_generation_duration_seconds",
    "Generation time in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

TOKENS_PER_SECOND = Histogram(
    "llm_inference_tokens_per_second",
    "Tokens generated per second",
    buckets=[1, 5, 10, 20, 50, 100, 200, 500],
)

# Queue metrics
QUEUE_SIZE = Gauge(
    "llm_inference_queue_size",
    "Current number of requests in queue",
)

ACTIVE_REQUESTS = Gauge(
    "llm_inference_active_requests",
    "Number of requests currently being processed",
)

# GPU metrics
GPU_MEMORY_USED = Gauge(
    "llm_inference_gpu_memory_used_bytes",
    "GPU memory currently used in bytes",
    ["gpu_id"],
)

GPU_MEMORY_TOTAL = Gauge(
    "llm_inference_gpu_memory_total_bytes",
    "Total GPU memory in bytes",
    ["gpu_id"],
)

GPU_UTILIZATION = Gauge(
    "llm_inference_gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_id"],
)

GPU_TEMPERATURE = Gauge(
    "llm_inference_gpu_temperature_celsius",
    "GPU temperature in Celsius",
    ["gpu_id"],
)

GPU_POWER = Gauge(
    "llm_inference_gpu_power_watts",
    "GPU power consumption in watts",
    ["gpu_id"],
)

# Error metrics
ERROR_COUNT = Counter(
    "llm_inference_errors_total",
    "Total number of errors",
    ["error_type"],
)

# Cache metrics
CACHE_HITS = Counter(
    "llm_inference_cache_hits_total",
    "Total number of cache hits",
)

CACHE_MISSES = Counter(
    "llm_inference_cache_misses_total",
    "Total number of cache misses",
)

# Rate limiting metrics
RATE_LIMIT_EXCEEDED = Counter(
    "llm_inference_rate_limit_exceeded_total",
    "Total number of rate limit exceeded events",
)


def setup_metrics() -> None:
    """Initialize metrics with application info."""
    from app import __version__
    from app.config import get_settings

    settings = get_settings()

    APP_INFO.info({
        "version": __version__,
        "model_name": settings.model_name,
        "max_model_length": str(settings.max_model_length),
        "gpu_memory_utilization": str(settings.gpu_memory_utilization),
    })


def record_request(endpoint: str, status: str, duration: float) -> None:
    """Record request metrics.

    Args:
        endpoint: API endpoint.
        status: Response status (success/error).
        duration: Request duration in seconds.
    """
    REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)


def record_generation(
    prompt_tokens: int,
    completion_tokens: int,
    generation_time: float,
) -> None:
    """Record generation metrics.

    Args:
        prompt_tokens: Number of prompt tokens.
        completion_tokens: Number of generated tokens.
        generation_time: Time taken for generation.
    """
    TOKENS_PROMPT.inc(prompt_tokens)
    TOKENS_GENERATED.inc(completion_tokens)
    GENERATION_TIME.observe(generation_time)

    if generation_time > 0:
        tokens_per_sec = completion_tokens / generation_time
        TOKENS_PER_SECOND.observe(tokens_per_sec)


def record_gpu_metrics(
    gpu_id: int,
    memory_used: float,
    memory_total: float,
    utilization: float,
    temperature: float,
    power: float,
) -> None:
    """Record GPU metrics.

    Args:
        gpu_id: GPU device ID.
        memory_used: Memory used in bytes.
        memory_total: Total memory in bytes.
        utilization: GPU utilization percentage.
        temperature: Temperature in Celsius.
        power: Power consumption in watts.
    """
    gpu_label = str(gpu_id)
    GPU_MEMORY_USED.labels(gpu_id=gpu_label).set(memory_used)
    GPU_MEMORY_TOTAL.labels(gpu_id=gpu_label).set(memory_total)
    GPU_UTILIZATION.labels(gpu_id=gpu_label).set(utilization)
    GPU_TEMPERATURE.labels(gpu_id=gpu_label).set(temperature)
    GPU_POWER.labels(gpu_id=gpu_label).set(power)


def record_error(error_type: str) -> None:
    """Record error metric.

    Args:
        error_type: Type of error (e.g., "oom", "timeout", "validation").
    """
    ERROR_COUNT.labels(error_type=error_type).inc()


def update_queue_metrics(queue_size: int, active_requests: int) -> None:
    """Update queue metrics.

    Args:
        queue_size: Current queue size.
        active_requests: Number of active requests.
    """
    QUEUE_SIZE.set(queue_size)
    ACTIVE_REQUESTS.set(active_requests)


def record_cache_hit() -> None:
    """Record cache hit."""
    CACHE_HITS.inc()


def record_cache_miss() -> None:
    """Record cache miss."""
    CACHE_MISSES.inc()
