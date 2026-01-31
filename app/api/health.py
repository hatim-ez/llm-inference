"""Health check endpoints."""

import structlog
from fastapi import APIRouter

from app import __version__
from app.models import HealthStatus
from app.utils.gpu_monitor import GPUMonitor

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get("/health")
async def health_check() -> HealthStatus:
    """Health check endpoint.

    Returns system health status including GPU metrics.
    """
    from app.main import get_engine, get_uptime

    engine = get_engine()
    model_loaded = engine is not None and engine.is_ready

    # Get GPU metrics
    gpu_monitor = GPUMonitor()
    gpu_stats = gpu_monitor.get_stats()

    gpu_available = gpu_stats is not None
    gpu_memory_used = None
    gpu_memory_total = None
    gpu_utilization = None

    if gpu_stats:
        gpu_memory_used = gpu_stats.get("memory_used_mb")
        gpu_memory_total = gpu_stats.get("memory_total_mb")
        gpu_utilization = gpu_stats.get("gpu_utilization")

    # Determine status
    if not gpu_available:
        status = "unhealthy"
    elif not model_loaded:
        status = "degraded"
    else:
        status = "healthy"

    return HealthStatus(
        status=status,
        model_loaded=model_loaded,
        gpu_available=gpu_available,
        gpu_memory_used_mb=gpu_memory_used,
        gpu_memory_total_mb=gpu_memory_total,
        gpu_utilization_percent=gpu_utilization,
        uptime_seconds=get_uptime(),
        version=__version__,
    )


@router.get("/ready")
async def readiness_check() -> dict:
    """Readiness check for load balancers.

    Returns 200 only when the model is loaded and ready.
    """
    from app.main import get_engine

    engine = get_engine()
    if engine is None or not engine.is_ready:
        return {"ready": False, "reason": "Model not loaded"}

    return {"ready": True}


@router.get("/live")
async def liveness_check() -> dict:
    """Liveness check for container orchestration.

    Returns 200 as long as the process is running.
    """
    return {"alive": True}
