"""GPU monitoring utilities using nvidia-smi."""

import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GPUStats:
    """GPU statistics."""

    gpu_id: int
    name: str
    temperature: float  # Celsius
    gpu_utilization: float  # Percentage
    memory_utilization: float  # Percentage
    memory_used_mb: float
    memory_total_mb: float
    power_watts: float
    clock_graphics_mhz: float
    clock_memory_mhz: float
    timestamp: float


class GPUMonitor:
    """Monitor NVIDIA GPU metrics."""

    # nvidia-smi query fields
    QUERY_FIELDS = [
        "index",
        "name",
        "temperature.gpu",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "power.draw",
        "clocks.gr",
        "clocks.mem",
    ]

    def __init__(self):
        """Initialize GPU monitor."""
        self._nvidia_smi_available: Optional[bool] = None

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available.

        Returns:
            True if nvidia-smi is available.
        """
        if self._nvidia_smi_available is not None:
            return self._nvidia_smi_available

        try:
            result = subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self._nvidia_smi_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._nvidia_smi_available = False
            logger.warning("nvidia_smi_not_available")

        return self._nvidia_smi_available

    def get_stats(self, gpu_id: int = 0) -> Optional[dict]:
        """Get GPU statistics.

        Args:
            gpu_id: GPU device ID.

        Returns:
            Dictionary with GPU stats or None if unavailable.
        """
        if not self._check_nvidia_smi():
            return None

        try:
            query = ",".join(self.QUERY_FIELDS)
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                    f"--id={gpu_id}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                logger.warning(
                    "nvidia_smi_query_failed",
                    stderr=result.stderr,
                )
                return None

            # Parse output
            parts = [p.strip() for p in result.stdout.strip().split(",")]

            if len(parts) < len(self.QUERY_FIELDS):
                logger.warning(
                    "nvidia_smi_unexpected_output",
                    output=result.stdout,
                )
                return None

            return {
                "gpu_id": int(parts[0]),
                "name": parts[1],
                "temperature": float(parts[2]),
                "gpu_utilization": float(parts[3]),
                "memory_utilization": float(parts[4]),
                "memory_used_mb": float(parts[5]),
                "memory_total_mb": float(parts[6]),
                "power_watts": float(parts[7]) if parts[7] != "[N/A]" else 0.0,
                "clock_graphics_mhz": float(parts[8]) if parts[8] != "[N/A]" else 0.0,
                "clock_memory_mhz": float(parts[9]) if parts[9] != "[N/A]" else 0.0,
                "timestamp": time.time(),
            }

        except subprocess.TimeoutExpired:
            logger.warning("nvidia_smi_timeout")
            return None
        except (ValueError, IndexError) as e:
            logger.warning("nvidia_smi_parse_error", error=str(e))
            return None

    def get_all_stats(self) -> list[dict]:
        """Get statistics for all GPUs.

        Returns:
            List of GPU stats dictionaries.
        """
        if not self._check_nvidia_smi():
            return []

        try:
            # Get number of GPUs
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return []

            num_gpus = len(result.stdout.strip().split("\n"))

            stats = []
            for gpu_id in range(num_gpus):
                gpu_stats = self.get_stats(gpu_id)
                if gpu_stats:
                    stats.append(gpu_stats)

            return stats

        except subprocess.TimeoutExpired:
            return []

    def get_memory_info(self, gpu_id: int = 0) -> Optional[dict]:
        """Get GPU memory information.

        Args:
            gpu_id: GPU device ID.

        Returns:
            Dictionary with memory info or None.
        """
        stats = self.get_stats(gpu_id)
        if not stats:
            return None

        return {
            "used_mb": stats["memory_used_mb"],
            "total_mb": stats["memory_total_mb"],
            "free_mb": stats["memory_total_mb"] - stats["memory_used_mb"],
            "utilization": stats["memory_utilization"],
        }

    def is_healthy(
        self,
        gpu_id: int = 0,
        max_temperature: float = 85.0,
        max_memory_utilization: float = 98.0,
    ) -> tuple[bool, Optional[str]]:
        """Check if GPU is healthy.

        Args:
            gpu_id: GPU device ID.
            max_temperature: Maximum acceptable temperature.
            max_memory_utilization: Maximum acceptable memory utilization.

        Returns:
            Tuple of (healthy, reason).
        """
        stats = self.get_stats(gpu_id)

        if not stats:
            return False, "Unable to query GPU"

        if stats["temperature"] > max_temperature:
            return False, f"Temperature too high: {stats['temperature']}°C"

        if stats["memory_utilization"] > max_memory_utilization:
            return False, f"Memory utilization too high: {stats['memory_utilization']}%"

        return True, None


class GPUMetricsCollector:
    """Collect GPU metrics for Prometheus."""

    def __init__(self, interval: float = 10.0):
        """Initialize metrics collector.

        Args:
            interval: Collection interval in seconds.
        """
        self.interval = interval
        self.monitor = GPUMonitor()
        self._running = False

    async def start(self) -> None:
        """Start collecting metrics."""
        import asyncio

        from app.utils.metrics import record_gpu_metrics

        self._running = True
        logger.info("gpu_metrics_collector_started", interval=self.interval)

        while self._running:
            try:
                stats_list = self.monitor.get_all_stats()

                for stats in stats_list:
                    record_gpu_metrics(
                        gpu_id=stats["gpu_id"],
                        memory_used=stats["memory_used_mb"] * 1024 * 1024,
                        memory_total=stats["memory_total_mb"] * 1024 * 1024,
                        utilization=stats["gpu_utilization"],
                        temperature=stats["temperature"],
                        power=stats["power_watts"],
                    )

            except Exception as e:
                logger.warning("gpu_metrics_collection_failed", error=str(e))

            await asyncio.sleep(self.interval)

    def stop(self) -> None:
        """Stop collecting metrics."""
        self._running = False
        logger.info("gpu_metrics_collector_stopped")
