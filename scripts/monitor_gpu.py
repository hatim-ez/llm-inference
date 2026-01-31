#!/usr/bin/env python3
"""GPU monitoring script with real-time display and alerts."""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class GPUStats:
    """GPU statistics snapshot."""

    timestamp: str
    gpu_id: int
    name: str
    temperature: float
    gpu_utilization: float
    memory_utilization: float
    memory_used_mb: float
    memory_total_mb: float
    power_watts: float
    clock_graphics_mhz: float
    clock_memory_mhz: float


def get_gpu_stats() -> Optional[list[GPUStats]]:
    """Get current GPU statistics for all GPUs."""
    try:
        query = (
            "index,name,temperature.gpu,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,power.draw,clocks.gr,clocks.mem"
        )
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return None

        stats_list = []
        timestamp = datetime.now().isoformat()

        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 10:
                continue

            stats = GPUStats(
                timestamp=timestamp,
                gpu_id=int(parts[0]),
                name=parts[1],
                temperature=float(parts[2]),
                gpu_utilization=float(parts[3]),
                memory_utilization=float(parts[4]),
                memory_used_mb=float(parts[5]),
                memory_total_mb=float(parts[6]),
                power_watts=float(parts[7]) if parts[7] != "[N/A]" else 0,
                clock_graphics_mhz=float(parts[8]) if parts[8] != "[N/A]" else 0,
                clock_memory_mhz=float(parts[9]) if parts[9] != "[N/A]" else 0,
            )
            stats_list.append(stats)

        return stats_list

    except Exception as e:
        print(f"Error querying GPU: {e}", file=sys.stderr)
        return None


def format_bar(value: float, max_value: float = 100, width: int = 30) -> str:
    """Format a value as a progress bar."""
    filled = int(width * min(value, max_value) / max_value)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def display_stats(stats_list: list[GPUStats], alerts: dict) -> None:
    """Display GPU statistics with colored output."""
    # Clear screen
    print("\033[2J\033[H", end="")

    print("=" * 70)
    print(f"GPU Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for stats in stats_list:
        print(f"\nGPU {stats.gpu_id}: {stats.name}")
        print("-" * 50)

        # Temperature
        temp_color = ""
        if stats.temperature >= alerts["temp_critical"]:
            temp_color = "\033[91m"  # Red
        elif stats.temperature >= alerts["temp_warning"]:
            temp_color = "\033[93m"  # Yellow
        else:
            temp_color = "\033[92m"  # Green
        print(
            f"  Temperature: {temp_color}{stats.temperature:5.1f}°C\033[0m "
            f"[{format_bar(stats.temperature, 100, 20)}]"
        )

        # GPU Utilization
        print(
            f"  GPU Util:    {stats.gpu_utilization:5.1f}%  "
            f"[{format_bar(stats.gpu_utilization, 100, 20)}]"
        )

        # Memory
        mem_pct = (stats.memory_used_mb / stats.memory_total_mb) * 100
        mem_color = ""
        if mem_pct >= alerts["mem_critical"]:
            mem_color = "\033[91m"
        elif mem_pct >= alerts["mem_warning"]:
            mem_color = "\033[93m"
        else:
            mem_color = "\033[92m"
        print(
            f"  Memory:      {mem_color}{stats.memory_used_mb:5.0f}MB / {stats.memory_total_mb:.0f}MB "
            f"({mem_pct:.1f}%)\033[0m"
        )
        print(f"               [{format_bar(mem_pct, 100, 20)}]")

        # Power
        print(f"  Power:       {stats.power_watts:5.1f}W")

        # Clocks
        print(
            f"  Clocks:      Graphics: {stats.clock_graphics_mhz:.0f}MHz, "
            f"Memory: {stats.clock_memory_mhz:.0f}MHz"
        )

        # Alerts
        if stats.temperature >= alerts["temp_critical"]:
            print(f"\n  \033[91m⚠️  ALERT: Temperature critical!\033[0m")
        elif stats.temperature >= alerts["temp_warning"]:
            print(f"\n  \033[93m⚠️  WARNING: Temperature high\033[0m")

        if mem_pct >= alerts["mem_critical"]:
            print(f"  \033[91m⚠️  ALERT: Memory nearly exhausted!\033[0m")
        elif mem_pct >= alerts["mem_warning"]:
            print(f"  \033[93m⚠️  WARNING: Memory usage high\033[0m")

    print("\n" + "-" * 70)
    print("Press Ctrl+C to exit")


def save_stats(stats_list: list[GPUStats], output_file: Path) -> None:
    """Append stats to JSON log file."""
    data = [
        {
            "timestamp": s.timestamp,
            "gpu_id": s.gpu_id,
            "temperature": s.temperature,
            "gpu_utilization": s.gpu_utilization,
            "memory_used_mb": s.memory_used_mb,
            "memory_total_mb": s.memory_total_mb,
            "power_watts": s.power_watts,
        }
        for s in stats_list
    ]

    # Append to file
    with open(output_file, "a") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor GPU metrics")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Update interval in seconds",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Log file to save stats (JSON lines format)",
    )
    parser.add_argument(
        "--temp-warning",
        type=float,
        default=80,
        help="Temperature warning threshold (°C)",
    )
    parser.add_argument(
        "--temp-critical",
        type=float,
        default=85,
        help="Temperature critical threshold (°C)",
    )
    parser.add_argument(
        "--mem-warning",
        type=float,
        default=90,
        help="Memory warning threshold (%)",
    )
    parser.add_argument(
        "--mem-critical",
        type=float,
        default=95,
        help="Memory critical threshold (%)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print stats once and exit",
    )

    args = parser.parse_args()

    alerts = {
        "temp_warning": args.temp_warning,
        "temp_critical": args.temp_critical,
        "mem_warning": args.mem_warning,
        "mem_critical": args.mem_critical,
    }

    output_file = Path(args.log) if args.log else None

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Logging to: {output_file}")

    try:
        while True:
            stats = get_gpu_stats()

            if stats is None:
                print("Failed to get GPU stats. Is nvidia-smi available?")
                if args.once:
                    sys.exit(1)
                time.sleep(args.interval)
                continue

            display_stats(stats, alerts)

            if output_file:
                save_stats(stats, output_file)

            if args.once:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    main()
