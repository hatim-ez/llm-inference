"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Model Configuration
    model_name: str = Field(
        default="Llama-3.2-11B-Vision-Instruct",
        description="Name of the model",
    )
    model_path: str = Field(
        default="/home/ubuntu/models/llama-3.2-11b-vision",
        description="Path to the model weights",
    )

    # vLLM Configuration
    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for tensor parallelism",
    )
    gpu_memory_utilization: float = Field(
        default=0.90,
        ge=0.1,
        le=0.99,
        description="Fraction of GPU memory to use",
    )
    max_model_length: int = Field(
        default=4096,
        ge=512,
        description="Maximum sequence length",
    )
    max_num_seqs: int = Field(
        default=8,
        ge=1,
        description="Maximum number of sequences to batch together (8 recommended for T4)",
    )
    enforce_eager: bool = Field(
        default=False,
        description="Whether to enforce eager execution",
    )
    swap_space: int = Field(
        default=4,
        ge=0,
        description="CPU swap space in GB",
    )

    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API host address",
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API port",
    )
    rate_limit: int = Field(
        default=1000,
        ge=1,
        description="Requests per minute rate limit",
    )
    rate_limit_burst: int = Field(
        default=100,
        ge=1,
        description="Maximum burst requests per second",
    )
    request_timeout: int = Field(
        default=300,
        ge=1,
        description="Request timeout in seconds",
    )

    # Security
    require_api_key: bool = Field(
        default=False,
        description="Whether to require API key authentication",
    )
    api_key_secret: Optional[str] = Field(
        default=None,
        description="Secret key for API authentication",
    )

    # Monitoring
    enable_metrics: bool = Field(
        default=True,
        description="Whether to enable Prometheus metrics",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: str = Field(
        default="json",
        description="Log format (json or text)",
    )

    # AWS Configuration
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region",
    )
    s3_bucket: Optional[str] = Field(
        default=None,
        description="S3 bucket for logs",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
